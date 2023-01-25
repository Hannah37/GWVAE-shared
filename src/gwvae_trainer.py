import sys
import time
import math
from metrics.stats import *
from tqdm import tqdm
import networkx as nx       
from utils import *
from gwvae import GWVAE

import torch
import torch.nn.functional as F
from torch_sparse import spspmm, spmm
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx

from tensorboardX import SummaryWriter

class GWVAE_Trainer(object):
    def __init__(self, args, train_loader, test_loader, ncount, nclass, nfeature):
        self.args = args
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.ncount = ncount
        self.nclass = nclass
        self.nfeature = nfeature
        self.device = torch.device('cuda')
        self.setup_model()

    def setup_model(self):
        self.model = GWVAE(self.args, self.ncount, self.nclass, self.nfeature, self.device)
        self.model = self.model.to(self.device)

    def get_wavelet_inv(self, batch):
        wavelet, wavelet_inv = [], []
        for graph in batch: 
            num_node = graph.G.number_of_nodes()
            scale = self.model.get_scales()
            graph = pygsp.graphs.Graph(nx.adjacency_matrix(graph.G))

            wavelet_, wavelet_inv_ = torch.zeros(num_node, num_node), torch.zeros(num_node, num_node).cuda()
            for i in range(self.args.nscale):
                # ftr = pygsp.filters.Abspline(graph, Nf=1, scales=[scale[i]])
                ftr = pygsp.filters.Heat(graph, tau=[scale[i]])
                graph.estimate_lmax()
                # w = ftr.compute_frame()
                w_inv = ftr.inverse().compute_frame()
                # wavelet_ += w
                wavelet_inv_ += w_inv
            wavelet_inv_ = torch.tensor(wavelet_inv_, dtype=torch.float32).cuda()
            # wavelet.append(wavelet_)
            wavelet_inv.append(wavelet_inv_)

            
        return wavelet, wavelet_inv

    def a_loss(self, recon, original, nwx_graph, wavelet_inv, MU, log_VAR): 
        CE, KLD = 0, 0
        for recon_data, data, g, w_inv, mu, log_var in zip(recon, original, nwx_graph, wavelet_inv, MU, log_VAR):
            recon_data = torch.mm(w_inv, recon_data) # N x F 

            data = torch.tensor(g.node_feature).cuda() # N x F           
            data = torch.where(data==1)[1] # N

            ce_loss = torch.nn.CrossEntropyLoss(reduction='mean') 
            CE += ce_loss(recon_data, data)
            KLD += -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        return CE + KLD

    def x_loss(self, recon, original, wavelet_inv, MU, log_VAR): 
        CE, KLD = 0, 0
        for recon_data, data, w_inv, mu, log_var in zip(recon, original, wavelet_inv, MU, log_VAR):
            recon_data = torch.mm(w_inv, recon_data) # N x N

            ce_loss = torch.nn.CrossEntropyLoss(reduction='mean') 
            CE += ce_loss(recon_data, data)
            KLD += -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        return CE + KLD
        
    def fit(self):
        tb_dir = self.args.log_path + '/logs/{}'.format(time.strftime("%b%d_%d-%H-%M", time.localtime()))
        logger = SummaryWriter(log_dir=tb_dir)

        print("Training.\n")
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                      lr=self.args.learning_rate,
                                      weight_decay=self.args.weight_decay)

        num_data = len(self.train_loader.dataset)
        niters_per_epoch = int(math.ceil(num_data*1.0 // self.args.batch_size))
        


        self.time = time.time()
        for epoch in range(self.args.epochs): 
            bar_format = '{desc}[{elapsed}<{remaining}]'
            pbar = tqdm(range(niters_per_epoch), bar_format=bar_format) #, file=sys.stdout)

            self.model.train()
            dataloader = iter(self.train_loader)
            total_loss = 0
            for idx in pbar:
                self.optimizer.zero_grad()
                batch = dataloader.next()
               
                recon_x, x_mu, x_logvar, recon_a, a_mu, a_logvar = self.model(batch)
                _, w_inv = self.get_wavelet_inv(batch['networkX_graphs'])

                loss = self.x_loss(recon_x, batch['features'], w_inv, x_mu, x_logvar) \
                    + self.a_loss(recon_a, batch['graphs'], batch['networkX_graphs'], w_inv, a_mu, a_logvar)
            
                loss.backward()

                # for param in self.model.parameters():
                #     print(param)
                #     print(param.grad.data.sum())

                # # start debugger
                # import pdb; pdb.set_trace()

                self.optimizer.step()
                total_loss += loss.item() / batch['num_graphs']

                curr_lr = self.optimizer.param_groups[0]['lr']

                print_str = 'Epoch{}/{}'.format(epoch, self.args.epochs) \
                        + ' Iter{}/{}:'.format(idx + 1, niters_per_epoch) \
                        + ' lr=%.2e' % curr_lr \
                        + ' loss=%.2f' % (loss.item() / batch['num_graphs'])     
                pbar.set_description(print_str, refresh=False)

                # for i, (name, param) in enumerate(self.model.named_parameters()):
                #     if (param.requires_grad) and (i<5):
                #         print(name, param.data)

            scales = self.model.get_scales()
            for i in range(len(scales)):
                logger.add_scalar('scale_' + str(i), scales[i].item(), epoch)

            logger.add_scalar('train_loss', round(total_loss, 2), epoch)
            
            self.model.eval()
            with torch.no_grad():
                test_batch = next(iter(self.test_loader))
                recon_x, x_mu, x_logvar, recon_a, a_mu, a_logvar = self.model(test_batch)
                test_loss = self.test_loss(test_batch, recon_x, x_mu, x_logvar, recon_a, a_mu, a_logvar)
                logger.add_scalar('test_loss', round(test_loss.item(), 2), epoch)

                # pred_ls = self.reconstruct_inv(test_batch, recon_x, recon_a)
                # mmd_degree = self.MMD_degree(pred_ls, test_batch['networkX_graphs'])

            if (epoch% 10 == 0) or (epoch == (self.args.epochs - 1)):
                torch.save(self.model.state_dict(), os.path.join(self.args.log_path, 'snapshot', 'epoch_' + str(epoch) +'.pt')) 


    def test_loss(self, test_batch, recon_x, x_mu, x_logvar, recon_a, a_mu, a_logvar ):
        _, w_inv = self.get_wavelet_inv(test_batch['networkX_graphs'])

        loss = self.x_loss(recon_x, test_batch['features'], w_inv, x_mu, x_logvar) \
                + self.a_loss(recon_a, test_batch['graphs'], test_batch['networkX_graphs'], w_inv, a_mu, a_logvar)
        loss /= test_batch['num_graphs']
        return loss
    
    def reconstruct_inv(self, test_batch, recon_x, recon_a):
        pred_ls = []
        for w_inv, w_inv_idx, x, a in zip(test_batch['inv_wavelets'], test_batch['inv_wavelet_indices'], recon_x, recon_a):
            num_node = x.shape[0]
            edge_pred = spmm(w_inv_idx, 
                        w_inv, 
                        num_node, 
                        num_node, 
                        a) 
            node_pred = spmm(w_inv_idx, 
                        w_inv, 
                        num_node, 
                        num_node, 
                        x) 

            # print('p edge: ', edge_pred, edge_pred.shape)
            # print('p node: ', node_pred, node_pred.shape)
            node_pred, edge_pred = node_pred.squeeze(), edge_pred.squeeze()

            edge_pred_sparse = edge_pred.to_sparse()
            edge_pred_indices, edge_pred_val = edge_pred_sparse.indices(), edge_pred_sparse.values()
            # print('edge_pred_indices: ', edge_pred_indices)
            

            # recon = Data(x=node_pred, edge_index=edge_pred_indices, edge_attr=edge_pred_val)
            # networkX_recon = to_networkx(recon, node_attrs=["x"], edge_attrs=["edge_attr"])
            # pred_ls.append(networkX_recon)
        # return edge and node in graph domain
        return pred_ls

    def MMD_degree(self, pred_ls, gt_ls):        
        mmd = degree_stats(pred_ls, gt_ls)
        print('mmd: ', mmd)
        return mmd
        

    # def MMD_clustering(self):
    # def MMD_orbit(self):
    # def visualize(self):
