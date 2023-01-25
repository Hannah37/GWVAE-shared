"""GWVAE layers."""
import torch
from torch_sparse import spspmm, spmm
from einops import rearrange, reduce, repeat
import torch.nn.functional as F
from torch_geometric.utils import *
from torch_scatter import scatter_add
import torch.nn as nn
import pygsp
import networkx as nx

"""
 spspmm: sparse matrix * sparse matrix
 spmm: sparse matrix * dense matrix
"""

"""
    param ncount: Number of nodes.
    param feature_number: Number of features.
    param class_number: Number of classes.
"""
class GWVAE(torch.nn.Module):
    def __init__(self, args, ncount, nclass, nfeature, device):
        super(GWVAE, self).__init__()
        self.args = args
        self.ncount = ncount
        self.nclass = nclass
        self.nfeature = nfeature
        self.device = device

        "random scales init"
        self.scales = nn.ParameterList([nn.Parameter(torch.rand(1), requires_grad=True) for _ in range(self.args.nscale)])

        "encoder layers"
        self.conv_x1 = WaveletGraphConv(self.ncount, self.nclass, self.args.dropout, self.scales, self.device)
        self.conv_x2 = WaveletGraphConv(self.ncount, self.nclass, self.args.dropout, self.scales, self.device)
        self.conv_a1 = WaveletGraphConv(self.ncount, self.nclass, self.args.dropout, self.scales, self.device)
        self.conv_a2 = WaveletGraphConv(self.ncount, self.nclass, self.args.dropout, self.scales, self.device)
        
        "decoder layers"
        self.fc_x = nn.Linear(self.ncount + self.nclass, self.ncount * self.nfeature)
        self.fc_a = nn.Linear(self.ncount + self.nclass, self.ncount)


    def get_scales(self):
        return self.scales

    def sampling(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)

        # return z sample
        return eps.mul(std).add_(mu) 

    def decoder_x(self, z, y, X):
        """
        z: N x 1 
        y: C x 1
        """

        z = torch.cat([z, y], dim=1)
        z = self.fc_x(z)

        recon_x = []
        for idx, x in enumerate(X):
            num_node = x.shape[0]
            num_drop = self.nfeature * (self.ncount - num_node)

            if num_drop > 0:
                z_ = z[idx, :-num_drop]
            else:
                z_ = z[idx]

            z_ = rearrange(z_, '(n f) -> n f', n=num_node, f=self.nfeature)
            z_ = F.sigmoid(z_)
            recon_x.append(z_)

        return recon_x

    def decoder_a(self, z, y, X):
        """
        z: N x 1 
        y: C x 1
        """
        z = torch.cat([z, y], dim=1)
        z = self.fc_a(z)

        recon_a = []
        for idx, x in enumerate(X):
            num_node = x.shape[0]
            num_drop = self.ncount - num_node

            if num_drop > 0:
                z_ = z[idx, :-num_drop]
            else:
                z_ = z[idx]

            z_ = z_.unsqueeze(dim=1)
            # z_ = F.sigmoid(torch.mm(z_, z_.T))
            z_ = F.sigmoid(z_ * z_.T)
            recon_a.append(z_)

        return recon_a


    def forward(self, batch):

        # print(batch)

        # phi_indices, phi_values, phi_inverse_indices, phi_inverse_values, graph, feature, y
        """
        param phi_indices: Sparse wavelet matrix index pairs.
        param phi_values: Sparse wavelet matrix values.
        param phi_inverse_indices: Inverse wavelet matrix index pairs.
        param phi_inverse_values: Inverse wavelet matrix values.
        """

        x_mu = self.conv_x1(batch['graphs'],
                            batch['features'],
                            batch['networkX_graphs'])

        x_logvar = self.conv_x2(batch['graphs'],
                            batch['features'],
                            batch['networkX_graphs'])

        a_mu = self.conv_a1(batch['graphs'],
                            batch['features'],
                            batch['networkX_graphs'])

        a_logvar = self.conv_a2(batch['graphs'],
                            batch['features'],
                            batch['networkX_graphs'])
        
        xz = self.sampling(x_mu, x_logvar)
        az = self.sampling(a_mu, a_logvar)

        ys = torch.stack(batch['ys']) 
        recon_x = self.decoder_x(xz, ys, batch['features'])
        recon_a = self.decoder_a(az, ys, batch['features'])

        return recon_x, x_mu, x_logvar, recon_a, a_mu, a_logvar

class WaveletGraphConv(torch.nn.Module):
    """
    param ncount: Number of nodes.
    """
    def __init__(self, ncount, nclass, dropout, scales, device):
        super(WaveletGraphConv, self).__init__()
        self.ncount = ncount 
        self.nclass = nclass
        self.device = device
        self.dropout = dropout 
        self.scales = scales
        self.define_parameters()
        self.init_parameters()

    def define_parameters(self):
        """
        Defining diagonal filter matrix and weight matrix F.
        """
        self.diag_weight_indices = torch.LongTensor([[node for node in range(self.ncount)],
                                                         [node for node in range(self.ncount)]])

        self.diag_weight_indices = self.diag_weight_indices.to(self.device)
        self.diag_weight_filter = torch.nn.Parameter(torch.Tensor(self.ncount, 1))

    def init_parameters(self):
        """
        Initializing the diagonal filter.
        """
        torch.nn.init.uniform_(self.diag_weight_filter, 0.9, 1.1)

    def get_wavelet(self, graph):
        num_node = graph.G.number_of_nodes()
        graph = pygsp.graphs.Graph(nx.adjacency_matrix(graph.G))

        wavelet, wavelet_inv = torch.zeros(num_node, num_node).cuda(), torch.zeros(num_node, num_node)
        for i in range(len(self.scales)):
            # ftr = pygsp.filters.Abspline(graph, Nf=1, scales=[self.scales[i]])
            ftr = pygsp.filters.Heat(graph, tau=[self.scales[i]])
            graph.estimate_lmax()
            w = ftr.compute_frame()
            # w_inv = ftr.inverse().compute_frame()
            wavelet += w
            # wavelet_inv += w_inv
        
        # wavelet = torch.tensor(wavelet, dtype=torch.float32).cuda()
        return wavelet, wavelet_inv

    def forward(self, A, X, G):
        """
        Forward propagation pass for a graph set with heterogeneous number of nodes
        :param phi_indices: Sparse wavelet matrix index pairs.
        :param phi_values: Sparse wavelet matrix values.
        :param phi_inverse_indices: Inverse wavelet matrix index pairs.
        :param phi_inverse_values: Inverse wavelet matrix values.
        :param dropout: Dropout rate.
        :return dropout_features: Filtered feature matrix extracted.
        """

        xf = [] # transform graph convolution into frequency domain
        for x, a, g in zip(X, A, G):
            num_node = x.shape[0]
            phi, _ = self.get_wavelet(g)

            # '''concat y'''
            # y = repeat(y, 'c -> n c', n = num_node)
            # x_ = torch.cat([x, y], dim=1)
            
            a_, _ = add_remaining_self_loops(a) # a = a + I
            row, col = a_[0], a_[1]
            a_weight = torch.ones(a_.shape[1]).to(self.device)
            deg = scatter_add(a_weight, row, dim=0, dim_size=num_node)
            deg_inv_sqrt = deg.pow_(-0.5)
            deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
            a_norm_weight = deg_inv_sqrt[row] * a_weight * deg_inv_sqrt[col]  # -D^{-1/2} * A * D^{-1/2}
            gc = spmm(a_, 
                    a_norm_weight, 
                    num_node, 
                    num_node, 
                    x) # A * (X||Y), shape: N * (F+C) # change x -> x_ if concat y

            # waxy = spmm(w_idx,
            #         w_val,
            #         num_node,
            #         num_node,
            #         gc) # wavelet * A * (X||Y)
            # waxy = torch.squeeze(waxy)
            
            waxy = torch.mm(phi.cuda(), gc) 

            waxy = F.pad(waxy, (0, 0, 0, self.ncount-num_node), "constant", 0) # shape: N_max * (F+C)
            xf.append(waxy)

        xf = torch.stack(xf) # .to(self.device) # shape: batch * N_max * (F+C)

        wgc = spmm(self.diag_weight_indices,
                self.diag_weight_filter.view(-1),
                self.ncount,
                self.ncount,
                xf)                   

        wgc = torch.sum(wgc, dim=-1) # sum all latent features, shape: batch * N_max
        wgc = F.dropout(wgc, p=self.dropout)

        return wgc