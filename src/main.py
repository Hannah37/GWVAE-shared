"""Running GWVAE."""
import torch
import numpy as np
from utils import *
from gwvae_trainer import GWVAE_Trainer
from param_parser import parameter_parser
from torch.utils.data import DataLoader, Dataset
import warnings

class GWVAE_Dataset(Dataset): 
    def __init__(self, graph, feature, y, networkX_graph):
        self.graph = graph
        self.feature = feature
        self.y = y
        self.networkX_graph = networkX_graph
        self.device = torch.device('cuda')

    def __len__(self): 
        return len(self.y)

    def __getitem__(self, idx): 
        graph = self.graph[idx].to(self.device)
        feature = self.feature[idx].to(self.device)
        y = self.y[idx].to(self.device)
        networkX_graph = self.networkX_graph[idx]

        sample = {'graph': graph, 'feature': feature, 'y': y, 'networkX_graph': networkX_graph}
        return sample

def GWVAE_collate_fn(samples):
    networkX_graphs = [sample['networkX_graph'] for sample in samples]
    graphs = [sample['graph'] for sample in samples]
    features = [sample['feature'] for sample in samples]
    ys = [sample['y'] for sample in samples]

    batch = {
            'graphs': graphs, 
            'features': features, 
            'ys': ys,
            'num_graphs': len(samples),
            'networkX_graphs': networkX_graphs
            }
    return batch

def main():
    warnings.filterwarnings("ignore")

    args = parameter_parser()
    torch.manual_seed(args.seed) 
    torch.cuda.manual_seed(args.seed)

    # tab_printer(args)
    train_graphset, test_graphset, train_Y, test_Y, ncount, nclass, nfeature = load_Enzyme(args)

    train_graph, train_feature, train_networkX, test_graph, test_feature, test_networkX = [], [], [], [], [], []

    for g in train_graphset:
        # print(g.G) # Graph with 24 nodes and 43 edges
        # print(g) # Graph(G=[], edge_index=[2, 86], edge_label_index=[2, 86], graph_label=[1], node_feature=[24, 3], node_label_index=[24])
        # print(g.edge_index)
        # print(g.edge_label_index)
        # print(g.edge_index)
        # train_networkX.append(g.G)
        train_networkX.append(g)
        train_graph.append(g.edge_index)
        train_feature.append(g.node_feature)
    

    for g in test_graphset:
        test_graph.append(g.edge_index)
        test_feature.append(g.node_feature)
        test_networkX.append(g)

    train_data = GWVAE_Dataset(train_graph, train_feature, train_Y, train_networkX)
    train_loader = DataLoader(train_data, collate_fn=GWVAE_collate_fn, batch_size=args.batch_size, shuffle=True)

    test_data = GWVAE_Dataset(test_graph, test_feature, test_Y, test_networkX)
    test_loader = DataLoader(test_data, collate_fn=GWVAE_collate_fn, batch_size=test_Y.shape[0], shuffle=False)

    trainer = GWVAE_Trainer(args, train_loader, test_loader, ncount, nclass, nfeature)
    trainer.fit()
    # tab_printer(args)
    
if __name__ == "__main__":
    main()
