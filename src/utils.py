"""GWNN data reading utils."""

import os
import sys
import json
from re import I

import numpy as np
import pandas as pd
import networkx as nx
from scipy import sparse
from texttable import Texttable
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split as sk_train_test_split
from torch_geometric.datasets import TUDataset
from deepsnap.dataset import GraphDataset
from deepsnap.batch import Batch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.nn.functional as F

# sys.path.append('C:\\Users\\puser\\anaconda3\\envs\\python38_torch\\lib\\site-packages')
# print(sys.path)
import pygsp
# sys.path.insert(0, 'D:/user/puser/Documents/code/GWVAE/src/python_pygsp/pygsp/filters/')
# from python_pygsp.pygsp.filters import abspline
# from python_pygsp.pygsp.filters.filter import inverse
# import filter

def tab_printer(args):
    """
    Function to print the logs in a nice tabular format.
    :param args: Parameters used for the model.
    """
    args = vars(args)
    keys = sorted(args.keys())
    t = Texttable()
    t.add_rows([["Parameter", "Value"]])
    t.add_rows([[k.replace("_", " ").capitalize(), args[k]] for k in keys])
    print(t.draw())

def load_Enzyme(args):
    """
    Function to create an NX graph object.
    :return dataset: graphs.
    """
    dataset = TUDataset(root='/tmp/ENZYMES', name='ENZYMES')
    
    # Visualization
    # graphs = GraphDataset.pyg_to_graphs(dataset)
    # dataset = GraphDataset(graphs, task="graph", minimum_node_per_graph=0)
    # visualize(dataset)

    # print("#data:", len(dataset))
    # print("#class:", dataset.num_classes)
    # print("#node features:", dataset.num_node_features)
    # print("#is_undirected: ", dataset[0].is_undirected())

    nfeature = dataset.num_node_features
    nclass = dataset.num_classes

    labels = []
    ncount = -1
    for i in range (len(dataset)):
        labels.append(dataset[i].y.item())
        if ncount < dataset[i].x.shape[0]:
            ncount = dataset[i].x.shape[0]
    
    # train test split
    E_train, E_test, train_Y, test_Y = sk_train_test_split(dataset, 
                                                        labels, 
                                                        test_size=args.test_size,
                                                        random_state=args.seed,
                                                        stratify=labels)

    assert (len(E_train) == len(train_Y)) and (len(E_test) == len(test_Y))   
    
    train_Y = F.one_hot(torch.tensor(train_Y), num_classes = nclass)
    test_Y = F.one_hot(torch.tensor(test_Y), num_classes = nclass)
    # print(train_Y.shape) # 480 * 6
    # train_Y = train_Y.tolist()
    # test_Y = test_Y.tolist()


    train_graphset = GraphDataset.pyg_to_graphs(E_train)
    # train_graphs = GraphDataset(train_graphs, task="graph", minimum_node_per_graph=0)
    # visualize(train_dataset)

    test_graphset = GraphDataset.pyg_to_graphs(E_test)
    # test_graphs = GraphDataset(test_graphs, task="graph", minimum_node_per_graph=0)   

    # print('#E_train:', len(E_train))
    # print('#E_test:', len(E_test))
    # print('#Y_train:', len(Y_train))
    # print('#Y_test:', len(Y_test))
    # print(Y_train.count(1), Y_train.count(2), Y_train.count(3), Y_train.count(4), Y_train.count(5), Y_train.count(6))

    return train_graphset, test_graphset, train_Y, test_Y, ncount, nclass, nfeature

def visualize(dataset):
    color_mapping = {
        0 : 'black',
        1 : 'red',
        2 : 'green',
        3 : 'yellow',
        4 : 'blue',
        5 : 'purple',
    }
    num_graphs_i = 3
    num_graphs_j = 3
    fig, ax = plt.subplots(num_graphs_i, num_graphs_j, figsize=(11, 11))
    fig.suptitle("ENZYMES Graphs Visualization", fontsize=16)
    indices = np.random.choice(np.arange(0, len(dataset)), size=9, replace=False)
    indices = indices.reshape(3, 3)
    for i in range(num_graphs_i):
        for j in range(num_graphs_j):
            index = int(indices[i, j])
            G = dataset[index].G
            label = dataset[index].graph_label.item()
            pos = nx.spring_layout(G, seed=1)
            colors = [color_mapping[label]] * dataset[index].num_nodes
            nodes = nx.draw_networkx_nodes(G, pos=pos, cmap=plt.get_cmap('coolwarm'), \
                node_size=30, ax=ax[(i, j)], node_color=colors)
            nodes.set_edgecolor('black')
            nx.draw_networkx_edges(G, pos=pos, ax=ax[(i, j)])
            ax[(i, j)].set_title("Sample graph with lable {}".format(label), fontsize=13)
            ax[(i, j)].set_axis_off()
    plt.show()


def load_Enzyme_dataset(path):
    """
    Function to create an NX graph object.
    :return dataset: NetworkX graphs.
    """
    
    data_adj = np.loadtxt(os.path.join(path, 'ENZYMES/ENZYMES_A.txt'), delimiter=',').astype(int)
    data_node_att = np.loadtxt(os.path.join(path, 'ENZYMES/ENZYMES_node_attributes.txt'), delimiter=',')
    data_node_label = np.loadtxt(os.path.join(path, 'ENZYMES/ENZYMES_node_labels.txt'), delimiter=',').astype(int)
    data_graph_indicator = np.loadtxt(os.path.join(path, 'ENZYMES/ENZYMES_graph_indicator.txt'), delimiter=',').astype(int)
    data_graph_labels = np.loadtxt(os.path.join(path, 'ENZYMES/ENZYMES_graph_labels.txt'), delimiter=',').astype(int) # 6 classes

    data_tuple = list(map(tuple, data_adj))

    G = nx.Graph()
    # add edges
    G.add_edges_from(data_tuple)
    # add node attributes
    for i in range(data_node_att.shape[0]):
        G.add_node(i+1, feature = data_node_att[i])
        G.add_node(i+1, label = data_node_label[i])

    # split into graphs
    graph_num = len(data_graph_labels) # 600
    node_list = np.arange(data_graph_indicator.shape[0])+1
    dataset = [] 
    for i in range(graph_num):
        # find the nodes for each graph
        nodes = node_list[data_graph_indicator==i+1]
        G_sub = G.subgraph(nodes)
        G_sub.graph['label'] = data_graph_labels[i]
        print(G_sub.number_of_nodes())
        # m = nx.attr_matrix(G_sub, node_attr = "feature")
        print(G_sub.nodes)
        # print([list(G_sub.nodes[i].values()) for i in range(G_sub.number_of_nodes())])
        dataset.append(G_sub)
        
    return dataset, data_graph_labels 


def feature_reader(path):
    """
    Reading the feature matrix stored as JSON from the disk.
    :param feature_path: Path to the JSON file.
    :return features: Feature sparse COO matrix.
    """
    features = json.load(open(path))
    index_1 = [int(k) for k, v in features.items() for fet in v]
    index_2 = [int(fet) for k, v in features.items() for fet in v]
    values = [1.0]*len(index_1)
    nodes = [int(k) for k, v in features.items()]
    node_count = max(nodes)+1
    feature_count = max(index_2)+1
    features = sparse.coo_matrix((values, (index_1, index_2)),
                                 shape=(node_count, feature_count),
                                 dtype=np.float32)
    return features

def graph_reader(path):
    """
    Function to create an NX graph object.
    :return graph: NetworkX graph.
    """

    graph = nx.from_edgelist(pd.read_csv(path).values.tolist())
    graph.remove_edges_from(nx.selfloop_edges(graph))
    return graph

def target_reader(path):
    """
    Reading thetarget vector to a numpy column vector.
    :param path: Path to the target csv.
    :return target: Target vector.
    """
    target = np.array(pd.read_csv(path)["target"])
    return target

def save_logs(args, logs):
    """
    Save the logs at the path.
    :param args: Arguments objects.
    :param logs: Log dictionary.
    """
    with open(args.log_path, "w") as f:
        json.dump(logs, f)
        f.write("\n")

class WaveletSparsifier(object):
    """
    Object to sparsify the wavelet coefficients for a graph.
    """
    def __init__(self, graph, scale):
        """
        :param graph: NetworkX graph object.
        :param scale: Kernel scale length parameter.
        """
        self.graph = graph
        self.pygsp_graph = pygsp.graphs.Graph(nx.adjacency_matrix(self.graph))
        self.pygsp_graph.estimate_lmax()
        self.scales = [-scale, scale]
        self.phi_matrices = [] # contain the wavelet and inverse wavelet matrices.

    def calculate_all_wavelets(self):
        """
        Graph wavelet calculation.
        """

        self.filter = pygsp.filters.Abspline(self.pygsp_graph, Nf=1, scales=[2])
        self.pygsp_graph.estimate_lmax()
        # self.filter = pygsp.filters.Heat(self.pygsp_graph, tau=[1.0])

        frame = self.filter.compute_frame()
        h = self.filter.inverse()
        frame_inv = h.compute_frame()
        # print(self.filter.shape, h.shape)
        # print(h[0])
        # print(frame.shape, frame_inv.shape)


        # self.chebyshev = pygsp.filters.approximations.compute_cheby_coeff(self.filter,
        #                                                                 m=self.approximation_order)
        # sparsified_wavelets = self.calculate_wavelet() # shape: n x n
        # self.phi_matrices.append(sparsified_wavelets)

        # print("\nWavelet calculation and sparsification started.\n")
        # for _, scale in enumerate(self.scales):
        #     self.heat_filter = pygsp.filters.Heat(self.pygsp_graph,
        #                                           tau=[scale])
        #     self.chebyshev = pygsp.filters.approximations.compute_cheby_coeff(self.heat_filter,
        #                                                                     m=self.approximation_order)
        #     sparsified_wavelets = self.calculate_wavelet() # shape: n x n
        #     print('sparsified_wavelets: ', sparsified_wavelets.shape)
        #     self.phi_matrices.append(sparsified_wavelets)

        self.normalize_matrices()
        # self.calculate_density()

    def calculate_wavelet(self):
        """
        Creating sparse wavelets.
        :return remaining_waves: Sparse matrix of attenuated wavelets.
        """
        impulse = np.eye(self.graph.number_of_nodes(), dtype=int)
        wavelet_coefficients = pygsp.filters.approximations.cheby_op(self.pygsp_graph,
                                                                     self.chebyshev,
                                                                     impulse)                                                 
        wavelet_coefficients[wavelet_coefficients < self.tolerance] = 0
        ind_1, ind_2 = wavelet_coefficients.nonzero()
        n_count = self.graph.number_of_nodes()
        # print('graph info: ', print(self.graph)) # Graph with 24 nodes and 43 edges
        print('wavelet_coefficients.shape: ', wavelet_coefficients.shape) #(72, 24)
        remaining_waves = sparse.csr_matrix((wavelet_coefficients[ind_1, ind_2], (ind_1, ind_2)),
                                            shape=(n_count, n_count),
                                            dtype=np.float32) ##
        return remaining_waves

    def normalize_matrices(self):
        """
        Normalizing the wavelet and inverse wavelet matrices.
        """
        # print("\nNormalizing the sparsified wavelets.\n")
        for i, phi_matrix in enumerate(self.phi_matrices):
            self.phi_matrices[i] = normalize(self.phi_matrices[i], norm='l1', axis=1)

    def calculate_density(self):
        """
        Calculating the density of the sparsified wavelet matrices.
        """
        wavelet_density = len(self.phi_matrices[0].nonzero()[0])/(self.graph.number_of_nodes()**2)
        wavelet_density = str(round(100*wavelet_density, 2))
        inverse_wavelet_density = len(self.phi_matrices[1].nonzero()[0])/(self.graph.number_of_nodes()**2)
        inverse_wavelet_density = str(round(100*inverse_wavelet_density, 2))
        print("Density of wavelets: " + wavelet_density+"%.")
        print("Density of inverse wavelets: " + inverse_wavelet_density+"%.\n")

