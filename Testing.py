import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv
from torch_geometric.nn import SAGEConv
import networkx as nx
import random
import numpy as np
from torch_geometric.utils import add_self_loops
import matplotlib.pyplot as plt
from itertools import product
def create_random_graph(num_nodes, prob):
    num_nodes = num_nodes
    G = nx.fast_gnp_random_graph(num_nodes, prob)
    return G
def solve_graph(G):
    node_colors, _ = welsh_powell_coloring(G)
    color_list = [node_colors.get(node, -1) for node in G.nodes()]
    return G, color_list
def welsh_powell_coloring(G):
    vertices = sorted(list(G.nodes()), key=lambda x: G.degree(x), reverse=True)
    colors = {}
    colors[vertices[0]] = 1
    for vertex in vertices[1:]:
        neighbor_colors = [colors.get(neigh) for neigh in G.neighbors(vertex)]
        available_colors = set(range(1, len(vertices) + 1)) - set(neighbor_colors)
        colors[vertex] = min(available_colors)
    num_colors = len(set(colors.values()))
    return colors, num_colors
def generate_dataset(num_iterations):
    adj_matrices_list = []
    node_feat_matrices_list = []
    labels_list = []
    for i in range(num_iterations):
        G = create_random_graph(15, 0.2)
        _, color_list = solve_graph(G)
        adj_matrix = nx.to_numpy_array(G)
        node_feat_matrix = np.array([G.degree(node) for node in G.nodes()]).reshape(-1, 1)
        adj_matrices_list.append(adj_matrix)
        node_feat_matrices_list.append(node_feat_matrix)
        labels_list.append(color_list)

    return adj_matrices_list, node_feat_matrices_list, labels_list
def predict_list(model, test_data, labels):
    labels = [torch.tensor(label, dtype=torch.long) for label in labels]
    model.eval()
    predicted_labels = []
    with torch.no_grad():
        for data in test_data:
            out = model(data.x, data.edge_index)
            predicted_labels.append(torch.argmax(out, dim=1))
        labels_tensor = torch.cat(labels, dim=0)    
        predicted_labels = torch.cat(predicted_labels)
        correct = (predicted_labels == labels_tensor).sum().item()
        total = len(labels_tensor)
        accuracy = correct / total
    return predicted_labels, accuracy
def predict(model, data):
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        predicted_labels = torch.argmax(out, dim=1)
    return predicted_labels
def convert(test_graph):
    # Convert the test graph to PyTorch tensors
    test_adj_matrix = nx.to_numpy_array(test_graph)
    test_node_feat_matrix = np.array([test_graph.degree(node) for node in test_graph.nodes()]).reshape(-1, 1)
    test_adj_tensor = torch.tensor(test_adj_matrix, dtype=torch.float)
    test_node_feat_tensor = torch.tensor(test_node_feat_matrix, dtype=torch.float)

    # Create a PyTorch Geometric Data object for the test data
    test_edge_index = torch.tensor(add_self_loops(np.argwhere(test_adj_tensor)[:, [0, 1]].T)[0], dtype=torch.long)
    test_data = Data(x=test_node_feat_tensor, edge_index=test_edge_index)
    return test_data
class GraphSAGE(torch.nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super(GraphSAGE, self).__init__()

        self.hidden_layers = nn.ModuleList()
        self.hidden_layers.append(SAGEConv(input_dim, hidden_dims[0]))
        for i in range(1, len(hidden_dims)):
            self.hidden_layers.append(SAGEConv(hidden_dims[i-1], hidden_dims[i]))

        self.conv_final = SAGEConv(hidden_dims[-1], output_dim)

    def forward(self, x, edge_index):
        for conv in self.hidden_layers:
            x = F.relu(conv(x, edge_index))
        x = self.conv_final(x, edge_index)
        return F.log_softmax(x, dim=1)
    def l2_regularization(self):
        l2_reg = None
        for name, param in self.named_parameters():
            if 'weight' in name:
                if l2_reg is None:
                    l2_reg = param.norm(2)
                else:
                    l2_reg = l2_reg + param.norm(2)
        return l2_reg
class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super(GCN, self).__init__()
        self.hidden_layers = nn.ModuleList()
        self.hidden_layers.append(GCNConv(input_dim, hidden_dims[0]))
        for i in range(1, len(hidden_dims)):
            self.hidden_layers.append(GCNConv(hidden_dims[i-1], hidden_dims[i]))
        self.output_layer = GCNConv(hidden_dims[-1], output_dim)

    def forward(self, x, edge_index):
        for layer in self.hidden_layers:
            x = F.relu(layer(x, edge_index))
        x = self.output_layer(x, edge_index)
        return F.log_softmax(x, dim=1)
    
    def l2_regularization(self):
        l2_reg = None
        for name, param in self.named_parameters():
            if 'weight' in name:
                if l2_reg is None:
                    l2_reg = param.norm(2)
                else:
                    l2_reg = l2_reg + param.norm(2)
        return l2_reg

test_graph = nx.Graph()
test_graph.add_edges_from([(0,7),(0,8),(0,10),(0,13),(1,6),(1,8),(1,11),(1,14),(2,6),(2,13),(3,5),(14,3),(8,5),(6,8),(7,9)
,(7,13),(8,10),(8,14),(9,10),(11,12),(10,15),(5,14)])
test_data, test_labels = solve_graph(test_graph)
test_data = convert(test_data)
model = GraphSAGE(1, [32, 64, 128], 8)  # Create an instance of the model
model.load_state_dict(torch.load('C:\\Users\\dell\\bb\\GraphSage\\GraphSage100000.pth'))  # Load the saved parameters
predictions= predict(model, test_data)
pos=nx.spring_layout(test_graph)
plt.figure(num="Input graph")
nx.draw(test_graph, pos=pos, with_labels=True)
plt.show()
plt.figure(num="Welsh Powell output graph")
nx.draw(test_graph, pos=pos, node_color=test_labels, with_labels=True)
plt.show()
plt.figure(num="Machine learning output graph")
nx.draw(test_graph, pos=pos, node_color=predictions, with_labels=True)
plt.show()
#_______________________________________________________________________________________#
test_graph = nx.Graph()
test_graph.add_edges_from([(0,7),(0,11),(0,13),(0,12),(2,1),(2,13),(2,11),(4,11),(4,14),(5,8),(5,12),(6,8),(6,12),(6,13),(10,6),(8,10),(11,14),(12,13),(13,14)])
test_data, test_labels = solve_graph(test_graph)
test_data = convert(test_data)
model = GCN(1, [16, 32, 64], 8)  # Create an instance of the model
model.load_state_dict(torch.load('C:\\Users\\dell\\bb\\GCN\\GCN100001.pth'))  # Load the saved parameters
predictions= predict(model, test_data)
pos=nx.spring_layout(test_graph)
plt.figure(num="Input graph")
nx.draw(test_graph, pos=pos, with_labels=True)
plt.show()
plt.figure(num="Welsh Powell output graph")
nx.draw(test_graph, pos=pos, node_color=test_labels, with_labels=True)
plt.show()
plt.figure(num="Machine learning output graph")
nx.draw(test_graph, pos=pos, node_color=predictions, with_labels=True)
plt.show()