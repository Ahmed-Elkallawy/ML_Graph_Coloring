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

def create_random_graph(num_nodes, prob):
    num_nodes = num_nodes
    prob = random.uniform(prob, prob+0.2)
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

def train(model, data_list, labels, num_epochs, learning_rate):
    optimizer_Adam = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()  
    lambda_value = 0.001 
    model.train()
    for epoch in range(num_epochs):
        optimizer_Adam.zero_grad()
        batch = Batch.from_data_list(data_list)
        out = model(batch.x, batch.edge_index)
        labels_tensor = torch.cat(labels, dim=0)  
        loss = criterion(out, labels_tensor)
        l2_reg = model.l2_regularization()
        loss += lambda_value * l2_reg
        loss.backward()
        optimizer_Adam.step()
        print(f"Epoch: {epoch+1}, Loss: {loss.item()}")
    return model

def predict_list(model, test_data, labels):
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

def evaluate_model(model, test_data, labels1):
    model.eval()
    with torch.no_grad():
        batch = Batch.from_data_list(test_data)
        out = model(batch.x, batch.edge_index)
        predicted_labels = torch.argmax(out, dim=1)
        labels_tensor = torch.cat(labels1, dim=0)    
        correct = (predicted_labels == labels_tensor).sum().item()
        total = len(labels_tensor)
        accuracy = correct / total
    return accuracy

def convert(test_graph):
    test_adj_matrix = nx.to_numpy_array(test_graph)
    test_node_feat_matrix = np.array([test_graph.degree(node) for node in test_graph.nodes()]).reshape(-1, 1)
    test_adj_tensor = torch.tensor(test_adj_matrix, dtype=torch.float)
    test_node_feat_tensor = torch.tensor(test_node_feat_matrix, dtype=torch.float)
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
            x = F.relu_(conv(x, edge_index))
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

num_iterations = 16000
num_iterations1 = 4000
adj_list, node_feat_list, labels_list = generate_dataset(num_iterations)
adj_list1, node_feat_list1, labels_list1 = generate_dataset(num_iterations1)
# save the data as numpy array files
np.save('adj_list20.npy', adj_list)
np.save('node_feat_list20.npy', node_feat_list)
np.save('labels_list.npy20', labels_list)

# save the data as numpy array files for evaluation
np.save('adj_list21.npy', adj_list1)
np.save('node_feat_list21.npy', node_feat_list1)
np.save('labels_list21.npy', labels_list1)

# load the data from the numpy array files
# adj_list = np.load('adj_list.npy')
# node_feat_list = np.load('node_feat_list.npy')
# labels_list = np.load('labels_list.npy')

# # load the data from the numpy array files
# adj_list1 = np.load('adj_list1.npy')
# node_feat_list1 = np.load('node_feat_list1.npy')
# labels_list1 = np.load('labels_list1.npy')

# Convert the adjacency list and node features to PyTorch tensors
adj_list = [torch.tensor(adj_matrix, dtype=torch.float) for adj_matrix in adj_list]
node_feat_list = [torch.tensor(node_feat_matrix, dtype=torch.float) for node_feat_matrix in node_feat_list]

# for Evaluation
adj_list1 = [torch.tensor(adj_matrix1, dtype=torch.float) for adj_matrix1 in adj_list1]
node_feat_list1 = [torch.tensor(node_feat_matrix1, dtype=torch.float) for node_feat_matrix1 in node_feat_list1]

# Create a PyTorch Geometric Data object
data_list = []
for adj, feat in zip(adj_list, node_feat_list):
    edge_index = torch.tensor(add_self_loops(np.argwhere(adj)[:, [0, 1]].T)[0], dtype=torch.long)
    data = Data(x=feat, edge_index=edge_index)
    data_list.append(data)

# for Evaluation
data_list1 = []
for adj1, feat1 in zip(adj_list1, node_feat_list1):
    edge_index1 = torch.tensor(add_self_loops(np.argwhere(adj1)[:, [0, 1]].T)[0], dtype=torch.long)
    data1 = Data(x=feat1, edge_index=edge_index1)
    data_list1.append(data1)

labels = [torch.tensor(label, dtype=torch.long) for label in labels_list]
labels1 = [torch.tensor(label1, dtype=torch.long) for label1 in labels_list1]

# Instantiate the GCN model
output_dim = max(max(label) for label in labels_list) + 1
GCN_model = GCN(1, [16, 32, 64], output_dim)

GCN_trained_model = train(GCN_model, data_list, labels, num_epochs=200, learning_rate=0.01)
accuracy = evaluate_model(GCN_trained_model ,data_list1, labels1)   
print(accuracy)
torch.save(GCN_trained_model.state_dict(), 'C:\\Users\\dell\\bb\\GCN\\GCN20.pth')

# Instantiate the GraphSAGE model
output_dim = max(max(label) for label in labels_list) + 1
GraphSAGE_model = GraphSAGE(1, [32, 64, 128], output_dim)

GraphSAGE_trained_model = train(GraphSAGE_model, data_list, labels, num_epochs=200, learning_rate=0.001)
accuracy = evaluate_model(GraphSAGE_trained_model ,data_list1, labels1)   
print(accuracy)
torch.save(GraphSAGE_trained_model.state_dict(), 'C:\\Users\\dell\\bb\\GraphSage\\GraphSage20.pth')

