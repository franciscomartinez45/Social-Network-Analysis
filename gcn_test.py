import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt



class GraphConvolutionLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GraphConvolutionLayer, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.bias = nn.Parameter(torch.FloatTensor(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, input, adjacency_matrix):
        support = torch.matmul(input, self.weight)
        output = torch.matmul(adjacency_matrix, support) + self.bias
        return output

class SocialGCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SocialGCN, self).__init__()
        self.gc1 = GraphConvolutionLayer(input_dim, hidden_dim)
        self.gc2 = GraphConvolutionLayer(hidden_dim, hidden_dim)
        self.gc3 = GraphConvolutionLayer(hidden_dim, output_dim)

    def forward(self, features, adjacency_matrix):
        # Perform message passing
        hidden1 = F.relu(self.gc1(features, adjacency_matrix))
        hidden2 = F.relu(self.gc2(hidden1, adjacency_matrix))
        output = self.gc3(hidden2, adjacency_matrix)
        return output

def visualize_predictions(graph, labels, predicted_labels):
    pos = nx.fruchterman_reingold_layout(graph) 
    nx.draw(graph, pos, node_color=labels.numpy(), cmap=plt.cm.Paired, node_size=500, with_labels=False)
    nx.draw_networkx_labels(graph, pos, labels={i: str(predicted_labels[i].item()) for i in range(len(predicted_labels))}, font_color='red')
    plt.show()


""" generate edges from "data.txt" """
print(np.genfromtxt("data.txt"))
adjacency_matrix = torch.tensor(np.genfromtxt("data.txt")).float()

""" Generate example data """
num_nodes = adjacency_matrix.size(0)

""" n number of features per node """
num_features = 7

""" binary node labels """
num_classes = 2 

""" random node features """
X = torch.randn((num_nodes,num_features)).float()

""" generate edges to analyze most influential nodes and number of cliques per graph to come up with a number of influential nodes """
B = np.genfromtxt("data.txt",dtype=float)

""" use networkx to create a graph for analysing """
G = nx.from_numpy_array(B)

""" create a sorted dictionary based on eigenvetor centrality """
D = sorted(nx.eigenvector_centrality(G).items(), key=lambda x:x[1],reverse=True)

""" get the highest number of cliques """
C = max(nx.number_of_cliques(G).items(), key = lambda x:x[1])[1]

""" store indices of top C number of cliques and store them  in an array """
arr_= []
for i in range(0,C):
    arr_.append(D[i][0])

""" set all nodes """
ls = []
for i in range(num_nodes):
    ls.append(0)


""" modify labels """
for index in range(0,len(X)):
    if index in arr_:
        ls[index] = 1.0

""" create labels for the adjecency graph """
labels = torch.tensor(ls).long() 

""" random node features """
features = torch.randn((num_nodes, num_features))

""" create the model """
model = SocialGCN(input_dim=num_features, hidden_dim=32, output_dim=num_classes)

""" Define loss function and optimizer """
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

epochs = 5



for epoch in range(epochs):
    optimizer.zero_grad()
    output = model(features, adjacency_matrix)
    loss = criterion(output, labels)
    loss.backward()
    optimizer.step()
    acc = (torch.argmax(output, dim=1) == labels).float().mean().item()
    print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item()}, Accuracy: {acc}')


graph = nx.from_numpy_array(adjacency_matrix.numpy())
predicted_labels = torch.argmax(model(features, adjacency_matrix), dim=1)
pos = nx.fruchterman_reingold_layout(graph) 
nx.draw(graph, pos, node_color=labels.numpy(), cmap=plt.cm.Paired, node_size=500, with_labels=False)
nx.draw_networkx_labels(graph, pos, labels={i: str(predicted_labels[i].item()) for i in range(len(predicted_labels))}, font_color='red')
plt.show()