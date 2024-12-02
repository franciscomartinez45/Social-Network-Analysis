import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from random import randint
from textblob import TextBlob

  

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
        hidden1 = F.relu(self.gc1(features, adjacency_matrix))
        hidden2 = F.relu(self.gc2(hidden1, adjacency_matrix))
        output = self.gc3(hidden2, adjacency_matrix)
        return output

def visualize_predictions(graph, labels, predicted_labels):
    pos = nx.fruchterman_reingold_layout(graph) 
    nx.draw(graph, pos, node_color=labels.numpy(), cmap=plt.cm.Paired, node_size=500, with_labels=False)
    nx.draw_networkx_labels(graph, pos, labels={i: str(predicted_labels[i].item()) for i in range(len(predicted_labels))}, font_color='blue')
    plt.show()

def extractFeatures(fileName):
    lowercase_lambda = lambda x: x.lower()
    arr = np.loadtxt(fileName, delimiter=",", dtype=str,usecols=range(1,len(pd.read_csv(fileName,nrows=1).columns)), converters={i: lowercase_lambda for i in range(1, len(pd.read_csv(fileName, nrows=1).columns))})[2:]
    
    arr = arr.tolist()
    # arr = np.delete(arr,[0,3,5], axis=1)
    dic = {("male","m") : 0.495 ,("yes","y","ye"): 1.0, "united states" : 0.0423, "year 4" : 0.652}
    for i in range(len(arr)):
        for j in range(len(arr[i])):
            if arr[i][j][0] == '"':
                arr[i][j] = arr[i][j][1:-1]
            if arr[i][j] in dic:
                arr[i][j] = 1
            else:
                blob = TextBlob(arr[i][j])
                """ Perform sentiment analysis """
                arr[i][j] = blob.sentiment.polarity
                
    #print(arr)
    return np.array(arr,dtype=float)
    
    
def extractEdges(file):
    data = np.loadtxt(file,usecols=(1,2),skiprows=8,dtype=int)
    G = nx.Graph()
    for item in data:
        G.add_edge(item[0],item[1])
    return nx.to_numpy_array(G)
    
""" generate edges from "GTgraph_data" 128 nodes X 125 edges as input to match feature list to match UCI dataset """
adjacency_matrix= torch.tensor(extractEdges("data.txt")).float()


""" generate from networkX and Standford social media dataset """
facebook = pd.read_csv(
    "data/facebook_combined.txt.gz",
    compression="gzip",
    sep=" ",
    names=["start_node", "end_node"],
)
""" Facebook Dataset"""
# adjacency_matrix = nx.from_pandas_edgelist(facebook, "start_node", "end_node")
# degree_centrality = nx.centrality.degree_centrality(adjacency_matrix)  
# adjacency_matrix = torch.tensor(nx.to_numpy_array(adjacency_matrix)).float()


"""Zachary's Karate Club adjacency matrix"""
# karate_club = nx.karate_club_graph()
# degree_centrality = nx.centrality.degree_centrality(karate_club) 
# adjacency_matrix = torch.tensor(nx.to_numpy_array(karate_club)).float()

"""Generate example data"""
num_nodes = int(adjacency_matrix.size(0))


""" n number of features per node"""
num_features = 15


""" binary node labels """
num_classes = 2


"""random node features"""
X = torch.randn((num_nodes,num_features)).float()


""""custom features data from UCI database"""
# X = torch.tensor(extractFeatures("StudentMentalhealth.csv")).float()
#X = torch.tensor(extractFeatures("data/TechMentalHealth.csv")).float()

""""generate edges to analyze most influential nodes and number of cliques per graph to come up with a number of influential nodes"""
#B = extractEdges("data.txt")


"""get top 8 nodes with highest centrality"""
centrality_arr = [item[0] for item in(sorted(degree_centrality.items(), key=lambda item: item[1], reverse=True,))[:5]]

"""set all nodes """
ls = []
for i in range(num_nodes):
    ls.append(0)


"""modify labels"""
for index in range(0,len(X)):
    if index in centrality_arr:
        ls[index] = 1.0


""" create labels for the adjecency graph """
labels = torch.tensor(ls).long() 

""" random node features """
features = X


""" create the model """
model = SocialGCN(input_dim=num_features, hidden_dim=4, output_dim=num_classes)



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