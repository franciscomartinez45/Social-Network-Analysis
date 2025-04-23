import random
import os
from datetime import datetime
import torch
import torch.nn.functional as F
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.data import Data
from torch_geometric.nn import GATConv
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import  explained_variance_score, mean_absolute_error, mean_squared_error, r2_score
import shap
import kagglehub
import pandas as pd
import copy
# Kaggle datasets

dataset1= {"handle":"emirhanai/social-media-usage-and-emotional-well-being","file":"/train.csv", "columns":["Daily_Usage_Time (minutes)", "Posts_Per_Day", "Likes_Received_Per_Day", "Comments_Received_Per_Day", "Messages_Sent_Per_Day"]} 
dataset2={"handle":"muhammadroshaanriaz/time-wasters-on-social-media","file":"/Time-Wasters on Social Media.csv","columns":["Income","Total Time Spent","Number of Sessions","Importance Score","Scroll Rate"]}

# RMAT graph path

rmat_path = "/home/fmartinez/research/GAT-test/GTgraph_data/data.txt"

# Define parameters for model training
hidden_layers = 32
learning_rate = 0.001
epochs_ = 200
# Select the top % percent nodes 
influential_percent = 0.5


# Initialize the social network


# Initialize directories
DIRECTORY = "IMG/"
SUB_DIRECTORY = f"{datetime.now()}-lr:{learning_rate}-e:{epochs_}-dh:{hidden_layers}"

def make_directories():
    try: 
        os.makedirs(F"{DIRECTORY}",exist_ok=True)
        print(f"Folder `{DIRECTORY}` created.")
    except FileExistsError:
        print(f"Folder `{DIRECTORY}` already exists.")
        
    try: 
        os.makedirs(F"{DIRECTORY}{SUB_DIRECTORY}")
    except FileExistsError:
        print(f"Folder `{SUB_DIRECTORY}` in `{DIRECTORY}` already exists.")


class GATPropagation(torch.nn.Module):
        def __init__(self, in_dim, hidden_dim, out_dim, heads=4):
            super(GATPropagation, self).__init__()
            self.conv1 = GATConv(in_dim, hidden_dim, heads=heads, concat=True)
            self.conv2 = GATConv(hidden_dim * heads, out_dim, heads=1, concat=False)
          
        def forward(self, x, edge_index):
            x = self.conv1(x, edge_index).relu()
            x = self.conv2(x, edge_index)
            return torch.sigmoid(x)  
        
def graph_features(**kwargs):
    G = kwargs.get("graph")
    dataset = kwargs.get("dataset")
    dataset["columns"] = ["Degree", "Eigenvector Centrality", "Betweenness Centrality", "PageRank"]
     # Generate node features
    pagerank = nx.pagerank(G)
    clustering_coeff = nx.clustering(G)
    features = []
    for node in G.nodes():
        features.append([ 
            nx.degree(G, node),
            nx.eigenvector_centrality(G, max_iter=100)[node],
            nx.betweenness_centrality(G)[node],
            pagerank[node],
            #clustering_coeff[node]
        ])
    return np.array(features, dtype=np.float32)


def get_kaggle_dataset(**kwargs):
    dataset = kwargs.get("dataset")
    size = kwargs.get("size")
    dataset_path = kagglehub.dataset_download(dataset.get("handle"))
    df = pd.read_csv(dataset_path+dataset.get("file"))
    return np.array(df[dataset.get("columns")][:size], dtype=np.float32)

def influential_hubs(**kwargs):
    standardized_features = kwargs.get("features")
    num_influential = kwargs.get("percent")
    n_nodes = kwargs.get("size")
    combined_scores = standardized_features.sum(axis=1)
    # Sort nodes based on combined influence scores
    sorted_nodes = torch.argsort(combined_scores, descending=True)
    influential_nodes = sorted_nodes[:num_influential]
    return influential_nodes




def SocialNetwork(**kwargs):
    
    G = kwargs.get("graph")
    n_nodes = kwargs.get("nodes")
    graph_name = kwargs.get("graph_name")
    dataset = kwargs.get("dataset")
    dataset_ = copy.deepcopy(dataset)
    
    

    # Convert to torch tensor
    edge_index = torch.tensor(np.array(G.edges()).T, dtype=torch.long)
   
    graph_nodes = G.nodes
    
    # dataset is only used to add features columns
    #features = graph_features(graph = G, dataset = dataset)

    # #concatenate graph topoligical features with another dataset
    features_ = graph_features(graph = G, dataset = dataset)
    features = np.concatenate((features_,get_kaggle_dataset(dataset=dataset_, size = len(graph_nodes))),axis=1)
    dataset["columns"] = dataset["columns"] + dataset_["columns"]
 
    #features = get_kaggle_dataset(dataset=dataset,size = len(graph_nodes))
    # Standardize features
    scaler = StandardScaler()
    standardized_features = scaler.fit_transform(features)
    standardized_features = torch.tensor(standardized_features, dtype=torch.float)
    num_influential = int(n_nodes * influential_percent)  

    # Compute combined influence score for each node
    influential_nodes = influential_hubs(features = standardized_features, size = len(graph_nodes), percent = num_influential)


    # Pick random nodes to influence
    #influential_nodes = random.sample(range(1,len(graph_nodes)),num_influential)
    

   
    # Print influential nodes
    #print(f"Influential nodes: {influential_nodes.numpy()}")
    
    # Initial information spread setup
    node_states = torch.zeros(n_nodes, dtype=torch.float)

    # Set influential nodes to 1
    node_states[influential_nodes] = 1.0 


    # Define PyG data object
    graph_data = Data(x=standardized_features, edge_index=edge_index, y=node_states)
   


    # Initialize model
    model = GATPropagation(standardized_features.shape[1], hidden_layers, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=5e-4)
    criterion = F.mse_loss  
   
    loss_values = []
    activation_over_epochs = {}
    # Train model
    epochs = epochs_
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(graph_data.x, graph_data.edge_index).squeeze()  
     
        loss = criterion(out, graph_data.y)
        loss.backward()
        optimizer.step()
        loss_values.append(loss.item()) 
        if epoch % 10 == 0:
            #print(f"Epoch {epoch}/{epochs}, Loss: {loss.item()}")
            model.eval()
            with torch.no_grad():
                activation_over_epochs[epoch] = model(graph_data.x, graph_data.edge_index).squeeze().numpy()
    
    
    for node_id in range(n_nodes): 
        activations = [activation_over_epochs[epoch][node_id] for epoch in activation_over_epochs.keys()]
        plt.plot(list(activation_over_epochs.keys()), activations)
    

    make_directories()
    plt.xlabel("Epochs")
    plt.ylabel("Activation Value")
    plt.title("Node Activation Over Time(Change in node's influence)")
    plt.grid(True)
    plt.savefig(f"{DIRECTORY}{SUB_DIRECTORY}/node_activation.jpg")
    plt.close()
    
    # Training Loss plot 
    plt.figure(figsize=(8, 5))
    plt.plot(range(epochs), loss_values, label="Training Loss", color='blue')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training Loss Over Time")
    plt.grid(True)
    plt.savefig(f"{DIRECTORY}{SUB_DIRECTORY}/training_loss.jpg")
    plt.close()


    # Compute final node activation levels
    model.eval()
    with torch.no_grad():
        output = model(graph_data.x, graph_data.edge_index).squeeze()
        # Calculate accuracy metrics
    true_labels = graph_data.y.numpy()
    predicted_labels = output.numpy()
     # Mean Squared Error
    mse = mean_squared_error(true_labels, predicted_labels)
    # Mean Absolute Error
    mae = mean_absolute_error(true_labels, predicted_labels)
    # R^2 Score
    r2 = r2_score(true_labels, predicted_labels)
    # Explained Variance Score
    evs = explained_variance_score(true_labels, predicted_labels)
    # Create file to save data
    with open(f"{DIRECTORY}{SUB_DIRECTORY}/metrics.txt","w") as File:
        File.writelines("Training data:\n")
        File.writelines(f"Graph Name: {graph_name}\nNumber of nodes: {n_nodes} - Epochs: {epochs_} - Hidden Layers: {hidden_layers} - Learning Rate: {learning_rate}\n")
        File.writelines(f"Mean Squared Error (MSE): {mse:.4f}\n")
        File.writelines(f"Mean Absolute Error (MAE): {mae:.4f}\n")
        File.writelines(f"R^2 Score: {r2:.4f}\n")
        File.writelines(f"Explained Variance Score: {evs:.4f}\n")
        File.write(f"{'Index':<6} {'Output (Confidence)':<22} {'True Label':<11} {'Predicted Label':<17} {'Correct?':<8}\n")
        for i, (conf, true, pred) in enumerate(zip(output, true_labels, predicted_labels)):
            pred_binary = int(pred > 0.3)
            correct = "Y" if pred_binary == true else "N"
            File.write(f"{i+1:<6} {conf:<22.4f} {int(true):<11} {pred_binary:<17.1f} {correct:<8}\n")
    
    #SHAP summary
    plt.figure(figsize=(10, 10))
    def predict_fn(x):
        x_tensor = torch.tensor(x, dtype=torch.float32)
        with torch.no_grad():
            return model(x_tensor, graph_data.edge_index).squeeze().numpy()  
    explainer = shap.Explainer(predict_fn, graph_data.x.numpy())
    shap_values = explainer(graph_data.x.numpy(),main_effects=True,error_bounds=True)
    feature_data = dataset["columns"]
    fig = plt.figure()
    shap.summary_plot(shap_values, graph_data.x.numpy(),feature_names = feature_data,show=False)
    fig.tight_layout()
    fig.savefig(f"{DIRECTORY}{SUB_DIRECTORY}/shap_summary.jpg", format="jpg", bbox_inches='tight')
    plt.close()


    # Visualization of influence spread
    pos = nx.spring_layout(G, seed=10)
    plt.figure(figsize=(10, 10))

    #True Labels for graph
    node_colors_2 = [
    "orange" if node in influential_nodes else
    "green" if true_labels[node]>0 else
    "white"
    for node in range(n_nodes)
    ]
    node_options_2 = {
        "font_size": 15,
        "node_size":750,
        "node_color":node_colors_2,
        "edgecolors":"black",
        "width":3
    }
    nx.draw(G, pos, with_labels=False, **node_options_2)
    node_labels_2 = {node: f"{true_labels[node]:.1f})" for node in range(n_nodes)}
    nx.draw_networkx_labels(G, pos, labels=node_labels_2, font_color='black', font_size=7)
    plt.savefig(f"{DIRECTORY}/{SUB_DIRECTORY}/graphTrueLabels.jpg")
    plt.close()

    # Predicted Labels graph
    node_colors_1 = [
    #'green' if predicted_labels[node] > 0.3 else
    'orange' if node in influential_nodes else
    'white'
    for node in range(n_nodes)
]
    node_options_1 = {
        "font_size": 15,
        "node_size":750,
        "node_color":node_colors_1,
        "edgecolors":"black",
        "width":3
    }
    pos = nx.spring_layout(G, seed=10)
    plt.figure(figsize=(10, 10))
    nx.draw(G, pos, with_labels=False, **node_options_1)
    node_labels_1 = {node: f"{predicted_labels[node]:.2f})" for node in range(n_nodes)}
    nx.draw_networkx_labels(G, pos, labels=node_labels_1, font_color='black', font_size=7)
    plt.savefig(f"{DIRECTORY}/{SUB_DIRECTORY}/graphPredictedLabels.jpg")
    plt.close()

def main():

    #NetworkXGraphs()
    RMAT()



def RMAT():
    B = nx.read_edgelist(rmat_path, nodetype=int, data=(("weight", float),))
    node_mapping = {node: i for i, node in enumerate(B.nodes())}
    remapped_edges = [(node_mapping[u], node_mapping[v]) for u, v in B.edges()]
    G = nx.Graph()
    G.add_edges_from(remapped_edges)
    graph_name = "RMAT"
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    SocialNetwork(graph=G, nodes = num_nodes, edges = num_edges,graph_name=graph_name, dataset=dataset1)

def NetworkXGraphs():
    n_nodes = 80
    m_edges = 1
    graph_name = "Barabasi Albert Graph"
    G = nx.barabasi_albert_graph(n_nodes, m_edges, seed=1)
    SocialNetwork(nodes=n_nodes, edges=m_edges, graph=G, graph_name=graph_name, dataset=dataset2)

if __name__ == "__main__":
    main()