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

# RMAT graph path

rmat_path = "/home/fmartinez/research/GAT-test/GTgraph_data/data.txt"

# Define parameters for model training
hidden_layers = 32
learning_rate = 0.001
epochs_ = 120

# Initialize the social network


# Initialize directories
DIRECTORY = "IMG/"
try: 
    os.makedirs(F"{DIRECTORY}",exist_ok=True)
    print(f"Folder `{DIRECTORY}` created.")
except FileExistsError:
    print(f"Folder `{DIRECTORY}` already exists.")
    
SUB_DIRECTORY = f"{datetime.now()}-lr:{learning_rate}-e:{epochs_}-dh:{hidden_layers}"
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

def SocialNetwork(**kwargs):
    G = kwargs.get("graph")
    n_nodes = kwargs.get("nodes")
    m_edges = kwargs.get("edges")
    graph_name = kwargs.get("graph_name")
    # Convert to torch tensor
    edge_index = torch.tensor(np.array(G.edges()).T, dtype=torch.long)

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
    
    features = np.array(features, dtype=np.float32)

    # Standardize features
    scaler = StandardScaler()
    standardized_features = scaler.fit_transform(features)
    standardized_features = torch.tensor(standardized_features, dtype=torch.float)

    # Compute combined influence score for each node
    combined_scores = standardized_features.sum(axis=1)

    # Sort nodes based on combined influence scores
    sorted_nodes = torch.argsort(combined_scores, descending=True)

    # Select the top % percent nodes 
    influential_percent = 0.2
    num_influential = int(n_nodes * influential_percent)  
    influential_nodes = sorted_nodes[:num_influential]

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
            print(f"Epoch {epoch}/{epochs}, Loss: {loss.item()}")
            model.eval()
            with torch.no_grad():
                activation_over_epochs[epoch] = model(graph_data.x, graph_data.edge_index).squeeze().numpy()
    
    # plt.figure(figsize=(10, 6))

    for node_id in range(n_nodes): 
        activations = [activation_over_epochs[epoch][node_id] for epoch in activation_over_epochs.keys()]
        plt.plot(list(activation_over_epochs.keys()), activations)
    


    
    plt.xlabel("Epochs")
    plt.ylabel("Activation Value")
    plt.title("Node Activation Over Time(Change in node's influence)")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{DIRECTORY}{SUB_DIRECTORY}/node_activation.jpg")
    plt.close()
    
    # Training Loss plot 
    plt.figure(figsize=(8, 5))
    plt.plot(range(epochs), loss_values, label="Training Loss", color='blue')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training Loss Over Time")
    plt.legend()
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
        File.writelines(f"Graph Name: {graph_name}\n# of nodes: {n_nodes} - Epochs: {epochs_} - Hidden Layers: {hidden_layers} - Learning Rate: {learning_rate}\n")
        File.writelines(f"Mean Squared Error (MSE): {mse:.4f}\n")
        File.writelines(f"Mean Absolute Error (MAE): {mae:.4f}\n")
        File.writelines(f"R^2 Score: {r2:.4f}\n")
        File.writelines(f"Explained Variance Score: {evs:.4f}\n")
        File.writelines(f"Predicted Labels: \n{str(predicted_labels)}")

    #SHAP summary
    def predict_fn(x):
        x_tensor = torch.tensor(x, dtype=torch.float32)
        with torch.no_grad():
            return model(x_tensor, graph_data.edge_index).squeeze().numpy()  


    explainer = shap.Explainer(predict_fn, graph_data.x.numpy())
    shap_values = explainer(graph_data.x.numpy(),main_effects=True,error_bounds=True)
    feature_data = [
            "Degree", "Eigenvector Centrality", "Betweenness Centrality", "PageRank", "Clustering Coefficient"
        ]
    fig = plt.figure()
    shap.summary_plot(shap_values, graph_data.x.numpy(),feature_names = feature_data,show=False)
    fig.tight_layout()
    fig.savefig(f"{DIRECTORY}{SUB_DIRECTORY}/shap_summary.png", format="png", bbox_inches='tight')
    plt.close()

    # Visualization of influence spread
    pos = nx.spring_layout(G, seed=10)
    plt.figure(figsize=(10, 10))
   
    # Node color based on influence state/probability
    node_colors = ['red' if node in influential_nodes else 'green' if output[node].item()>=0.3 else 'white' for node in range(n_nodes)]
    node_options = {
        "font_size": 15,
        "node_size":750,
        "node_color":node_colors,
        "edgecolors":"black",
        "width":3
    }
    nx.draw(G, pos, with_labels=False, **node_options)

    # Label nodes with influence probabilities
    node_labels = {node: f"{output[node].item():.2f})" for node in range(n_nodes)}
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_color='black', font_size=7)
    plt.savefig(f"{DIRECTORY}/{SUB_DIRECTORY}/graph.jpg")
    plt.close()

def main():

    #NetworkXGraphs()
    RMAT()



def RMAT():
    B = nx.read_edgelist(rmat_path, nodetype=int, data=(("weight", float),))
   
    G = nx.DiGraph(B)
    graph_name = "RMAT"
    max_node = -1
    for from_, to_ in G.edges():
        current_max = max(max_node,max(from_,to_))
        max_node=current_max

    for i in range(max_node):
        if i not in G.nodes:
            G.add_node(i)

    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    # for u, v, data in G.edges(data=True):
    #     print(f"{u} -- {v}, weight={data['weight']}")
    SocialNetwork(graph=G, nodes = num_nodes, edges = num_edges,graph_name=graph_name)

def NetworkXGraphs():
    n_nodes = 80
    m_edges = 1
    graph_name = "Barabasi Albert Graph"
    G = nx.barabasi_albert_graph(n_nodes, m_edges, seed=1)
    SocialNetwork(nodes=n_nodes, edges=m_edges, graph=G, graph_name=graph_name)

if __name__ == "__main__":
    main()