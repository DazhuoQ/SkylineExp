import random
import sys
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, GINConv, SAGEConv
from torch.nn import Sequential as Seq, Linear, ReLU, BatchNorm1d
import torch.optim as optim

from src.utils import *

class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, output_dim)

    def forward(self, x, edge_index):

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv3(x, edge_index)

        return F.log_softmax(x, dim=1)


class GCN_1(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCN_1, self).__init__()
        self.conv1 = GCNConv(input_dim, output_dim)

    def forward(self, x, edge_index):

        x = self.conv1(x, edge_index)

        return F.log_softmax(x, dim=1)


class GCN_2(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCN_2, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, x, edge_index):

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)


class GCN_4(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCN_4, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.conv4 = GCNConv(hidden_dim, output_dim)

    def forward(self, x, edge_index):

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv4(x, edge_index)

        return F.log_softmax(x, dim=1)



class GAT(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GAT, self).__init__()
        heads = 8

        self.conv1 = GATConv(input_dim, hidden_dim, heads=heads, dropout=0.6)
        self.conv2 = GATConv(hidden_dim * heads, hidden_dim, heads=heads, dropout=0.6)
        self.conv3 = GATConv(hidden_dim * heads, output_dim, heads=1, concat=True, dropout=0.6)

    def forward(self, x, edge_index):
        
        x, edge_index = x, edge_index
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv2(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv3(x, edge_index)

        return F.log_softmax(x, dim=-1)


class GIN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GIN, self).__init__()

        self.conv1 = GINConv(Seq(Linear(input_dim, hidden_dim),
                                 ReLU(),
                                 Linear(hidden_dim, hidden_dim),
                                 ReLU(),
                                 BatchNorm1d(hidden_dim)), train_eps=True)
        
        self.conv2 = GINConv(Seq(Linear(hidden_dim, hidden_dim),
                                 ReLU(),
                                 Linear(hidden_dim, hidden_dim),
                                 ReLU(),
                                 BatchNorm1d(hidden_dim)), train_eps=True)
        
        self.conv3 = GINConv(Seq(Linear(hidden_dim, hidden_dim),
                                 ReLU(),
                                 Linear(hidden_dim, hidden_dim),
                                 ReLU(),
                                 BatchNorm1d(hidden_dim)), train_eps=True)
        
        self.lin = Linear(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        x, edge_index = x, edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))
        return self.lin(x)


# class GIN(torch.nn.Module):
#     def __init__(self, input_dim, hidden_dim, output_dim):
#         super(GIN, self).__init__()

#         self.conv1 = GINConv(Seq(Linear(input_dim, hidden_dim),
#                                  ReLU(),
#                                  Linear(hidden_dim, hidden_dim),
#                                  ReLU(),
#                                  BatchNorm1d(hidden_dim)), train_eps=True)
        
#         self.lin = Linear(hidden_dim, output_dim)

#     def forward(self, x, edge_index):
#         x, edge_index = x, edge_index
#         x = F.relu(self.conv1(x, edge_index))
#         return self.lin(x)


class GraphSAGE(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GraphSAGE, self).__init__()
        
        # Define the layers
        self.conv1 = SAGEConv(input_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.conv3 = SAGEConv(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        # First layer with ReLU activation
        x = self.conv1(x, edge_index)
        x = F.relu(x)

        # Second layer with ReLU activation
        x = self.conv2(x, edge_index)
        x = F.relu(x)

        # Third layer (output layer)
        x = self.conv3(x, edge_index)

        return x



def get_model(config):
    model_name = config['model_name']
    input_dim = config['input_dim']
    hidden_dim = config['hidden_dim']
    output_dim = config['output_dim']
    data_name = config['data_name']

    if model_name == 'gcn':
        model = GCN(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)
        return model
    elif model_name == 'gat':
        model = GAT(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)
        return model 
    elif model_name == 'gin':
        model = GIN(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)
        return model 
    elif model_name == 'sage':
        model = GraphSAGE(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)
        return model 
    elif model_name == 'gcn1':
        model = GCN_1(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)
        return model 
    elif model_name == 'gcn2':
        model = GCN_2(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)
        return model 
    elif model_name == 'gcn4':
        model = GCN_4(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)
        return model 



def main(config_file, output_dir):
    # Load configuration
    config = load_config(config_file)
    data_name = config['data_name']
    model_name = config['model_name']
    random_seed = config['random_seed']
    set_seed(config['random_seed'])

    # Device config
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Get input graph
    data = dataset_func(config)
    data.to(device)

    # Get the model for training
    model = get_model(config)
    model.to(device)
    best_loss = float('inf')
    best_acc = 0
    best_model = None

    # train and save the model
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    model.train()
    for epoch in range(200):
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        if model_name=='gin':
            loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        else:
            loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        print(f'Epoch {epoch+1}, Loss: {loss.item()}')
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_model = model.state_dict()
        
        # acc = test_model(model, data)
        # if acc > best_acc:
        #     best_acc = acc
        #     best_model = model.state_dict()
    torch.save(best_model, 'models/{}_{}_model.pth'.format(data_name, model_name))

    # Save experiment settings
    print('Seed: '+str(config['random_seed']))
    print('Dataset: '+str(config['data_name']))
    print('Model: '+str(config['model_name']))

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python model.py <config_file> <output_dir>")
        sys.exit(1)
    
    config_file = sys.argv[1]
    output_dir = sys.argv[2]
    main(config_file, output_dir)

