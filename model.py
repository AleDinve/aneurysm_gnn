import torch
import torch.nn as nn
from torch.nn import Linear, Sequential, BatchNorm1d, Tanh, Sigmoid
from torch_geometric.nn import GraphConv, GINConv, GCNConv, TransformerConv
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class GNN(nn.Module):
    """
    GNN model class.
    The GNN modules can be chosen among the following three:
    - 'gconv': it refers to the GraphConv module (Morris, 2019)
    - 'gcn': it refers to the Graph Convolutional Network single module (Kipf, 2016)
    - 'gin': it refers to the Graph Isomorphism Network module (Xu, 2018)
    """
    def __init__(self, 
                 input_size, hidden_size, output_size, 
                 num_layers, conv_type='gconv', device = torch.device('cpu')):
        super().__init__()
        self.conv_type = conv_type
        self.layers = torch.nn.ModuleList()
        self.device = device
        
        if conv_type == 'gconv':
            self.gconv = lambda i, o: GraphConv(i, o, aggr='add', bias=True)
        elif conv_type == 'gin': #GIN
            self.gconv = lambda i, o: GINConv(Sequential(Linear(i, hidden_size),
                    BatchNorm1d(hidden_size),
                    Tanh(),
                    Linear(hidden_size, o), Tanh()))
        elif conv_type == 'gcn':
            self.gconv = lambda i, o: GCNConv(i,o)
        elif conv_type == 'gtr':
            self.gconv = lambda i, o: TransformerConv(i,o, heads = 1)
        for _ in range(num_layers):
            self.layers.append(self.gconv(input_size,hidden_size).to(self.device))
            input_size = hidden_size
        if self.conv_type == 'gin':
            self.final = GINConv(Sequential(Linear(input_size, hidden_size),
                    BatchNorm1d(hidden_size),
                    Tanh(),
                    Linear(hidden_size, output_size), Sigmoid())).to(self.device)
        else:
            self.final = self.gconv(input_size, output_size).to(self.device)
        #self.linear = nn.Linear(hidden_size, output_size).to(device)
        
    def forward(self, x, edge_index):
        x = x.float()
        for layer in self.layers:  
            if self.conv_type == 'gin':
                x = layer(x, edge_index)
            else:
                x = torch.tanh(layer(x, edge_index))
        #x = global_add_pool(x, batch)
        if self.conv_type == 'gin':
            x = self.final(x, edge_index)
        else:
            x = torch.sigmoid(self.final(x,edge_index))
        return x