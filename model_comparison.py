import numpy as np 
import matplotlib.pyplot as plt 
import torch_geometric as pyg
import torch
import pandas as pd
from torch_geometric.utils import to_undirected
import networkx as nx
from utils import dataset_gen, minmaxscaler
import seaborn as sns
from model import GNN
import os
from torch_geometric.loader import DataLoader
from IPython.display import clear_output

device = torch.device('cpu')


indicators = ['WSS','OSI']#

sub_ind = [indicators[0]]

conv_list = ['gconv','gin','gcn', 'gtr']
long_conv_list = ['GraphConv', 'GIN', 'GCN', 'Graph Transformer']
input_size = 7 # dataset features: Time, Press_SA, Press_abd, FlowRate, coord(x), coord(y), coord(z)
hidden_size = 32
num_layers = 3
perc = 75
output_size = len(sub_ind)
dataset= dataset_gen(perc,sub_ind)
num_nodes = dataset[0].num_nodes
T = len(dataset)
loader = DataLoader(dataset, batch_size = T, shuffle=False)

y = torch.zeros((T,num_nodes))
for i, data in enumerate(loader):
    y = data.y.reshape((T,num_nodes))
t = torch.linspace(0,1,T)


save_folder = '_'.join(['store', 'architecture','interp', sub_ind[0] +'/' ])

nodes = [60, 735, 1167, 1083, 1415]
data_raw = []
for i, conv_type in enumerate(conv_list):
    for it in range(5):  
        save_filename = '_'.join(['pred',sub_ind[0],conv_type, str(it)])
     
        for node in nodes:
            for j in range(500):
                data_raw.append({'conv_type': conv_type, 'it': it, 'node': node, 't':float(t[j]), 'pred': float(torch.load(save_folder+save_filename)[j,node])})
df = pd.DataFrame.from_records(data_raw)

df.to_csv('architecture_WSS')

for node in nodes:
    fig, axes = plt.subplots(1,4, sharey=True, figsize=(16, 8) )
    axes[0].set_ylabel(sub_ind[0], fontsize = '16')
    for i,conv_type in enumerate(conv_list):
        red = df[df['conv_type']==conv_type][df['node']==node][['t','pred','it']]
        axes[i].set_xlabel('t', fontsize = '16')
        axes[i].tick_params(axis = 'both', which='minor', labelsize= '14')
        sns.lineplot(data = red, x="t", y="pred", ax=axes[i], label = 'Prediction')
        sns.lineplot(x = t, y = y[:,node].detach(),ax=axes[i], label = 'Ground Truth')
        axes[i].legend(fontsize = '14', loc = 'upper right')
        axes[i].set_title(long_conv_list[i])

    os.makedirs('seaborn/pred_layers_architecture/', exist_ok=True)
    plt.savefig('seaborn/pred_layers_architecture/pred'+str(node)+'.pdf')
    plt.close()