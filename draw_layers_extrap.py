import numpy as np 
import matplotlib.pyplot as plt 
import torch
from utils import dataset_gen, minmaxscaler
from model import GNN
import pandas as pd
import os
import seaborn as sns
from torch_geometric.loader import DataLoader

device = torch.device('cpu')

indicators = ['WSS','OSI']#

sub_ind = [indicators[0]]

task_type = 'extrap'
param_type = 'layers'
conv_type = 'gtr'
input_size = 7 # dataset features: Time, Press_SA, Press_abd, FlowRate, coord(x), coord(y), coord(z)
hidden_size = 32
layers_list = [2,3,4,5]
perc = 125
output_size = len(sub_ind)
dataset = dataset_gen(perc,sub_ind)
num_nodes = dataset[0].num_nodes
T = len(dataset)
loader = DataLoader(dataset, batch_size = 1, shuffle=False)

y = torch.zeros((T,num_nodes))
for i, data in enumerate(loader):
    y[i,:] = data.y.squeeze()
t = torch.linspace(0,1,T)



model_folder = '_'.join(['store', param_type, task_type, sub_ind[0]])+'/'
if not os.path.exists(model_folder):
    os.makedirs(model_folder)



nodes = [i for i in range(num_nodes)]
data = []
for i, num_layers in enumerate(layers_list):
    for it in range(5):       
        model_name ='_'.join(['pred',indicators[0],str(num_layers),str(it)+'.pt'])
        for node in nodes:
            for j in range(500):
                data.append({'num_layer': num_layers, 'it': it, 'node': node, 't':float(t[j]), 'pred': float(torch.load(model_folder+model_name).detach().numpy()[j,node])})
            #pred_plot[i,j,:,:] = torch.load('_'.join(['pred',indicators[0],str(num_layers),str(j)])).detach()
df = pd.DataFrame.from_records(data)

df.to_csv('layers_extrap_WSS.csv')


for node in nodes:
    fig, axes = plt.subplots(1,4, sharey=True, figsize=(16, 8) )
    axes[0].set_ylabel(sub_ind[0], fontsize = '16')
    for i,l in enumerate(layers_list):
        red = df[df['num_layer']==l][df['node']==node][['t','pred','it']]
        axes[i].set_xlabel('t', fontsize = '16')
        axes[i].tick_params(axis = 'both', which='minor', labelsize= 12)
        sns.lineplot(data = red, x="t", y="pred", ax=axes[i], label = 'Prediction')
        sns.lineplot(x = t, y = y[:,node].detach(),ax=axes[i], label = 'Ground Truth')
        axes[i].legend(fontsize = '14', loc = 'upper right')
        axes[i].set_title(str(l)+' layers')

    os.makedirs('seaborn/pred_layers_extrap/', exist_ok=True)
    plt.savefig('seaborn/pred_layers_extrap/pred'+str(node)+'.pdf')
    plt.close()