import numpy as np 
import matplotlib.pyplot as plt 
import torch_geometric as pyg
import torch
import pandas as pd
from torch_geometric.utils import to_undirected
import networkx as nx
from utils import dataset_gen, sublists
import seaborn as sns
import os
import argparse
from torch_geometric.loader import DataLoader
import random
import torch.nn as nn
from torch.nn import Linear, Sequential, BatchNorm1d, ReLU, Tanh, Sigmoid
import torch.nn.functional as F
from torch_geometric.nn import GraphConv, GINConv, GCNConv, TransformerConv
from model import GNN
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
def main(epochs, conv_types):
    abl_type = 'architecture'
    log = []
    lr = 0.001
    num_it = 5
    indicators = ['WSS','OSI'] #,'TAWSS'
    for sub_ind in sublists(indicators):
        if len(sub_ind)>0: #only pairs of indicators, or the three of them
            for conv_type in conv_types:
                print(conv_type)
                train_perc = [25,50,100]
                train_set = []
                for perc in train_perc:
                    train_set+=dataset_gen(perc,sub_ind)
                test_perc = 75
                test_set = dataset_gen(test_perc,sub_ind)

                log_name = '_'.join([conv_type,'loss'])
                log_path = 'interp/' + '_'.join([abl_type]+sub_ind)

                if not os.path.exists(log_path):
                    os.makedirs(log_path)
                input_size = 7 # dataset features: Time, Press_SA, Press_abd, FlowRate, coord(x), coord(y), coord(z)
                hidden_size = 32
                output_size = len(sub_ind)
                num_layers = 3
                for it in range(num_it):

                    print(f'Indicators: {sub_ind}, Conv type: {conv_type}, Iteration: {it}')
                    pyg.seed_everything((it+1)*10)
                    random.shuffle(train_set)
                    random.shuffle(test_set)
                    train_loader = DataLoader(train_set, batch_size = 25, shuffle=True)
                    test_loader = DataLoader(test_set, batch_size = 25, shuffle=True)

                    model = GNN(input_size,hidden_size,output_size,num_layers,conv_type=conv_type, device = device)
                    optimizer = torch.optim.Adam(model.parameters(), lr=lr, amsgrad=True)

                    @torch.enable_grad()
                    def train():
                        model.train()

                        total_loss = 0
                        for data in train_loader:
                            data = data.to(device)
                            optimizer.zero_grad()
                            pred = model(data.x, data.edge_index)
                            loss = F.mse_loss(pred, data.y.float())
                            loss.backward()
                            optimizer.step()
                            total_loss += float(loss) * data.num_graphs
                        return total_loss / len(train_loader.dataset)

                    @torch.no_grad()
                    def test(loader):
                        model.eval()
                        total_loss = 0
                        for data in loader:
                            data = data.to(device)
                            pred = model(data.x, data.edge_index)            
                            loss = F.mse_loss(pred, data.y.float())
                            total_loss += float(loss) * data.num_graphs
                        return total_loss / len(train_loader.dataset)

                    #training phase
                    for epoch in range(1, epochs+1):
                        train_loss = train()
                        test_loss = test(test_loader)
                        log.append({'epoch':epoch, 'train_loss': train_loss, 'test_loss': test_loss, 'it': it })
                        if epoch%50 == 0:
                            torch.save(model.state_dict(),log_path+'/'+log_name + str(it)+'.pt')
                        if epoch%1000 == 0:
                            print(f'Epoch {epoch}')
                            print(f'Train loss: {train_loss}')
                            print(f'Test loss: {test_loss}' )

                loss_df = pd.DataFrame.from_records(log)
                loss_df.to_csv(log_path+'/'+log_name + '.csv')

                #visualization of training and test loss over the epochs
                plt.figure()
                plt.title(conv_type)
                sns.lineplot(data = loss_df[['epoch', 'train_loss', 'it']], x='epoch', y='train_loss',legend='brief', label='Train Loss')
                plt.xlabel("Epoch")
                plt.ylabel("Loss")
                sns.lineplot(data = loss_df[['epoch', 'test_loss', 'it']], x='epoch', y='test_loss',legend='brief', label='Test Loss')
                plt.yscale('log')
                plt.savefig(log_path+'/'+log_name + '.png')
                plt.close()


# plt.savefig(str(num_layers)+'_layers_test_loss.png')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Aneurysm rupture risk prediction")
    #parser.add_argument("--conv", help="type(s) of Graph Convolutional Layer (passed as list string)", type=str, default='[gconv]')
    parser.add_argument("--epochs", help="extra features", type=int, default=5000)
    args = parser.parse_args()
    conv_types = ['gconv','gin','gcn','gtr']
    main(args.epochs, conv_types)
