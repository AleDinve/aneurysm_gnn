import numpy as np 
from scipy.interpolate import CubicSpline
import pandas as pd
import matplotlib.pyplot as plt 
import torch_geometric as pyg
import torch
from torch_geometric.utils import to_undirected
import networkx as nx
from itertools import combinations

def interp(x, y, xnew, type='linear'):
    if type=='linear':
        ynew = np.interp(xnew, x, y)
    else:
        spl = CubicSpline(x,y)
        ynew = spl(xnew)
    return ynew

def sublists(lst):
    sublists_list = []
    for i in range(0, len(lst) + 1):
        # Use the 'combinations' function to generate all combinations of 'my_list' of length 'i'
        sublists_list += [list(x) for x in combinations(lst, i)]
    return sublists_list

def minmaxscaler(v):
    # scaler between 0 and 1
    v_oldmin = v.min()
    v_oldmax = v.max()
    return (v-v_oldmin)/(v_oldmax-v_oldmin)

def data_processing():
    from utils import interp, minmaxscaler
    df = pd.read_excel('Boundary_Condition_Flow_Rate_Pressure.xlsx',sheet_name='Pressure (2)',header=2)
    # SA outlet pressure
    dfr = df[['Time Scale','Pressure(mmHg)']]
    dfr = dfr[dfr['Time Scale']<=1]
    x = np.array(df['Time Scale'].values)
    y = np.array(df['Pressure(mmHg)'].values)
    xnew = np.linspace(0,1, 500)
    ynew = interp(x,y, xnew, 'linear')
    raw = {'Time':xnew, 'Pressure_SA':minmaxscaler(ynew)}
    # abd outlet pressure
    dfr = df[['Time Scale.1','Pressure(mmHg).1']]
    dfr = dfr[dfr['Time Scale.1']<=1]
    x = np.array(df['Time Scale.1'].values)
    y = np.array(df['Pressure(mmHg).1'].values)
    xnew = np.linspace(0,1, 500)
    ynew = interp(x,y, xnew, 'linear')
    raw['Pressure_abd'] = minmaxscaler(ynew)
    df2 = pd.read_excel('Boundary_Condition_Flow_Rate_Pressure.xlsx',sheet_name='Flow Rate (2)',header=2)
    dfr2 = df2[['Time Scale.2','Flow Rate(mL/s).2' ]]
    dfr2 = dfr2[dfr2['Time Scale.2']<=1]
    x = np.array(dfr2['Time Scale.2'].values)
    y = np.array(dfr2['Flow Rate(mL/s).2'].values)
    xnew = np.linspace(0,1, 500)
    ynew = interp(x,y, xnew, 'linear')
    raw['FlowRate'] = minmaxscaler(ynew)
    df_processed = pd.DataFrame.from_records(raw)[['Time', 'FlowRate', 'Pressure_SA', 'Pressure_abd']]
    df_processed.to_csv('processed_features.csv')

def dataset_gen(percentage, indicators_names = ['WSS','OSI']):
    dataset = []
    data = np.load(str(percentage)+'Percent.npz')
    #nodes coordinates
    x_loaded = data['x']
    y_loaded = data['y']
    z_loaded = data['z']
    #triangular elements
    connectivity = data['connectivity'] 
    #indicators: list of strings, sublist of ['WSS', 'OSI', 'TAWSS']
    indicators = np.stack([minmaxscaler(data[ind_name]) for ind_name in indicators_names],axis = -1)

    edge_index = [[],[]]
    # edge extraction from triangular elements
    for i in range(connectivity.shape[0]):
        edge_index[0].append(connectivity[i,0])
        edge_index[1].append(connectivity[i,1])
        edge_index[0].append(connectivity[i,0])
        edge_index[1].append(connectivity[i,2])
        edge_index[0].append(connectivity[i,1])
        edge_index[1].append(connectivity[i,2])
    a = np.unique(edge_index, axis=1)
    #mesh = pyg.data.Data(edge_index=to_undirected(torch.tensor(a, dtype=torch.int64)))
    
    coord = np.stack([x_loaded,y_loaded,z_loaded]).T
    # G = pyg.utils.to_networkx(mesh, to_undirected=True)
    # node_xyz = np.array([coord[v] for v in sorted(G)])
    # edge_xyz = np.array([(coord[u], coord[v]) for u, v in G.edges()])
    df_processed = pd.read_csv('processed_features.csv', index_col=0)
    features = df_processed.to_numpy()
    F = torch.from_numpy(features)
    
    for i in range(F.shape[0]):
        x = torch.cat([F[i,:].repeat(x_loaded.shape[0],1),torch.from_numpy(coord)],dim=1)
        # mesh.y = torch.from_numpy(WSS_loaded.T)[:,i].unsqueeze(-1)
        y = torch.from_numpy((indicators[i,:,:]))
        mesh = pyg.data.Data(x=x, y=y, edge_index=to_undirected(torch.tensor(a, dtype=torch.int64)))
        mesh.num_nodes = x_loaded.shape[0]
        dataset.append(mesh)
    return dataset


if __name__=='__main__':
    dataset_gen(50,['WSS'])