import torch
import pickle
import numpy as np
import pandas as pd

def normalize(mx):
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = np.diag(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def load_london_data(path='./data/london_data/'):
    print("Loading london dataset...")

    # Load Features - FloatTensor - torch.Size([786,72])
    df_zonevec = pd.read_csv(path+'df_zonevec.csv')
    zonevec = df_zonevec.values[:,1:]
    features = torch.FloatTensor(zonevec)

    # Load Labels - FloatTensor - torch.Size([786,8])
    df_area = pd.read_csv(path+'df_LUarea.csv')
    df_labels = df_area.div(df_area.sum(axis=1),axis=0)
    labels = df_labels.values
    labels = torch.FloatTensor(labels)

    # Load Spatial Interaction (Flow - ndarray (786,786) /Distance - ndarray (786,786)/Adjacency - ndarray (786,786))
    flow = np.load(path+'bike_od_mx.npy')
    df_dis = pd.read_csv(path+'df_dis.csv')
    dis = df_dis.values
    with open(path+"ls_edge.pkl","rb") as f:
        ls_edge=pickle.load(f)
    adj = np.zeros(dis.shape)
    for i in ls_edge:
        adj[i[0],i[1]] = 1
        adj[i[1],i[0]] = 1
    adj = adj + np.eye(adj.shape[0])

    return features, labels, flow, dis, adj

def load_shenzhen_data(path='./data/shenzhen_data/'):
    print("Loading shenzhen dataset...")

    # Load Features - FloatTensor - torch.Size([12345,72])
    df_zonevec = pd.read_csv(path+'df_zonevec.csv')
    zonevec = df_zonevec.values[:,2:]
    features = torch.FloatTensor(zonevec)

    # Load Labels - FloatTensor - torch.Size([12345,8])
    df_area = pd.read_csv(path+'df_area.csv')
    labels = df_area.values[:,2:]
    labels = torch.FloatTensor(labels)

    # Load Spatial Interaction (Flow - ndarray (12345,12345)/Distance ndarray (12345,12345)/Adjacency ndarray (12345,12345))
    flow = np.load(path+'flow_mx.npy')
    dis = np.load(path+'wdis_normalize.npy')
    adj = np.load(path+'adj.npy')

    return features,labels,flow,dis,adj