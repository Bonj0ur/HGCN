import time
import torch
import warnings
import argparse
import numpy as np
import pandas as pd
import torch.optim as optim
import torch.nn.functional as F
from utils import load_london_data, load_shenzhen_data, normalize
from sklearn.model_selection import ShuffleSplit
from models import MLP, GCN, BiGCN, TriGCN, AdjBiGCN, AdjTriGCN

# Group A : Train Multi-Layer Perception
def train_MLP(epoch,model,optimizer,loss_fn,features,labels,train_index,test_index):
    mae_loss_fn = torch.nn.L1Loss()
    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(features)
    loss_train = loss_fn(F.log_softmax(output[train_index],dim=1),labels[train_index])
    mae_train = mae_loss_fn(F.softmax(output[train_index],dim=1),labels[train_index])
    cos_train = cos(F.softmax(output[train_index],dim=1),labels[train_index]).mean()
    loss_train.backward()
    optimizer.step()
    # test
    model.eval()
    output = model(features)
    loss_test = loss_fn(F.log_softmax(output[test_index],dim=1),labels[test_index])
    mae_test = mae_loss_fn(F.softmax(output[test_index],dim=1),labels[test_index])
    cos_test = cos(F.softmax(output[test_index],dim=1),labels[test_index]).mean()
    global total_loss
    global total_mae
    global total_cos
    if loss_test.item()<total_loss:
        total_loss = loss_test.item()
        total_mae = mae_test.item()
        total_cos = cos_test.item()
    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train(KL_Div): {:.4f}'.format(loss_train.item()),
          'MAE_train: {:.4f}'.format(mae_train.item()),
          'COS_train: {:.4f}'.format(cos_train.item()),
          'loss_test(KL_Div): {:.4f}'.format(loss_test.item()),
          'MAE_test: {:.4f}'.format(mae_test.item()),
          'COS_test: {:.4f}'.format(cos_test.item()),
          'time: {:.4f}s'.format(time.time() - t))

# Group B : Train Graph Convolution Neural Network
def train_GCN(epoch,model,optimizer,loss_fn,features,labels,mx,train_index,test_index):
    mae_loss_fn = torch.nn.L1Loss()
    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(features,mx)
    loss_train = loss_fn(F.log_softmax(output[train_index],dim=1),labels[train_index])
    mae_train = mae_loss_fn(F.softmax(output[train_index],dim=1),labels[train_index])
    cos_train = cos(F.softmax(output[train_index],dim=1),labels[train_index]).mean()
    loss_train.backward()
    optimizer.step()
    # test
    model.eval()
    output = model(features,mx)
    loss_test = loss_fn(F.log_softmax(output[test_index],dim=1),labels[test_index])
    mae_test = mae_loss_fn(F.softmax(output[test_index],dim=1),labels[test_index])
    cos_test = cos(F.softmax(output[test_index],dim=1),labels[test_index]).mean()
    global total_loss
    global total_mae
    global total_cos
    if loss_test.item()<total_loss:
        total_loss = loss_test.item()
        total_mae = mae_test.item()
        total_cos = cos_test.item()
    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train(KL_Div): {:.4f}'.format(loss_train.item()),
          'MAE_train: {:.4f}'.format(mae_train.item()),
          'COS_train: {:.4f}'.format(cos_train.item()),
          'loss_test(KL_Div): {:.4f}'.format(loss_test.item()),
          'MAE_test: {:.4f}'.format(mae_test.item()),
          'COS_test: {:.4f}'.format(cos_test.item()),
          'time: {:.4f}s'.format(time.time() - t))

# Group C : Train Downgraded Heterogeneous Graph Convolution Network (BiGCN)
def train_BiGCN(epoch,model,optimizer,loss_fn,features,labels,mx,train_index,test_index):
    mae_loss_fn = torch.nn.L1Loss()
    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(features,mx)
    loss_train = loss_fn(F.log_softmax(output[train_index],dim=1),labels[train_index])
    mae_train = mae_loss_fn(F.softmax(output[train_index],dim=1),labels[train_index])
    cos_train = cos(F.softmax(output[train_index],dim=1),labels[train_index]).mean()
    loss_train.backward()
    optimizer.step()
    # test
    model.eval()
    output = model(features,mx)
    loss_test = loss_fn(F.log_softmax(output[test_index],dim=1),labels[test_index])
    mae_test = mae_loss_fn(F.softmax(output[test_index],dim=1),labels[test_index])
    cos_test = cos(F.softmax(output[test_index],dim=1),labels[test_index]).mean()
    global total_loss
    global total_mae
    global total_cos
    if loss_test.item()<total_loss:
        total_loss = loss_test.item()
        total_mae = mae_test.item()
        total_cos = cos_test.item()
    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train(KL_Div): {:.4f}'.format(loss_train.item()),
          'MAE_train: {:.4f}'.format(mae_train.item()),
          'COS_train: {:.4f}'.format(cos_train.item()),
          'loss_test(KL_Div): {:.4f}'.format(loss_test.item()),
          'MAE_test: {:.4f}'.format(mae_test.item()),
          'COS_test: {:.4f}'.format(cos_test.item()),
          'time: {:.4f}s'.format(time.time() - t))

# Group D : Train Standard Heterogeneous Graph Convolution Network (TriGCN)
def train_TriGCN(epoch,model,optimizer,loss_fn,features,labels,mx,train_index,test_index):
    mae_loss_fn = torch.nn.L1Loss()
    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(features,mx)
    loss_train = loss_fn(F.log_softmax(output[train_index],dim=1),labels[train_index])
    mae_train = mae_loss_fn(F.softmax(output[train_index],dim=1),labels[train_index])
    cos_train = cos(F.softmax(output[train_index],dim=1),labels[train_index]).mean()
    loss_train.backward()
    optimizer.step()
    # test
    model.eval()
    output = model(features,mx)
    loss_test = loss_fn(F.log_softmax(output[test_index],dim=1),labels[test_index])
    mae_test = mae_loss_fn(F.softmax(output[test_index],dim=1),labels[test_index])
    cos_test = cos(F.softmax(output[test_index],dim=1),labels[test_index]).mean()
    global total_loss
    global total_mae
    global total_cos
    if loss_test.item()<total_loss:
        total_loss = loss_test.item()
        total_mae = mae_test.item()
        total_cos = cos_test.item()
    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train(KL_Div): {:.4f}'.format(loss_train.item()),
          'MAE_train: {:.4f}'.format(mae_train.item()),
          'COS_train: {:.4f}'.format(cos_train.item()),
          'loss_test(KL_Div): {:.4f}'.format(loss_test.item()),
          'MAE_test: {:.4f}'.format(mae_test.item()),
          'COS_test: {:.4f}'.format(cos_test.item()),
          'time: {:.4f}s'.format(time.time() - t))

# Group E-1 : Extended Heterogeneous Graph Convolution Network (Adj-Bi-GCN)
def train_AdjBiGCN(epoch,model,optimizer,loss_fn,features,labels,mx,adj,train_index,test_index):
    mae_loss_fn = torch.nn.L1Loss()
    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(features,mx,adj)
    loss_train = loss_fn(F.log_softmax(output[train_index],dim=1),labels[train_index])
    mae_train = mae_loss_fn(F.softmax(output[train_index],dim=1),labels[train_index])
    cos_train = cos(F.softmax(output[train_index],dim=1),labels[train_index]).mean()
    loss_train.backward()
    optimizer.step()
    # test
    model.eval()
    output = model(features,mx,adj)
    loss_test = loss_fn(F.log_softmax(output[test_index],dim=1),labels[test_index])
    mae_test = mae_loss_fn(F.softmax(output[test_index],dim=1),labels[test_index])
    cos_test = cos(F.softmax(output[test_index],dim=1),labels[test_index]).mean()
    global total_loss
    global total_mae
    global total_cos
    if loss_test.item()<total_loss:
        total_loss = loss_test.item()
        total_mae = mae_test.item()
        total_cos = cos_test.item()
    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train(KL_Div): {:.4f}'.format(loss_train.item()),
          'MAE_train: {:.4f}'.format(mae_train.item()),
          'COS_train: {:.4f}'.format(cos_train.item()),
          'loss_test(KL_Div): {:.4f}'.format(loss_test.item()),
          'MAE_test: {:.4f}'.format(mae_test.item()),
          'COS_test: {:.4f}'.format(cos_test.item()),
          'time: {:.4f}s'.format(time.time() - t))

# Croup E-2 : Extended Heterogeneous Graph Convolution Network (Adj-Tri-GCN)
def train_AdjTriGCN(epoch,model,optimizer,loss_fn,features,labels,mx,adj,train_index,test_index):
    mae_loss_fn = torch.nn.L1Loss()
    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(features,mx,adj)
    loss_train = loss_fn(F.log_softmax(output[train_index],dim=1),labels[train_index])
    mae_train = mae_loss_fn(F.softmax(output[train_index],dim=1),labels[train_index])
    cos_train = cos(F.softmax(output[train_index],dim=1),labels[train_index]).mean()
    loss_train.backward()
    optimizer.step()
    # test
    model.eval()
    output = model(features,mx,adj)
    loss_test = loss_fn(F.log_softmax(output[test_index],dim=1),labels[test_index])
    mae_test = mae_loss_fn(F.softmax(output[test_index],dim=1),labels[test_index])
    cos_test = cos(F.softmax(output[test_index],dim=1),labels[test_index]).mean()
    global total_loss
    global total_mae
    global total_cos
    if loss_test.item()<total_loss:
        total_loss = loss_test.item()
        total_mae = mae_test.item()
        total_cos = cos_test.item()
    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train(KL_Div): {:.4f}'.format(loss_train.item()),
          'MAE_train: {:.4f}'.format(mae_train.item()),
          'COS_train: {:.4f}'.format(cos_train.item()),
          'loss_test(KL_Div): {:.4f}'.format(loss_test.item()),
          'MAE_test: {:.4f}'.format(mae_test.item()),
          'COS_test: {:.4f}'.format(cos_test.item()),
          'time: {:.4f}s'.format(time.time() - t))

# Training settings
warnings.filterwarnings('ignore')
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=30, help='Random seed.')
parser.add_argument('--epochs', type=int, default=400, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
parser.add_argument('--k-fold', type=int, default=30, help='K-fold validation.')
parser.add_argument('--train_size', type=float, default=0.7, help='Train size.')
parser.add_argument('--train_mode',type=int,default=1,help='Train mode.')
parser.add_argument('--dist_beta',type=float,default=1.5,help='Distance Beta.')
parser.add_argument('--proportion',type=int,default=32,help='Flow proportion')
parser.add_argument('--dataset',type=int,default=1,help='1 for London & 2 for Shenzhen')

# Enable cuda
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

# Set seed
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Load data
if args.dataset == 1:
    features, labels, flow, dis, adj = load_london_data()
else:
    features, labels, flow, dis, adj = load_shenzhen_data()

# Mode setting
if args.train_mode == 1:
    pass
elif args.train_mode == 2:
    if args.dataset == 1:
        beta = args.dist_beta
        def weight(x,beta=beta):
            return np.log((1+max_data**beta)/(1+x**beta))
        df_dis = pd.DataFrame(dis)
        max_data = df_dis.max().max()
        df_wdis = df_dis.applymap(weight)
        mx = df_wdis.values
        mx = normalize(mx)
        mx = torch.FloatTensor(mx)
    else:
        mx = torch.FloatTensor(dis)
elif args.train_mode == 3:
    flow = flow.T
    flow = normalize(flow)
    mx = torch.FloatTensor(flow)
elif args.train_mode == 4:
    flow = normalize(flow)
    mx = torch.FloatTensor(flow)
elif args.train_mode == 5:
    flow = (flow + flow.T)/2.0
    flow = normalize(flow)
    mx = torch.FloatTensor(flow)
elif args.train_mode == 6:
    adj = normalize(adj)
    mx = torch.FloatTensor(adj)
elif args.train_mode == 7:
    np.fill_diagonal(flow,0.0)
    flow = flow.T
    flow = normalize(flow)
    mx = torch.FloatTensor(flow)
elif args.train_mode == 8:
    np.fill_diagonal(flow,0.0)
    flow = (flow + flow.T)/2.0
    flow = normalize(flow)
    mx = torch.FloatTensor(flow)
elif args.train_mode == 9:
    np.fill_diagonal(flow,0.0)
    flow = normalize(flow)
    mx = torch.FloatTensor(flow)
elif args.train_mode == 10:
    np.fill_diagonal(flow,0.0)
    flow = normalize(flow)
    mx = torch.FloatTensor(flow)
elif args.train_mode == 11:
    np.fill_diagonal(flow,0.0)
    flow = flow.T
    flow = normalize(flow)
    mx = torch.FloatTensor(flow)
    adj = adj + np.eye(adj.shape[0])
    adj = normalize(adj)
    adj = torch.FloatTensor(adj)
elif args.train_mode == 12:
    np.fill_diagonal(flow,0.0)
    flow = (flow+flow.T)/2.0
    flow = normalize(flow)
    mx = torch.FloatTensor(flow)
    adj = adj + np.eye(adj.shape[0])
    adj = normalize(adj)
    adj = torch.FloatTensor(adj)
elif args.train_mode == 13:
    np.fill_diagonal(flow,0.0)
    flow = normalize(flow)
    mx = torch.FloatTensor(flow)
    adj = adj + np.eye(adj.shape[0])
    adj = normalize(adj)
    adj = torch.FloatTensor(adj)
elif args.train_mode == 14:
    np.fill_diagonal(flow,0.0)
    flow = normalize(flow)
    mx = torch.FloatTensor(flow)
    adj = adj + np.eye(adj.shape[0])
    adj = normalize(adj)
    adj = torch.FloatTensor(adj)

# K-fold split
shuffle_folds = ShuffleSplit(n_splits=args.k_fold,train_size=args.train_size,random_state=args.seed).split(range(features.shape[0]))

# Start
total_loss = 10
total_mae = 10
total_cos = 10
avg_loss = []
avg_cos = []
avg_mae = []
for i, (train_index,test_index) in enumerate(shuffle_folds):
    print("------Fold {}:".format(i+1))
    train_index = torch.LongTensor(train_index)
    test_index = torch.LongTensor(test_index)
    # Model and optimizer
    if args.train_mode == 1:
        model = MLP(nfeat=features.shape[1],nout=labels.shape[1],dropout=args.dropout)
    elif args.train_mode == 2:
        model = GCN(nfeat=features.shape[1],nout=labels.shape[1],dropout=args.dropout)
    elif args.train_mode == 3:
        model = GCN(nfeat=features.shape[1],nout=labels.shape[1],dropout=args.dropout)
    elif args.train_mode == 4:
        model = GCN(nfeat=features.shape[1],nout=labels.shape[1],dropout=args.dropout)
    elif args.train_mode == 5:
        model = GCN(nfeat=features.shape[1],nout=labels.shape[1],dropout=args.dropout)
    elif args.train_mode == 6:
        model = GCN(nfeat=features.shape[1],nout=labels.shape[1],dropout=args.dropout)
    elif args.train_mode == 7:
        model = BiGCN(nfeat=features.shape[1],nout=labels.shape[1],dropout=args.dropout,proportion=args.proportion)
    elif args.train_mode == 8:
        model = BiGCN(nfeat=features.shape[1],nout=labels.shape[1],dropout=args.dropout,proportion=args.proportion)
    elif args.train_mode == 9:
        model = BiGCN(nfeat=features.shape[1],nout=labels.shape[1],dropout=args.dropout,proportion=args.proportion)
    elif args.train_mode == 10:
        model = TriGCN(nfeat=features.shape[1],nout=labels.shape[1],dropout=args.dropout,proportion=args.proportion)
    elif args.train_mode == 11:
        model = AdjBiGCN(nfeat=features.shape[1],nout=labels.shape[1],dropout=args.dropout,proportion=args.proportion)
    elif args.train_mode == 12:
        model = AdjBiGCN(nfeat=features.shape[1],nout=labels.shape[1],dropout=args.dropout,proportion=args.proportion)
    elif args.train_mode == 13:
        model = AdjBiGCN(nfeat=features.shape[1],nout=labels.shape[1],dropout=args.dropout,proportion=args.proportion)
    elif args.train_mode == 14:
        model = AdjTriGCN(nfeat=features.shape[1],nout=labels.shape[1],dropout=args.dropout,proportion=args.proportion)

    optimizer = optim.Adam(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)
    loss_fn = torch.nn.KLDivLoss(reduction='batchmean')
    # Cuda settings
    if args.cuda:
        model.cuda()
        features = features.cuda()
        labels = labels.cuda()
        train_index = train_index.cuda()
        test_index = test_index.cuda()
        # Mode setting
        if args.train_mode == 1:
            pass
        elif args.train_mode == 2:
            mx = mx.cuda()
        elif args.train_mode == 3:
            mx = mx.cuda()
        elif args.train_mode == 4:
            mx = mx.cuda()
        elif args.train_mode == 5:
            mx = mx.cuda()
        elif args.train_mode == 6:
            mx = mx.cuda()
        elif args.train_mode == 7:
            mx = mx.cuda()
        elif args.train_mode == 8:
            mx = mx.cuda()
        elif args.train_mode == 9:
            mx = mx.cuda()
        elif args.train_mode == 10:
            mx = mx.cuda()
        elif args.train_mode == 11:
            mx = mx.cuda()
            adj = adj.cuda()
        elif args.train_mode == 12:
            mx = mx.cuda()
            adj = adj.cuda()
        elif args.train_mode == 13:
            mx = mx.cuda()
            adj = adj.cuda()
        elif args.train_mode == 14:
            mx = mx.cuda()
            adj = adj.cuda()

    total_loss = 10
    # Train model
    t_total = time.time()
    for epoch in range(args.epochs):
        if args.train_mode == 1:
            train_MLP(epoch,model,optimizer,loss_fn,features,labels,train_index,test_index)
        elif args.train_mode == 2:
            train_GCN(epoch,model,optimizer,loss_fn,features,labels,mx,train_index,test_index)
        elif args.train_mode == 3:
            train_GCN(epoch,model,optimizer,loss_fn,features,labels,mx,train_index,test_index)
        elif args.train_mode == 4:
            train_GCN(epoch,model,optimizer,loss_fn,features,labels,mx,train_index,test_index)
        elif args.train_mode == 5:
            train_GCN(epoch,model,optimizer,loss_fn,features,labels,mx,train_index,test_index)
        elif args.train_mode == 6:
            train_GCN(epoch,model,optimizer,loss_fn,features,labels,mx,train_index,test_index)
        elif args.train_mode == 7:
            train_BiGCN(epoch,model,optimizer,loss_fn,features,labels,mx,train_index,test_index)
        elif args.train_mode == 8:
            train_BiGCN(epoch,model,optimizer,loss_fn,features,labels,mx,train_index,test_index)
        elif args.train_mode == 9:
            train_BiGCN(epoch,model,optimizer,loss_fn,features,labels,mx,train_index,test_index)
        elif args.train_mode == 10:
            train_TriGCN(epoch,model,optimizer,loss_fn,features,labels,mx,train_index,test_index)
        elif args.train_mode == 11:
            train_AdjBiGCN(epoch,model,optimizer,loss_fn,features,labels,mx,adj,train_index,test_index)
        elif args.train_mode == 12:
            train_AdjBiGCN(epoch,model,optimizer,loss_fn,features,labels,mx,adj,train_index,test_index)
        elif args.train_mode == 13:
            train_AdjBiGCN(epoch,model,optimizer,loss_fn,features,labels,mx,adj,train_index,test_index)
        elif args.train_mode == 14:
            train_AdjTriGCN(epoch,model,optimizer,loss_fn,features,labels,mx,adj,train_index,test_index)
    
    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
    print("Best Loss:{}".format(total_loss))
    avg_loss.append(total_loss)
    avg_mae.append(total_mae)
    avg_cos.append(total_cos)

# Log out
avg_loss = np.array(avg_loss)
avg_mae = np.array(avg_mae)
avg_cos = np.array(avg_cos)
print("Average Loss:{:.4f}".format(np.mean(avg_loss)))
print("Average MAE:{:.4f}".format(np.mean(avg_mae)))
print("Average COS:{:.4f}".format(np.mean(avg_cos)))
print("Std Loss:{:.4f}".format(np.std(avg_loss)))
print("Std MAE:{:.4f}".format(np.std(avg_mae)))
print("Std COS:{:.4f}".format(np.std(avg_cos)))