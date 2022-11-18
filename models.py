import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

# Group A : Baseline model -- Multi-Layer Perception
class MLP(nn.Module):
    def __init__(self,nfeat,nout,dropout):
        super(MLP, self).__init__()
        self.dropout = dropout
        self.layer1 = Parameter(torch.FloatTensor(nfeat, 64))
        self.bias1 = Parameter(torch.FloatTensor(64))
        self.layer2 = Parameter(torch.FloatTensor(64, 32))
        self.bias2 = Parameter(torch.FloatTensor(32))
        self.layer3 = Parameter(torch.FloatTensor(32, 16))
        self.bias3 = Parameter(torch.FloatTensor(16))
        self.layer4 = Parameter(torch.FloatTensor(16, nout))
        self.bias4 = Parameter(torch.FloatTensor(nout))
        self.reset_parameters()
    
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.layer3.size(1))
        self.layer3.data.uniform_(-stdv, stdv)
        self.bias3.data.uniform_(-stdv, stdv)
        stdv = 1. / math.sqrt(self.layer2.size(1))
        self.layer2.data.uniform_(-stdv, stdv)
        self.bias2.data.uniform_(-stdv, stdv)
        stdv = 1. / math.sqrt(self.layer1.size(1))
        self.layer1.data.uniform_(-stdv, stdv)
        self.bias1.data.uniform_(-stdv, stdv)
        stdv = 1. / math.sqrt(self.layer4.size(1))
        self.layer4.data.uniform_(-stdv, stdv)
        self.bias4.data.uniform_(-stdv, stdv)
    
    def forward(self,x):
        x = torch.mm(x,self.layer1)
        x = x + self.bias1
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.mm(x,self.layer2)
        x = x + self.bias2
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.mm(x,self.layer3)
        x = x + self.bias3
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.mm(x,self.layer4)
        x = x + self.bias4
        return x

# Group B : Baseline model -- Graph Convolution Neural Network 
class GCN(nn.Module):
    def __init__(self,nfeat,nout,dropout):
        super(GCN, self).__init__()
        self.gcn = GraphConvolution(nfeat, 64)
        self.dropout = dropout
        self.layer1 = Parameter(torch.FloatTensor(64, 32))
        self.bias1 = Parameter(torch.FloatTensor(32))
        self.layer2 = Parameter(torch.FloatTensor(32, 16))
        self.bias2 = Parameter(torch.FloatTensor(16))
        self.layer3 = Parameter(torch.FloatTensor(16, nout))
        self.bias3 = Parameter(torch.FloatTensor(nout))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.layer1.size(1))
        self.layer1.data.uniform_(-stdv, stdv)
        self.bias1.data.uniform_(-stdv, stdv)
        stdv = 1. / math.sqrt(self.layer2.size(1))
        self.layer2.data.uniform_(-stdv, stdv)
        self.bias2.data.uniform_(-stdv, stdv)
        stdv = 1. / math.sqrt(self.layer3.size(1))
        self.layer3.data.uniform_(-stdv, stdv)
        self.bias3.data.uniform_(-stdv, stdv)

    def forward(self,x,mx):
        x = F.relu(self.gcn(x,mx))
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.mm(x,self.layer1)
        x = x + self.bias1
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.mm(x,self.layer2)
        x = x + self.bias2
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.mm(x,self.layer3)
        x = x + self.bias3
        return x

# Group C : Downgraded Heterogeneous Graph Convolution Network (BiGCN)
class BiGCN(nn.Module):
    def __init__(self,nfeat,nout,dropout,proportion):
        super(BiGCN,self).__init__()
        self.gcn = GraphConvolution(nfeat,proportion)
        self.dropout = dropout
        self.layer0 = Parameter(torch.FloatTensor(nfeat, 32))
        self.bias0 = Parameter(torch.FloatTensor(32))
        self.layer1 = Parameter(torch.FloatTensor(64, 32))
        self.bias1 = Parameter(torch.FloatTensor(32))
        self.layer2 = Parameter(torch.FloatTensor(32, 16))
        self.bias2 = Parameter(torch.FloatTensor(16))
        self.layer3 = Parameter(torch.FloatTensor(16, nout))
        self.bias3 = Parameter(torch.FloatTensor(nout))
        self.reset_parameters()
    
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.layer0.size(1))
        self.layer0.data.uniform_(-stdv, stdv)
        self.bias0.data.uniform_(-stdv, stdv)
        stdv = 1. / math.sqrt(self.layer1.size(1))
        self.layer1.data.uniform_(-stdv, stdv)
        self.bias1.data.uniform_(-stdv, stdv)
        stdv = 1. / math.sqrt(self.layer2.size(1))
        self.layer2.data.uniform_(-stdv, stdv)
        self.bias2.data.uniform_(-stdv, stdv)
        stdv = 1. / math.sqrt(self.layer3.size(1))
        self.layer3.data.uniform_(-stdv, stdv)
        self.bias3.data.uniform_(-stdv, stdv)
    
    def forward(self,x,mx):
        feat = F.relu(self.gcn(x,mx))
        x = torch.mm(x,self.layer0)
        x = x + self.bias0
        x = torch.cat([feat,x],1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.mm(x,self.layer1)
        x = x + self.bias1
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.mm(x,self.layer2)
        x = x + self.bias2
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.mm(x,self.layer3)
        x = x + self.bias3
        return x

# Group D : Standard Heterogeneous Graph Convolution Network (TriGCN)
class TriGCN(nn.Module):
    def __init__(self,nfeat,nout,dropout,proportion):
        super(TriGCN,self).__init__()
        self.gc_out = GraphConvolution(nfeat,int(proportion/2))
        self.gc_in = GraphConvolution(nfeat,int(proportion/2))
        self.dropout = dropout
        self.layer0 = Parameter(torch.FloatTensor(nfeat, (64-proportion)))
        self.bias0 = Parameter(torch.FloatTensor((64-proportion)))
        self.layer1 = Parameter(torch.FloatTensor(64, 32))
        self.bias1 = Parameter(torch.FloatTensor(32))
        self.layer2 = Parameter(torch.FloatTensor(32, 16))
        self.bias2 = Parameter(torch.FloatTensor(16))
        self.layer3 = Parameter(torch.FloatTensor(16, nout))
        self.bias3 = Parameter(torch.FloatTensor(nout))
        self.reset_parameters()
    
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.layer0.size(1))
        self.layer0.data.uniform_(-stdv, stdv)
        self.bias0.data.uniform_(-stdv, stdv)
        stdv = 1. / math.sqrt(self.layer1.size(1))
        self.layer1.data.uniform_(-stdv, stdv)
        self.bias1.data.uniform_(-stdv, stdv)
        stdv = 1. / math.sqrt(self.layer3.size(1))
        self.layer3.data.uniform_(-stdv, stdv)
        self.bias3.data.uniform_(-stdv, stdv)
        stdv = 1. / math.sqrt(self.layer2.size(1))
        self.layer2.data.uniform_(-stdv, stdv)
        self.bias2.data.uniform_(-stdv, stdv)
        
    def forward(self,x,mx):
        out_feat = F.relu(self.gc_out(x,mx))
        in_feat = F.relu(self.gc_in(x,mx.T))
        x = torch.mm(x,self.layer0)
        x = x + self.bias0
        x = torch.cat([x,in_feat,out_feat],1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.mm(x,self.layer1)
        x = x + self.bias1
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.mm(x,self.layer2)
        x = x + self.bias2
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.mm(x,self.layer3)
        x = x + self.bias3
        return x

# Group E-1 : Extended Heterogeneous Graph Convolution Network (Adj-Bi-GCN)
class AdjBiGCN(nn.Module):
    def __init__(self,nfeat,nout,dropout,proportion):
        super(AdjBiGCN,self).__init__()
        self.gc_flow = GraphConvolution(nfeat, proportion)
        self.gc_self = GraphConvolution(nfeat, (64-proportion))
        self.dropout = dropout
        self.layer1 = Parameter(torch.FloatTensor(64, 32))
        self.bias1 = Parameter(torch.FloatTensor(32))
        self.layer2 = Parameter(torch.FloatTensor(32, 16))
        self.bias2 = Parameter(torch.FloatTensor(16))
        self.layer3 = Parameter(torch.FloatTensor(16, nout))
        self.bias3 = Parameter(torch.FloatTensor(nout))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.layer3.size(1))
        self.layer3.data.uniform_(-stdv, stdv)
        self.bias3.data.uniform_(-stdv, stdv)
        stdv = 1. / math.sqrt(self.layer2.size(1))
        self.layer2.data.uniform_(-stdv, stdv)
        self.bias2.data.uniform_(-stdv, stdv)
        stdv = 1. / math.sqrt(self.layer1.size(1))
        self.layer1.data.uniform_(-stdv, stdv)
        self.bias1.data.uniform_(-stdv, stdv)
    
    def forward(self,x,mx,adj):
        out_feat = F.relu(self.gc_flow(x,mx))
        x = F.relu(self.gc_self(x,adj))
        x = torch.cat([x,out_feat],1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.mm(x,self.layer1)
        x = x + self.bias1
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.mm(x,self.layer2)
        x = x + self.bias2
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.mm(x,self.layer3)
        x = x + self.bias3
        return x

# Croup E-2 : Extended Heterogeneous Graph Convolution Network (Adj-Tri-GCN)
class AdjTriGCN(nn.Module):
    def __init__(self,nfeat,nout,dropout,proportion):
        super(AdjTriGCN,self).__init__()
        self.gc_out = GraphConvolution(nfeat, int(proportion/2))
        self.gc_in = GraphConvolution(nfeat, int(proportion/2))
        self.gc_self = GraphConvolution(nfeat, (64-proportion))
        self.dropout = dropout
        self.layer1 = Parameter(torch.FloatTensor(64, 32))
        self.bias1 = Parameter(torch.FloatTensor(32))
        self.layer2 = Parameter(torch.FloatTensor(32, 16))
        self.bias2 = Parameter(torch.FloatTensor(16))
        self.layer3 = Parameter(torch.FloatTensor(16, nout))
        self.bias3 = Parameter(torch.FloatTensor(nout))
        self.reset_parameters()
    
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.layer3.size(1))
        self.layer3.data.uniform_(-stdv, stdv)
        self.bias3.data.uniform_(-stdv, stdv)
        stdv = 1. / math.sqrt(self.layer2.size(1))
        self.layer2.data.uniform_(-stdv, stdv)
        self.bias2.data.uniform_(-stdv, stdv)
        stdv = 1. / math.sqrt(self.layer1.size(1))
        self.layer1.data.uniform_(-stdv, stdv)
        self.bias1.data.uniform_(-stdv, stdv)
    
    def forward(self,x,mx,adj):
        out_feat = F.relu(self.gc_out(x,mx))
        in_feat = F.relu(self.gc_in(x,mx.T))
        x = F.relu(self.gc_self(x,adj))
        x = torch.cat([out_feat,in_feat,x],1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.mm(x,self.layer1)
        x = x + self.bias1
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.mm(x,self.layer2)
        x = x + self.bias2
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.mm(x,self.layer3)
        x = x + self.bias3
        return x

# Graph Convolution Layer
class GraphConvolution(Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
    
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self,input,adj):
        support = torch.mm(input, self.weight)
        output = torch.mm(adj,support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output