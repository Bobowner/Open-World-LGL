""" A simple yet generic MLP """
import torch
import numpy as np
import scipy.sparse as sp
import torch.nn as nn
import torch.nn
#from . import MLP
from torch.nn import Dropout, Linear, LayerNorm
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.utils import to_dense_adj

def get_A_r(adj, r):
    adj_label = adj
    if r == 1:
        adj_label = adj_label
    elif r == 2:
        adj_label = adj_label@adj_label
    elif r == 3:
        adj_label = adj_label@adj_label@adj_label
    elif r == 4:
        adj_label = adj_label@adj_label@adj_label@adj_label
    return adj_label



def getAr(edge_index, n_nodes, r):
    
    #nnodes = edge_index.shape[1]
    edge_index, edge_weight = gcn_norm(edge_index, edge_weight=None, num_nodes=nodes)
    A = to_dense_adj(edge_index, batch=None, edge_attr=edge_weight, max_num_nodes=n_nodes).squeeze()
    tmp = A
    
    for _ in range(r-1):
        tmp = torch.mm(tmp, A)
    return tmp

#deprecated
def aug_normalized_adjacency(adj):
    
    adj = adj.to_dense() + torch.eye(adj.shape[0])
    row_sum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(row_sum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    res = d_mat_inv_sqrt.dot(adj)
    res=res.dot(d_mat_inv_sqrt)

    return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo()
        
        
def Ncontrast(x_dis, adj_label, tau):
    """
    compute the Ncontrast loss
    """
    
    if type(x_dis) == int and x_dis == 0:
        raise Exception('Error: ', 'Call model first, to estimate the Ncontrast loss.')
    
  
    if adj_label.shape[0]==0:
        raise Exception('Error: ', 'Set adj_potenz!=null in yaml to use graph_mlp')
        
    
    x_dis = torch.exp( tau * x_dis)
    x_dis_sum = torch.sum(x_dis, 1)
    x_dis_sum_pos = torch.sum(x_dis*adj_label, 1)
    loss = -torch.log(x_dis_sum_pos * (x_dis_sum**(-1))+1e-8).mean()
    return loss


def get_feature_dis(x):
    """
    x :           batch_size x nhid
    x_dis(i,j):   item means the similarity between x(i) and x(j).
    """

    x_dis = x@x.T
    mask = torch.eye(x_dis.shape[0], device=x.device)
    x_sum = torch.sum(x**2, 1).reshape(-1, 1)
    x_sum = torch.sqrt(x_sum).reshape(-1, 1)
    x_sum = x_sum @ x_sum.T
    x_dis = x_dis*(x_sum**(-1))
    x_dis = (1-mask) * x_dis
    return x_dis

class GraphMLP(nn.Module):
    def __init__(self, in_feats, n_hidden, n_classes, n_layers, activation, dropout, tau):
        super(GraphMLP, self).__init__()
        self.n_hidden = n_hidden
        self.n_layers = n_layers
       
        self.fc_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()
        
        self.fc1 = Linear(in_feats, n_hidden)
        self.act_fn = torch.nn.functional.gelu
       
        for l in range(self.n_layers-1):
            self.fc_layers.append(Linear(n_hidden, n_hidden))
            self.norm_layers.append(LayerNorm(n_hidden, eps=1e-6))

        
        self.tau = tau
        
        
        for l in range(self.n_layers-1):
            nn.init.xavier_uniform_(self.fc_layers[l].weight)
            nn.init.normal_(self.fc_layers[l].bias, std=1e-6)

        self.dropout = Dropout(dropout)
        self.layernorm = LayerNorm(n_hidden, eps=1e-6)
        
        self.classifier = nn.Linear(self.n_hidden, n_classes)
        
        self.x_dis = 0
        
    def forward(self, data):
        
        x, edge_index, edge_weights = data.x, data.edge_index, data.edge_attr
                
      
        
        x = self.fc1(x)
        for l in range(self.n_layers-1):
            x = self.act_fn(x)
            x = self.norm_layers[l](x)
            x = self.dropout(x)
            x = self.fc_layers[l](x)
                
        
        feature_cls = x
        Z = x
        
        if self.training:
            x_dis = get_feature_dis(Z)
       
        class_feature = self.classifier(feature_cls)
        #class_logits = F.log_softmax(class_feature, dim=1)

        if self.training:
            self.x_dis = x_dis
            
        return class_feature
        
    def model_loss(self, normalized_adj):
        n_contrast_loss = Ncontrast(self.x_dis, normalized_adj, tau=self.tau)
        return n_contrast_loss
    def reset_parameters(self):
        #self.mlp.reset_parameters()
        self.reset_final_parameters()

    def reset_final_parameters(self):
        self.classifier.reset_parameters()

    def final_parameters(self):
        yield self.classifier.weight
        yield self.classifier.bias