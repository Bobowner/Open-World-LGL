import torch.nn as nn
from models.BasicNet import BasicNet
from torch_geometric.nn import GraphConv, GATConv

class GAT(BasicNet):
    def __init__(self,
                 in_feats,
                 n_hidden_per_head,
                 n_classes,
                 activation,
                 dropout,
                 attn_dropout,
                 heads,
                 add_self_loops=True):
        # sense-free call to superclass
        super(GAT, self).__init__(1, 1, 1, 1, 1, 1, GraphConv)

        # now override the network architecture, because GAT is special
        self.layers = nn.ModuleList()
        self.layers.append(GATConv(in_feats, n_hidden_per_head, heads=heads, dropout=attn_dropout, add_self_loops=add_self_loops))
        # hidden layers
        self.layers.append(GATConv(n_hidden_per_head * heads, n_classes, heads=1, concat=True,
                                   dropout=attn_dropout, add_self_loops=add_self_loops))
        # output layer
        self.activation = activation
        self.dropout = dropout

    def final_parameters(self):
        yield self.layers[-1].lin.weight
        yield self.layers[-1].bias
