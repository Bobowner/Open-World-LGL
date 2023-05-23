from ood_models.BasicNet import BasicNet
from torch_geometric.nn import GraphConv, GCNConv

class GCN(BasicNet):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout,
                 improved=False):
        super(GCN, self).__init__(in_feats, n_hidden, n_classes, n_layers, activation, dropout, GCNConv, improved=improved)

    def final_parameters(self):
        yield self.layers[-1].lin.weight
        yield self.layers[-1].lin.bias
        yield self.layers[-1].weight