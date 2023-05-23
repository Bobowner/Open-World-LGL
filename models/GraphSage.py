from models.BasicNet import BasicNet
from torch_geometric.nn import SAGEConv, GraphConv

class GraphSAGE(BasicNet):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout):
        super(GraphSAGE, self).__init__(in_feats, n_hidden, n_classes, n_layers, activation, dropout, SAGEConv)

    def final_parameters(self):
        yield self.layers[-1].lin_rel.weight
        yield self.layers[-1].lin_rel.bias
        yield self.layers[-1].lin_root.weight