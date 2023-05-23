import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC
from torch_geometric.nn import GraphConv, GATConv, SAGEConv


class BasicNet(nn.Module, ABC):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout,
                 layer, **layer_kwargs):
        super(BasicNet, self).__init__()
        self.layers = nn.ModuleList()
        
        if n_layers == 1:
            self.layers.append(layer(in_feats, n_classes, **layer_kwargs))
            self.dropout = dropout
        
        else:
            self.layers.append(layer(in_feats, n_hidden, **layer_kwargs))
            # hidden layers
            for i in range(n_layers - 1):
                self.layers.append(layer(n_hidden, n_hidden, **layer_kwargs))
            # output layer
            self.layers.append(layer(n_hidden, n_classes, **layer_kwargs))
            self.activation = activation
            self.dropout = dropout

    def forward(self, data, edge_norm=None, return_attention_weights=False):
        x, edge_index, edge_weights = data.x, data.edge_index, data.edge_attr
        h = x
        for layer in self.layers[:-1]:
            if isinstance(layer, GATConv):
                h = layer(x=h, edge_index=edge_index, edge_attr=None)
            elif isinstance(layer, SAGEConv):
                h = layer(x=h, edge_index=edge_index)
            else:
                h = layer(x=h, edge_index=edge_index, edge_weight=edge_weights)
          
            h = self.activation(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            
        
        final_layer = self.layers[-1]
        if isinstance(final_layer, GATConv):
            if return_attention_weights:
                h, weights = final_layer(x=h, edge_index=edge_index, edge_attr=None, return_attention_weights=return_attention_weights)
                
            else:
                h = final_layer(x=h, edge_index=edge_index, edge_attr=None)
        elif isinstance(final_layer, SAGEConv):
            h = final_layer(x=h, edge_index=edge_index)
        else:
            h = final_layer(x=h, edge_index=edge_index, edge_weight=edge_weights)
                
            
        if return_attention_weights:
            return h, weights
        else:
            return h
    
    def model_loss(self, normalized_adj):
        return 0

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()

    def reset_final_parameters(self):
        self.layers[-1].reset_parameters()

    def final_parameters(self):
        pass