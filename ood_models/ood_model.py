import  torch
import torch.nn.functional as F
from abc import ABC, abstractmethod
from ood_models.GCN import GCN
from torch_geometric.data import Data



class OODModel(torch.nn.Module, ABC):
    """ Abstract base class for open world learning """
              
    @abstractmethod
    def loss(self, inputs, labels, known_classes=None):
        """ Return loss score to train model """
        raise NotImplementedError("Abstract method called")

    @abstractmethod
    def fit(self, inputs, setting=None, edge_weight_model=None, mask=None, known_classes=None):
        """ Hook to learn additional parameters on whole training set """
        return self
   
    @abstractmethod
    def validate(self, graph, mask):
        """ Hook to learn additional parameters/tune hyper parameters on validation set """
        return self

    @abstractmethod
    def predict(self, inputs = None, mask=None):
        """ Return most likely classes per instance """
        raise NotImplementedError("Abstract method called")
        
    @abstractmethod
    def reject(self, inputs, mask=None):
        """ Return example-wise mask to emit 1 if reject and 0 otherwise """
        raise NotImplementedError("Abstract method called")
    
    @abstractmethod
    def calc_ood_scores(self,inputs=None, known_classes=None):
        """ Return OOD scores, without explicit theshold determination"""          
        
        raise NotImplementedError("Abstract method called")
     
    @abstractmethod
    def forward(self, inputs):
        raise NotImplementedError("Abstract method called")
        
    @abstractmethod
    def reset_parameters(self):
        raise NotImplementedError("Abstract method reset parameters called")

        
def train_threshold_model(test_graph, scores, mask ,known_classes, n_epochs, ood_ratio, setting):
    
    sorted_scores, indices = torch.sort(scores[mask])
    
    indices = indices[int(sorted_scores.shape[0]*(1-ood_ratio)):]
    
    pseudo_labels = torch.zeros_like(sorted_scores)
    pseudo_labels[indices] = 1 #highes ood_ratio percent set label
    pseudo_labels = pseudo_labels.type(torch.LongTensor).to(setting.device)

    
    ood_classifier = GCN(test_graph.x.shape[1], setting.params["n_hidden"], 
                         n_classes=2, n_layers=2, activation=F.relu, 
                         dropout=0.7).to(setting.device)
    
    optimizer = torch.optim.Adam(ood_classifier.parameters(), lr=setting.params["lr"] , weight_decay=0)

    if setting.params["ood_weight"]:
        weights = torch.Tensor([1-ood_ratio, ood_ratio]).to(setting.device)
    else:
        weights = [0.5, 0.5]
        
    loss_fn =torch.nn.CrossEntropyLoss(weight=weights)
    
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        out = ood_classifier(test_graph)
        out = out[mask]
        loss = loss_fn(out, pseudo_labels)
        print(loss)
        loss.backward()
        optimizer.step()
        
    return ood_classifier