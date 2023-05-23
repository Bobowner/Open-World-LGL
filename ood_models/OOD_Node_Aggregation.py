import torch
import numpy as np
import torch.nn.functional as F
from torch.distributions import Categorical
from ood_models.ood_model import train_threshold_model
from ood_models.ood_model import OODModel
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree, segregate_self_loops
from scipy.stats import entropy

class OOD_Node_Aggregation(OODModel):
    def __init__(self, 
                 base_model,
                 task_loss,
                 threshold: float = 0.5,
                 threshold_method = "pure",
                 neighbourhood_influence = 0,
                 num_classes=None, 
                 ood_model = None,
                 device = None,
                 use_edge_weights = False):
        
        super().__init__()
        self.threshold = float(threshold)
        self.threshold_method = threshold_method
        self.num_classes = num_classes
        self.neighbourhood_influence = neighbourhood_influence
        self.use_edge_weights = use_edge_weights
        self.aggregation_fn = OOD_Score_Aggregation(1,1, neighbourhood_influence, device)
        self.ood_model = ood_model
        
        gpu_number = device
        device = torch.device(*('cuda', gpu_number) if torch.cuda.is_available() else 'cpu')
        self.device = device

    def forward(self, inputs):
        return self.ood_model(inputs)
        
    def loss(self, inputs, labels, mask, adj_r, known_classes=None):

        return self.ood_model.loss(inputs, labels, mask, adj_r)
    
    def fit(self, inputs, setting=None, edge_weight_model=None, mask=None, known_classes=None):
        """ Hook to learn additional parameters/tune hyper parameters on train set """
  

        if self.threshold_method == "pure":
            #use plaint threshold
            self.threshold = self.threshold 
            
            
        elif self.threshold_method == "quantile":
            
            # use threshold based training set quantile
            scores = self.calc_ood_scores(inputs, known_classes)
            self.threshold = torch.quantile(scores, self.threshold)
            
                   
            
        elif self.threshold_method == "conf_based":
            
            scores = self.calc_ood_scores(inputs, known_classes)
            self.ood_classifier = train_threshold_model(test_graph=inputs, scores=scores, mask=mask, known_classes=known_classes, 
                                                        n_epochs=setting.n_epochs, ood_ratio=self.threshold, setting=setting)
            
            
        elif self.threshold_method == "entropy":
  

            logits = F.softmax(self.ood_model(inputs), dim=1)            
            
            cat = Categorical(probs=logits)
            # compute entropy per sample
            entropy_per_sample = cat.entropy()
            # get the indices of the top 10% of samples with the highest entropy
            top_indices = entropy_per_sample.topk(int(self.threshold * len(entropy_per_sample)))[1]
            # get the maximum class probabilities for the top 10% of samples
            max_class_probs = logits[top_indices].max(dim=1)[0]
            # calculate the average of the maximum class probabilities
            avg_E_unseen = max_class_probs.mean()

            avg_seen = np.mean(np.max(logits.cpu().detach().numpy(), axis=1))
             
            self.threshold = (avg_seen + avg_E_unseen)/2
            
        
        else:
            raise NotImplementedError("Your thresholding method is not defined!")
            
            
        
        if self.use_edge_weights:
            self.edge_weight_model = edge_weight_model
            optimizer = torch.optim.Adam(edge_weight_model.parameters(), 
                                         lr=setting.params["lr"] , 
                                         weight_decay=setting.params["weight_decay"])
            self.__train(setting.dataset, optimizer, setting.n_epochs, self.device)
            
        return self
    
        
    def validate(self, graph, mask):
        """ Hook to learn additional parameters/tune hyper parameters on validation set """
        return self
    
    
    def reject(self, inputs, mask=None):
        
        if self.threshold_method == "conf_based":
            
            if mask != None:
                reject_mask = self.ood_classifier(inputs)
                return torch.max(reject_mask[mask], axis=1).indices
            else:
                reject_mask = self.ood_classifier(inputs)
                return torch.max(reject_mask, axis=1).indices
                
        else:    
            scores = self.calc_ood_scores(inputs)

            if mask != None:
                scores = scores[mask]

            reject_mask = scores <= self.threshold

            return reject_mask
        
    
    def predict(self, inputs = None, mask=None):
            
        max_indices = self.ood_model.predict(inputs, mask)
   
        return max_indices
    
    def calc_ood_scores(self, inputs, known_classes=None):
        
        if self.use_edge_weights:
            x, edge_index, edge_attr = inputs.x, inputs.edge_index, inputs.edge_attr
            
            logits, edge_weights = self.edge_weight_model(data=inputs, return_attention_weights=True)
            edge_weights = torch.squeeze(edge_weights[1])
            
            scores = self.ood_model.calc_ood_scores(inputs, known_classes)
            scores = self.aggregation_fn(torch.unsqueeze(scores,dim=1), edge_index, edge_weights)
            return torch.squeeze(scores)    
        else:
            x, edge_index, _ = inputs.x, inputs.edge_index, inputs.edge_attr
            scores = self.ood_model.calc_ood_scores(inputs, known_classes)
            scores = scores.to(self.device)
            scores = self.aggregation_fn(torch.unsqueeze(scores,dim=1), edge_index)
            return torch.squeeze(scores)
    
    def reset_parameters(self):
        self.ood_model.reset_parameters()
        
        
    #private methods
    def __train(self, dataset, optimizer, n_epochs, device):
    
        self.edge_weight_model.train()
       
        for epoch in range(n_epochs):
            for graph, mask, adj_r in zip(dataset.train_graphs, dataset.train_unseen_masks, dataset.ar_list):
                graph = graph.to(device)
                mask = mask.to(device)
                adj_r = adj_r.to(device)
                
                optimizer.zero_grad()
                logits = self.edge_weight_model(graph)
                loss = dataset.task_loss(logits[mask], graph.y[mask])
                loss.backward()
                optimizer.step()
                
        return loss
        


class OOD_Score_Aggregation(MessagePassing):
    def __init__(self, in_channels, out_channels, neighbourhood_influence, device):
        super().__init__(aggr='add')
        self.neighbourhood_influence = neighbourhood_influence
        self.device = device

    def forward(self, x, edge_index_org, edge_weights=None):
        # Step 1: Add self-loops to the adjacency matrix.
        
        if edge_weights==None:
            
            edge_index, _ = add_self_loops(edge_index_org, num_nodes=x.size(0))
            row, col = edge_index_org

            norm = torch.zeros(edge_index.shape[1]).to(self.device)


            deg = degree(col, x.size(0), dtype=x.dtype)
            deg_inv= deg.pow(-1)
            deg_inv[deg_inv == float('inf')] = 0


            norm[:-x.size(0)] = deg_inv[col]
            norm[:-x.size(0)] = norm[:-x.size(0)]*self.neighbourhood_influence
            norm[-x.size(0):] = (1 - self.neighbourhood_influence)
            norm[-x.size(0):][(deg==0)] = 1
        
        else:
            edge_index, _ = add_self_loops(edge_index_org, num_nodes=x.size(0))
            row, col = edge_index_org

            norm = torch.zeros(edge_index.shape[1]).to(self.device)
            deg = degree(col, x.size(0), dtype=x.dtype)
            norm[:-x.size(0)] = edge_weights*self.neighbourhood_influence
            norm[-x.size(0):] = (1 - self.neighbourhood_influence)
            norm[-x.size(0):][(deg==0)] = 1
            
        # Step 4-5: Start propagating messages.
        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j, norm):
        # Step 4: Normalize node features.
        return norm.view(-1, 1) * x_j