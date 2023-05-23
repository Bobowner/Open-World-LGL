import torch
import numpy as np
import torch.nn.functional as F
from ood_models.ood_model import OODModel
from ood_models.ood_model import train_threshold_model
from scipy.stats import entropy


from torch_geometric.data import Data
from models.GCN import GCN


class ODIN(OODModel):
    def __init__(self, 
                 base_model,
                 task_loss,
                 threshold: float = 0.5,
                 threshold_method = "pure",
                 eps = 0.08,
                 temperature = 10,
                 num_classes=None,                 
                 aggregation_fn = "max",
                 normalize = True,
                 perturbation_kind = (True, True),
                 model_loss_weight = 10):
        
        super().__init__()
        self.threshold = float(threshold)

        self.num_classes = num_classes
        

        # Minimum threshold if reduce_risk is True,
        # allows to call fit() multiple times
        self.threshold_method = threshold_method
        
        self.base_model = base_model
        
        self.task_loss = task_loss
        
        self.aggregation_fn = aggregation_fn
        
        self.perturbation_kind = perturbation_kind
        
        self.normalize = normalize
        
        self.temperature = temperature
        
        self.eps = eps
        
        self.model_loss_weight = model_loss_weight
        
    def forward(self, inputs):
        return self.base_model(inputs)
   
    def calc_ood_scores(self, inputs, known_classes=None):
        
       
        data_perurbated = self.__perturbate_input(inputs, self.eps)
        
        logits = self.base_model(data_perurbated)
        if type(known_classes) != type(None):
            logits = logits[:,list(known_classes)]
            
        scaled_logits = logits/self.temperature
        scores =  scaled_logits
       
        scores = F.softmax(scores, dim = 1)#.max(dim=1).values
     
        if self.aggregation_fn == "max":
            scores =  torch.max(scores, dim=1).values
        elif self.aggregation_fn == "sum":
            scores =   torch.sum(scores, dim=1)
        else:
            Exception('Specified aggreation method not implemented!')
            
        
        
        return scores
    
    
    def loss(self, inputs, labels, mask, adj_r, known_classes=None):
        logits = self.base_model(inputs)
        loss = self.task_loss(logits[mask], labels[mask]) + self.model_loss_weight*self.base_model.model_loss(adj_r)
        return loss
    
    def fit(self, inputs, setting=None, edge_weight_model=None, mask=None, known_classes=None):
        
        
        if self.threshold_method == "pure":
            self.threshold = self.threshold 
            
            
        elif self.threshold_method == "quantile":
            
            scores = self.calc_ood_scores(inputs, known_classes)
            self.threshold = torch.quantile(scores, self.threshold)
            
        elif self.threshold_method == "conf_based":
            
            scores = self.calc_ood_scores(inputs, known_classes)
            self.ood_classifier = train_threshold_model(test_graph=inputs, scores=scores, mask=mask, known_classes=known_classes, 
                                                        n_epochs=setting.n_epochs, ood_ratio=self.threshold, setting=setting)
            
        elif self.threshold_method == "entropy":
            
            logits = F.softmax(self.base_model(inputs), dim=1).cpu().detach().numpy()
            entropy_scores  = entropy(logits, base=2, axis=1)
            avg_seen = np.mean(np.max(logits, axis=1))
            
            k = int(0.1 * len(entropy_scores)) # calculate k as 10% of the length of the array
            sorted_entropy_scores = np.sort(entropy_scores)[-k:]
            
            
            avg_E_unseen = np.mean(sorted_entropy_scores)
            self.threshold = (avg_seen + avg_E_unseen)/2
        
        else:
            raise NotImplementedError("Your thresholding method is not defined!")
            
            
        
        
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
        
        with torch.no_grad():
            logits = self.base_model(inputs)
            
            if mask != None:
                logits = logits[mask]
            
          
            prediction = torch.max(logits, axis=1).indices
            
        return prediction
        
    def __perturbate_input(self,data, eps):
       
        if self.perturbation_kind == (True, True):
            x_perturbated = self.__perturbate_features(data, eps)
            edge_perturbated = self.__perturbate_edges(data, eps)
            data_peturbated = Data(x=x_perturbated, edge_index=data.edge_index, y=data.y, edge_attr=edge_perturbated)
            return data_peturbated
        
        elif self.perturbation_kind == (True, False):
            x_perturbated = self.__perturbate_features(data, eps)
            data_peturbated = Data(x=x_perturbated, edge_index=data.edge_index, y=data.y)
            return data_peturbated
        
        elif self.perturbation_kind == (False, True):
            edge_perturbated = self.__perturbate_edges(data, eps)
            data_peturbated = Data(x=data.x, edge_index=data.edge_index, y=data.y, edge_attr=edge_perturbated)
            
            return data_peturbated
        
        elif self.perturbation_kind == (False, False):
            return data
        else:
            
            raise Exception("Your perturbation was undefined!")
            
    def __perturbate_features(self, data, eps):
        
        (g, feats) = (data.edge_index, data.x)
        data.x.requires_grad_(True)
        score = self.base_model(data)
        feats.retain_grad()
        score.backward(gradient = torch.ones_like(score))
        grad = data.x.grad
        res = feats - eps*torch.sign(-grad)
        res = res*feats
               
        return res


    def __perturbate_edges(self ,data, eps, use_numerator=True):
        x, edge_index, edge_weights = data.x, data.edge_index, data.edge_attr
        

        edge_weights.requires_grad_(True)

        score = self.base_model(data)
        score.backward(gradient = torch.ones_like(score))
        grad = edge_weights.grad
        if grad != None:
            res = edge_weights - self.eps*torch.sign(-grad)
        else:
            res = edge_weights

        return res
    
    def optimize_hyper_parameters(self,inputs, validation_class):
        temperatures = [1,10,20,50,100,1000,10000]
        if self.eps_opt == None:
            self.eps_opt = 0.08
            return
            self.eps_opt = self.optimize_eps(inputs)
        
        
        for temperature in temperatures: 
                return    
    
    def optimize_eps(self, data):
        score_sums = []
        epsilons = [0.0025, 0.005, 0.01, 0.02, 0.04, 0.08, 0.1]
        
        for eps in epsilons:
            data_perturbated = self.__perturbate_input(data, eps)
            logits = self.base_model(data_perturbated)
            scores = F.softmax(logits, dim=1)
            x = torch.sum(torch.max(scores[data.val_mask], dim=1).values)
            score_sums.append(x)

        
        return max(zip(epsilons, score_sums), key=lambda x: x[1])[0]
    
    
    def reset_parameters(self):
        self.base_model.reset_parameters()