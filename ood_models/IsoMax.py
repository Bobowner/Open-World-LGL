import torch
import torch.nn.functional as F
import time
from torch import nn
from ood_models.ood_model import OODModel
from torch_geometric.nn import GCNConv
from torch.nn import Module
from torch.nn.functional import binary_cross_entropy
from torch.nn import Linear



class IsoMaxPlusSoftmax(Module):
    """This part replaces the model classifier output layer nn.Linear()"""
    def __init__(self, hidden_dim, num_classes):
        super(IsoMaxPlusSoftmax, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.prototypes = nn.Parameter(torch.Tensor(num_classes, hidden_dim))
        torch.nn.init.normal_(self.prototypes, mean=0.0, std=10.0)
        self.distance_scale = torch.nn.Parameter(torch.Tensor(1))
        torch.nn.init.constant_(self.distance_scale, 1.0)
        
        
        
    def forward(self, features):
            
        distances = torch.abs(self.distance_scale) * torch.cdist(F.normalize(features), F.normalize(self.prototypes),p=2.0, compute_mode="donot_use_mm_for_euclid_dist")
        logits = -distances
        
        return logits
  
    
class IsoMaxPlusLossSecondPart(Module):
    """This part replaces the nn.CrossEntropyLoss()"""
    def __init__(self, entropic_scale = 10.0):
        super(IsoMaxPlusLossSecondPart, self).__init__()
        self.entropic_scale = entropic_scale
        
        
        
    def forward(self, logits, targets):
        """Probabilities and logarithms are calculated separately and sequentially"""
        """Therefore, nn.CrossEntropyLoss() must not be used to calculate the loss"""      
        
        distances = -logits
            
        probabilities_for_training = torch.nn.Softmax(dim=1)(-self.entropic_scale * distances)
        probabilities_at_targets = probabilities_for_training[range(distances.size(0)), targets]
        loss = -torch.log(probabilities_at_targets).mean()
        
        return loss
    
    
class IsoMax(OODModel):
    def __init__(self, base_model, task_loss, threshold, threshold_method, hidden_dim, num_classes, model_loss_weight=10):
        super().__init__()
        self.threshold = threshold
        self.base_model = base_model
        self.task_loss = task_loss
        self.model_loss_weight = model_loss_weight
        self.threshold_method = threshold_method
        self.loss_fn = IsoMaxPlusLossSecondPart()
        self.final_layer = IsoMaxPlusSoftmax(hidden_dim, num_classes)
        
        
    def forward(self, inputs):
        x, edge_index = inputs.x, inputs.edge_index
        features = self.base_model(inputs)
        logits = self.final_layer(features)
        return logits
        
    def loss(self, inputs, labels, mask, adj_r, known_classes=None):
        x, edge_index = inputs.x, inputs.edge_index
        features = self.base_model(inputs)
        logits = self.final_layer(features)     
                
        loss = self.loss_fn(logits[mask], labels[mask]) + self.model_loss_weight*self.base_model.model_loss(adj_r)
        return loss
    
    def reject(self, inputs, mask=None):
        
        scores = self.calc_ood_scores(inputs)
        reject_mask = scores >= self.threshold
        
        return reject_mask
    
        
    def validate(self, graph, mask):
        """ Hook to learn additional parameters/tune hyperparameters on validation set """
        return self
    
    
    def calc_ood_scores(self, inputs, known_classes=None):
        
        logits = self.forward(inputs)
        
        if type(known_classes) != type(None):
            logits = logits[:,list(known_classes)]
                
        #TODO: Add other methods
        scores = torch.max(logits, axis=1).values
        
        return scores

    def predict(self, inputs, mask=None):
        x, edge_index = inputs.x, inputs.edge_index
        features = self.base_model(inputs)
        logits = self.final_layer(features)
        
        prediction = torch.max(logits, axis=1).indices
        
        return prediction
    
    def fit(self, inputs, setting=None, edge_weight_model=None, mask=None, known_classes=None):
        #self.eps = self.optimize_eps(inputs)
        
        
        if self.threshold_method == "pure":
            #use plaint threshold
            self.threshold = self.threshold 
            
            
        elif self.threshold_method == "quantile":
            
            # use threshold based training set quantile
            scores = self.calc_ood_scores(inputs, known_classes)
            self.threshold = torch.quantile(scores, self.threshold)
            
        elif self.threshold_method == "conf_based":
            
            scores = self.calc_ood_scores(inputs, known_classes)
            self.ood_classifier = train_threshold_model(inputs, mask, known_classes, 
                                                        n_epochs=setting.n_epochs, 
                                                        ood_ratio=self.threshold, 
                                                        setting=setting)
            
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
    
    def reset_parameters(self):
        self.base_model.reset_parameters()
        torch.nn.init.normal_(self.final_layer.prototypes, mean=0.0, std=10.0)
        torch.nn.init.constant_(self.final_layer.distance_scale, 1.0)