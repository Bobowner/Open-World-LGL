import torch
from ood_models.ood_model import OODModel



class Identity(OODModel):
    def __init__(self, base_model, task_loss, threshold, threshold_method, num_classes, model_loss_weight=10):
        super().__init__()
        self.threshold = threshold
        self.threshold_method = threshold_method
        self.base_model = base_model
        self.task_loss = task_loss
        self.model_loss_weight = model_loss_weight
        
    def forward(self, inputs):
        
        return self.base_model(inputs)
        
    def loss(self, inputs, labels, mask, adj_r, known_classes=None):
       
        logits = self.base_model(inputs)
        loss = self.task_loss(logits[mask], labels[mask]) + self.model_loss_weight*self.base_model.model_loss(adj_r)
        

        return loss
    
    def reject(self, inputs, mask=None):
        
        reject_mask = torch.zeros(inputs.x.shape[0])
        return reject_mask
    
    
    def calc_ood_scores(self, inputs, known_classes=None):
        
        scores = torch.zeros(inputs.x.shape[0])
        
        return scores

    def predict(self, inputs, mask=None):
        logits = self.base_model(inputs)
        
        prediction = torch.max(logits, axis=1).indices
        
        return prediction
    
    def fit(self, inputs, setting=None, edge_weight_model=None, mask=None, known_classes=None):
        
        return self
    
        
    def validate(self, graph, mask):
        """ Hook to learn additional parameters/tune hyper parameters on validation set """
        return self
    
    
    def reset_parameters(self):
        self.base_model.reset_parameters()