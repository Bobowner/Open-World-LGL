import torch
from ood_models.ood_model import OODModel



class gDoc(OODModel):
    """
    Deep Open ClassificatioN: Sigmoidal activation + Threshold based rejection
    Inputs should *not* be activated in any way.
    This module will apply sigmoid activations.
    """
    def __init__(self,
                 base_model,
                 task_loss,
                 threshold: float = 0.5,
                 threshold_method = "pure",
                 reduce_risk: bool = True, 
                 alpha: float = 3.0,
                 num_classes=None,
                 model_loss_weight = 10,
                 use_class_weights=True):
        """
        Arguments
        ---------
        threshold: Threshold for class rejection
        alpha: Factor of standard deviation to reduce open space risk
        **kwargs: will be passed to BCEWithLogitsLoss
        """
        
        super().__init__()
        
        self.base_model = base_model
        self.task_loss = task_loss
        
        self.reduce_risk = bool(reduce_risk)
        self.alpha = float(alpha)
        self.threshold = float(threshold)
        self.threshold_method = threshold_method

        self.num_classes = num_classes
        
        self.use_class_weights = use_class_weights
        self.model_loss_weight = model_loss_weight

        # Minimum threshold if reduce_risk is True,
        # allows to call fit() multiple times
        self.min_threshold = threshold

    def forward(self, inputs):
        return self.base_model(inputs)
    
    
    def loss(self, inputs, labels, mask, adj_r, known_classes=None):
        
        logits = self.base_model(inputs)
        labels = inputs.y
        
        if self.use_class_weights:
            
            with torch.no_grad():
                values, counts = torch.unique(labels,
                                              return_counts=True)
               
                total = counts.sum()
                # Neg examples / positive examples *per class*
                class_weights = (total - counts) / counts

                pos_weight = torch.zeros(self.num_classes,
                                          device=class_weights.device)
                pos_weight[values] = class_weights
        else:
            pos_weight = None

        criterion = torch.nn.BCEWithLogitsLoss(reduction='mean',
                                               pos_weight=pos_weight)
        targets = torch.nn.functional.one_hot(labels,
                                              num_classes=logits.size(1))       
        
        loss = criterion(logits[mask], targets[mask].float()) + self.model_loss_weight*self.base_model.model_loss(adj_r)
        return loss

    
    def fit(self, inputs, setting=None, edge_weight_model=None, mask=None, known_classes=None):
        """ Gaussian fitting of the thresholds per class.
        To be called on the full training set after actual training,
        but before evaluation!
        """
        inputs = inputs.to(setting.device)
        logits = self.base_model(inputs)
        labels = inputs.y
        
        if not self.reduce_risk:
            print("[DOC/warning] fit() called but reduce_risk is False. Pass.")
            return self

        y = logits.detach().sigmoid()  # [num_examples, num_classes]

        # posterior "probabilities" p(y=l_i | x_j, y_j = li)
        uniq_labels = labels.unique()
        if self.num_classes is None:
            # Infer #classes
            num_classes = len(uniq_labels)
        else:
            num_classes = self.num_classes

        std_per_class = torch.zeros(num_classes, device=logits.device)

        for i in uniq_labels:
            # Filter for y_j == li
            y_i = y[labels == i, i]

            # for each existing point,
            # create a mirror point (not a probability),
            # mirrored on the mean of 1
            y_i_mirror = 1 + (1 - y_i)  # [num_examples, num_classes]

            # estimate the standard deviation per class
            # using both existing and the created points
            y_i_all = torch.cat([y_i, y_i_mirror], dim=0)
            # TODO: unbiased SD? orig work did not specify...
            std_i = y_i_all.std(dim=0, unbiased=True)  # scalar

            std_per_class[i] = std_i

        # Set the probability threshold t_i = max(0.5, 1 - alpha * SD_i)
        # Orig paper uses base threshold 0.5,
        # but we use a specified minimum threshold
        thresholds_per_class = (1 - self.alpha * std_per_class).clamp(self.min_threshold)

        self.threshold = thresholds_per_class  # [num_classes]

        return self

    def reject(self, inputs, mask=None):
        
        with torch.no_grad():
                
            logits = self.base_model(inputs)
            
            # Reduce view on thresholds if subset is given,
            # AND if self.threshold is not just a float
            if not isinstance(self.threshold, float):
                threshold = self.threshold
            else:
                threshold = self.threshold

            y_proba = logits.sigmoid()
            # Dim1 is reduced by 'all' anyways, no mapping back needed
            reject_mask = (y_proba < threshold).all(dim=1)
       
        return reject_mask
    
    def calc_ood_scores(self, inputs, known_classes=None):
        logits = self.base_model(inputs)
        
        if type(known_classes) != type(None):
            logits = logits[:,list(known_classes)]
        
        y_proba = logits.sigmoid()
        
        return torch.max(y_proba, dim=1).values

    def predict(self, inputs, mask=None):
        
        with torch.no_grad():
            logits = self.base_model(inputs)

            y_proba = logits.sigmoid()

            # Basic argmax
            __max_vals, max_indices = torch.max(y_proba, dim=1)
        return max_indices
    
        
    def validate(self, graph, mask):
        """ Hook to learn additional parameters/tune hyper parameters on validation set """
        return self
    
    
    def reset_parameters(self):
        self.base_model.reset_parameters()