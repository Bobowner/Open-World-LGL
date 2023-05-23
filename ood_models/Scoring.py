from ood_model import OODModel

class ScoringBasedOOD(OODModel):
    def __init__(self, threshold: float = 0.5,
                 num_classes=None, use_class_weights=False, device=torch.device("cpu"),
                 backend = "dgl",
                 underlying_model = None,
                 scoring_function="odin",
                 aggregation = "max",
                 normalize = True,
                 perturbation_kind = (True, False)):
        
        super().__init__()
        self.threshold = float(threshold)

        self.num_classes = num_classes
        
        self.use_class_weights = use_class_weights

        self.min_threshold = threshold
        
        self.backend = backend
        
        self.underlying_model = underlying_model
        
        self.scoring_function = scoring_function
        
        self.aggregation = aggregation
        
        self.perturbation_kind = perturbation_kind
        
        self.normalize = normalize
        
        self.temperature = 1000
        
        self.eps_opt = None
        
   
   
    def ood_score(self, data, subset, eps ,temperature):
        
        data_perurbated = self.__perturbate_input(data, eps, self.perturbation_kind)
        logits = self.underlying_model(*data_perurbated)
        if subset is not None:
            logits = logits[:, subset]

        
        if self.scoring_function == "odin":
            scaled_logits = logits/temperature
            scores =  scaled_logits
        
        elif self.scoring_function == "energy":
            logits = self.underlying_model(*data_perurbated)
            scaled_logits = logits/temperature
            if self.aggregation == "max":
                energy = -temperature*torch.log(torch.max(scaled_logits,dim=1).values)
            elif self.aggregation == "sum":
                energy = -temperature*torch.log(torch.sum(scaled_logits,dim=1))
            
            scores = energy
            return scores
        
        elif self.scoring_function == "joint_energy":
            data_perurbated = self.perturbate_input(data, eps, perturbation_kind)
            logits = self.underlying_model(*data_perurbated)
            scaled_logits = logits/temperature
            e_k =torch.logaddexp(torch.ones_like(scaled_logits), scaled_logits)
            scores = e_k
            
        else:
            Exception('Specified scoring function not implemented')
        
        if self.normalize:
            scores = F.softmax(scores, dim = 1)
        
        if self.aggregation == "max":
            scores =  torch.max(scores, dim=1).values
        elif self.aggregation == "sum":
            scores =   torch.sum(scores, dim=1)
        else:
            Exception('Specified aggreation method not implemented')
            

        
        return scores
    
    
    
    def calc_ood_scores(self, logits, inputs=None, subset=None):
        if subset is not None:
            logits = logits[:, subset]
            
        
        scores = self.ood_score(inputs, subset, self.eps_opt , self.temperature)
        
        return scores 
    
    
    def loss(self, logits, labels):

        return F.cross_entropy(logits, labels)
    
    #update at each time step?
    def fit(self, inputs, setting=None, edge_weight_model=None, mask=None, known_classes=None):
        known_labels = labels.unique()
        self.optimize_hyper_parameters(inputs, known_labels[0])
        return self
    
    
    def reject(self, logits, inputs, subset=None, test_mask=None):
       
        if subset is not None:
            logits = logits[:, subset]
            
        scores = self.ood_score(inputs, subset, self.eps_opt ,self.temperature)
 
        if test_mask != None:
            scores = scores[test_mask]
        
        reject_mask = scores <= torch.quantile(scores, self.threshold)
        
        return reject_mask

    def predict(self, logits, inputs = None, subset=None, test_mask=None):
        
        with torch.no_grad():
            if subset is not None:
                print(f"Reducing view to {len(subset)} known classes")
                logits = logits[:, subset]


            logits = self.underlying_model(*inputs)
            
            if test_mask != None:
                logits = logits[test_mask]
            
            # Basic argmax
            __max_vals, max_indices = torch.max(logits, dim=1)
            
        return max_indices
        
    def __perturbate_input(self ,data, eps, perturbation_kind):
       
        if perturbation_kind == (True, True):
            x_perturbated = self.__perturbate_features(data, eps)
            edge_perturbated = self.__perturbate_edges(data, eps)
            data_perturbated.x = x_perturbated
            data_perturbated.edge_index = data.edge_index
            data_perturbated.edge_attr = edge_perturbated
            return (g_perturbated, feats_perturbated)
        elif perturbation_kind == (True, False):
            x_perturbated = self.__perturbate_features(data, eps)
            feats_perturbated = x_perturbated
            g_perturbated = data[0]
            return (g_perturbated, feats_perturbated)
        elif perturbation_kind == (False, True):
            edge_perturbated = self.__perturbate_edges(data, eps)
            data_perturbated.x = data.x
            data_perturbated.edge_index = data.edge_index
            data_perturbated.edge_attr = edge_perturbated
            return (g_perturbated, feats_perturbated)
        else:
            raise Exception("You have to use some kind of perturbation!")
            
    def __perturbate_features(self, data, eps):
        (g, feats) = data
        feats.requires_grad_(True)
        
        score = self.underlying_model(*data)
        score.backward(gradient = torch.ones_like(score))
        grad = feats.grad
        res = feats - eps*torch.sign(-grad)
        res = res*feats
               
        return res


    def __perturbate_edges(self ,data, eps, use_numerator=True):
        (g, feats) = data
        g.edges().requires_grad_(True)

        score = self.underlying_model(*data)

        score.backward(gradient = torch.ones_like(score))
        grad = g.edges().grad
        res = g.edges() - eps*torch.sign(-grad)

        return res
    
    def optimize_hyper_parameters(self,inputs, validation_class):
        temperatures = [1,10,20,50,100,1000,10000]
        if self.eps_opt == None:
            self.eps_opt = 0.08
            return
            self.eps_opt = self.optimize_eps(inputs, self.perturbation_kind)
        
        
        for temperature in temperatures: 
                return
    
    
    def optimize_eps(self, data,  perturbation_kind):
        score_sums = []
        epsilons = [0.0025, 0.005, 0.01, 0.02, 0.04, 0.08, 0.1, 0.2]
        for eps in epsilons:
            data_perturbated = self.__perturbate_input(data, eps, perturbation_kind)
            logits = self.underlying_model(*data_perturbated)
            scores = F.softmax(logits, dim=1)
            x = torch.sum(torch.max(scores[data.val_mask], dim=1).values)
            score_sums.append(x)

        eps_opt = max(zip(epsilons, score_sums), key=lambda x: x[1])[0]
        return eps_opt