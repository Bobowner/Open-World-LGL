import torch
import torch.nn.functional as F
from models.GCN import GCN
from models.GAT import GAT
from models.GraphSage import GraphSAGE
from models.graph_mlp import GraphMLP
from ood_models.Odin import ODIN
from ood_models.IsoMax import IsoMax
from ood_models.Identity import Identity
from ood_models.gDoc import gDoc
from ood_models.OOD_Node_Aggregation import OOD_Node_Aggregation
from handle_meta_data import fill_defaults
from load_dataset import OOD_Dataset, ds_exists, store_OOD_dataset, load_OOD_dataset
from pathlib import Path
import time


class Setting:
    
    def __init__(self, dataset, ood_model, base_model, optimizer, device, temporal, params):
        self.dataset = dataset
        self.ood_model = ood_model.to(device)
        self.base_model = base_model.to(device)
        self.optimizer = optimizer
        self.device = device
        self.n_epochs = params["n_epochs"]
        self.num_repeats = params["num_repeats"]
        self.return_attention_weights = params["return_attention_weights"]
        self.temporal = temporal
        self.validate = params["validate"]
        self.params = params
        
        

def built_dataset(params):
    
    dataset = OOD_Dataset(name=params["dataset"], inductive=params["inductive"], 
                          disjunct=params["disjunkt"], validate =params["validate"] , 
                          unseen_classes=params["unseen_classes"], temporal=params["temporal"],
                          custom_split=params["custom_split"],
                          r=params["adj_potenz"], t0=params["t_start"])
    
    
    return dataset

def built_model(params, dataset):
    activation_functions = {
        "relu" : F.relu
    }
    activation = activation_functions[params["activation"]]
    
    
    if params["ood_model"] in ["iso_max"]:
        num_classes = params["n_hidden"]
    else:
        num_classes = dataset.num_classes
    
    #TODO: check this
    #num_classes = dataset.num_classes
    if params["base_model"]=="gcn":
         model = GCN(dataset.in_feats, params["n_hidden"], num_classes, params["n_layers"], activation, params["dropout"])
    elif params["base_model"]=="graph_mlp":
        
        model = GraphMLP(in_feats=dataset.in_feats, n_hidden=params["n_hidden"], n_classes=num_classes, 
                         n_layers=params["n_layers"], activation=activation, dropout=params["dropout"], tau=params["tau"])
        
    elif params["base_model"]=="gat":
        model = GAT(dataset.in_feats, params["n_hidden_per_head"], num_classes, activation, params["dropout"], params["attn_dropout"], params["attn_heads"])
    elif params["base_model"]=="sage":
        model = GraphSAGE(dataset.in_feats, params["n_hidden"], num_classes, params["n_layers"], activation, params["dropout"])
    else:
        raise NotImplementedError("Your base model architecture is not implemented yet.")
    return model

def built_ood_model(params, dataset, base_model):
    
    if params["ood_model"]=="identity":
        
        ood_model = Identity(base_model, dataset.task_loss, threshold=params["threshold"], threshold_method = params["threshold_method"], 
                             num_classes=dataset.num_classes, model_loss_weight = params["model_loss_weight"])
        
    elif params["ood_model"]=="iso_max":
        
        ood_model = IsoMax(base_model=base_model, task_loss=dataset.task_loss, threshold=params["threshold"], threshold_method = params["threshold_method"], 
                           hidden_dim=params["n_hidden"],  num_classes=dataset.num_classes, model_loss_weight = params["model_loss_weight"])
        
    elif params["ood_model"]=="odin":
        ood_model = ODIN(base_model, dataset.task_loss, threshold=params["threshold"], threshold_method = params["threshold_method"], 
                         eps= params["odin_eps"] , temperature = params["temperature"],
                         num_classes=dataset.num_classes, 
                         aggregation_fn=params["score_aggregation"], normalize=params["score_normalize"], 
                         perturbation_kind = (params["odin_perturbation"][0],params["odin_perturbation"][1]) , model_loss_weight = params["model_loss_weight"])
        
    elif params["ood_model"] == "gdoc":
        
        ood_model = gDoc(base_model, dataset.task_loss, threshold =params["threshold"], threshold_method = params["threshold_method"], 
                         num_classes=dataset.num_classes, model_loss_weight = params["model_loss_weight"])
    else:
        raise NotImplementedError("Your ood model architecture is not implemented yet.")
        
    if params["node_aggregation"]:
        ood_model = OOD_Node_Aggregation(base_model, dataset.task_loss, threshold = params["threshold"],
                            threshold_method = params["threshold_method"],
                            neighbourhood_influence = params["neigh_influence"], 
                            num_classes=dataset.num_classes, ood_model = ood_model, device=params["device"][0],
                            use_edge_weights = params["return_attention_weights"])

    return ood_model
        
def built_optmizer(ood_model, params):
    if params["optimizer"] == "adam":
        optimizer = torch.optim.Adam(ood_model.parameters(), lr=params["lr"] , weight_decay=params["weight_decay"])
    else:
        raise NotImplementedError("Your optimizer is not implemented yet.")
    return optimizer

def setup_experiment(params):
    full_params = fill_defaults(params)
    
    gpu_number = full_params["device"][0]
    device = torch.device(*('cuda', gpu_number) if torch.cuda.is_available() else 'cpu')
    
    dataset = built_dataset(full_params)
   
    base_model = built_model(full_params, dataset).to(device)
   
    ood_model = built_ood_model(full_params, dataset, base_model).to(device)
  
    optimizer = built_optmizer(ood_model, full_params)
    setting = Setting(dataset, ood_model, base_model, optimizer, device, temporal=full_params["temporal"], params=full_params)
  
    p = Path("results/"+full_params["filename"])
    p.mkdir(parents=True, exist_ok=True)

    return setting 