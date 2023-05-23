import yaml
import torch
import torch.nn.functional as F
import numpy as np
from torch_geometric.loader import NeighborLoader
from pathlib import Path
from evaluation import evaluate_classifier, evaluate_ood,eval_with_sampler, evaluate_thresholds
from results_writer import write_scores, write_threshold_scores
from models.GAT import GAT
from tqdm import tqdm

def run_one_setting(setting, idx):

    
    for seed in tqdm(range(setting.num_repeats)):
        
        #rest parameter for independet runs
        setting.ood_model.reset_parameters()
        scores = run_one_experiment(setting, idx, seed)
        if isinstance(scores, list):
            for score in scores:
                write_scores(score, setting, idx, seed)
        else:
            write_scores(scores, setting, idx, seed)
            
    #torch.cuda.empty_cache()
    
def run_one_experiment(setting, idx, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    #Use these to create completly reproducable results - but slower algorithms
    
    #os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
    #torch.backends.cudnn.benchmark = False
    #torch.use_deterministic_algorithms(True)
    
    #add seed to params
    setting.params["seed"] = seed
    
    #temporal setting
    if setting.temporal:
        scores = process_temporal_setting(setting)
        
    #static setting    
    else:
        #train model
        
        #eval graph is val or test set, specified in yml file
        for train_graph, train_mask, eval_graph, eval_mask, ar in setting.dataset.batches:
            
            t0 = setting.params["t_start"]
            
         
            train_loss = train(train_graph, train_mask, ar, setting.ood_model, 
                               setting.optimizer, setting.n_epochs, 
                               device=setting.device, setting=setting)

            #evaluate model
            #notimplemented otherwise
            only_use_known_classes = False

            if only_use_known_classes:
                 raise NotImplementedError("Accuracy extended to unknown classes implementation needs to be checked.")
            else:
                test_masks = eval_mask
 
     
            known_classes = setting.dataset.known_classes
        
            #fit ood model per graph?
            eval_graph = eval_graph.to(setting.device)
            eval_mask = eval_mask.to(setting.device)
            if not setting.params["disjunkt"]:
                
                setting.ood_model.threshold = setting.params["threshold"]
                
                setting.ood_model = setting.ood_model.fit(train_graph, 
                                                  setting=setting, 
                                                  edge_weight_model=None, 
                                                  mask=train_mask, 
                                                  known_classes=known_classes)
                
            else:
                setting.ood_model.threshold = setting.params["threshold"]

             
                setting.ood_model = setting.ood_model.fit(eval_graph, setting=setting, 
                                                  edge_weight_model=None, mask=eval_mask, 
                                                  known_classes=known_classes)
        
            classification_scores = evaluate_classifier(eval_graph, test_masks, setting, 
                                                        known_classes, 
                                                        reduce_to_known_classes=setting.params["eval_on_known"], 
                                                        device=setting.device)
            

            ood_scores = evaluate_ood(eval_graph, test_masks, known_classes, setting.ood_model, setting.device)
            
            if setting.params["eval_mult_thresholds"]:
                
                threshold_scores = {}
                for t in np.arange(0.0, 1.1, 0.1):
                    
                    setting.ood_model.threshold = t
                    
                    disjunkt = True
                    setting.ood_model = setting.ood_model.fit(train_graph, 
                                                  setting=setting, edge_weight_model=None, 
                                                  mask=train_mask, known_classes=known_classes)
                    
                    
                     
                        
                    t_scores = evaluate_thresholds(eval_graph, test_masks, 
                                                   known_classes, ood_model, t, 
                                                   ood_model.threshold, 
                                                   disjunkt ,setting.device)
                    
                    write_threshold_scores(t_scores, setting, idx, 0)

                    
                    threshold_scores.update(t_scores)

                    
                    disjunkt = False 
                    setting.ood_model.threshold = t
                    setting.ood_model = setting.ood_model.fit(eval_graph, setting=setting, 
                                                      edge_weight_model=None, mask=eval_mask, 
                                                      known_classes=known_classes)
                    
                    t_scores = evaluate_thresholds(eval_graph, test_masks, 
                                                   known_classes, ood_model, t, 
                                                   ood_model.threshold, 
                                                   disjunkt ,setting.device)
                    
                    threshold_scores.update((k, threshold_scores[k] + t_scores[k]) for k in threshold_scores.keys() | t_scores.keys())

                    write_threshold_scores(t_scores, setting, idx, 0)

                    threshold_scores.update(t_scores)
                    
                    
                    
                    
                    
                    
            else:
                threshold_scores = {}
            
        
        #TODO: Work on more flexible metric calculation
        scores = {**classification_scores, **ood_scores}
    
    return scores

def process_temporal_setting(setting):
     # get train and eval data
    setting_scores = []
    
        
    for train_graph, test_graph, ar, train_mask, test_mask, known_classes in setting.dataset.batches:
        
        
        scores = process_one_timestep(train_graph, train_mask, ar, test_graph, test_mask, known_classes, setting)
        
        node_year = test_graph.node_year[test_mask][0].cpu().detach().numpy()
        scores.update( {'year' : node_year})
        print("Year: ", node_year)
        setting_scores.append(scores)
                      
        #stop after validation step
        
                
        if setting.validate:
            break
        
    # extend model
    # extend data
    return setting_scores

def process_one_timestep(train_graph, train_mask, ar, test_graph, test_mask, known_classes, setting):
    # train model on current classes and data
    
    
    train_loss = train_one_timestep(train_graph, train_mask, ar, 
                                    setting.ood_model, setting.optimizer, setting.n_epochs, 
                                    setting.device, known_classes, setting)
       
     
    if not setting.params["disjunkt"]:
                
        setting.ood_model.threshold = setting.params["threshold"]

        setting.ood_model = setting.ood_model.fit(train_graph, 
                                          setting=setting, 
                                          edge_weight_model=None, 
                                          mask=train_mask, 
                                          known_classes=known_classes)

    else:
        setting.ood_model.threshold = setting.params["threshold"]


        setting.ood_model = setting.ood_model.fit(eval_graph, setting=setting, 
                                          edge_weight_model=None, mask=eval_mask, 
                                          known_classes=known_classes)
    
    threshold_scores = eval_thresholds(train_graph, train_mask, test_graph, test_mask, known_classes, setting)
    
    train_graph.cpu()
    train_mask.cpu()
    # evaluate model on t+1

    #with torch.no_grad():
    scores = eval_one_timestep(test_graph, test_mask, known_classes, 
                               setting.ood_model, setting.device, setting)
    
    scores.update(threshold_scores)
    
    test_graph.cpu()
    test_mask.cpu()
    return scores
    
    
    
def train_one_timestep(train_graph, train_mask, ar, ood_model, optimizer, n_epochs, device, known_classes, setting):
    
    ood_model.to(setting.device)
    ood_model.train()

    
    known_classes = torch.Tensor(list(known_classes)).type(torch.long).to(device)
    
    if setting.params["sampler"] == "neighbor":
        
        train_graph.train_mask = train_mask.to(device)
        loss = train_with_sampler(train_graph, train_mask, ar, ood_model, known_classes, optimizer, n_epochs, device, setting)
    
    else:
        train_graph = train_graph.to(device)
        mask = train_mask.to(device)
        ar = ar.to(device)
  
        for epoch in range(n_epochs):

            optimizer.zero_grad()
            
            loss = ood_model.loss(train_graph, train_graph.y, mask ,ar, known_classes)
            #torch.cuda.empty_cache()
            loss.backward()
            optimizer.step()
        
   
    return loss


def eval_thresholds(train_graph, train_mask, eval_graph, eval_mask, known_classes, setting):
    
    if setting.params["eval_mult_thresholds"]:
                
                threshold_scores = {}
                for t in np.arange(0.0, 1.1, 0.1):
                    
                    setting.ood_model.threshold = t
                    
                    disjunkt = True
                    ood_model = setting.ood_model.fit(train_graph, 
                                                  setting=setting, edge_weight_model=None, 
                                                  mask=train_mask, known_classes=known_classes)
                    
                    
                     
                    eval_graph.to(setting.device)
                    eval_mask.to(setting.device)

                    t_scores = evaluate_thresholds(eval_graph, eval_mask, known_classes, setting.ood_model, t, disjunkt ,setting.device)
                    threshold_scores.update(t_scores)
               

                    
                    disjunkt = False 
                    ood_model = setting.ood_model.fit(eval_graph, setting=setting, 
                                                      edge_weight_model=None, mask=eval_mask, 
                                                      known_classes=known_classes)
                    
                    t_scores = evaluate_thresholds(eval_graph, eval_mask, known_classes, setting.ood_model, t, disjunkt ,setting.device)
                    
                    eval_graph.cpu()
                    eval_mask.cpu()
                    threshold_scores.update(t_scores)
                    
    else:
        threshold_scores = {}
        
    return threshold_scores

def eval_one_timestep(test_graph, test_mask, known_classes, ood_model, device, setting):
    
    if setting.params["sampler"] == "neighbor":
        classification_scores, ood_scores = eval_with_sampler(test_graph, test_mask, ood_model, setting, 
                                                              known_classes, reduce_to_known_classes=False ,device=device)
    else:
        classification_scores = evaluate_classifier(test_graph, test_mask, setting, known_classes, device=device)
        ood_scores = evaluate_ood(test_graph, test_mask, known_classes, ood_model, device)
        
    
    scores = {**classification_scores, **ood_scores}
    
    return scores
    
    
    

def train(train_graph, train_mask, ar, ood_model, optimizer, n_epochs, device, setting):

    known_classes = setting.dataset.known_classes
    known_classes = torch.Tensor(list(known_classes)).type(torch.long).to(device)
    edge_weight_model = None
    
    ood_model.train()
    #TODO: include if on edge weight use for ood
    if setting.params["return_attention_weights"]:
        edge_weight_model = GAT(setting.dataset.in_feats, setting.params["n_hidden_per_head"], setting.dataset.num_classes, F.relu, 
                                setting.params["dropout"], setting.params["attn_dropout"], 
                                setting.params["attn_heads"], add_self_loops = False).to(device)
    
   
    
   
   
    #batch size = start nodes
    if setting.params["sampler"] == "neighbor":
        
        
        loss = train_with_sampler(train_graph=train_graph, train_mask=train_mask,ar=ar, ood_model=ood_model, 
                                  known_classes=known_classes, optimizer=optimizer, n_epochs=n_epochs, device=device, setting=setting)
    
    else:
        graph = train_graph.to(device)
        mask = train_mask.to(device)
        adj_r = ar.to(device)
    
        for epoch in range(n_epochs):
            optimizer.zero_grad()
            loss = ood_model.loss(graph, graph.y, mask ,adj_r, known_classes)
            loss.backward()
            optimizer.step()
            
    
    return loss

def train_with_sampler(train_graph, train_mask, ar, ood_model, known_classes, optimizer, n_epochs, device, setting):
    
    train_graph = train_graph.cpu()#to(device)
    
    if setting.params["temporal"]:
        batch_size = int(0.2*train_graph.x.shape[0])
        num_neighbors=[50,20]
    else:
        batch_size = setting.params["n_sampler_batch_size"]
        num_neighbors=[25,5]
        
    loader = iter(NeighborLoader(data=train_graph, input_nodes=train_mask, 
                                     batch_size= batch_size,
                                     num_neighbors = num_neighbors))
        
    for epoch in range(n_epochs):
        for graph in loader:
            graph.to(device)
            optimizer.zero_grad()
            loss = ood_model.loss(graph, graph.y, graph.train_mask ,ar, known_classes)
            loss.backward()
            optimizer.step()
            graph.cpu()
            
    return loss


def validate(dataset, ood_model, device):
    
    for graph, mask, adj_r in zip(dataset.val_graphs, dataset.val_unseen_masks, dataset.ar_list):
        graph.to(device)
        mask.to(device)
        adj_r.to(device)
        ood_model = ood_model.validate(graph,mask)