import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch_geometric.loader import NeighborLoader
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import roc_auc_score, auc, roc_curve


def bool2pmone(x):
    """ Converts boolean mask to {-1,1}^N int array """
    x = np.asarray(x, dtype=int)
    return x * 2 - 1


def calc_precision_recall(correct, pred):
    
    precision_micro = precision_score(correct,pred, average="micro", zero_division = 0)
    recall_micro = recall_score(correct,pred, average="micro", zero_division = 0)
    precision_macro = precision_score(correct,pred, average="macro", zero_division = 0)
    recall_macro = recall_score(correct,pred, average="macro", zero_division = 0)
    
    return precision_micro, recall_micro, precision_macro, recall_macro



def evaluate_classifier(test_graph, test_mask, setting, known_classes, reduce_to_known_classes=False ,device=None):
    ood_model = setting.ood_model
       
    ood_model.eval()
    correct_sum = 0
    node_sum = 0
    precision_scores = []
    recall_scores = []

    test_graph = test_graph.to(device)
    if reduce_to_known_classes:
        test_mask = test_mask.cpu().detach().numpy()
        known_classes = np.asarray(list(known_classes))
        known_mask = np.isin(test_graph.y.cpu().detach().numpy(), known_classes) #True for known class
        test_mask = np.logical_and(known_mask,test_mask) #True: known class test node
        test_mask = torch.tensor(test_mask)
    else:
        test_mask = test_mask.to(device)

        
    pred = ood_model(test_graph)
    pred = pred.argmax(dim=1)

    correct = (pred[test_mask] == test_graph.y[test_mask]).sum()
    
    correct_sum = correct
  
    node_sum = test_graph.y[test_mask].shape[0]

    pred = pred[test_mask].cpu().detach().numpy()
    label = test_graph.y[test_mask].cpu().detach().numpy()
    
    precision_micro, recall_micro, precision_macro, recall_macro = calc_precision_recall(label, pred)
    
    micro_f1 = f1_score(pred, label, average="micro")
    macro_f1 = f1_score(pred, label, average="macro")
        
    accuracy = correct_sum/node_sum
    accuracy = accuracy.cpu().detach().numpy()
        
    classification_scores = {"test accuracy" : accuracy, "precision_micro" : precision_micro, "recall_micro" : recall_micro, 
                             "precision_macro" : precision_macro, 
                             "recall_macro" : recall_macro, "micro_f1" : micro_f1, "macro_f1": macro_f1}

    return classification_scores



def evaluate_ood(test_graph, test_mask, known_classes, ood_model, device):
  
    ood_model=ood_model.eval()
    known_classes = np.asarray(list(known_classes))
    known_mask = np.isin(test_graph.y.cpu().detach().numpy(), known_classes) #True for known class
 
    test_mask = test_mask.cpu().detach().numpy() #True: Test node arbitrary class
    unseen_mask = np.logical_and(~known_mask,test_mask) #True: Unseen class test node
    test_graph = test_graph#.to(device)
    scores = ood_model.calc_ood_scores(test_graph, known_classes)
    
    
    
    rejects = ood_model.reject(test_graph).cpu().detach().numpy()
    
    
    pred = ood_model.predict(test_graph)
    
    
    
    mcc = matthews_corrcoef(bool2pmone(unseen_mask[test_mask]),
                           bool2pmone(rejects[test_mask]))



    scores = scores[test_mask].cpu().detach().numpy()

    if torch.sum(torch.Tensor(unseen_mask[test_mask])) == 0:
        
        au_roc = -1
        tnr95_score = -1
    
    else:

        fprs, tprs, thresholds = roc_curve(~unseen_mask[test_mask], scores)
        tnr95_score = 1 - fprs[np.argmax(tprs>=.95)]
        
        y = test_graph.y.cpu().detach().numpy()
        au_roc = roc_auc_score(~unseen_mask[test_mask], scores)
    

    # Open F1 Macro
    labels = test_graph.y[test_mask].clone().cpu().detach().numpy()
    pred = pred[test_mask].clone().cpu().detach().numpy()
    labels[unseen_mask[test_mask]] = -100
    pred[np.asarray(rejects[test_mask], dtype=bool)] = -100
    f1_macro = f1_score(labels, pred, average='macro')
        
    labels[unseen_mask[test_mask]] = 1
    labels[~unseen_mask[test_mask]] = 0
    rejects = ood_model.reject(test_graph).cpu().detach().numpy()[test_mask]
    ood_f1_micro = f1_score(labels, rejects, average='micro')
    ood_f1_macro = f1_score(labels, rejects, average='macro')
    
    

            
    ood_scores = {"mcc" : mcc, "au_roc" : au_roc, "tnr95_score": tnr95_score, "open_f1_macro": f1_macro, 
                  "ood_f1_micro" : ood_f1_micro,  "ood_f1_macro" : ood_f1_macro}
    
    return ood_scores


def eval_with_sampler(test_graph, test_mask, ood_model, setting, known_classes, reduce_to_known_classes=False ,device=None):
    
    test_graph.test_mask = test_mask
    if setting.params["temporal"]:
        batch_size = int(0.2*test_graph.x.shape[0])
        num_neighbors=[50,20]
    else:
        batch_size = setting.params["n_sampler_batch_size"]
        num_neighbors=[25,5]
    
    loader = iter(NeighborLoader(data=test_graph, input_nodes=test_mask, 
                                     batch_size= batch_size,
                                     num_neighbors = num_neighbors))
    classification_scores_list = []
    ood_scores_list = []
    for graph in loader:
        graph = graph.to(device)
        classification_scores = evaluate_classifier(graph, graph.test_mask, setting, 
                                                    known_classes, reduce_to_known_classes=reduce_to_known_classes ,device=device)
        ood_scores = evaluate_ood(graph, graph.test_mask, known_classes, 
                                  ood_model, device)
        
        classification_scores_list.append(classification_scores)
        ood_scores_list.append(ood_scores)
        graph.cpu()
        
    class_scores_df = pd.DataFrame(classification_scores_list)
    ood_scores_df = pd.DataFrame(ood_scores_list)
    
    class_scores_dict = class_scores_df.mean().to_frame().to_dict()[0]
    ood_scores_dict = ood_scores_df.mean().to_frame().to_dict()[0]
    
    return class_scores_dict, ood_scores_dict


def evaluate_thresholds(test_graph, test_mask, known_classes, ood_model, t, ood_threshold, disjunkt, device):
    
  
    
    known_classes = np.asarray(list(known_classes))
    known_mask = np.isin(test_graph.y.cpu().detach().numpy(), known_classes) #True for known class
 
    test_mask = test_mask.cpu().detach().numpy() #True: Test node arbitrary class
    unseen_mask = np.logical_and(~known_mask,test_mask) #True: Unseen class test node
    
    labels = test_graph.y[test_mask].clone().cpu().detach().numpy()
    labels[unseen_mask[test_mask]] = 1
    labels[~unseen_mask[test_mask]] = 0
    
    
    rejects = ood_model.reject(test_graph).cpu().detach().numpy()[test_mask]
    ood_f1_micro = f1_score(labels, rejects, average='micro')
    ood_f1_macro = f1_score(labels, rejects, average='macro')
    
    ood_threshold = ood_threshold.cpu().detach().numpy()
    
    t_ood_scores = {"threshold": t, 
                    "model_threshold" : ood_threshold,
                    "test_nodes": disjunkt, 
                    "ood_f1_micro_" : ood_f1_micro, 
                    "ood_f1_macro_" : ood_f1_macro}
    
    return t_ood_scores
    
    

    
def evaluate_multi_graph():
    pass