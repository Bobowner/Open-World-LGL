import torch
import numpy as np
import copy
import torch.nn.functional as F
import pickle
import networkx as nx
import torch_geometric as tg
from torch_geometric.datasets import Planetoid
from torch_geometric.datasets import PPI
from torch_geometric.utils import subgraph, to_undirected
from torch_geometric.data import Data
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.utils import to_dense_adj
from ogb.nodeproppred import PygNodePropPredDataset
from pathlib import Path
from itertools import compress
from torch_sparse import SparseTensor
from data_iterators import BatchIterTemporal, BatchIterStatic, GraphIter, make_inductive, sample_graph_mlp

import time

class OOD_Dataset(torch.utils.data.Dataset):
    def __init__(self, name : str, inductive :bool, disjunct : bool, validate : bool ,
                 unseen_classes, custom_split=None, temporal=False, 
                 cumulative=True, t0=0, seed=None, r=None):
 
        assert name != "PPI", "PPI not supported anymore"
        assert not(temporal) or unseen_classes == [], "Temporal and predfined unseenclass not defined together"
    
        super(OOD_Dataset, self).__init__()
        
 
        self.ds_name = name
        self.unseen_classes = unseen_classes
        self.unseen_class_list = []
        self.inductive = inductive
        self.disjunct = disjunct
        self.temporal = temporal
        self.cumulative = cumulative
        self.t0 = t0
        self.r = r
        ds = load_dataset(name, custom_split)
        self.num_classes = ds.num_classes
        self.validate = validate
        self.graph = get_graph(ds, name)
        self.n_nodes = self.graph.x.shape[0]
        self.in_feats = self.graph.x.shape[1]
        self.task_loss = get_task_loss(name)
        self.batch_size = 20000
        
        self.known_classes = set(range(self.num_classes))-set(unseen_classes)
        
        
        #load the right ar into RAM
        if r != None:
            if temporal:
                self.ar = None
            else:
                if name in ["PubMed"]:

                    if r == 2:
                        self.ar = torch.load("dataset/PubMed/PubMed_r2.pt")

                    elif r == 3:

                        self.ar = torch.load("dataset/PubMed/PubMed_r3.pt")

                    else:
                        self.ar = getAr(self.graph.edge_index, self.graph.x.shape[0], r)

                elif name in ["ogb-arxiv"]:
                    if r == 2:
                        self.ar = torch.load("dataset/ogbn_arxiv/A2_undir_sparse.pt")
                    elif r == 3:

                        self.ar = torch.load("dataset/ogbn_arxiv/A3_undir_dense.pt")

                    else:
                        self.ar = getAr(self.graph.edge_index, self.graph.x.shape[0], r)

                elif name in ["dblp-easy"]:

                    if r == 2:
                        self.ar = torch.load("data/dblp-easy/dblp-easy_r2.pt")

                    else:

                        self.ar = getAr(self.graph.edge_index, self.graph.x.shape[0], r)

                elif name in ["dblp-hard"]:

                    if r == 2:
                        self.ar = torch.load("data/dblp-hard/dblp-hard_r2.pt")

                    else:

                        self.ar = getAr(self.graph.edge_index, self.graph.x.shape[0], r)

                else:
                    # All other Datasets: Compute Ar
                    self.ar = getAr(self.graph.edge_index, self.graph.x.shape[0], r)
        else:
            self.ar = None
       
      
       
       
        #built temporal dataset/iterator
        if temporal:
                   
            self.known_class_list = built_known_class_list(iter(GraphIter(self.graph, t0, cumulative, unseen_classes)))
          
            self.built_unknown_class_list = built_unknown_class_list(self.known_class_list, self.num_classes)
            
            self.years = torch.unique(self.graph.node_year)
        
        #built planatoid split iterator
        elif custom_split == None:
            
            self.unseen_graph_mask = get_unseen_class_graph_mask(self.graph, self.unseen_classes)
            self.train_graph_mask, self.test_graph_mask = get_graph_masks(self.graph, 
                                                                          inductive= self.inductive, 
                                                                          validate= self.validate, 
                                                                          unseen_graph_mask=self.unseen_graph_mask)
            
            self.train_mask, self.test_mask = get_task_masks(self.graph, self.validate, self.train_graph_mask, self.test_graph_mask)
            

          
        #built custom split dataset
        else:
            if name in ["Cora", "CiteSeer",  "PubMed", "ogb-arxiv"]:
                
                self.graph = load_dataset(name, custom_split)[0]
            
                train_mask, val_mask, test_mask = create_split(self.graph, custom_split[0], custom_split[1], custom_split[2], seed)
                
                self.graph.train_mask = train_mask
                self.graph.val_mask = val_mask
                self.graph.test_mask = test_mask
                
                                
                self.unseen_graph_mask = get_unseen_class_graph_mask(self.graph, self.unseen_classes)
                self.train_graph_mask, self.test_graph_mask = get_graph_masks(self.graph, 
                                                                          inductive= self.inductive, 
                                                                          validate= self.validate, 
                                                                          unseen_graph_mask=self.unseen_graph_mask)
                
            
                self.train_mask, self.test_mask = get_task_masks(self.graph, 
                                                                 self.validate, 
                                                                 self.train_graph_mask, 
                                                                 self.test_graph_mask)
                
                                
            else:
                NotImplementedError("Custom split not implemented for this type of dataset")

      
    @property
    def graph_list(self):
    
        if self.temporal:
            return iter(GraphIter(self.graph, self.t0, self.cumulative, []))
            #return self.graph_list_intern
        
        else:
            return self.graph_list_intern
        
    @property
    def batches(self):
    
        if self.temporal:
            
            return iter(BatchIterTemporal(self.graph, 
                                          self.t0, 
                                          self.cumulative, 
                                          self.known_class_list, 
                                          self.ar, self.n_nodes, 
                                          self.batch_size, 
                                          self.r, 
                                          self.ds_name))
            
        else:
        
            return iter(BatchIterStatic(self.graph, 
                                        train_graph_mask=self.train_graph_mask, 
                                        test_graph_mask=self.test_graph_mask,
                                        train_mask = self.train_mask,
                                        test_mask = self.test_mask,
                                        ar = self.ar, n_nodes=self.n_nodes, 
                                        batch_size = self.batch_size, 
                                        r = self.r))
        
        
    @property
    def train_graph_list(self):
    
        if self.temporal:
            return iter(GraphIter(self.graph, self.t0, self.cumulative, []))
        
        else:
            raise NotImplementedError("Train graph list not defined in temporal") 
        
    @property
    def test_graph_list(self):
    
        if self.temporal:
            return iter(GraphIter(self.graph, self.t0+1, self.cumulative, []))
        
        else:
            raise NotImplementedError("Test graph list not defined in temporal")
                   
    @property
    def ar_list(self):
    
        n_nodes_list = list(map(lambda g : g.x.shape[0], self.graph_list_intern))
        if self.r != None:
            if self.inductive:
                masks = list(map(lambda n_nodes : torch.ones(n_nodes, dtype=torch.bool), n_nodes_list))
                return iter(ArIter(self.ar, n_nodes_list , batch_size=self.batch_size, r=self.r, masks = masks))
            else:
                
                return iter(ArIter(self.ar, n_nodes_list , batch_size=self.batch_size, r=self.r, masks = None))
            
        else:
            return [torch.zeros(0)] * len(self.train_graphs)
           
    @property
    def train_masks(self):
        if self.inductive:
            return list(compress(self.train_masks_attr, self.train_graph_mask))
        else:
            return self.train_masks_attr
    @property
    def val_masks(self):
        if self.inductive:
            return list(compress(self.val_masks_attr, self.val_graph_mask))
        else:
            return self.val_masks_attr
    @property
    def test_masks(self):
        if self.inductive:
            
            return list(compress(self.test_masks_attr, self.test_graph_mask))
        else:
            return self.test_masks_attr
    
    @property
    def train_graphs(self):
        if self.inductive:
            return list(compress(self.graph_list, self.train_graph_mask))
        else:
            return list(compress(self.graph_list, self.train_graph_mask))
    
    @property
    def val_graphs(self):
        if self.inductive:
            return list(compress(self.graph_list, self.val_graph_mask))
        else:
            return list(compress(self.graph_list, self.val_graph_mask))
    
    @property
    def test_graphs(self):
        if self.inductive:
            return list(compress(self.graph_list, self.test_graph_mask))
        else:
            return list(compress(self.graph_list, self.test_graph_mask))
     
    @property    
    def train_unseen_masks(self):
        # unseeen_class_masks true for known classes
        if self.inductive:
            self.train_graph_masks = compress(self.train_masks_attr, self.train_graph_mask)
       
            return list(map(lambda masks : torch.logical_and(masks[0], masks[1]),   
                            zip(self.train_graph_masks, self.unseeen_class_masks)))
        else:
            return list(map(lambda elem : torch.logical_and(elem[0], elem[1]), zip(self.unseeen_class_masks, self.train_masks)))
    
    @property    
    def val_unseen_masks(self):
        if self.inductive:
            return [torch.logical_and(self.val_masks_attr[0], self.unseeen_class_masks[1])]
        else:
            return list(map(lambda elem : torch.logical_and(elem[0], elem[1]), zip(self.unseeen_class_masks, self.val_masks)))
    
    @property    
    def test_unseen_masks(self):
        if self.inductive:
            
            return [torch.logical_and(self.test_masks_attr[0], self.unseeen_class_masks[2])]
            
        else:
            return list(map(lambda elem : torch.logical_and(elem[0], elem[1]), zip(self.unseeen_class_masks, self.test_masks)))
        


def get_graph(ds, name):
    graph = ds[0]
    if name in ["ogb-arxiv", "dblp-easy", "dblp-hard"]:
        graph.edge_index =  to_undirected(graph.edge_index)
        return graph
    else:
        return graph



def get_graph_masks(graph, inductive, validate, unseen_graph_mask):
    train_mask = graph.train_mask
    val_mask = graph.val_mask
    test_mask = graph.test_mask
    
    if inductive:
        if validate:
            
            train_graph_mask = torch.logical_and(unseen_graph_mask, ~torch.logical_or(val_mask, test_mask))
            test_graph_mask = torch.logical_and(torch.ones(graph.x.shape[0], dtype=torch.bool), ~test_mask)

        else:
            train_graph_mask = torch.logical_and(unseen_graph_mask, ~test_mask)
            test_graph_mask = torch.ones(graph.x.shape[0], dtype=torch.bool)

    else:
        train_graph_mask = torch.ones(graph.x.shape[0], dtype=torch.bool)
        test_graph_mask = torch.ones(graph.x.shape[0], dtype=torch.bool)
    
    return train_graph_mask, test_graph_mask

def get_task_masks(graph, validate, train_graph_mask, test_graph_mask):
    
    if validate:
        train_mask = graph.train_mask[train_graph_mask]
        test_mask = graph.val_mask[test_graph_mask]
    else:
        train_mask = graph.train_mask[train_graph_mask]
        test_mask = graph.test_mask[test_graph_mask]
    return train_mask, test_mask


def get_graphs(name, inductive, disjunct, validate, unseen_classes):
    if name in ["Cora", "CiteSeer",  "PubMed"]:
        dataset = load_dataset(name, "train")
        graph = dataset[0]
        graph.y = torch.unsqueeze(torch.tensor(graph.y), 1)


        if inductive:


            if disjunct:
                train_mask = get_mask(name, graph, "train", unseen_classes, inductive)
                val_mask = get_mask(name, graph, "val", unseen_classes, inductive)
                test_mask = get_mask(name, graph, "test", unseen_classes, inductive)

                train_graph = make_inductive(graph, train_mask)
                val_graph = make_inductive(graph, val_mask)
                test_graph = make_inductive(graph, test_mask)
                return [train_graph, val_graph, test_graph], [True, False, False], [False, True, False], [False, False, True]

            else:
                train_mask_graph = get_mask(name, graph, "train", unseen_classes, inductive)
                val_mask_graph = get_mask(name, graph, "val", unseen_classes, inductive)
                test_mask_graph = get_mask(name, graph, "test", unseen_classes, inductive)

                if validate:
                    
                    train_graph = make_inductive(graph, train_mask_graph)
                    val_graph = make_inductive(graph, val_mask_graph)
                    test_graph = make_inductive(graph, test_mask_graph)

                    train_mask = graph.train_mask[train_mask_graph]
                    val_mask = graph.val_mask[val_mask_graph]
                    test_mask = graph.test_mask
                    
                else:
                    val_graph = make_inductive(graph, val_mask_graph)
                    train_graph = val_graph
                    test_graph = make_inductive(graph, test_mask_graph)
                    
                    train_mask = graph.train_mask[val_mask_graph]
                    val_mask = graph.val_mask[val_mask_graph]
                    test_mask = graph.test_mask

                
                
                train_masks_attr = []
                val_masks_attr = []
                test_masks_attr = []
                train_masks_attr.append(train_mask)
                val_masks_attr.append(val_mask)
                test_masks_attr.append(test_mask)
                

                return [train_graph, val_graph, test_graph], [True, False, False], [False, True, False], [False, False, True], train_masks_attr, val_masks_attr, test_masks_attr

        else:
            return [graph], [True], [True], [True]

    if name in ["ogb-arxiv"]:
        dataset = load_dataset(name, "train")

        if inductive:
            train_mask = get_mask(name, dataset[0], "train", unseen_classes, inductive)
            val_mask = get_mask(name, dataset[0], "val", unseen_classes, inductive)
            test_mask = get_mask(name, dataset[0], "test", unseen_classes, inductive)

            train_graph = make_inductive(dataset[0], train_mask)
            val_graph = make_inductive(dataset[0], val_mask)
            test_graph = make_inductive(dataset[0], test_mask)
            return [train_graph, val_graph, test_graph], [True, False, False], [False, True, False], [False, False, True]

        else:
            return [dataset[0]], [True], [True], [True]

    elif name in ["PPI"]:
        train_set = load_dataset(name, "train")
        val_set = load_dataset(name, "val")
        test_set = load_dataset(name, "test")
        batches = []
        train_masks = []
        graphs = []
        indices = torch.arange(0, len(train_set + val_set + test_set))

        for ds in train_set + val_set + test_set:
            graphs.append(ds)
        return graphs, indices<len(train_set), torch.logical_and(len(train_set)<=indices,indices<len(train_set+val_set)), indices>=len(train_set+val_set)
        
        
def getAr(edge_index, n_nodes, r):
    nnodes = edge_index.shape[1]
    
    edge_index, edge_weight = gcn_norm(edge_index, edge_weight=None, num_nodes=n_nodes)

    #A = to_dense_adj(edge_index, batch=None, edge_attr=edge_weight, max_num_nodes=n_nodes).squeeze()
    A = SparseTensor(row=edge_index[0], col=edge_index[1], value=edge_weight, sparse_sizes=(n_nodes, n_nodes))
    
    tmp = A
    for i in range(r-1):
        tmp = tmp.matmul(A)
        
    tmp = tmp.to_dense()
    
    return tmp

def get_A_r(adj, r):
    adj_label = adj
    if r == 1:
        adj_label = adj_label
    elif r == 2:
        adj_label = adj_label@adj_label
    elif r == 3:
        adj_label = adj_label@adj_label@adj_label
    elif r == 4:
        adj_label = adj_label@adj_label@adj_label@adj_label
    return adj_label



def add_edge_weight(graph):
    graph.edge_attr = torch.ones(graph.edge_index.shape[1],1)
    return graph


def load_dataset(name, split):
    if name in ["Cora", "CiteSeer",  "PubMed"]:
        dataset = Planetoid(root='dataset/' + name, name=name)
    elif name in ["ogb-arxiv"]:
        
        dataset = PygNodePropPredDataset(name = "ogbn-arxiv", root = 'dataset/')
    
    elif name in ["dblp-easy", "dblp-hard"]:
        data_file = Path("data/"+name+"/"+name+"-graph.pt")
        dataset = built_dblp(name)
        
    elif name in ["PPI"]:
        dataset = PPI(root = 'dataset/', split=split, transform=None, pre_transform=None, pre_filter=None)
    else:
        error_message = "The dataset: \"" + name + "\" is not supported yet."
        raise NotImplementedError(error_message) 
    return dataset



def get_unseen_class_graph_mask(graph, unseen_classes):
    unseen_class_mask = torch.ones(graph.y.shape[0], dtype=torch.bool)
    for c in unseen_classes:
        unseen_class_mask = torch.logical_and(unseen_class_mask, graph.y != c)
    return unseen_class_mask

def get_mask_without_unseen_classes(graph_list, unseen_classes):
    
    unseen_class_masks = []
    
    for g in graph_list:
        g.y = torch.squeeze(g.y)
        unseen_class_mask = torch.ones(g.y.shape[0], dtype=torch.bool)
        
        for c in unseen_classes:        
            unseen_class_mask = torch.logical_and(unseen_class_mask, g.y != c)
            
        unseen_class_masks.append(unseen_class_mask)
    return unseen_class_masks
    
    
def get_mask(name, graph, split, unseen_classes, inductive):
    y = torch.squeeze(graph.y)

    unseen_class_mask = torch.ones(y.shape[0], dtype=torch.bool)
    for c in unseen_classes:
        
        unseen_class_mask = torch.logical_and(unseen_class_mask, y != c)
        
  
    if name in ["Cora", "CiteSeer",  "PubMed"]:
        if split == "train":
            
            if inductive:
                
                val_mask = graph.val_mask
                test_mask = graph.test_mask
                train_mask = torch.ones(graph.x.shape[0], dtype=torch.bool)
                
                train_mask = torch.logical_and(train_mask,~val_mask)
                train_mask = torch.logical_and(train_mask,~test_mask)
                mask = torch.logical_and(train_mask, unseen_class_mask)
                
            else:
                
                mask = graph.train_mask
                
        elif split == "val":
            
            if inductive:
                
                test_mask = graph.test_mask
                
                val_mask = torch.ones(graph.x.shape[0], dtype=torch.bool)
                val_mask = torch.logical_and(val_mask,~test_mask)
                
                mask = val_mask

            else:
                
                mask = graph.val_mask
                
        elif split == "test":
            if inductive:
                mask = torch.ones(graph.x.shape[0], dtype=torch.bool)
            else:
                mask = graph.test_mask
            
        return mask
        
    elif name in ["ogb-arxiv"]:
        if split == "train":
            return (graph.node_year <= 2017).squeeze()
        elif split == "val":
            return (graph.node_year == 2018).squeeze()
        elif split == "test":
            return (graph.node_year >= 2019).squeeze()
        
    elif name in ["PPI"]:
        
        mask = torch.ones(graph.x.shape[0], dtype=torch.bool)
        
        return mask
    
    
def create_batch_masks(graphs, name, inductive, temporal, cumulative, unseeen_class_masks):
    
    train_masks_list = []
    val_masks_list = []
    test_masks_list = []
    
    for g in graphs:
        if cumulative and temporal:
            
            max_year = torch.max(g.node_year)
            train_mask = torch.ones(g.x.shape[0], dtype=torch.bool)
            val_mask = torch.ones(g.x.shape[0], dtype=torch.bool)
            test_mask = torch.squeeze(g.node_year == max_year)
            
        elif temporal and not cumulative:
            train_mask = torch.ones(g.x.shape[0], dtype=torch.bool)
            val_mask = torch.ones(g.x.shape[0], dtype=torch.bool)
            test_mask = torch.ones(g.x.shape[0], dtype=torch.bool)
            
            
        elif inductive:
            #may not work for multiple graphs!
            ds = load_dataset(name, None)
    
            train_mask = ds[0].train_mask
            val_mask = ds[0].val_mask
            test_mask = ds[0].test_mask
            
            
            
            train_subset = torch.logical_not(torch.logical_or(val_mask, test_mask))
            train_mask = train_mask[train_subset]
            val_mask = val_mask[~test_mask]
            
        else:
            ds = load_dataset(name, None)
            train_mask = ds[0].train_mask 
            val_mask = ds[0].val_mask 
            test_mask = ds[0].test_mask 
            
        train_masks_list.append(train_mask)
        val_masks_list.append(val_mask)
        test_masks_list.append(test_mask)
        
    return train_masks_list, val_masks_list, test_masks_list



def create_split(dataset, train_portion, val_portion, test_portion, seed):
    
    y = dataset.y.cpu().detach().numpy()
    unique, counts = np.unique(y, return_counts=True)

    rng = np.random.default_rng(seed)
    train = []
    val = []
    test = []

    for cl in unique:
        
        tmp = np.argwhere(y==cl)
        c1 = int(len(tmp)*train_portion)
        c2 = int(len(tmp)*(train_portion+val_portion))
        rng.shuffle(tmp)
        train.append(tmp[:c1])
        val.append(tmp[c1:c2])
        test.append(tmp[c2:])
        
    train_ix = np.concatenate(train)
    val_ix = np.concatenate(val)
    test_ix = np.concatenate(test)

    train = torch.full_like(dataset.y, False, dtype=torch.bool)
    train[train_ix] = True
    val = torch.full_like(dataset.y, False, dtype=torch.bool)
    val[val_ix] = True
    test = torch.full_like(dataset.y, False, dtype=torch.bool)
    test[test_ix] = True
    return train, val, test



def get_task_loss(name):
    if name in ["Cora", "CiteSeer",  "PubMed", "ogb-arxiv", "dblp-easy", "dblp-hard"]:
        
        return torch.nn.CrossEntropyLoss()
    
    elif name in ["PPI"]:
        
        return torch.nn.BCEWithLogitsLoss()
    else:
        raise NotImplementedError("The loss for your dataset is not implemented!")
    

    
def built_temporal_dataset(name, t0, unseen_classes, cumulative):
    assert name in ["ogb-arxiv", "dblp-easy", "dblp-hard"], "Your data set is not supported for a temporal built"
    
    ds = load_dataset(name, "")
    years =  torch.unique(ds[0].node_year)
    if t0 < torch.min(years):
        t0 = torch.min(years)
  
    unseen_class_mask = torch.ones(ds[0].y.shape[0], dtype=torch.bool)
    for c in unseen_classes:  
        unseen_class_mask = torch.logical_and(unseen_class_mask, (ds[0].y != c).squeeze())
    
  
    #disassemble graph
    graphs = []


    train_years_mask = torch.logical_and((ds[0].node_year <= t0).squeeze(), unseen_class_mask)
    train_ds = make_inductive(ds[0], train_years_mask)
    train_ds.node_year = ds[0].node_year[train_years_mask]
    graphs.append(train_ds)
    eval_years = [year for year in years if year > t0]

    for year in eval_years:
        if cumulative:
            eval_year_mask = (ds[0].node_year <= year).squeeze()
        else:
            eval_year_mask = (ds[0].node_year == year).squeeze()
        eval_ds = make_inductive(ds[0], eval_year_mask)
        eval_ds.node_year = ds[0].node_year[eval_year_mask]
        graphs.append(eval_ds)

        
    return graphs


def built_temporal_graph_masks(name, t0):
    train_graph_masks = [True]
    val_graph_masks = [False]
    test_graphs_masks = [False]
    
    ds = load_dataset(name, "")
    years =  torch.unique(ds[0].node_year)
    
    eval_years = [year for year in years if year > t0]
    for year in eval_years:
        train_graph_masks.append(False)
        val_graph_masks.append(False)
        test_graphs_masks.append(True)
    
    return train_graph_masks, val_graph_masks, test_graphs_masks

def built_known_class_list(graphs):
    known_classes = set()
    known_class_list = []
    for gt in graphs:
        
        current_classes = torch.unique(gt.y)
        
        for one_class in current_classes.tolist():
            known_classes.add(one_class)
        
        known_class_list.append(copy.deepcopy(known_classes))
    
    return known_class_list

def built_unknown_class_list(known_class_list, n_classes):
    all_classes = set(range(n_classes))
    unkown_class_list = []
    for classes in known_class_list:
        unknown_classes = all_classes - classes
        unkown_class_list.append(unknown_classes)
    
    return unkown_class_list



#This class is just to unify api with torch_geoemtric data classes
class Dataset:
    """Class for keeping track of an item in inventory."""
    def __init__(self, ds, num_classes):
        self.ds = ds
        self.num_classes = num_classes

    def __getitem__(self, key):
        return self.ds
    
def built_dblp(name):
    path = Path('data')
    y = np.load(path/name/"y.npy")
    x = np.load(path/name/"X.npy")
    node_year = np.load(path/name/"t.npy")
    nx_graph = nx.read_adjlist(path/name/"adjlist.txt", nodetype=int)
    data = tg.utils.from_networkx(nx_graph)
    data.x = torch.tensor(x, dtype=torch.float32)
    data.y = torch.unsqueeze(torch.tensor(y), 1)
    data.node_year = torch.unsqueeze(torch.tensor(node_year),1)
    num_classes = np.unique(y).shape[0]
    ds = Dataset(data, num_classes)
    return ds


def get_split(name, custom_split, seed):

    path = Path("splits/"+name+"_split_"+str(seed))
    if path.is_file():
        splits = torch.load("splits/"+name+"_split_"+str(seed))
    else:
        print("Create new split file!")
        train_mask, val_mask, test_mask = create_split(self.graph, custom_split[0], custom_split[1], custom_split[2], seed)
        split = {
              "train": train_mask,
              "val": val_mask,
              "test": test_mask
        }
        torch.save(split, "splits/"+name+"_split_"+str(seed))

    return splits

#methods for already existing dataset
def ds_exists(ds_path):
    path = Path('data/'+ds_path)
    return path.is_dir()


def store_OOD_dataset(ds_path, ds):
    path = Path('data/'+ds_path)
    path = path.with_suffix('.pkl')
   
    with path.open(mode='wb') as outp:
        pickle.dump(ds, outp, pickle.HIGHEST_PROTOCOL)

def load_OOD_dataset(ds_path):
    path = Path('data/'+ds_path)
    path.with_suffix('.pkl')
    with open(path, 'rb') as inp:
        ds = pickle.load(inp)
    
    return ds