import torch
from torch_geometric.data import Data
from torch_geometric.utils import subgraph

#Iterator for Ar to compute per graph
class ArIter:

    def __init__(self, ar, n_nodes_list, batch_size, r, masks):
        self.ar = ar
        self.r = r
        self.n_nodes_list = n_nodes_list
        self.batch_size = batch_size
        self.masks = masks
        
    
    def __iter__(self):
        self.idx = 0
        
        return self

    def __next__(self):

        if self.masks != None:
           
            ar, _  = sample_graph_mlp(self.ar, self.n_nodes_list[self.idx], self.batch_size, self.masks[self.idx])
        else:
            
            ar, _  = sample_graph_mlp(self.ar, self.n_nodes_list[self.idx], self.batch_size, None)
            
        self.idx+=1
        
        return ar
    
    
#Iterator for train graphs in temporal setting
class GraphIter:

    def __init__(self, graph, t0, cumulative, unseen_classes):
        self.graph = graph
        self.t0 = t0
        self.unseen_classes = unseen_classes
        self.cumulative = cumulative
    
    def __iter__(self):
        years =  torch.unique(self.graph.node_year)
        
        if self.t0 < torch.min(years):
            self.t0 = torch.min(years)
            
        self.unseen_class_mask = torch.ones(self.graph.y.shape[0], dtype=torch.bool)
        
        for c in self.unseen_classes:  
            self.unseen_class_mask = torch.logical_and(self.unseen_class_mask, (self.graph.y != c).squeeze())
            
        self.years =  iter(year for year in years if year >= self.t0)
        #self.current_year = self.t0
        
        return self
    
    def __next__(self):
        self.current_year = next(self.years)

        if self.cumulative:
            mask = (self.graph.node_year <= self.current_year).squeeze(1)
        else:
            mask = (self.graph.node_year == self.current_year).squeeze(1)
            
        
        current_graph = make_inductive(self.graph, mask)
        current_graph.node_year = self.graph.node_year[mask]
        current_graph = add_edge_weight(current_graph)
        
        return current_graph
    
    
class BatchIterTemporal:

    def __init__(self, graph, t0, cumulative, known_class_list,  ar, n_nodes, batch_size, r, name = None):
        self.graph = graph
        self.t0 = t0
        self.cumulative = cumulative
        self.name = name
        
        self.ar = ar
        self.r = r
        self.n_nodes = n_nodes
        self.batch_size = batch_size
        self.known_class_list = known_class_list
      
    
    def __iter__(self):
        years =  torch.unique(self.graph.node_year)
        
        if self.t0 < torch.min(years):
            self.t0 = torch.min(years)
            
        self.known_class_list_iter = iter(self.known_class_list)
        
        self.years =  iter(year for year in years if year >= self.t0)
        self.current_year = next(self.years)
        
        
        return self
    
    def __next__(self):
        
        train_year = self.current_year
        test_year = next(self.years)
                    
        if self.cumulative:
            year_mask = torch.squeeze(self.graph.node_year <= train_year)
            test_graph_mask = (self.graph.node_year <= test_year).squeeze()
            
            
        else:
            year_mask = torch.squeeze(self.graph.node_year == train_year)
            test_graph_mask = (self.graph.node_year == test_year).squeeze()
            
        
                
        if self.r != None:

            if self.name in ["dblp-easy", "dblp-hard", "ogb-arxiv"]:
                if self.r in [2,3]:
                   
                    ar = torch.load("data/"+self.name+"/"+self.name+"_r"+str(self.r)+"_cumulative_"+str(int(train_year))+".pt")
                    ar, sample_mask, indices = sample_graph_mlp(ar=ar, n_nodes=ar.shape[0], 
                                                                batch_size=self.batch_size, mask=None, get_indices=True)
                else:
                    raise Exception("Check this implementation again!")
                    ar, sample_mask, indices = sample_graph_mlp(self.ar, self.n_nodes, 
                                                                self.batch_size, mask=year_mask, get_indices=True)
                
            else:
                ar, sample_mask, indices = sample_graph_mlp(self.ar, self.n_nodes, 
                                                            self.batch_size, mask=year_mask, get_indices=True)
                
                
            batch_mask = year_mask
          
            train_graph_edges = subgraph(batch_mask, self.graph.edge_index, relabel_nodes=True, num_nodes=self.graph.x.shape[0])[0]
            
            train_graph_edges = subgraph(sample_mask, train_graph_edges, relabel_nodes=True, num_nodes=torch.sum(batch_mask))[0]
         
            
            
            
            node_year = self.graph.node_year[batch_mask]
            node_year = node_year[sample_mask]
            
            features = self.graph.x[batch_mask][sample_mask,:]
            labels = self.graph.y[batch_mask][sample_mask]

            train_graph = Data(x=features,edge_index=train_graph_edges, y=labels, node_year=node_year)
            train_graph.y = torch.squeeze(train_graph.y)

            
        else:
            ar = torch.zeros(0)
            batch_mask = year_mask
            
            train_graph = make_inductive(self.graph, batch_mask)
            train_graph.node_year = self.graph.node_year[batch_mask]
            
        
        
        train_graph = add_edge_weight(train_graph)
        test_graph = make_inductive(self.graph, test_graph_mask)
        test_graph.node_year = self.graph.node_year[test_graph_mask]

        train_mask = torch.ones(train_graph.x.shape[0], dtype=torch.bool)
        
        test_mask = torch.zeros(test_graph.x.shape[0], dtype=torch.bool)
        test_mask[torch.squeeze(test_graph.node_year==test_year)] = True
        
        test_graph = add_edge_weight(test_graph)
        
        self.current_year = test_year
        
        return train_graph, test_graph, ar, train_mask, test_mask, next(self.known_class_list_iter)
    
    
class BatchIterStatic:

    def __init__(self, graph, train_graph_mask, test_graph_mask, train_mask, test_mask, ar, n_nodes, batch_size, r):
        
        self.graph = graph
        self.train_graph_mask = train_graph_mask
        self. test_graph_mask = test_graph_mask
        self.train_mask = train_mask
        self.test_mask = test_mask
        
        
        self.ar = ar
        self.r = r
        self.n_nodes = n_nodes
        self.batch_size = batch_size

    def __iter__(self):
        
        self.idx_list = iter(range(1))
            
        #self.train_unseen_masks_iter = iter(self.train_unseen_masks)
        return self
    
    def __next__(self):
        
        idx = next(self.idx_list)
        
        if self.r != None:
            
            ar, sample_mask, indices = sample_graph_mlp(self.ar, self.n_nodes, self.batch_size, mask=self.train_graph_mask, get_indices=True)
            
            features = self.graph.x[sample_mask,:]
            
            labels = self.graph.y[sample_mask]
            
            train_graph_edges = subgraph(self.train_graph_mask, self.graph.edge_index, relabel_nodes=True, num_nodes=self.n_nodes)[0]
            
            train_mask = self.train_mask[indices]

            train_graph = Data(x=features,edge_index=train_graph_edges, y=labels)
            
        else:
            
            train_graph = self.graph.subgraph(self.train_graph_mask)
            train_graph = add_edge_weight(train_graph)
            train_mask = self.train_mask
            ar = torch.zeros(0)

            
        test_graph = self.graph.subgraph(self.test_graph_mask)
        test_mask = self.test_mask        
        test_graph = add_edge_weight(test_graph)
            
        return train_graph, train_mask, test_graph, test_mask, ar
    
    
    
def make_inductive(dataset, mask):
   
    num_nodes = dataset.x.shape[0]
    edges = subgraph(mask, dataset.edge_index, relabel_nodes=True, num_nodes=num_nodes)[0]

    nodes = dataset.x[mask,:]
    labels = torch.squeeze(dataset.y[mask],1)
    dataset_new = Data(x=nodes,edge_index=edges, y=labels)
    return dataset_new


def add_edge_weight(graph):
    graph.edge_attr = torch.ones(graph.edge_index.shape[1],1)
    return graph


def sample_graph_mlp(ar, n_nodes, batch_size, mask, get_indices = False):
       
    
    if mask != None:
        # not finished
        task_indices = torch.arange(0, n_nodes)
        task_indices = task_indices[mask]
        sub_ar = ar[task_indices, task_indices]
        
        n_task_nodes = torch.sum(mask)
        indices = torch.arange(0, n_task_nodes)
        perm = torch.randperm(n_task_nodes)
        sample = perm[:batch_size]
        
        
    else:
        indices = torch.arange(0,n_nodes) #n_nodes = yearnodes
        perm = torch.randperm(n_nodes)
        sample = perm[:batch_size]
    
    indices = sample
    sample_mask = torch.zeros(n_nodes, dtype=torch.bool)
    sample_mask[indices] = True
       
    if get_indices:
        return ar[sample_mask,:][:,sample_mask], sample_mask, indices
    else:
        return ar[sample_mask,:][:,sample_mask], sample_mask
