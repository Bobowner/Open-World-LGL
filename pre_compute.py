from load_dataset import OOD_Dataset, load_dataset
from pathlib import Path
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.utils import to_undirected
from torch_sparse import SparseTensor
import torch


def getAr(edge_index, n_nodes, r):
    nnodes = edge_index.shape[1]
    
    edge_index, edge_weight = gcn_norm(edge_index, edge_weight=None, num_nodes=n_nodes)
    A = SparseTensor(row=edge_index[0], col=edge_index[1], value=edge_weight, sparse_sizes=(n_nodes, n_nodes))#.to(device)
    
    tmp = A
    for i in range(r-1):
        tmp = tmp.matmul(A)
        
    tmp = tmp.to_dense()
    
    return tmp

#path where to save matrices
r = 2
t0 = 2004
dataset = "dblp-easy"
path_start = "data"
temporal = True


ds = OOD_Dataset(dataset, inductive=True, disjunct=False, unseen_classes=[], temporal=temporal, validate = True, cumulative=True, t0=t0,r=None)
count = 0
for g in ds.graph_list:
    print(g)
    year = torch.max(torch.unique(g.node_year)).cpu().detach().numpy()
    g.edge_index = to_undirected(g.edge_index)
    ar = getAr(g.edge_index, g.x.shape[0],r)
   
    save_path = Path(path_start) /dataset/(dataset+'_r'+str(r)+'_cumulative_'+str(year)+'.pt')
    torch.save(ar, save_path)
    count += 1
    print(year)
