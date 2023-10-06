import os
import scipy.io as sio
import torch
from torch_geometric.data import HeteroData
from funcset.utils import adj_to_coo
import pandas as pd
import pickle
import re

data = HeteroData()
path_node = 'dataset/Yelp/node.dat'
node_info = pd.read_csv(path_node, delimiter='\t', header=None)
print(node_info)

node_idx = node_info[0].values
node_types = torch.from_numpy(node_info[2].values)

node_names = ['BUSINESS', 'LOCATION', 'STARS', 'PHRASE']
# Reorder all nodes
node_re_idx = []
count = [-1, -1, -1, -1]
for k in range(82465):
    count[node_types[k]] += 1
    node_re_idx.append(count[node_types[k]])
node_re_idx = torch.tensor(node_re_idx)

data[node_names[0]]['num_nodes'] = count[0]+1
data[node_names[1]]['num_nodes'] = count[1]+1
data[node_names[2]]['num_nodes'] = count[2]+1
data[node_names[3]]['num_nodes'] = count[3]+1

path_link = 'dataset/Yelp/link.dat'
link_info = pd.read_csv(path_link, delimiter='\t', header=None)
print(link_info)
edge_types = torch.tensor(link_info[2].values)
edge_weight = torch.tensor(link_info[3].values)
rows = torch.tensor(link_info[0].values)
cols = torch.tensor(link_info[1].values)
edge_name = [[('BUSINESS', 'to', 'LOCATION'), ('LOCATION', 'to', 'BUSINESS')],
             [('BUSINESS', 'to', 'STARS'), ('STARS', 'to', 'BUSINESS')],
             [('BUSINESS', 'to', 'PHRASE'), ('PHRASE', 'to', 'BUSINESS')]]
             # [('PHRASE', 'to', 'PHRASE')]]
for i in range(3):
    sub_rows = rows[edge_types == i]
    re_sub_rows = node_re_idx[sub_rows].reshape(1, -1)
    sub_cols = cols[edge_types == i]
    re_sub_cols = node_re_idx[sub_cols].reshape(1, -1)
    sub_edge_weight = edge_weight[edge_types == i]
    # if i < 3:
    data[edge_name[i][0]]['edge_index'] = torch.cat([re_sub_rows, re_sub_cols], dim=0)
    data[edge_name[i][1]]['edge_index'] = torch.cat([re_sub_cols, re_sub_rows], dim=0)
    data[edge_name[i][0]]['edge_attr'] = sub_edge_weight
    data[edge_name[i][1]]['edge_attr'] = sub_edge_weight
    # else:
    #     data[edge_name[i][0]]['edge_index'] = torch.cat([re_sub_rows, re_sub_cols], dim=0)
    #     data[edge_name[i][0]]['edge_attr'] = sub_edge_weight


torch.save(data, f='Yelp.pt')
