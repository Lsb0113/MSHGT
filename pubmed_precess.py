import os
import scipy.io as sio
import torch
from torch_geometric.data import HeteroData
from funcset.utils import adj_to_coo
import pandas as pd
import pickle
import re

data = HeteroData()
path_node = 'dataset/PubMed/node.dat'
path_node_add = 'dataset/PubMed/record.dat'
node_info = pd.read_csv(path_node, delimiter='\t', header=None)
print(node_info)
node_info.drop(index=18399, inplace=True)
node_info_add = pd.read_csv(path_node_add, delimiter='\t', header=None)
print(node_info_add)
all_node_info = pd.concat([node_info, node_info_add], axis=0)
all_node_info.sort_values(by=0, inplace=True)
print(all_node_info)

node_idx = all_node_info[0].values
node_types = torch.from_numpy(all_node_info[2].values)

node_names = ['GENE', 'DISEASE', 'CHEMICAL', 'SPECIES']
# reorder all nodes
node_re_idx = []
count = [-1, -1, -1, -1]
for k in range(63109):
    count[node_types[k]] += 1
    node_re_idx.append(count[node_types[k]])
node_re_idx = torch.tensor(node_re_idx)

data[node_names[0]]['num_nodes'] = count[0] + 1
data[node_names[1]]['num_nodes'] = count[1] + 1
data[node_names[2]]['num_nodes'] = count[2] + 1
data[node_names[3]]['num_nodes'] = count[3] + 1

str_feats = all_node_info[3].values
node_feats = {0: [], 1: [], 2: [], 3: []}
count_idx = 0
for str_data in str_feats:
    data_list = re.findall(pattern=r'-?\d+\.\d+|[1-9]\d*|0$', string=str_data)
    data_list = list(map(float, data_list[:]))
    nt = all_node_info[2].values[count_idx]
    feat = torch.tensor(data_list[0:200]).reshape(1, -1)
    node_feats[nt].append(feat)

    count_idx += 1

# Distribute features for nodes
for j in range(4):
    data[node_names[j]]['x'] = torch.cat(node_feats[j], dim=0)

path_link = 'dataset/PubMed/link.dat'
link_info = pd.read_csv(path_link, delimiter='\t', header=None)
print(link_info)
edge_types = torch.tensor(link_info[2].values)
edge_weight = torch.tensor(link_info[3].values)
rows = torch.tensor(link_info[0].values)
cols = torch.tensor(link_info[1].values)
edge_name = [('GENE', 'to', 'GENE'), ('GENE', 'to', 'DISEASE'), ('DISEASE', 'to', 'DISEASE'),
             ('CHEMICAL', 'to', 'GENE'), ('CHEMICAL', 'to', 'DISEASE'), ('CHEMICAL', 'to', 'CHEMICAL'),
             ('CHEMICAL', 'to', 'SPECIES'), ('SPECIES', 'to', 'GENE'), ('SPECIES', 'to', 'DISEASE'),
             ('SPECIES', 'to', 'SPECIES')]
for i in range(10):
    sub_rows = rows[edge_types == i]
    re_sub_rows = node_re_idx[sub_rows].reshape(1, -1)
    sub_cols = cols[edge_types == i]
    re_sub_cols = node_re_idx[sub_cols].reshape(1, -1)
    sub_edge_weight = edge_weight[edge_types == i]
    sub_edge_index = torch.cat([re_sub_rows, re_sub_cols], dim=0)
    data[edge_name[i]]['edge_index'] = sub_edge_index
    data[edge_name[i]]['edge_attr'] = sub_edge_weight

print(data)
torch.save(data, f='PubMed.pt')
