import torch
import torch.nn as nn

import torch.nn.functional as F
from MSHGT.model.LastFM.GTLayer import GraphTransformerLayer
from MSHGT.model.LastFM.mf import get_all_pe
from torch_geometric.nn import GATConv, to_hetero
from torch_geometric.nn import Node2Vec


class GAT(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, num_layers, heads, dropout=0.4):
        super().__init__()

        self.num_layers = num_layers
        self.conv_in = GATConv(in_channels=hidden_channels, out_channels=hidden_channels, heads=heads, dropout=dropout,
                               add_self_loops=False)
        self.hidden_layers = nn.Sequential()
        for i in range(num_layers - 2):
            self.hidden_layers.append(GATConv(in_channels=heads * hidden_channels, out_channels=hidden_channels,
                                              heads=heads, dropout=dropout, add_self_loops=False))
        self.conv_out = GATConv(in_channels=heads * hidden_channels, out_channels=out_channels, heads=1,
                                dropout=dropout,
                                add_self_loops=False)

    def forward(self, x, edge_index):
        x = self.conv_in(x, edge_index).relu()
        if self.num_layers > 2:
            for i, block in enumerate(self.hidden_layers):
                x = block(x, edge_index).relu()
        x = self.conv_out(x, edge_index)
        return x


class GraphTransformer(nn.Module):
    def __init__(self, in_dim, hid_dim, num_edge_types, heads, num_layers, dropout):
        super().__init__()
        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.heads = heads
        self.num_layers = num_layers
        self.dropout = dropout
        self.trans = nn.Sequential(nn.Linear(in_features=in_dim, out_features=2 * hid_dim), nn.ReLU(),
                                   nn.Linear(in_features=2 * hid_dim, out_features=hid_dim))

        self.transformer_blocks = nn.Sequential()

        for i in range(num_layers):
            self.transformer_blocks.add_module('block' + str(i + 1),
                                               GraphTransformerLayer(in_dim=hid_dim, hid_dim=hid_dim,
                                                                     num_edge_types=num_edge_types, heads=heads,
                                                                     dropout=dropout))
        self.lin_cat = nn.Linear(in_features=(num_layers + 1) * hid_dim, out_features=hid_dim)

    def forward(self, x, pe_Q, pe_K, deg):
        output_list = [x]

        for i, blk in enumerate(self.transformer_blocks):
            gtl_in = output_list[-1]
            gtl_out = blk(gtl_in, pe_Q, pe_K, deg)
            output_list.append(gtl_out)

        concat_layer_output = F.dropout(torch.cat(output_list, dim=-1), p=self.dropout)
        output_x = F.relu(self.lin_cat(concat_layer_output))

        return output_x


class MSHGTModel(nn.Module):
    def __init__(self, num_nodes, homo_edge_index, hidden_dim, num_edge_types, heads,
                 num_layers_gnn, num_layers_gt, dropout, metadata):
        super(MSHGTModel, self).__init__()
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        self.num_edge_types = num_edge_types
        self.heads = heads
        self.dropout = dropout

        self.get_embs = Node2Vec(edge_index=homo_edge_index, embedding_dim=hidden_dim, walk_length=16,
                                 context_size=8, walks_per_node=4, p=1, q=1, num_negative_samples=1)
        self.hgnn = to_hetero(GAT(hidden_channels=hidden_dim, out_channels=hidden_dim, num_layers=num_layers_gnn,
                                  heads=heads, dropout=dropout), metadata=metadata, aggr='sum')
        self.get_pe = get_all_pe(num_nodes=num_nodes, hidden_dim=hidden_dim,
                                 num_edge_types=num_edge_types)
        self.net = GraphTransformer(in_dim=hidden_dim, hid_dim=hidden_dim, num_edge_types=num_edge_types, heads=heads,
                                    num_layers=num_layers_gt, dropout=dropout)
        self.mlp = nn.Sequential(nn.Linear(in_features=hidden_dim, out_features=2 * hidden_dim), nn.ReLU(),
                                 nn.Linear(in_features=2 * hidden_dim, out_features=hidden_dim))
        self.pred_dot = nn.Linear(in_features=hidden_dim, out_features=hidden_dim)

    def forward(self, data, original_A, pos_index, neg_index, deg):
        embs = self.get_embs()  # (N,64)
        x_dict = {}
        x_dict['user'] = embs[0:1892, :]
        x_dict['artist'] = embs[1892:(17632 + 1892), :]
        x_dict['tag'] = embs[(17632 + 1892):, :]

        edge_index_dict = data.edge_index_dict
        x = self.hgnn(x_dict, edge_index_dict)
        loss_pe, pe_Q, pe_K = self.get_pe(original_A)
        x_gt = self.net(x['user'], pe_Q, pe_K, deg)
        x_gt = F.dropout(x_gt, p=self.dropout)
        out = self.mlp(x_gt)

        # pred_adj = torch.sigmoid(self.pred_dot(out) @ out.t())# U-U
        pred_adj = torch.sigmoid(self.pred_dot(out) @ x['artist'].t() / self.hidden_dim)  # U-A
        pos = pred_adj[pos_index[0], pos_index[1]]
        loss_pos = F.binary_cross_entropy(pos, torch.ones(size=(pos_index.shape[1], 1)).squeeze())
        neg = pred_adj[neg_index[0], neg_index[1]]
        loss_neg = F.binary_cross_entropy(neg, torch.zeros(size=(neg_index.shape[1], 1)).squeeze())
        loss = loss_pos + loss_neg + loss_pe

        return loss, pred_adj
