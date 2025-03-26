import torch
import torch.nn as nn

import torch.nn.functional as F
from MSHGT.model.IMDB.GTLayer import GraphTransformerLayer
from MSHGT.model.IMDB.mf import get_all_pe
from torch_geometric.nn import GATConv, to_hetero


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
        self.conv_out = GATConv(in_channels=heads * hidden_channels, out_channels=out_channels, heads=1, dropout=0.4,
                                add_self_loops=False)

    def forward(self, x, edge_index):
        x = self.conv_in(x, edge_index).relu()
        if self.num_layers > 2:
            for i, block in enumerate(self.hidden_layers):
                x = block(x, edge_index).relu()
        x = self.conv_out(x, edge_index)
        return x


class GraphTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hid_dim = config.hidden_dim
        self.heads = config.num_heads
        self.num_layers = config.num_blocks
        self.dropout = config.dropout

        self.transformer_blocks = nn.Sequential(*[GraphTransformerLayer(config=config) for _ in range(self.num_layers)])

        self.lin_cat = nn.Linear(in_features=(self.num_layers + 1) * self.hid_dim, out_features=self.hid_dim)

    def forward(self, x, pe_Q, pe_K, deg):
        save_x = []
        save_x.append(x)
        for i, blk in enumerate(self.transformer_blocks):
            gtl_in = save_x[-1]
            gtl_out = blk(gtl_in, pe_Q, pe_K, deg)
            save_x.append(gtl_out)
        save_x=torch.stack(save_x,dim=0)
        save_x = save_x.transpose(0, 1).reshape(x.shape[0], -1)
        out_x = F.relu(self.lin_cat(save_x))

        return out_x

class MSHGTModel(nn.Module):
    def __init__(self, config, metadata):
        super(MSHGTModel, self).__init__()
        self.num_nodes = config.num_nodes
        self.in_dim = config.x_input_dim
        self.hidden_dim = config.hidden_dim
        self.svd_dim = config.svd_dim
        self.out_dim = config.classes
        self.num_edge_types = config.num_metapaths
        self.heads = config.num_heads
        self.dropout = config.dropout
        self.num_layers_gnn = config.num_gnns

        self.trans_1 = nn.Sequential(nn.Linear(in_features=self.in_dim, out_features=self.hidden_dim), nn.ReLU())
        self.trans_2 = nn.Sequential(nn.Linear(in_features=self.in_dim, out_features=self.hidden_dim), nn.ReLU())
        self.trans_3 = nn.Sequential(nn.Linear(in_features=self.in_dim, out_features=self.hidden_dim), nn.ReLU())
        self.hgnn = to_hetero(GAT(hidden_channels=self.hidden_dim, out_channels=self.hidden_dim, num_layers=self.
                                  num_layers_gnn, heads=self.heads), metadata=metadata, aggr='sum')
        self.get_pe = get_all_pe(num_nodes=self.num_nodes, hidden_dim=self.svd_dim,
                                 num_edge_types=self.num_edge_types)
        self.net = GraphTransformer(config=config)
        self.mlp = nn.Sequential(nn.Linear(in_features=self.hidden_dim, out_features=2 * self.hidden_dim), nn.ReLU(),
                                 nn.Linear(in_features=2 * self.hidden_dim, out_features=self.out_dim, bias=False))
        self.alpha = nn.Parameter(torch.ones(1), requires_grad=True)

    def forward(self, data, original_A, deg):
        x_m = self.trans_1(data['movie']['x'])
        x_d = self.trans_2(data['director']['x'])
        x_a = self.trans_3(data['actor']['x'])
        x_dict, edge_index_dict = data.x_dict, data.edge_index_dict  
        x_dict['movie'] = x_m
        x_dict['director'] = x_d
        x_dict['actor'] = x_a

        x = self.hgnn(x_dict, edge_index_dict)
        loss_pe, pe_Q, pe_K = self.get_pe(original_A)
        x_gt = self.net(x['movie'], pe_Q, pe_K, deg)
        x_gt = F.dropout(x_gt, p=self.dropout)
        out = self.mlp(x_gt)
        out_logits = F.softmax(out, dim=-1)
        return self.alpha * loss_pe, out_logits, x_gt, pe_Q, pe_K
