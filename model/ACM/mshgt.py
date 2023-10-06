import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from MSHGT.model.ACM.GTLayer import GraphTransformerLayer
from MSHGT.model.ACM.mf import get_all_pe


class GAT(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, num_layers, heads, dropout=0.4):
        super().__init__()

        self.num_layers = num_layers
        self.conv_in = GATConv(in_channels=hidden_channels, out_channels=hidden_channels, heads=heads, dropout=dropout,
                               add_self_loops=False)
        self.hidden_layers = nn.Sequential()
        for i in range(num_layers - 2):
            self.hidden_layers.add_module(name=str(i),module=GATConv(in_channels=heads * hidden_channels,
                                                                     out_channels=hidden_channels,heads=heads,
                                                                     dropout=dropout, add_self_loops=False))

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
    def __init__(self, num_nodes, in_dim, hidden_dim, out_dim, num_edge_types, heads,
                 num_layers_gnn, num_layers_gt, dropout):
        super(MSHGTModel, self).__init__()
        self.num_nodes = num_nodes
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.num_edge_types = num_edge_types
        self.heads = heads
        self.dropout = dropout

        self.trans = nn.Sequential(nn.Linear(in_features=in_dim, out_features=hidden_dim), nn.ReLU())
        self.gnn = GAT(hidden_channels=hidden_dim, out_channels=hidden_dim, num_layers=num_layers_gnn, heads=heads,
                       dropout=dropout)
        self.norm = nn.BatchNorm1d(hidden_dim)
        self.get_pe = get_all_pe(num_nodes=num_nodes, hidden_dim=hidden_dim, num_edge_types=num_edge_types)
        self.gt = GraphTransformer(in_dim=hidden_dim, hid_dim=hidden_dim, num_edge_types=num_edge_types, heads=heads,
                                   num_layers=num_layers_gt, dropout=dropout)
        self.mlp = nn.Sequential(nn.Linear(in_features=hidden_dim, out_features=2 * hidden_dim), nn.ReLU(),
                                 nn.Linear(in_features=2 * hidden_dim, out_features=out_dim))

    def forward(self, data, original_A, deg):
        x_p = self.trans(data['P'].x)
        # edge_index_pap, edge_index_ptp, edge_index_plp = data['PAP'].edge_index, data['PTP'].edge_index, data[
        #     'PLP'].edge_index
        edge_index_pap, edge_index_plp = data['PAP'].edge_index, data['PLP'].edge_index
        x_pap = self.gnn(x_p, edge_index_pap)
        # x_ptp = self.gnn(x_p, edge_index_ptp)
        x_plp = self.gnn(x_p, edge_index_plp)
        # x_all = self.norm(x_plp + x_ptp + x_pap)
        x_all = self.norm(x_plp + x_pap)
        loss_pe, pe_Q, pe_K = self.get_pe(original_A)
        x_gt = self.gt(x_all, pe_Q, pe_K, deg)
        x_gt = F.dropout(x_gt, p=self.dropout)
        out = self.mlp(x_gt)

        return loss_pe, F.softmax(out, dim=-1), x_gt
