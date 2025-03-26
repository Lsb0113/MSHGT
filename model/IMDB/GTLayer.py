import torch
import torch.nn as nn
from torch.nn.init import normal_
from torch_geometric.nn.inits import glorot
import torch.nn.functional as F
import math


class Single_Attention_layer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.svd_dim = config.svd_dim
        self.hid_dim = config.hidden_dim
        self.heads = config.num_heads
        self.num_metapaths = config.num_metapaths
        self.dropout = config.dropout
        self.bias = config.bias

        self.Q_lin = nn.Linear(in_features=self.num_metapaths * self.svd_dim + self.hid_dim,
                               out_features=self.hid_dim // self.heads, bias=self.bias)
        self.K_lin = nn.Linear(in_features=self.num_metapaths * self.svd_dim + self.hid_dim,
                               out_features=self.hid_dim // self.heads, bias=self.bias)
        self.V_lin = nn.Linear(in_features=self.hid_dim, out_features=self.hid_dim // self.heads,
                               bias=self.bias)


    def forward(self, input_x, pe_Q, pe_K):
        x_Q = torch.concat((input_x, pe_Q), dim=-1)
        x_K = torch.concat((input_x, pe_K), dim=-1)

        Q = self.Q_lin(x_Q)  # (N,dh)
        K = self.K_lin(x_K)  # (N,dh)
        V = self.V_lin(input_x)  # (N,dh)
        QKT = Q @ K.t()
        QKT = F.dropout(QKT, p=self.dropout, training=self.training)
        attn = F.softmax(QKT / math.sqrt(self.hid_dim // self.heads), dim=-1)
        out = attn @ V
        return out


class GraphTransformerLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hid_dim = config.hidden_dim
        self.heads = config.num_heads
        self.dropout = config.dropout

        self.attn_layers_list = nn.Sequential(*[Single_Attention_layer(config) for _ in range(self.heads)])

        self.cat_heads_out = nn.Linear(in_features=self.hid_dim, out_features=self.hid_dim)


        self.norm1 = nn.BatchNorm1d(self.hid_dim)
        self.fnn_1 = nn.Linear(in_features=self.hid_dim, out_features=self.hid_dim)

        self.norm2 = nn.BatchNorm1d(self.hid_dim)
        self.fnn_2 = nn.Linear(in_features=self.hid_dim, out_features=self.hid_dim)

    def forward(self, x, pe_Q, pe_K, deg):
        head_outs_list = []
        for i, blocks in enumerate(self.attn_layers_list):
            head_outs_list.append(blocks(x, pe_Q, pe_K))
        head_outs_list = torch.stack(head_outs_list, dim=0)
        head_outs = head_outs_list.transpose(0, 1).reshape(-1, self.hid_dim).contiguous()
        attn_x = self.cat_heads_out(head_outs)
        attn_x = F.relu(attn_x)

        x_1 = x + attn_x / torch.sqrt(deg).reshape(-1, 1)
        x_1 = self.norm1(x_1)
        # FNN
        x_2 = self.fnn_1(x_1)
        # x_2 = F.relu(x_2)
        # x_2 = F.dropout(x_2, p=self.dropout, training=self.training)
        x_2 = self.fnn_2(x_2)

        x_2 = x_1 + x_2
        out = self.norm2(x_2)

        return out
