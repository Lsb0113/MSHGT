import torch
import torch.nn as nn
from torch.nn.init import normal_
from torch_geometric.nn.inits import glorot
import torch.nn.functional as F
import math


class Attention_layer(nn.Module):
    def __init__(self, in_dim, hid_dim, num_edge_types, heads=1):
        super().__init__()
        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.heads = heads
        self.use_self_seg = True

        self.WQ = nn.Parameter(torch.Tensor(heads, (num_edge_types + 1) * hid_dim, hid_dim // heads),
                               requires_grad=True)
        self.WK = nn.Parameter(torch.Tensor(heads, (num_edge_types + 1) * hid_dim, hid_dim // heads),
                               requires_grad=True)
        self.WV = nn.Parameter(torch.Tensor(heads, in_dim, hid_dim // heads), requires_grad=True)
        self.Q_bias = nn.Parameter(torch.zeros((heads, 1, hid_dim // heads)))
        self.K_bias = nn.Parameter(torch.zeros((heads, 1, hid_dim // heads)))
        self.V_bias = nn.Parameter(torch.zeros((heads, 1, hid_dim // heads)))
        self.lin = nn.Linear(in_features=hid_dim, out_features=hid_dim)
        self.reset_parameters()

    def forward(self, input_x, pe_Q, pe_K):
        x_Q = torch.cat([input_x, pe_Q], dim=-1)
        x_K = torch.cat([input_x, pe_K], dim=-1)
        Q = x_Q @ self.WQ + self.Q_bias
        K = x_K @ self.WK + self.K_bias
        V = input_x @ self.WV + self.V_bias

        KT = torch.transpose(K, 2, 1)
        QKT = Q @ KT
        attn = F.softmax(QKT / math.sqrt(self.hid_dim), dim=-1)
        attn_x = attn @ V
        list_attn_x = [x for x in attn_x]
        x_cat = torch.cat(list_attn_x, dim=1)
        return self.lin(x_cat)

    def reset_parameters(self):
        glorot(self.WK)
        glorot(self.WQ)
        glorot(self.WV)


class GraphTransformerLayer(nn.Module):
    def __init__(self, in_dim, hid_dim, num_edge_types, heads, dropout):
        super().__init__()
        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.heads = heads
        self.dropout = dropout

        self.attn = Attention_layer(in_dim=in_dim, hid_dim=hid_dim, num_edge_types=num_edge_types, heads=heads)
        self.norm1 = nn.LayerNorm(hid_dim)
        self.fnn_1 = nn.Linear(in_features=hid_dim, out_features=hid_dim)
        self.norm2 = nn.LayerNorm(hid_dim)
        self.fnn_2 = nn.Linear(in_features=hid_dim, out_features=hid_dim)

    def forward(self, x, pe_Q, pe_K, deg):
        attn_x = self.attn(x, pe_Q, pe_K)
        x_1 = x + attn_x / torch.sqrt(deg).reshape(-1, 1)
        x_1 = self.norm1(x_1)
        x_1_save = x_1
        # FNN
        x_2 = self.fnn_1(x_1)
        x_2 = F.relu(x_2)
        x_2 = F.dropout(x_2, p=self.dropout, training=self.training)
        x_2 = self.fnn_2(x_2)

        x_2 = x_1_save + x_2
        out = self.norm2(x_2)

        return out
