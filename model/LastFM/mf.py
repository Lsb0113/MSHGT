import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class SVD(nn.Module):
    def __init__(self, N, hidden_dim):
        super(SVD, self).__init__()
        self.hidden_dim = hidden_dim

        self.U = nn.Parameter(torch.rand(N, hidden_dim), requires_grad=True)
        self.S = nn.Parameter(torch.tensor(np.ones(shape=hidden_dim), dtype=torch.float32), requires_grad=True)
        self.V = nn.Parameter(torch.rand(hidden_dim, N), requires_grad=True)

    def forward(self, A):
        new_A = self.U @ torch.diag(self.S)
        new_A = (new_A @ self.V) / self.hidden_dim

        regular_U = 0.5 * torch.norm(self.U.t() @ self.U - torch.eye(n=self.hidden_dim), p=2)  
        regular_V = 0.5 * torch.norm(self.V @ self.V.t() - torch.eye(n=self.hidden_dim), p=2)
        loss = nn.MSELoss()(A, new_A) + regular_U + regular_V

        U = self.U @ torch.sqrt(torch.diag(self.S))
        V = torch.sqrt(torch.diag(self.S)) @ self.V
        pe_Q = U
        pe_K = V.t()
        return loss, pe_Q, pe_K


class get_all_pe(nn.Module):
    def __init__(self, num_nodes,hidden_dim, num_edge_types):
        super(get_all_pe, self).__init__()
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        self.num_edge_types = num_edge_types

        self.svd_list = nn.Sequential()
        for i in range(int(num_edge_types)):
            self.svd_list.add_module(name=f'svd_{i}', module=SVD(N=num_nodes, hidden_dim=hidden_dim))

    def forward(self, original_A):
        pe_Q_list = []
        pe_K_list = []
        loss_svd = 0
        for k, block in enumerate(self.svd_list):
            loss, pe_Q, pe_K = block(original_A[k])
            pe_Q_list.append(pe_Q)
            pe_K_list.append(pe_K)
            loss_svd += loss
        all_pe_Q = torch.cat(pe_Q_list, dim=-1)
        all_pe_K = torch.cat(pe_K_list, dim=-1)
        return loss_svd /self.num_edge_types, all_pe_Q, all_pe_K
