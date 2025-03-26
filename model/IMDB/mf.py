import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class SVD(nn.Module):
    def __init__(self, N, hidden_dim):
        super(SVD, self).__init__()
        self.n = N
        self.hidden_dim = hidden_dim

        self.U = nn.Parameter(torch.randn(N, hidden_dim), requires_grad=True)
        self.S = nn.Parameter(torch.ones(self.hidden_dim, dtype=torch.float32), requires_grad=True)
        self.V = nn.Parameter(torch.randn(hidden_dim, N), requires_grad=True)


        self.init_weight()

        self.register_buffer("Identity_matrix", torch.eye(n=self.hidden_dim, requires_grad=False))

    def init_weight(self):
        torch.nn.init.normal_(self.U)
        torch.nn.init.normal_(self.V)

    def forward(self, A):
        new_A = self.U @ torch.diag(self.S)
        new_A = (new_A @ self.V) / self.hidden_dim

        # The properties of the orthonormal matrix U,V are used as constraints
        regular_U = 0.5 * torch.norm(self.U.t() @ self.U - self.Identity_matrix, p=2) 
        regular_V = 0.5 * torch.norm(self.V @ self.V.t() - self.Identity_matrix, p=2)  
        loss = (nn.MSELoss()(A, new_A)) / (self.n ** 2) + regular_U + regular_V

        U = self.U @ torch.sqrt(torch.diag(self.S))
        V = torch.sqrt(torch.diag(self.S)) @ self.V
        pe_Q = U
        pe_K = V.t()
        return loss, pe_Q, pe_K


class get_all_pe(nn.Module):
    def __init__(self, num_nodes, hidden_dim, num_edge_types):
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
        pe_Q_list = torch.stack(pe_Q_list, dim=0)
        pe_K_list = torch.stack(pe_K_list, dim=0)
        
        all_pe_Q = pe_Q_list.transpose(0, 1).reshape(self.num_nodes, -1).contiguous()
        all_pe_K = pe_K_list.transpose(0, 1).reshape(self.num_nodes, -1).contiguous()
        return loss_svd / self.num_edge_types, all_pe_Q, all_pe_K

