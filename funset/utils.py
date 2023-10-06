import os
import random
import pandas as pd
import networkx as nx
import torch
import torch.nn as nn
from torch_geometric.utils import degree, add_self_loops, to_scipy_sparse_matrix, is_undirected, to_undirected
import numpy as np
from sklearn import metrics

def edge_index_to_adj(edge_index, edge_values=None, add_self=False, num_nodes=None):
    # 创建NXN的矩阵
    if add_self is True:
        edge_index = add_self_loops(edge_index=edge_index, num_nodes=num_nodes)
    adj_csc = to_scipy_sparse_matrix(edge_index=edge_index, edge_attr=edge_values, num_nodes=num_nodes).tocsc()
    adj = torch.tensor(adj_csc.toarray())
    return adj


def Hetero_edge_index_to_adj(edge_index, edge_values=None, add_self=False, rows_max=None, cols_max=None):
    if rows_max < cols_max:
        adj = edge_index_to_adj(edge_index, edge_values=edge_values, add_self=add_self, num_nodes=cols_max)[
              0:rows_max, :]
    else:
        adj = edge_index_to_adj(edge_index, edge_values=edge_values, add_self=add_self, num_nodes=rows_max)[:,
              0:cols_max]
    return adj


def edge_index_to_matrix(x, edge_index, edge_values=None, add_self=False):
    # 创建NXN的稀疏邻接矩阵
    if add_self is True:
        edge_index = add_self_loops(edge_index=edge_index)
    matrix_shape = np.zeros((x.shape[0], x.shape[0])).shape  # 邻接矩阵形状
    if edge_values is None:
        edge_values = torch.FloatTensor(np.ones(edge_index.shape[1]))  # 边上的权值默认为1
    matrix = torch.sparse_coo_tensor(edge_index, edge_values, matrix_shape)  # 邻接矩阵
    return matrix


def remove_edge(edge_index, u, v):
    del_src_u_mask = (edge_index[0] == u)
    del_dst_v_mask = (edge_index[1] == v)
    del_edge_uv_mask = del_dst_v_mask * del_src_u_mask
    del_src_v_mask = (edge_index[0] == v)
    del_dst_u_mask = (edge_index[1] == u)
    del_edge_vu_mask = del_dst_u_mask * del_src_v_mask

    del_edge_mask = del_edge_vu_mask + del_edge_uv_mask

    save_edge_mask = abs(del_edge_mask.int() - 1).bool()
    adjust_edge_index = edge_index[save_edge_mask]
    return adjust_edge_index


def adj_to_coo(adj):
    idx = torch.nonzero(adj).T
    data = adj[idx[0], idx[1]]
    adj_coo = torch.sparse_coo_tensor(idx, data, adj.shape)
    return adj_coo


def coo_to_adj(coo_mtx):
    return coo_mtx.to_dense()


def view_parameters(model, param_name=None):
    # 依次遍历模型每一层的参数 存储到dict中
    params = {}
    for name, param in model.named_parameters():
        params[name] = param.detach().cpu().numpy()
    if param_name is None:
        return params
    else:
        return params[param_name]


def count_parameters(model):
    # 计算模型大小
    return sum([p.numel() for p in model.parameters() if p.requires_grad])


def get_parameter(model, name):
    params = model.state_dict()
    return params[name]


def set_random_seed(seed):
    # 设定随机种子
    seed = int(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, save_path, patience=7, verbose=False, delta=0):
        """
        Args:
            save_path : 模型保存文件夹
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.save_path = save_path
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.test_acc = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta  # 设置一个可变动的范围，score < self.best_score - self.delta，尽管下一次迭代的score在变动不满足
        # <best_score 仍然可以继续迭代。

    def __call__(self, val_loss, test_acc, model):

        score = val_loss

        if self.best_score is None:
            self.best_score = score
            self.test_acc = test_acc
            self.save_checkpoint(val_loss, model)
        elif score >= self.best_score + self.delta:  # val_loss increase or stable
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.test_acc = test_acc
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation accuracy increased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        path = os.path.join(self.save_path, 'best_network.pth')
        torch.save(model.state_dict(), path)  # 这里会存储迄今最优模型的参数
        self.val_loss_min = val_loss


def l1_regularization(model, l1_alpha):
    l1_loss = []
    for module in model.modules():
        if type(module) is nn.BatchNorm2d:
            l1_loss.append(torch.abs(module.weight).sum())
    return l1_alpha * sum(l1_loss)


def l2_regularization(model, l2_alpha):
    l2_loss = []
    for module in model.modules():
        if type(module) is nn.Conv2d:
            l2_loss.append((module.weight ** 2).sum() / 2.0)
    return l2_alpha * sum(l2_loss)


def adj_mul(adj_i, adj, N):
    adj_i_sp = torch.sparse_coo_tensor(adj_i, torch.ones(adj_i.shape[1], dtype=torch.float).to(adj.device), (N, N))
    adj_sp = torch.sparse_coo_tensor(adj, torch.ones(adj.shape[1], dtype=torch.float).to(adj.device), (N, N))
    adj_j = torch.sparse.mm(adj_i_sp, adj_sp)
    adj_j = adj_j.coalesce().indices()
    return adj_j
