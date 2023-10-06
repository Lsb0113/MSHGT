import torch
from torch_geometric.datasets import IMDB, DBLP, AMiner,LastFM
import torch_geometric.transforms as T
import numpy as np


def get_dataset(name, Normalize=True):
    if name == 'IMDB':
        if Normalize == True:
            dataset = IMDB(root='dataset/' + name, transform=T.NormalizeFeatures())
        else:
            dataset = IMDB(root='dataset/' + name)
        return dataset
    if name == 'DBLP':
        if Normalize == True:
            dataset = DBLP(root='dataset/' + name, transform=T.NormalizeFeatures())
        else:
            dataset = DBLP(root='dataset/' + name)
        return dataset
    if name == 'LastFM':
        if Normalize == True:
            dataset = LastFM(root='dataset/' + name, transform=T.NormalizeFeatures())
        else:
            dataset = LastFM(root='dataset/' + name)
        return dataset


def train_val_test_split(num_nodes, y, train_p, val_p):
    node_index = torch.tensor(list(range(0, num_nodes)))
    num_classes = len(y.unique())
    train_idx = []
    val_idx = []
    test_idx = []
    for i in range(num_classes):
        label_i_idx = node_index[y == i].tolist()
        np.random.shuffle(label_i_idx)
        train_size, val_size = int(len(label_i_idx) * train_p), int(len(label_i_idx) * val_p)

        train_idx.append(label_i_idx[0:train_size])
        val_idx.append(label_i_idx[train_size:train_size + val_size])
        test_idx.append(label_i_idx[train_size + val_size:])

    all_train_idx = np.concatenate(train_idx)
    all_val_idx = np.concatenate(val_idx)
    all_test_idx = np.concatenate(test_idx)
    train_mask = torch.zeros(num_nodes)
    val_mask = torch.zeros(num_nodes)
    test_mask = torch.zeros(num_nodes)

    train_mask[all_train_idx] = 1
    val_mask[all_val_idx] = 1
    test_mask[all_test_idx] = 1
    return train_mask.bool(), val_mask.bool(), test_mask.bool()

