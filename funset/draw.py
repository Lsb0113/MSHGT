import matplotlib.pyplot as plt
import torch
import seaborn as sns
from torch_geometric.utils import degree, add_self_loops, to_scipy_sparse_matrix, is_undirected, \
    to_undirected, to_networkx
import networkx as nx
from sklearn.manifold import TSNE
import pandas as pd
import numpy as np

plt.rcParams["font.sans-serif"] = ["KaiTi"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams['svg.fonttype'] = 'none'


def visualize_embedding(outputs, labels, name):
    outputs = outputs.detach().numpy()
    labels = labels.detach().numpy()

    model = TSNE(n_components=2)  # 降维，变成2维
    node_pos = model.fit_transform(outputs)

    color_idx = {}
    for i in range(outputs.shape[0]):
        label = labels[i]
        color_idx.setdefault(label.item(), [])
        color_idx[label.item()].append(i)
    # color_idx重排序 0,1,2,3.....
    reorder_color_idx = {}
    for j in range(len(color_idx)):
        reorder_color_idx[j] = color_idx[j]

    for c, idx in reorder_color_idx.items():
        plt.scatter(node_pos[idx, 0], node_pos[idx, 1], label=c)
    plt.legend()
    plt.savefig('./figs/' + name + '_emb.svg', bbox_inches='tight')


def draw_heatmap(matrix, name, figsize):
    # 编辑做为参数的字典：
    dict_ = {'label': 'Edge weight'}

    fig, ax = plt.subplots(figsize=figsize)
    node_idx = np.array(list(range(matrix.shape[0])))

    sns.heatmap(pd.DataFrame(matrix, columns=node_idx,
                             index=node_idx), annot=False, vmax=matrix.max(), vmin=matrix.min(), xticklabels=True,
                yticklabels=True, square=True, cmap="YlGnBu", cbar_kws=dict_)
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=10)
    # plt.xticks(fontsize=30)
    # plt.yticks(fontsize=30)
    plt.savefig('./figs/' + name + '_reconstruct.svg', bbox_inches='tight')


def draw_loss(Loss_list, epochs, dataset, Type_name='Train'):
    plt.cla()
    x1 = range(1, epochs + 1)
    y1 = Loss_list
    plt.title(Type_name + ' loss vs. epoches', fontsize=20)
    plt.plot(x1, y1, '.-')
    plt.xlabel('epochs', fontsize=20)
    plt.ylabel(Type_name + ' loss', fontsize=20)
    plt.grid()
    plt.savefig('./figs/' + dataset + '/' + Type_name + '_loss.png')
    plt.show()


def draw_acc(acc_list, epochs, dataset, Type_name='Train'):
    plt.cla()
    x1 = range(1, epochs + 1)
    y1 = acc_list
    plt.title(Type_name + ' accuracy vs. epoch', fontsize=20)
    plt.plot(x1, y1, '.-')
    plt.xlabel('epoch', fontsize=20)
    plt.ylabel(Type_name + ' accuracy', fontsize=20)
    plt.grid()
    plt.savefig('./figs/' + dataset + '/' + Type_name + '_accuracy.png')
    plt.show()
