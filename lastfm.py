import os
import argparse
import torch

from MSHGT.funcset.data import get_dataset, train_val_test_split
import torch.nn as nn
from torch_geometric.utils import degree
from MSHGT.funcset.draw import draw_loss, draw_acc, draw_heatmap
from MSHGT.model.LastFM.mshgt import MSHGTModel
from torch_geometric.data import HeteroData
import numpy as np
from sklearn.metrics import roc_auc_score

device = torch.device('cuda' if torch.cuda.is_available() else "cpu")


def load_args():
    parser = argparse.ArgumentParser(description='MSHGT', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--seed', type=int, default=52, help='random seed')
    parser.add_argument('--dataset', type=str, default='LastFM', help='name of dataset')  # adjust
    parser.add_argument('--hidden-dim', type=int, default=32, help="hidden dimension of Transformer")  # adjust
    parser.add_argument('--dropout', type=float, default=0.4, help="dropout-rate")
    parser.add_argument('--epochs', type=int, default=500, help='number of epochs')
    parser.add_argument('--num-layers', type=int, default=2, help="number of transformer encoders")
    parser.add_argument('--heads', type=int, default=4, help="number of transformer multi-heads")
    parser.add_argument('--lr', type=float, default=0.01, help='initial learning rate')  # adjust
    parser.add_argument('--weight-decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--use_lr_schedule', type=bool, default=False, help='whether use warmup?')  # adjust
    parser.add_argument('--convergence_epoch', type=int, default=250, help="number of epochs for warmup")  # adjust
    parser.add_argument('--use_early_stopping', type=bool, default=True,
                        help="whether to use early stopping")  # adjust
    parser.add_argument('--patience', type=int, default=100, help='val_dataset loss increases or is stable '
                                                                  'before the maximum iterations')  # adjust

    args = parser.parse_args()

    return args


def mrr(y_pred, y_true):
    y_pred[y_pred >= 0.5] = 1
    y_pred[y_pred < 0.5] = 0
    count_true = (y_pred == y_true).sum()
    count_false = (y_pred != y_true).sum()
    return (count_true * 1 + 0.5 * count_false) * (1 / len(y_true))


def train(model, data, original_A, pos_index, neg_index, deg, optimizer):
    model.train()
    loss, pred_adj = model(data, original_A, pos_index, neg_index, deg)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    pred_adj = pred_adj.cpu().detach()
    pos = pred_adj[pos_index[0], pos_index[1]]
    neg = pred_adj[neg_index[0], neg_index[1]]
    y_pred = torch.cat([pos, neg], dim=-1)
    y_true = torch.cat([torch.ones(pos.shape), torch.zeros(neg.shape)], dim=-1)
    train_mrr = mrr(y_pred=y_pred, y_true=y_true)

    ROC_AUC = roc_auc_score(y_true=y_true, y_score=y_pred)

    return loss.cpu().detach().numpy(), ROC_AUC, train_mrr


def validate(model, data, original_A, pos_index, neg_index, deg):
    model.eval()
    with torch.no_grad():
        loss, pred_adj = model(data, original_A, pos_index, neg_index, deg)
        pred_adj = pred_adj.cpu().detach()
        pos = pred_adj[pos_index[0], pos_index[1]]
        neg = pred_adj[neg_index[0], neg_index[1]]
        y_pred = torch.cat([pos, neg], dim=-1)
        y_true = torch.cat([torch.ones(pos.shape), torch.zeros(neg.shape)], dim=-1)
        val_mrr = mrr(y_pred=y_pred, y_true=y_true)

        ROC_AUC = roc_auc_score(y_true=y_true, y_score=y_pred)

    return loss.cpu().detach().numpy(), ROC_AUC, val_mrr


def test(model, data, original_A, pos_index, neg_index, deg):
    model.eval()
    with torch.no_grad():
        loss, pred_adj = model(data, original_A, pos_index, neg_index, deg)
        pred_adj = pred_adj.cpu().detach()
        pos = pred_adj[pos_index[0], pos_index[1]]
        neg = pred_adj[neg_index[0], neg_index[1]]
        y_pred = torch.cat([pos, neg], dim=-1)
        y_true = torch.cat([torch.ones(pos.shape), torch.zeros(neg.shape)], dim=-1)
        test_mrr = mrr(y_pred=y_pred, y_true=y_true)

        ROC_AUC = roc_auc_score(y_true=y_true, y_score=y_pred)

    return ROC_AUC, test_mrr


def main():
    global args
    args = load_args()
    print(args)

    # set_random_seed(args.seed)
    dataset = get_dataset(name=args.dataset)
    data = dataset[0]
    print(data)

    datafm = HeteroData()
    node_types, edge_types = data.metadata()
    print(node_types)
    print(edge_types)
    datafm['user']['num_nodes'] = 1892
    datafm['artist']['num_nodes'] = 17632
    datafm['tag']['num_nodes'] = 1088
    datafm[('user', 'to', 'artist')]['edge_index'] = data[('user', 'to', 'artist')]['edge_index']
    datafm[('user', 'to', 'user')]['edge_index'] = data[('user', 'to', 'user')]['edge_index']
    datafm[('artist', 'to', 'user')]['edge_index'] = data[('artist', 'to', 'user')]['edge_index']
    datafm[('artist', 'to', 'tag')]['edge_index'] = data[('artist', 'to', 'tag')]['edge_index']
    datafm[('tag', 'to', 'artist')]['edge_index'] = data[('tag', 'to', 'artist')]['edge_index']
    print(datafm)
    homo_LFM = datafm.to_homogeneous()  # The order of nodes is user-artist-tag.
    print(homo_LFM)
    num_nodes = 1892
    num_edge_types = 3

    original_A = []
    # LsatFM
    edge_index_u_u = data[edge_types[1]]['edge_index']

    edge_index_u_a = data[edge_types[0]]['edge_index']
    edge_index_a_t = data[edge_types[3]]['edge_index']

    adj_u_u_sp = torch.sparse_coo_tensor(indices=edge_index_u_u, values=torch.ones(edge_index_u_u.shape[1]),
                                         size=(1892, 1892))
    adj_u_u = adj_u_u_sp.to_dense()
    adj_u_a_sp = torch.sparse_coo_tensor(indices=edge_index_u_a, values=torch.ones(edge_index_u_a.shape[1]),
                                         size=(1892, 17632))
    adj_u_a = adj_u_a_sp.to_dense()

    adj_a_t_sp = torch.sparse_coo_tensor(indices=edge_index_a_t, values=torch.ones(edge_index_a_t.shape[1]),
                                         size=(17632, 1088))
    adj_a_t = adj_a_t_sp.to_dense()

    adj_uau = torch.matmul(adj_u_a_sp, adj_u_a.t())  # 注意c没有属性，实现的时候直接作为常数矩阵
    adj_uat = torch.matmul(adj_u_a_sp, adj_a_t)
    adj_utu = torch.matmul(adj_uat, adj_uat.t())
    draw_heatmap(adj_uau[0:50, 0:50], name=args.dataset, figsize=(15, 15))

    original_A.append(adj_u_u)
    original_A.append(adj_uau)
    original_A.append(adj_utu)

    # Calculate degree
    all_A = original_A[0]
    for i in range(1, len(original_A)):
        all_A += original_A[i]
    deg = degree(index=all_A.nonzero().t()[0], num_nodes=1892)
    deg = deg.to(device)

    # Split dataset and predict link U-A
    pos_edge_index = edge_index_u_a

    neg_edge_index = (adj_u_a - 1).nonzero().t()
    num_pos_index = pos_edge_index.shape[1]
    num_neg_index = neg_edge_index.shape[1]
    pos_idx = list(range(num_pos_index))
    neg_idx = list(range(num_neg_index))
    np.random.shuffle(pos_idx)
    np.random.shuffle(neg_idx)
    train_pos_index = pos_edge_index[:, pos_idx[0:int(num_pos_index * 0.24)]]
    val_pos_index = pos_edge_index[:, pos_idx[int(num_pos_index * 0.24):int(num_pos_index * 0.3)]]
    test_pos_index = pos_edge_index[:, pos_idx[int(num_pos_index * 0.3):]]
    # positive links: negative links =1:1
    train_neg_index = neg_edge_index[:, neg_idx[0:int(num_pos_index * 0.24)]]
    val_neg_index = neg_edge_index[:, neg_idx[int(num_pos_index * 0.24):int(num_pos_index * 0.3)]]
    test_neg_index = neg_edge_index[:, neg_idx[int(num_pos_index * 0.3):num_pos_index]]

    data = data.to(device)
    original_A = torch.cat(original_A, dim=0).view(-1, num_nodes, num_nodes)
    original_A = original_A.to(device)
    model = MFHGTModel(num_nodes=num_nodes, homo_edge_index=homo_LFM.edge_index, hidden_dim=args.hidden_dim,
                       num_edge_types=num_edge_types, heads=args.heads, num_layers_gnn=3, num_layers_gt=args.num_layers,
                       dropout=args.dropout, metadata=datafm.metadata()).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=50, gamma=0.1, last_epoch=-1)

    save_path = 'models/' + args.dataset

    Train_Loss = []
    Train_ROC_AUC = []
    Val_Loss = []
    Val_ROC_AUC = []

    patience = args.patience
    count = 0
    best_val_ROC_AUC = 0
    best_ROC_AUC = 0
    for i in range(args.epochs):
        train_loss, train_ROC_AUC, train_mrr = train(model, data, original_A, train_pos_index, train_neg_index, deg,
                                                     optimizer)
        Train_Loss.append(train_loss)
        Train_ROC_AUC.append(train_ROC_AUC)
        val_loss, val_ROC_AUC, val_mrr = validate(model, data, original_A, val_pos_index, val_neg_index, deg)
        Val_ROC_AUC.append(val_ROC_AUC)
        Val_Loss.append(val_loss)
        test_ROC_AUC, test_mrr = test(model, data, original_A, test_pos_index, test_neg_index, deg)

        if i % 10 == 0:
            print(
                'Epoch {:03d}'.format(i),
                '|| train',
                'loss : {:.3f}'.format(train_loss),
                ',train_ROC_AUC : {:.2f}%'.format(train_ROC_AUC * 100),
                ',train_mrr : {:.2f}%'.format(train_mrr * 100),
                '|| val',
                'loss : {:.3f}'.format(val_loss),
                ', val_ROC_AUC : {:.2f}%'.format(val_ROC_AUC * 100),
                ',val_mrr : {:.2f}%'.format(val_mrr * 100),
                '|| test',
                ', test_ROC_AUC : {:.2f}%'.format(test_ROC_AUC * 100),
                ',test_mrr : {:.2f}%'.format(test_mrr * 100),
            )
        if i > args.convergence_epoch and args.use_lr_schedule:
            lr_scheduler.step()

        if args.use_early_stopping:
            if count <= patience:
                if best_val_ROC_AUC >= Val_ROC_AUC[-1]:
                    if count == 0:
                        best_ROC_AUC = test_ROC_AUC
                        path = os.path.join(save_path, 'best_network.pth')
                        torch.save(model.state_dict(), path)
                    count += 1
                else:
                    count = 0
                    best_val_ROC_AUC = Val_ROC_AUC[-1]
            else:
                break

    draw_loss(Train_Loss, len(Train_Loss), args.dataset, 'Train')
    draw_acc(Train_ROC_AUC, len(Train_ROC_AUC), args.dataset, 'Train_ROC_AUC')
    draw_loss(Val_Loss, len(Val_Loss), args.dataset, 'Val')
    draw_acc(Val_ROC_AUC, len(Val_ROC_AUC), args.dataset, 'Val_ROC_AUC')


if __name__ == "__main__":
    main()
