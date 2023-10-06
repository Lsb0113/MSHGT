import os
import argparse
import torch

from MSHGT.funcset.data import get_dataset, train_val_test_split
import torch.nn as nn
from torch_geometric.utils import degree, add_self_loops
from MSHGT.funcset.draw import draw_loss, draw_acc
from MSHGT.model.PubMed.mshgt import MSHGTModel
from torch_geometric.data import HeteroData
import numpy as np
from sklearn.metrics import roc_auc_score

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
device = torch.device('cuda' if torch.cuda.is_available() else "cpu")


def load_args():
    parser = argparse.ArgumentParser(description='MSHGT', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--seed', type=int, default=52, help='random seed')
    parser.add_argument('--dataset', type=str, default='PubMed', help='name of dataset')  # adjust
    parser.add_argument('--hidden-dim', type=int, default=16, help="hidden dimension of Transformer")  # adjust
    parser.add_argument('--dropout', type=float, default=0.4, help="dropout-rate")
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--num-layers', type=int, default=1, help="number of transformer encoders")
    parser.add_argument('--heads', type=int, default=1, help="number of transformer multi-heads")
    parser.add_argument('--lr', type=float, default=0.01, help='initial learning rate')  # adjust
    parser.add_argument('--weight-decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--use_lr_schedule', type=bool, default=False, help='whether use warmup?')  # adjust
    parser.add_argument('--convergence_epoch', type=int, default=250, help="number of epochs for warmup")  # adjust
    parser.add_argument('--use_early_stopping', type=bool, default=True,
                        help="whether to use early stopping")  # adjust
    parser.add_argument('--patience', type=int, default=200, help='val_dataset loss increases or is stable '
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

    data = torch.load(f='PubMed.pt')
    print(data)
    node_types, edge_types = data.metadata()
    num_nodes = 20163
    in_dim = 200
    num_edge_types = 1


    # PubMed
    edge_index_d_d = data[edge_types[2]]['edge_index']

    adj_d_d_sp = torch.sparse_coo_tensor(indices=edge_index_d_d, values=torch.ones(edge_index_d_d.shape[1]),
                                         size=(20163, 20163))
    adj_d_d = adj_d_d_sp.to_dense()

    # Calculate degree
    ed_d_d_sl = add_self_loops(edge_index=edge_index_d_d, num_nodes=20163)[0]
    deg = degree(index=ed_d_d_sl[0], num_nodes=num_nodes)
    deg = deg.to(device)

    # Predict link D-D
    pos_edge_index = data[edge_types[2]]['edge_index']
    neg_edge_index = (adj_d_d - 1).nonzero().t()
    num_pos_index = pos_edge_index.shape[1]
    num_neg_index = neg_edge_index.shape[1]
    pos_idx = list(range(num_pos_index))
    neg_idx = list(range(num_neg_index))
    np.random.shuffle(pos_idx)
    np.random.shuffle(neg_idx)
    train_pos_index = pos_edge_index[:, pos_idx[0:int(num_pos_index * 0.024)]]
    val_pos_index = pos_edge_index[:, pos_idx[int(num_pos_index * 0.024):int(num_pos_index * 0.03)]]
    test_pos_index = pos_edge_index[:, pos_idx[int(num_pos_index * 0.03):int(num_pos_index * 0.1)]]
    # Positive link: Negative links=1:1
    train_neg_index = neg_edge_index[:, neg_idx[0:int(num_pos_index * 0.024)]]
    val_neg_index = neg_edge_index[:, neg_idx[int(num_pos_index * 0.024):int(num_pos_index * 0.03)]]
    test_neg_index = neg_edge_index[:, neg_idx[int(num_pos_index * 0.03):int(num_pos_index * 0.1)]]

    data = data.to(device)
    original_A = adj_d_d.to(device)
    model = MFHGTModel(num_nodes=num_nodes, in_dim=in_dim, hidden_dim=args.hidden_dim,
                       num_edge_types=num_edge_types, heads=args.heads, num_layers_gnn=2, num_layers_gt=args.num_layers,
                       dropout=args.dropout, metadata=data.metadata()).to(device)

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
