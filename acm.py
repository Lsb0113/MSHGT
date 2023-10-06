import os
import argparse
import torch

from MSHGT.funcset.data import get_dataset, train_val_test_split
import torch.nn as nn
from torch_geometric.utils import degree
from MSHGT.funcset.utils import adj_to_coo
from MSHGT.funcset.draw import draw_loss, draw_acc, visualize_embedding
from MSHGT.model.ACM.mshgt import MSHGTModel
from sklearn.metrics import f1_score
import scipy.io as sio
from torch_geometric.data import HeteroData

device = torch.device('cuda' if torch.cuda.is_available() else "cpu")


def load_args():
    parser = argparse.ArgumentParser(description='MSHGT', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--seed', type=int, default=52, help='random seed')
    parser.add_argument('--dataset', type=str, default='ACM', help='name of dataset')  # adjust
    parser.add_argument('--hidden-dim', type=int, default=64, help="hidden dimension of Transformer")  # adjust
    parser.add_argument('--dropout', type=float, default=0.4, help="dropout-rate")
    parser.add_argument('--epochs', type=int, default=300, help='number of epochs')
    parser.add_argument('--num-layers', type=int, default=1, help="number of transformer encoders")
    parser.add_argument('--heads', type=int, default=4, help="number of transformer multi-heads")
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


def train(model, data, original_A, y, train_mask, deg, loss_fn, optimizer):
    model.train()
    loss_pe, out, embs = model(data, original_A, deg)
    loss_acc = loss_fn(out[train_mask], y[train_mask])
    l = loss_pe + loss_acc
    optimizer.zero_grad()
    l.backward()
    optimizer.step()

    pred = out.argmax(dim=1)
    marco_f1 = f1_score(y_true=y[train_mask].cpu().numpy(), y_pred=pred[train_mask].cpu().numpy(), average='macro')
    mirco_f1 = f1_score(y_true=y[train_mask].cpu().numpy(), y_pred=pred[train_mask].cpu().numpy(), average='micro')

    return l.cpu().detach().numpy(), marco_f1, mirco_f1


def validate(model, data, original_A, y, val_mask, deg, loss_fn):
    model.eval()
    with torch.no_grad():
        loss_pe, out, embs = model(data, original_A, deg)
        loss_acc = loss_fn(out[val_mask], y[val_mask])
        l = loss_pe + loss_acc
        pred = out.argmax(dim=1)
        marco_f1 = f1_score(y_true=y[val_mask].cpu().numpy(), y_pred=pred[val_mask].cpu().numpy(), average='macro')
        mirco_f1 = f1_score(y_true=y[val_mask].cpu().numpy(), y_pred=pred[val_mask].cpu().numpy(), average='micro')
    return l.cpu().detach().numpy(), marco_f1, mirco_f1


def test(model, data, original_A, y, test_mask, deg):
    model.eval()
    with torch.no_grad():
        loss_pe, out, embs = model(data, original_A, deg)
        pred = out.argmax(dim=1)
        marco_f1 = f1_score(y_true=y[test_mask].cpu().numpy(), y_pred=pred[test_mask].cpu().numpy(), average='macro')
        mirco_f1 = f1_score(y_true=y[test_mask].cpu().numpy(), y_pred=pred[test_mask].cpu().numpy(), average='micro')
    return marco_f1, mirco_f1, embs


def main():
    global args
    args = load_args()
    print(args)

    # set_random_seed(args.seed)
    path = 'ACM3025.mat'
    dataset = sio.loadmat(path)
    data = HeteroData()
    adj_ptp = torch.tensor(dataset['PTP'], dtype=torch.float32)
    data['PTP'].edge_index = adj_to_coo(adj_ptp)._indices()
    adj_plp = torch.tensor(dataset['PLP'], dtype=torch.float32)
    data['PLP'].edge_index = adj_to_coo(adj_plp)._indices()
    adj_pap = torch.tensor(dataset['PAP'], dtype=torch.float32)
    data['PAP'].edge_index = adj_to_coo(adj_pap)._indices()

    data['P'].x = torch.tensor(dataset['feature'], dtype=torch.float32)
    data['P'].y = torch.tensor(dataset['label'], dtype=torch.float32).nonzero()[:, 1]
    data['P'].train_idx = torch.tensor(dataset['train_idx'][0])
    data['P'].val_idx = torch.tensor(dataset['train_idx'][0])
    data['P'].test_idx = torch.tensor(dataset['train_idx'][0])
    print(data)

    num_nodes = adj_pap.shape[0]
    num_classes = len((data['P'].y).unique())
    in_dim = data['P'].x.shape[1]

    original_A = []
    original_A.append(adj_pap)
    original_A.append(adj_plp)
    # original_A.append(adj_ptp)

    # Calculate degree
    all_A = original_A[0]
    for i in range(1, 2):
        all_A += original_A[i]
    deg = degree(index=all_A.nonzero().t()[0], num_nodes=num_nodes)
    deg = deg.to(device)
    train_mask, val_mask, test_mask = train_val_test_split(num_nodes=num_nodes, y=data['P'].y, train_p=0.5,
                                                           val_p=0.25)
    # visualize_embedding(data['P'].x[test_mask], data['P'].y[test_mask], name='org_ACM')

    data = data.to(device)
    original_A = torch.cat(original_A, dim=0).view(-1, num_nodes, num_nodes)
    original_A = original_A.to(device)
    y = data['P'].y.to(device)
    model = MFHGTModel(num_nodes=num_nodes, in_dim=in_dim, hidden_dim=args.hidden_dim, out_dim=num_classes,
                       num_edge_types=2, heads=args.heads, num_layers_gnn=3,num_layers_gt=args.num_layers, dropout=args.dropout).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=50, gamma=0.1, last_epoch=-1)

    loss_fn = nn.CrossEntropyLoss().to(device)

    save_path = 'models/' + args.dataset

    Train_Loss = []
    Train_macro_f1 = []
    Train_micro_f1 = []
    Val_Loss = []
    Val_macro_f1 = []
    Val_micro_f1 = []

    patience = args.patience
    count = 0
    max_val_acc = 0
    best_macro_f1 = 0
    best_micro_f1 = 0
    for i in range(args.epochs):
        train_loss, train_macro_f1, train_micro_f1 = train(model, data, original_A, y, train_mask, deg,
                                                           loss_fn,
                                                           optimizer)
        Train_Loss.append(train_loss)
        Train_macro_f1.append(train_macro_f1)
        Train_micro_f1.append(train_micro_f1)
        val_loss, val_macro_f1, val_micro_f1 = validate(model, data, original_A, y, val_mask, deg, loss_fn)
        Val_macro_f1.append(val_macro_f1)
        Val_micro_f1.append(val_micro_f1)
        Val_Loss.append(val_loss)
        test_macro_f1, test_micro_f1, _ = test(model, data, original_A, y, test_mask, deg)

        if i % 10 == 0:
            print(
                'Epoch {:03d}'.format(i),
                '|| train',
                'loss : {:.3f}'.format(train_loss),
                ', macro_f1 : {:.2f}%'.format(train_macro_f1 * 100),
                ', micro_f1 : {:.2f}%'.format(train_micro_f1 * 100),
                '|| val',
                'loss : {:.3f}'.format(val_loss),
                ', macro_f1 : {:.2f}%'.format(val_macro_f1 * 100),
                ', micro_f1 : {:.2f}%'.format(val_micro_f1 * 100),
                '|| test',
                ', macro_f1 : {:.2f}%'.format(test_macro_f1 * 100),
                ', micro_f1 : {:.2f}%'.format(test_micro_f1 * 100),
            )
        if i > args.convergence_epoch and args.use_lr_schedule:
            lr_scheduler.step()

        if args.use_early_stopping:
            if count <= patience:
                if max_val_acc >= Val_macro_f1[-1]:
                    if count == 0:
                        best_macro_f1 = test_macro_f1
                        best_micro_f1 = test_micro_f1
                        path = os.path.join(save_path, 'best_network_DBLP.pth')
                        torch.save(model.state_dict(), path)
                    count += 1
                else:
                    count = 0
                    max_val_acc = Val_macro_f1[-1]
            else:
                break

    _, _, out = test(model, data, original_A, y, test_mask, deg)
    # visualize_embedding(outputs=out[test_mask], labels=y[test_mask], name='ACM')

    draw_loss(Train_Loss, len(Train_Loss), args.dataset, 'Train')
    draw_acc(Train_macro_f1, len(Train_macro_f1), args.dataset, 'Train_macro_f1')
    draw_acc(Train_micro_f1, len(Train_micro_f1), args.dataset, 'Train_micro_f1')
    draw_loss(Val_Loss, len(Val_Loss), args.dataset, 'Val')
    draw_acc(Val_macro_f1, len(Val_macro_f1), args.dataset, 'Val_macro_f1')
    draw_acc(Val_micro_f1, len(Val_micro_f1), args.dataset, 'Val_micro_f1')
    print('test_macro_f1:{:.2f}'.format(best_macro_f1 * 100))
    print('test_micro_f1:{:.2f}'.format(best_micro_f1 * 100))


if __name__ == "__main__":
    main()
