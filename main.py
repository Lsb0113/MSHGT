import os
import argparse
import torch

from MSHGT.funcset.data import get_dataset, train_val_test_split
import torch.nn as nn
from torch_geometric.utils import degree
from MSHGT.funcset.utils import set_random_seed
from MSHGT.funcset.draw import draw_loss, draw_acc, visualize_embedding
# from MSHGT.model.IMDB.mshgt import MSHGTModel, MSHGTModelSE, MSHGTModelLN, MSHGTModelSELN
from MSHGT.model.IMDB.mshgt import MSHGTModel
from sklearn.metrics import f1_score
import time
from dataclasses import dataclass

device = torch.device('cuda' if torch.cuda.is_available() else "cpu")


def load_args():
    parser = argparse.ArgumentParser(description='MSHGT', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--seed', type=int, default=52, help='random seed')
    parser.add_argument('--dataset', type=str, default='IMDB', help='name of dataset')  # adjust
    parser.add_argument('--hidden-dim', type=int, default=64, help="hidden dimension of Transformer")  # adjust
    parser.add_argument('--dropout', type=float, default=0.2, help="dropout-rate")
    parser.add_argument('--epochs', type=int, default=1000, help='number of epochs')
    parser.add_argument('--num-layers', type=int, default=2, help="number of transformer encoders")
    parser.add_argument('--heads', type=int, default=4, help="number of transformer multi-heads")
    parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate')  # adjust
    parser.add_argument('--weight-decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--use_lr_schedule', type=bool, default=False, help='whether use warmup?')  # adjust
    parser.add_argument('--convergence_epoch', type=int, default=50, help="number of epochs for warmup")  # adjust
    parser.add_argument('--use_early_stopping', type=bool, default=True,
                        help="whether to use early stopping")  # adjust
    parser.add_argument('--use_pre_train_se', type=bool, default=False,
                        help="whether to use early stopping")  # adjust
    parser.add_argument('--use_lgt', type=bool, default=False,
                        help="whether to use early stopping")  # adjust
    parser.add_argument('--use_kd', type=bool, default=False,
                        help="whether to use early stopping")  # adjust
    parser.add_argument('--patience', type=int, default=200, help='val_dataset loss increases or is stable '
                                                                  'before the maximum iterations')  # adjust

    args = parser.parse_args()

    return args



def train(model, data, original_A, y, train_mask, deg, loss_fn, optimizer):
    model.train()
    loss_pe, out, embs, _, _ = model(data, original_A, deg)
    loss_acc = loss_fn(out[train_mask], y[train_mask])
    l = loss_pe + loss_acc
    optimizer.zero_grad()
    l.backward()
    optimizer.step()

    pred = out.argmax(dim=1)
    marco_f1 = f1_score(y_true=y[train_mask], y_pred=pred[train_mask], average='macro')
    mirco_f1 = f1_score(y_true=y[train_mask], y_pred=pred[train_mask], average='micro')

    return l.cpu().detach().numpy(), marco_f1, mirco_f1


def validate(model, data, original_A, y, val_mask, deg, loss_fn):
    model.eval()
    with torch.no_grad():
        loss_pe, out, embs, _, _ = model(data, original_A, deg)
        loss_acc = loss_fn(out[val_mask], y[val_mask])
        l = loss_pe + loss_acc
        pred = out.argmax(dim=1)
        marco_f1 = f1_score(y_true=y[val_mask], y_pred=pred[val_mask], average='macro')
        mirco_f1 = f1_score(y_true=y[val_mask], y_pred=pred[val_mask], average='micro')
    return l.cpu().detach().numpy(), marco_f1, mirco_f1


def test(model, data, original_A, y, test_mask, deg):
    model.eval()
    with torch.no_grad():
        loss_pe, out, embs, pe_Q, pe_K = model(data, original_A, deg)
        pred = out.argmax(dim=1)
        torch.save(pe_Q, 'IMDB_Q.pth')
        torch.save(pe_K, 'IMDB_K.pth')
        marco_f1 = f1_score(y_true=y[test_mask], y_pred=pred[test_mask], average='macro')
        mirco_f1 = f1_score(y_true=y[test_mask], y_pred=pred[test_mask], average='micro')
    return marco_f1, mirco_f1, embs


def main():
    global args
    args = load_args()
    print(args)

    set_random_seed(args.seed)
    dataset = get_dataset(args.dataset, True)
    data = dataset[0]
    print(data)

    node_types, edge_types = data.metadata()
    num_node = (data['movie']['x']).shape[0]
    num_classes = len((data['movie']['y']).unique())
    num_edge_types = 2
    in_dim = data['movie']['x'].shape[1]
    if args.use_pre_train_se:
        homo_data = data.to_homogeneous()  # M-D-A

    if os.path.exists('E:\MultiModal\MSHGT\IMDB_origin_A.pth'):
        original_A = torch.load('E:\MultiModal\MSHGT\IMDB_origin_A.pth')
    else:

        original_A = []
        # Calculate metapath-based adjacency matrices
        edge_index_m_d = data[edge_types[0]]['edge_index']
        adj_m_d_sp = torch.sparse_coo_tensor(indices=edge_index_m_d, values=torch.ones(edge_index_m_d.shape[1]),
                                             size=(4278, 2081))
        adj_m_d = adj_m_d_sp.to_dense()
        edge_index_m_a = data[edge_types[1]]['edge_index']
        adj_m_a_sp = torch.sparse_coo_tensor(indices=edge_index_m_a, values=torch.ones(edge_index_m_a.shape[1]),
                                             size=(4278, 5257))
        adj_m_a = adj_m_a_sp.to_dense()
        adj_mam = torch.matmul(adj_m_a_sp, adj_m_a.t())
        adj_mdm = torch.matmul(adj_m_d_sp, adj_m_d.t())
        original_A.append(adj_mam)
        original_A.append(adj_mdm)

        original_A = torch.cat(original_A, dim=0).view(-1, num_node, num_node)
        torch.save(original_A, 'IMDB_origin_A.pth')

    # calculate degree
    all_A = original_A.sum(dim=0)
    deg = degree(index=all_A.nonzero().t()[0], num_nodes=num_node)
    deg = deg.to(device)
    train_mask, val_mask, test_mask = train_val_test_split(num_nodes=num_node, y=data[node_types[0]]['y'], train_p=0.5,
                                                           val_p=0.25)
    # visualize_embedding(data[node_types[0]]['x'][test_mask], data[node_types[0]]['y'][test_mask], name='org_IMDB')
    original_A = original_A.to(device)
    data = data.to(device)

    y = data[node_types[0]]['y'].to(device)

    @dataclass
    class IMDBconfig:
        num_nodes: int = num_node
        x_input_dim: int = in_dim
        hidden_dim: int = args.hidden_dim
        svd_dim: int = 16
        classes: int = num_classes
        num_heads: int = args.heads
        dropout: float = args.dropout
        bias: bool = True
        num_blocks: int = args.num_layers
        num_metapaths: int = num_edge_types
        num_gnns: int = 2

    model = MSHGTModel(config=IMDBconfig, metadata=data.metadata()).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=50, gamma=0.1, last_epoch=-1)

    loss_fn = nn.CrossEntropyLoss().to(device)

    # the path of model saving
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

    sum_train_time = 0
    sum_val_time = 0
    for i in range(args.epochs):
        train_begin = time.time()
        train_loss, train_macro_f1, train_micro_f1 = train(model, data, original_A, y, train_mask, deg, loss_fn,
                                                               optimizer)
        train_end = time.time()
        train_time = train_end - train_begin
        sum_train_time += train_time
        Train_Loss.append(train_loss)
        Train_macro_f1.append(train_macro_f1)
        Train_micro_f1.append(train_micro_f1)
       
        val_begin = time.time()
        val_loss, val_macro_f1, val_micro_f1 = validate(model, data, original_A, y, val_mask, deg, loss_fn)
        val_end = time.time()
        val_time = val_end - val_begin
        sum_val_time += val_time
        Val_macro_f1.append(val_macro_f1)
        Val_micro_f1.append(val_micro_f1)
        
        Val_Loss.append(val_loss)
        
        test_macro_f1, test_micro_f1, _ = test(model, data, original_A, y, test_mask, deg)
        
        if i % 10 == 0:
            print(
                'Epoch {:03d}'.format(i),
                '|| train',
                'loss : {:.3f}'.format(train_loss.item()),
                'train_time : {:.8f}'.format(train_time),
                ', macro_f1 : {:.2f}%'.format(train_macro_f1 * 100),
                ', micro_f1 : {:.2f}%'.format(train_micro_f1 * 100),
                '|| val',
                'loss : {:.3f}'.format(val_loss.item()),
                'val_time : {:.8f}'.format(val_time),
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
                        path = os.path.join(save_path, 'best_network.pth')
                        torch.save(model.state_dict(), path)  # 这里会存储迄今最优模型的参数
                    count += 1
                else:
                    count = 0
                    max_val_acc = Val_macro_f1[-1]
            else:
                break

    # _, _, out,_,_ = test(model, data, original_A, y, test_mask, deg)
    # visualize_embedding(outputs=out[test_mask], labels=y[test_mask], name='IMDB')

    draw_loss(Train_Loss, len(Train_Loss), args.dataset, 'Train')
    draw_acc(Train_macro_f1, len(Train_macro_f1), args.dataset, 'Train_macro_f1')
    draw_acc(Train_micro_f1, len(Train_micro_f1), args.dataset, 'Train_micro_f1')
    
    draw_loss(Val_Loss, len(Val_Loss), args.dataset, 'Val')
    draw_acc(Val_macro_f1, len(Val_macro_f1), args.dataset, 'Val_macro_f1')
    draw_acc(Val_micro_f1, len(Val_micro_f1), args.dataset, 'Val_micro_f1')
   
    print('test_macro_f1:{:.2f}'.format(best_macro_f1 * 100))
    print('test_micro_f1:{:.2f}'.format(best_micro_f1 * 100))
    print(f'Train time/epoch:{sum_train_time / args.epochs}')
    print(f'Infernce time/epoch:{sum_val_time / args.epochs}')


if __name__ == "__main__":
    main()
