import functools
import torch
from torch_geometric.loader import DataLoader
import torch.optim as optim
import torch.nn.functional as F
from models.gnn import GNN
from exp import expander_graph_generation

from tqdm import tqdm
import argparse
import time
import numpy as np

### importing OGB
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator

### importing utils
from models.utils import str2bool, set_seed

multicls_criterion = torch.nn.CrossEntropyLoss()


def train(model, device, loader, optimizer):
    model.train()

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)

        if batch.x.shape[0] == 1 or batch.batch[-1] == 0:
            pass
        else:
            pred = model(batch)
            optimizer.zero_grad()

            loss = multicls_criterion(pred.to(torch.float32), batch.y.view(-1, ))

            loss.backward()
            optimizer.step()


def eval(model, device, loader, evaluator):
    model.eval()
    y_true = []
    y_pred = []

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)

        if batch.x.shape[0] == 1:
            pass
        else:
            with torch.no_grad():
                pred = model(batch)

            y_true.append(batch.y.view(-1, 1).detach().cpu())
            y_pred.append(torch.argmax(pred.detach(), dim=1).view(-1, 1).cpu())

    y_true = torch.cat(y_true, dim=0).numpy()
    y_pred = torch.cat(y_pred, dim=0).numpy()

    input_dict = {"y_true": y_true, "y_pred": y_pred}

    return evaluator.eval(input_dict)


def add_zeros(data):
    data.x = torch.zeros(data.num_nodes, dtype=torch.long)
    return data


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='GNN baselines on ogbg-ppa data with Pytorch Geometrics')
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed for training')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--gnn', type=str, default='gin-virtual',
                        help='GNN gin, gin-virtual, or gcn, or gcn-virtual (default: gin-virtual)')
    parser.add_argument('--drop_ratio', type=float, default=0.5,
                        help='dropout ratio (default: 0.5)')
    parser.add_argument('--num_layer', type=int, default=5,
                        help='number of GNN message passing layers (default: 5)')
    parser.add_argument('--emb_dim', type=int, default=300,
                        help='dimensionality of hidden units in GNNs (default: 300)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='number of workers (default: 0)')
    parser.add_argument('--dataset', type=str, default="ogbg-ppa",
                        choices = ["ogbg-ppa"],
                        help='dataset name (default: ogbg-ppa)')
    parser.add_argument('--expander', dest='expander', type=str2bool, default=False,
                        help='whether to use expander graph propagation')
    parser.add_argument('--expander_graph_generation_method', type=str, default="ramanujan-bipartite",
                        choices=['perfect-matchings', 'ramanujan-bipartite'],
                        help='method for generating expander graph')
    parser.add_argument('--expander_graph_order', type=int, default=3,
                        help='order of hypergraph expander graph')
    # parser.add_argument('--random_seed', type=int, default=42,
    #                     help='random seed used when generating ramanujan bipartite graphs')
    parser.add_argument('--expander_edge_handling', type=str, default='masking',
                        choices=['masking', 'learn-features', 'summation', 'summation-mlp'],
                        help='method to handle expander edge nodes')
    parser.add_argument('--filename', type=str, default="",
                        help='filename to output result (default: )')
    args = parser.parse_args()

    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")

    set_seed(args.seed)

    expander_graph_generation_fn = None
    if args.expander_graph_generation_method == "perfect-matchings":
        expander_graph_generation_fn = functools.partial(
            expander_graph_generation.add_expander_edges_via_perfect_matchings,
            args.expander_graph_order,
            True)
    elif args.expander_graph_generation_method == "ramanujan-bipartite":
        expander_graph_generation_fn = functools.partial(
            expander_graph_generation.add_expander_edges_via_ramanujan_bipartite_graph,
            args.expander_graph_order,
            args.seed,
            True)

    ### automatic dataloading and splitting
    if not args.expander:
        dataset = PygGraphPropPredDataset(name=args.dataset, transform=add_zeros)
    else:
        dataset = PygGraphPropPredDataset(name=args.dataset, pre_transform=expander_graph_generation_fn)

    split_idx = dataset.get_idx_split()

    ### automatic evaluator. takes dataset name as input
    evaluator = Evaluator(args.dataset)

    train_loader = DataLoader(dataset[split_idx["train"]], batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers)
    valid_loader = DataLoader(dataset[split_idx["valid"]], batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers)
    test_loader = DataLoader(dataset[split_idx["test"]], batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers)

    if args.gnn == 'gin':
        model = GNN(gnn_type='gin', task="mol", num_class=dataset.num_classes, num_layer=args.num_layer, emb_dim=args.emb_dim,
                    drop_ratio=args.drop_ratio, expander=args.expander, expander_edge_handling=args.expander_edge_handling).to(device)
    elif args.gnn == 'gcn':
        model = GNN(gnn_type='gcn', task="mol", num_class=dataset.num_classes, num_layer=args.num_layer, emb_dim=args.emb_dim,
                    drop_ratio=args.drop_ratio, expander=args.expander, expander_edge_handling=args.expander_edge_handling).to(device)
    else:
        raise ValueError('Invalid GNN type')

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    valid_curve = []
    test_curve = []
    train_curve = []

    best_val_so_far = 0
    for epoch in range(1, args.epochs + 1):
        print("=====Epoch {}".format(epoch))
        print('Training...')
        train(model, device, train_loader, optimizer)

        print('Evaluating...')
        train_perf = eval(model, device, train_loader, evaluator)
        valid_perf = eval(model, device, valid_loader, evaluator)
        test_perf = eval(model, device, test_loader, evaluator)

        print({'Train': train_perf, 'Validation': valid_perf, 'Test': test_perf})

        train_curve.append(train_perf[dataset.eval_metric])
        valid_curve.append(valid_perf[dataset.eval_metric])
        test_curve.append(test_perf[dataset.eval_metric])
        if valid_perf > best_val_so_far:
            # TODO: Check how long saving the model takes (shouldn't be too long) so we don't slow the training process
            start_time = time.time()
            torch.save(model.state_dict(), args.filename + "best_val_model.pt")
            print(f"Time taken to save model: {time.time() - start_time}")

    best_val_epoch = np.argmax(np.array(valid_curve))
    best_train = max(train_curve)

    print('Finished training!')
    print('Best validation score: {}'.format(valid_curve[best_val_epoch]))
    print('Test score: {}'.format(test_curve[best_val_epoch]))

    if not args.filename == '':
        torch.save({'Val': valid_curve[best_val_epoch], 'Test': test_curve[best_val_epoch],
                    'Train': train_curve[best_val_epoch], 'BestTrain': best_train}, args.filename + "_best")
        torch.save({'Val': valid_curve, 'Test': test_curve, 'Train': train_curve}, args.filename + "_curves")
        torch.save(model.state_dict(), args.filename + "final_model.pt")


if __name__ == "__main__":
    main()