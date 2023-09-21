# Taken from https://github.com/tech-srl/bottleneck/blob/main/main.py and
# https://github.com/tech-srl/bottleneck/blob/main/experiment.py and adapted for
# enabling to run with expander graphs.
import torch
from torch_geometric.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from argparse import ArgumentParser
from models.gnn import GNN
from tree_neighbours_match.common import Task, STOP
from models.utils import str2bool, set_seed


class Experiment:
    def __init__(self, args):
        self.task = args.task
        self.gnn = args.gnn
        self.depth = args.depth
        self.num_layers = self.depth if args.num_layers is None else args.num_layers
        self.emb_dim = args.emb_dim
        self.train_fraction = args.train_fraction
        self.max_epochs = args.max_epochs
        self.batch_size = args.batch_size
        self.accum_grad = args.accum_grad
        self.eval_every = args.eval_every
        self.loader_workers = args.loader_workers
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.stopping_criterion = args.stop
        self.patience = args.patience
        self.filename = args.filename
        self.expander = args.expander
        self.hypergraph_order = args.hypergraph_order
        self.random_seed = args.random_seed
        self.expander_edge_handling = args.expander_edge_handling

        set_seed(self.random_seed)

        self.X_train, self.X_test, dim0, out_dim, self.criterion = \
            self.task.get_dataset(self.depth, self.train_fraction, expander=self.expander,
                                  hypergraph_order=self.hypergraph_order, random_seed=self.random_seed)

        if self.gnn == 'gin':
            self.model = GNN(gnn_type='gin', task="tree_neighbours_match", num_layer=self.num_layers,
                             emb_dim=self.emb_dim, expander=self.expander,
                             expander_edge_handling=self.expander_edge_handling, tree_neighbours_dim0=dim0,
                             tree_neighbours_out_dim=out_dim, residual=True).to(self.device)
        else:
            raise ValueError('Invalid GNN type. Only GIN is currently supported.')

        print(f'Starting experiment')
        self.print_args(args)
        print(f'Training examples: {len(self.X_train)}, test examples: {len(self.X_test)}')

    def print_args(self, args):
        for arg in vars(args):
            print(f"{arg}: {getattr(args, arg)}")
        print()

    def run(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        scheduler = ReduceLROnPlateau(optimizer, mode='max', threshold_mode='abs', factor=0.5, patience=10)
        print('Starting training')

        best_test_acc = 0.0
        best_train_acc = 0.0
        best_epoch = 0
        epochs_no_improve = 0
        train_accs = []
        test_accs = []
        for epoch in range(1, (self.max_epochs // self.eval_every) + 1):
            self.model.train()
            loader = DataLoader(self.X_train * self.eval_every, batch_size=self.batch_size, shuffle=True,
                                pin_memory=True, num_workers=self.loader_workers)

            total_loss = 0
            total_num_examples = 0
            train_correct = 0
            optimizer.zero_grad()
            for i, batch in enumerate(loader):
                batch = batch.to(self.device)
                out = self.model(batch)
                loss = self.criterion(input=out, target=batch.y)
                total_num_examples += batch.num_graphs
                total_loss += (loss.item() * batch.num_graphs)
                _, train_pred = out.max(dim=1)
                train_correct += train_pred.eq(batch.y).sum().item()

                loss = loss / self.accum_grad
                loss.backward()
                if (i + 1) % self.accum_grad == 0:
                    optimizer.step()
                    optimizer.zero_grad()

            avg_training_loss = total_loss / total_num_examples
            train_acc = train_correct / total_num_examples
            scheduler.step(train_acc)

            test_acc = self.eval()
            cur_lr = [g["lr"] for g in optimizer.param_groups]

            train_accs.append(train_acc)
            test_accs.append(test_acc)

            new_best_str = ''
            stopping_threshold = 0.0001
            stopping_value = 0
            if self.stopping_criterion is STOP.TEST:
                if test_acc > best_test_acc + stopping_threshold:
                    torch.save(self.model.state_dict(), self.filename + "_final_model.pt")
                    best_test_acc = test_acc
                    best_train_acc = train_acc
                    best_epoch = epoch
                    epochs_no_improve = 0
                    stopping_value = test_acc
                    new_best_str = ' (new best test)'
                else:
                    epochs_no_improve += 1
            elif self.stopping_criterion is STOP.TRAIN:
                if train_acc > best_train_acc + stopping_threshold:
                    torch.save(self.model.state_dict(), self.filename + "_final_model.pt")
                    best_train_acc = train_acc
                    best_test_acc = test_acc
                    best_epoch = epoch
                    epochs_no_improve = 0
                    stopping_value = train_acc
                    new_best_str = ' (new best train)'
                else:
                    epochs_no_improve += 1
            print(
                f'Epoch {epoch * self.eval_every}, LR: {cur_lr}: Train loss: {avg_training_loss:.7f}, Train acc: {train_acc:.4f}, Test accuracy: {test_acc:.4f}{new_best_str}')
            torch.save({'Train': best_train_acc, 'Test': best_test_acc, 'Epoch': best_epoch}, self.filename + "_best")
            torch.save({'Train': train_accs, 'Test': test_accs}, self.filename + "_curves")
            if stopping_value == 1.0:
                break
            if epochs_no_improve >= self.patience:
                print(
                    f'{self.patience} * {self.eval_every} epochs without {self.stopping_criterion} improvement, stopping. ')
                break
        print(f'Best train acc: {best_train_acc}, epoch: {best_epoch * self.eval_every}')

        torch.save({'Train': best_train_acc, 'Test': best_test_acc, 'Epoch': best_epoch}, self.filename + "_best")
        torch.save({'Train': train_accs, 'Test': test_accs}, self.filename + "_curves")

        return best_train_acc, best_test_acc, best_epoch

    def eval(self):
        self.model.eval()
        with torch.no_grad():
            loader = DataLoader(self.X_test, batch_size=self.batch_size, shuffle=False,
                                pin_memory=True, num_workers=self.loader_workers)

            total_correct = 0
            total_examples = 0
            for batch in loader:
                batch = batch.to(self.device)
                _, pred = self.model(batch).max(dim=1)
                total_correct += pred.eq(batch.y).sum().item()
                total_examples += batch.y.size(0)
            acc = total_correct / total_examples
            return acc


def main():
    parser = ArgumentParser()
    parser.add_argument("--task", dest="task", default=Task.NEIGHBORS_MATCH, type=Task.from_string, choices=list(Task),
                        required=False)
    parser.add_argument("--gnn", dest="gnn", default="gin", type=str,
                        help='GNN type (default: gin)', choices=['gin'])
    parser.add_argument("--emb_dim", dest="emb_dim", default=32, type=int,
                        required=False)  
    parser.add_argument("--depth", dest="depth", default=5, type=int, required=False)
    parser.add_argument("--num_layers", dest="num_layers", default=3, type=int,
                        required=False)  # Use (depth+1) in original paper
    parser.add_argument("--train_fraction", dest="train_fraction", default=0.8, type=float, required=False)
    parser.add_argument("--max_epochs", dest="max_epochs", default=10000, type=int, required=False)
    parser.add_argument("--eval_every", dest="eval_every", default=100, type=int, required=False)
    parser.add_argument("--batch_size", dest="batch_size", default=1024, type=int, required=False)
    parser.add_argument("--accum_grad", dest="accum_grad", default=1, type=int, required=False)
    parser.add_argument("--stop", dest="stop", default=STOP.TRAIN, type=STOP.from_string, choices=list(STOP),
                        required=False)
    parser.add_argument("--patience", dest="patience", default=20, type=int, required=False)
    parser.add_argument("--loader_workers", dest="loader_workers", default=0, type=int, required=False)
    parser.add_argument('--filename', type=str, default="tree_neighbours_match_expander_learn_features",
                        help='filename to output result (default: )')
    parser.add_argument('--expander', dest='expander', type=str2bool, default=True,
                        help='whether to use expander graph propagation')
    parser.add_argument('--hypergraph_order', type=int, default=3,
                        help='order of hypergraph expander graph')
    parser.add_argument('--random_seed', type=int, default=42,
                        help='random seed')
    parser.add_argument('--expander_edge_handling', type=str, default='learn-features',
                        choices=['masking', 'learn-features', 'summation', 'summation-mlp'],
                        help='method to handle expander edge nodes')
    args = parser.parse_args()

    Experiment(args).run()


if __name__ == "__main__":
    main()
