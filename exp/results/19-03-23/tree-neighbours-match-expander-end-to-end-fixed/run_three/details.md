``` 
task: Task.NEIGHBORS_MATCH
gnn: gin
emb_dim: 32
depth: 5
num_layers: 3
train_fraction: 0.8
max_epochs: 11000
eval_every: 100
batch_size: 1024
accum_grad: 1
stop: STOP.TRAIN
patience: 20
loader_workers: 0
filename: tree_neighbours_match_expander_learn_features_run_three
expander: True
hypergraph_order: 3
random_seed: 496
expander_edge_handling: learn-features
```