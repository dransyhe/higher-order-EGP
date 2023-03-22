``` 
task: Task.NEIGHBORS_MATCH
gnn: gin
emb_dim: 32
depth: 5
num_layers: 3
train_fraction: 0.8
max_epochs: 8500
eval_every: 100
batch_size: 1024
accum_grad: 1
stop: STOP.TRAIN
patience: 20
loader_workers: 0
filename: expander-summation-mlp-run-two
expander: True
hypergraph_order: 3
random_seed: 133
expander_edge_handling: summation-mlp
```