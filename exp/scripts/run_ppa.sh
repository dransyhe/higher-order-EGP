#!/bin/sh

python -m exp.run_ppa \
    --seed=1 \
    --device=0 \
    --gnn=gin \
    --drop_ratio=0.5 \
    --num_layer=5 \
    --emb_dim=300 \
    --batch_size=32 \
    --epochs=100 \
    --num_workers=0 \
    --dataset=ogbg-ppa \
    --expander=True \
    --expander_graph_generation_method=perfect-matchings \
    --expander_graph_order=3 \
    --expander_edge_handling=masking \
    --filename=mol