#!/bin/sh

python -m exp.run_code2 \
    --seed=1 \
    --device=0 \
    --gnn=gin \
    --drop_ratio=0 \
    --max_seq_len=5 \
    --num_vocab=5000 \
    --num_layer=5 \
    --emb_dim=300 \
    --batch_size=128 \
    --epochs=25 \
    --random_split=True \
    --num_workers=0 \
    --dataset=ogbg-code2 \
    --expander=True \
    --expander_graph_generation_method=perfect-matchings \
    --expander_graph_order=3 \
    --expander_edge_handling=masking \
    --filename=mol