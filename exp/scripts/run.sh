#!/bin/sh

python -m exp.run \
    --device=0 \
    --gnn=gcn \
    --drop_ratio=0.5 \
    --num_layer=5 \
    --emb_dim=300 \
    --batch_size=32 \
    --epochs=100 \
    --num_workers=0 \
    --dataset=ogbg-molhiv \
    --feature=full \
    --filename=mol