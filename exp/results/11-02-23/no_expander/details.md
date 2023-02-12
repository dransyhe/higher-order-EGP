Run Script:

`python -m exp.run \
    --device=0 \
    --gnn=gcn \
    --drop_ratio=0.5 \
    --num_layer=5 \
    --emb_dim=300 \
    --batch_size=32 \
    --epochs=100 \
    --num_workers=0 \
    --dataset=ogbg-molhiv \
    --expander=False \
    --expander_graph_generation_method=perfect-matchings \
    --expander_graph_order=3 \
    --feature=full \
    --filename=mol_no_expander`

Results:

- Best validation score: 0.8075549921614735
- Test score: 0.7639853994862782