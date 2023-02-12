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
    --expander=True \
    --expander_graph_generation_method=perfect-matchings \
    --expander_graph_order=3 \
    --feature=full \
    --filename=mol`

Results:

- Best validation score: 0.6737550215559474
- Test score: 0.6562139091137333