Run Script:

`!python -m exp.run --device=0 --gnn=gin --drop_ratio=0.5 --num_layer=5 --emb_dim=300 --batch_size=32 --epochs=100 --num_workers=0 --dataset=ogbg-molhiv \
--expander=True \
--expander_graph_generation_method=perfect-matchings \
--expander_graph_order=3 \
--expander_edge_handling=summation-mlp \
--feature=full \
--filename=mol_gin_summation_mlp`

Results:
- Best validation score: 0.827654994611013
- Test score: 0.7327430039205083

