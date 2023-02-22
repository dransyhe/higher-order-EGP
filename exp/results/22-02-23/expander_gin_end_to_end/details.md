Run Script:

`!python -m exp.run --device=0 --gnn=gin --drop_ratio=0.5 --num_layer=5 --emb_dim=300 --batch_size=32 --epochs=100 --num_workers=0 --dataset=ogbg-molhiv \
--expander=True \
--expander_graph_generation_method=perfect-matchings \
--expander_graph_order=3 \
--expander_edge_handling=learn-features \
--feature=full \
--filename=mol_gin_end_to_end`

Results:
- Best validation score: 0.8036418528316676
- Test score: 0.7665636648061955

