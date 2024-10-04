# aneurysm_gnn
Repo associated to the paper "Mesh-informed reduced order models for aneurysm rupture risk prediction" by G.A. D'Inverno, S. Moradizadeh, S. Salavatidezfouli, P. C. Africa and G. Rozza

draw_graph.ipynb : it draws the graphs highlighting which nodes we are selecting for nodewise comparison

aneurysm_architecture.py :  main code to compare different GNN modules (training: 25%, 50%, 100% ; test: 75%)

aneurysm_layers_interp.py :  main code to compare GNNs with different number of layers (training: 25%, 50%, 100% ; test: 75%)

paraview_export_generation.ipynb : notebook to generate original data and prediction (in Pytorch) to be exported in Paraview (needs to be processed by next notebook)

paraview_export_process.ipynb : notebook that export the generated data in Paraview format.

model_comparison.ipynb: notebook to generate node-wise comparisons w.r.t. different GNN modules

draw_layers_interp.ipynb: notebook to generate node-wise comparisons w.r.t GNNs with different number of layers

model.py : it contains the GNN class

utils.py :  miscellaneous (e.g. dataset generation)
