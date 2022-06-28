# GNN
Graph Neural Network for noise classification in particle detector event data

Use generate.py, which uses functions from toy_model.py to create PMT hit datasets including dark noise hits. There is a script which will make 16 datasets for
you in npz format (data_gen_toy_.cxx). Datasets can be used to train and test the GNNs [(GravNet/GarNet)](https://arxiv.org/abs/1902.07987) to classify PMT dark noise using new_block_GravNet_Model.py (or GarNet).
This uses functions from new_block_GNN.py. If you use datasets from the toy model, you will need to modify the training/testing file to accept the npz input files. 
This can be done by changing the open_pkl_file function to open_npz_file function from new_block_GNN.py. The training/testing file will output an npz file with the predicted probabilities for each hit, along with the true positive and false positive rates to generate an ROC curve to look at the effectiveness of the binary classifier. The GNN models in new_block_GNN.py have a block architecture.

Information about the layers used in the GravNet/GarNet model can be found in caloGraphNN_keras.py which was lifted from [here](https://github.com/jkiesele/caloGraphNN). 
