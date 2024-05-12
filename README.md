<<<<<<< HEAD
# Step-by-step running:
## 0. Install Python libraries needed
- Install pytorch_geometric following instruction at https://github.com/rusty1s/pytorch_geometric
- Install rdkit: conda install -y -c conda-forge rdkit
- Or run the following commands to install both pytorch_geometric and rdkit:<br>
```
conda install -y -c conda-forge rdkit
conda install pytorch torchvision cudatoolkit -c pytorch
pip install torch-scatter==latest+cu101 -f https://pytorch-geometric.com/whl/torch-1.4.0.html
pip install torch-sparse==latest+cu101 -f https://pytorch-geometric.com/whl/torch-1.4.0.html
pip install torch-cluster==latest+cu101 -f https://pytorch-geometric.com/whl/torch-1.4.0.html
pip install torch-spline-conv==latest+cu101 -f https://pytorch-geometric.com/whl/torch-1.4.0.html
pip install torch-geometric
```
## 1. Create data in pytorch format
Running<br>
```
python create_data.py
```
This returns kiba_train.csv, kiba_test.csv, davis_train.csv, and davis_test.csv, saved in data/ folder. These files are in turn input to create data in pytorch format, stored at data/processed/, consisting of kiba_train.pt, kiba_test.pt, davis_train.pt, and davis_test.pt
## 2. Train a prediction model
To train model using training data. The model is chosen if it gains the best MSE for testing data.</br>
Running<br>
```
python k_fold_training.py 0
```
where the first argument is for the index of the datasets, 0/1 for 'davis' or 'kiba', respectively
=======
# SPDTA
>>>>>>> cb37af75919f80f67e6e4c650d97a9c437740716
