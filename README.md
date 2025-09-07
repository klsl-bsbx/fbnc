

This project implements an algorithm for subspace clustering with missing data using a subspace-aware graph neural network. This algorithm is able to perceive the subspace structure of the data and perform differentiated learning on the data features corresponding to different labels, thereby improving the performance of subspace clustering.

## Algorithm Principle

### The Core Concept of Subspace-Aware GNNs

Traditional GNN imputation algorithms typically treat all data as having the same distribution, ignoring the fact that data may come from different subspaces. Our algorithm addresses this issue by:

1. **Subspace-aware graph construction**: Enhance connections between samples within the same subspace based on label information
2. **Subspace-specific parameter learning**: Learn independent weight matrices for each subspace
3. **Subspace-specific predictors**: Use independent predictors for each subspace to impute missing values

### Algorithm Flow

1. Constructing a Subspace-Aware Graph Structure
- Building a base graph based on KNN
- Enhancing connections within the same subspace based on label information

2. Subspace-Aware Graph Convolution
- Learning a separate weight matrix for each subspace
- Performing message passing within the subspace

3. Subspace-Specific Prediction
- Using a separate predictor for each subspace
- Differentiated imputation based on the data characteristics of different subspaces

## Runtime Environment
Python 3.12 is recommended.

### Installing Dependencies

```bash
pip install numpy, torch, scipy, scikit-learn, matplotlib, tqdm
```

### Running the Example

Run the subspace-aware GNN imputation algorithm using the command line:

```bash
python run_subspace_gnn.py --input data/datasets/your_dataset.mat --treat_zeros
```
Adding --auto_cluster to the command automatically performs subspace clustering on the imputed dataset after imputation.
python run_subspace_gnn.py --input data/datasets/ORL_32x32_zeroed.mat --treat_zeros --epochs 300 --alpha 0.95 --k 30 --auto_cluster
To perform subspace clustering directly, run:
python subspace_cluster.py --data coil100_zeroed
python subspace_cluster.py --data orl_zeroed
### Parameter Description

- `--input`: Input .mat file path (required)
- `--output_dir`: Output directory, defaults to data/datasets
- `--treat_zeros`: Treat zero values ​​as missing values
- `--k`: Number of neighbors in the KNN graph, defaults to 5
- `--epochs`: Number of training epochs, defaults to 200
- `--alpha`: Weight coefficient (0-1) for intra-subspace connections, controlling the influence of subspace structure. Defaults to 0.8.
- `--list_datasets`: Lists all available .mat files in the data/datasets directory.

### Input Data Format

The algorithm accepts .mat format data files, which should contain the following fields:
- `fea`: Feature matrix, shape [n_samples, n_features]
- `gnd`: Label vector, shape [n_samples, 1] or [n_samples]

## Algorithm Advantages

1. **Subspace Structure Aware**: Utilizes label information to capture the subspace structure of the data
2. **Differentiated Learning**: Learns independent parameters for different subspaces
3. **Label-Assisted Imputation**: Utilizes label information to guide the missing value imputation process
4. **End-to-End Learning**: Requires no pre-processing or post-processing steps




For the COIL00 dataset with a 10% missingness ratio, the optimal command is:
python run_subspace_gnn.py --input data/datasets/COIL100_zeroed.mat --treat_zeros --epochs 350 --alpha 0.95 --k 30 --auto_cluster
For the ORL_32x32 dataset with a 10% missingness ratio, the optimal command is:
python run_subspace_gnn.py --input data/datasets/ORL_32x32_zeroed.mat --treat_zeros --epochs 300
--alpha 0.95 --k 5 --auto_cluster
Start TendorBoard:
tensorboard --logdir=exps --port=6006 --reload_multifile=true --reload_interval=60 --load_fast=false