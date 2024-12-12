# Hyper-Graph-Neural-Networks
# Implementing a Geometric Graph Neural Networks for a small molecule classifier
# This script implements a pipeline for training a Hypergraph Neural Network (HGNN) on molecular data represented as SMILES strings.
# Here's a summary of what the script does:

Features:\
SMILES to Hypergraph Conversion:

1. Converts SMILES strings into hypergraph representations.
2. Nodes represent atoms, and hyperedges represent rings or functional groups.
3. Atom features include atomic number and degree.
4. Hyperedges represent ring systems (via GetSymmSSSR).
# HGNN Model:

1. A PyTorch Geometric-based implementation of Hypergraph Neural Networks.
2. Includes a configurable number of HypergraphConv layers.
3. Graph-level pooling is performed using mean aggregation.
4. Hyperparameter Optimization with Optuna:

# Optimizes:
1. hidden_dim: Hidden layer size.
2. num_layers: Number of layers in the HGNN.
3. batch_size: Batch size for training.
4. learning_rate: Learning rate for the optimizer.
5. Evaluates performance using 10 epochs for each trial.
   
# Training and Evaluation:

1. Uses a simple training loop with CrossEntropyLoss.
1.1 Splits data into 80% training and 20% testing.
1.2 Calculates accuracy on the test set.
# Command-Line Interface:

Accepts a CSV file with two columns: smiles (SMILES strings) and label (classification labels).
Allows setting the number of trials for Optuna.
