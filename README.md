# Hyper-Graph-Neural-Networks
# Implementing a Geometric Graph Neural Networks for a small molecule classifier
# This script implements a pipeline for training a Hypergraph Neural Network (HGNN) on molecular data represented as SMILES strings.
# Here's a summary of what the script does:

Features:\
SMILES to Hypergraph Conversion:\

Converts SMILES strings into hypergraph representations.
Nodes represent atoms, and hyperedges represent rings or functional groups.
Atom features include atomic number and degree.
Hyperedges represent ring systems (via GetSymmSSSR).
HGNN Model:

A PyTorch Geometric-based implementation of Hypergraph Neural Networks.
Includes a configurable number of HypergraphConv layers.
Graph-level pooling is performed using mean aggregation.
Hyperparameter Optimization with Optuna:

Optimizes:
hidden_dim: Hidden layer size.
num_layers: Number of layers in the HGNN.
batch_size: Batch size for training.
learning_rate: Learning rate for the optimizer.
Evaluates performance using 10 epochs for each trial.
Training and Evaluation:

Uses a simple training loop with CrossEntropyLoss.
Splits data into 80% training and 20% testing.
Calculates accuracy on the test set.
Command-Line Interface:

Accepts a CSV file with two columns: smiles (SMILES strings) and label (classification labels).
Allows setting the number of trials for Optuna.
