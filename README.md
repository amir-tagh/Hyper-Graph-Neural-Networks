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
# To test a set of SMILES with a trained model and predict the probability of each SMILES being a binder or non-binder, you can follow these steps:
# Steps to Test SMILES and Predict Probabilities
1. Prepare the Input SMILES: Ensure the test SMILES strings are in a CSV file, and each SMILES is processed into a hypergraph format using the smiles_to_hypergraph function.

2. Load the Trained Model: Use the load_model function to load the best-trained model from the saved file.

3. Create a DataLoader: Convert the test SMILES into hypergraph Data objects and create a DataLoader for batch processing.

4. Predict Probabilities: Use the trained model in evaluation mode (model.eval()) to compute probabilities using torch.nn.functional.softmax.

5. Output Predictions: Save the predictions (probabilities) for each SMILES into a CSV file.

# Changes:
1. K-Fold Cross-Validation:

1.1 Added KFold from sklearn.model_selection for splitting data into folds.
Calculates the accuracy for each fold and computes the average accuracy.
Loss Curve Plotting:

2. Plots training and validation loss curves for each fold.
# Explanation (Test SMILES)
1. Input File: The --input argument should point to a CSV file containing a column named smiles with the test SMILES strings.

2. Model File: Use the --model argument to specify the path to the saved trained model file (e.g., best_model.pth).

3. Output File: The --output argument specifies the CSV file to save the predicted probabilities.

4. Label Probabilities: The output CSV will contain:

4.1 Column SMILES: The input SMILES.
4.2 Column Non-binder Probability: Probability of the SMILES being a non-binder.
4.3 Column Binder Probability: Probability of the SMILES being a binder.
4.4 Softmax for Probabilities: The F.softmax function converts the model's logits into probabilities for each class.

