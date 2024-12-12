import argparse
import pandas as pd
from rdkit import Chem
import torch
import torch.nn.functional as F
from torch_geometric.data import DataLoader

from model import HGNN, smiles_to_hypergraph, load_model  # Assume these are in a module `model.py`


def test_model(model, test_loader, device):
    """
    Test the model on a set of SMILES and predict probabilities for each class.
    """
    model.eval()
    predictions = []
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            out = model(data)
            probs = F.softmax(out, dim=1)  # Convert logits to probabilities
            predictions.extend(probs.cpu().numpy())  # Move probabilities to CPU and convert to NumPy
    return predictions


def main():
    parser = argparse.ArgumentParser(description="Test a trained HGNN on a set of SMILES.")
    parser.add_argument("--input", type=str, required=True, help="Path to CSV file with test SMILES.")
    parser.add_argument("--model", type=str, required=True, help="Path to the trained model file.")
    parser.add_argument("--output", type=str, required=True, help="Path to save predictions as a CSV file.")
    parser.add_argument("--label_encoder", type=str, required=True, help="Path to the label encoder file (optional).")
    args = parser.parse_args()

    # Load the test SMILES
    test_df = pd.read_csv(args.input)
    smiles_list = test_df["smiles"].tolist()

    # Convert SMILES to hypergraphs
    data_list = []
    for smiles in smiles_list:
        try:
            data = smiles_to_hypergraph(smiles)
            data_list.append(data)
        except ValueError as e:
            print(f"Skipping invalid SMILES: {smiles} - {e}")

    # Create DataLoader for testing
    test_loader = DataLoader(data_list, batch_size=32, shuffle=False)

    # Load the trained model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.model, input_dim=data_list[0].x.size(1), hidden_dim=64, output_dim=2, num_layers=2, device=device)

    # Predict probabilities
    predictions = test_model(model, test_loader, device)

    # Save predictions to a CSV file
    pred_df = pd.DataFrame(predictions, columns=["Non-binder Probability", "Binder Probability"])
    pred_df.insert(0, "SMILES", smiles_list)
    pred_df.to_csv(args.output, index=False)
    print(f"Predictions saved to {args.output}")


if __name__ == "__main__":
    main()

