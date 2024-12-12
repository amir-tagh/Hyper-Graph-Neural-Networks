import argparse
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdmolops
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import HypergraphConv
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import optuna


def smiles_to_hypergraph(smiles):
    """
    Convert a SMILES string to a hypergraph.
    Nodes: Atoms
    Hyperedges: Rings and functional groups.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    
    # Atom features (atomic number, degree, etc.)
    node_features = []
    for atom in mol.GetAtoms():
        node_features.append([atom.GetAtomicNum(), atom.GetDegree()])
    x = torch.tensor(node_features, dtype=torch.float)
    
    # Edges: Atom connectivity
    edge_index = []
    for bond in mol.GetBonds():
        edge_index.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    
    # Hyperedges: Rings as hyperedges
    hyperedges = []
    for ring in rdmolops.GetSymmSSSR(mol):
        hyperedges.append(list(ring))
    hyperedge_index = torch.tensor(hyperedges, dtype=torch.long).t().contiguous()
    
    return Data(x=x, edge_index=edge_index, hyperedge_index=hyperedge_index)


class HGNN(torch.nn.Module):
    """
    Hypergraph Neural Network (HGNN) for molecular classification.
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(HGNN, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(HypergraphConv(input_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(HypergraphConv(hidden_dim, hidden_dim))
        self.fc = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index, hyperedge_index = data.x, data.edge_index, data.hyperedge_index
        for conv in self.convs:
            x = conv(x, edge_index, hyperedge_index)
            x = F.relu(x)
        x = torch.mean(x, dim=0)  # Graph-level pooling
        x = self.fc(x)
        return x


def train(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def test(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data)
            pred = out.argmax(dim=1)
            correct += (pred == data.y).sum().item()
            total += data.y.size(0)
    return correct / total


def objective(trial, data_list, num_classes, device):
    """
    Objective function for hyperparameter optimization.
    """
    # Hyperparameter search space
    hidden_dim = trial.suggest_int("hidden_dim", 32, 128)
    num_layers = trial.suggest_int("num_layers", 1, 4)
    batch_size = trial.suggest_int("batch_size", 16, 64)
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-4, 1e-2)
    
    # Train-test split
    train_data, test_data = train_test_split(data_list, test_size=0.2, random_state=42)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size)

    # Model setup
    model = HGNN(input_dim=2, hidden_dim=hidden_dim, output_dim=num_classes, num_layers=num_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(10):  # Use fewer epochs during hyperparameter optimization
        train(model, train_loader, optimizer, criterion, device)

    # Evaluate on test set
    accuracy = test(model, test_loader, device)
    return accuracy


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Train an HGNN using a list of SMILES.")
    parser.add_argument("--input", type=str, required=True, help="Path to CSV file with SMILES and labels.")
    parser.add_argument("--trials", type=int, default=50, help="Number of hyperparameter optimization trials.")
    args = parser.parse_args()

    # Load and preprocess data
    df = pd.read_csv(args.input)
    smiles_list = df["smiles"].tolist()
    labels = df["label"].tolist()

    # Encode labels
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels)
    num_classes = len(label_encoder.classes_)

    # Convert SMILES to hypergraphs
    data_list = []
    for smiles, label in zip(smiles_list, labels):
        try:
            data = smiles_to_hypergraph(smiles)
            data.y = torch.tensor([label], dtype=torch.long)
            data_list.append(data)
        except ValueError as e:
            print(e)

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Run hyperparameter optimization
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, data_list, num_classes, device), n_trials=args.trials)

    # Output best parameters
    print("Best parameters:", study.best_params)
    print("Best accuracy:", study.best_value)

