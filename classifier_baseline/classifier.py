import argparse
import os
import logging
from typing import Tuple
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import pandas as pd
from datetime import datetime
import json


class MLP(nn.Module):
    """Multi-Layer Perceptron (MLP) model."""

    def __init__(self, input_size: int, hidden_layers: list, dropout: float = 0.2):
        """
        Initialize the MLP model.

        Args:
            input_size (int): Number of input features.
            hidden_layers (list): List of integers specifying the number of neurons in each hidden layer.
            dropout (float, optional): Dropout rate. Defaults to 0.2.
        """
        super(MLP, self).__init__()
        layers = []
        previous_size = input_size

        for idx, layer_size in enumerate(hidden_layers):
            layers.append(nn.Linear(previous_size, layer_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            previous_size = layer_size
            logging.debug(f"Added layer {idx+1}: Linear({previous_size} -> {layer_size})")

        layers.append(nn.Linear(previous_size, 2))  # 2 output classes
        self.network = nn.Sequential(*layers)
        logging.info(f"MLP architecture: {self.network}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
    

def record_experiment(args, metrics, filename="./experiment_results.json"):
    """
    Record the command-line arguments and evaluation metrics into a JSON file.

    Args:
        args (argparse.Namespace): The parsed command-line arguments.
        metrics (dict): The dictionary containing evaluation metrics.
        filename (str, optional): The file where results will be stored. Defaults to 'experiment_results.json'.
    """
    # Convert the args to a dictionary
    args_dict = vars(args)

    # Combine args and metrics
    experiment_data = {
        "arguments": args_dict,
        "metrics": metrics
    }

    # Append results to the file (if exists) or create a new one, experiment_results.txt
    with open(filename, "a") as f:
        json.dump(experiment_data, f, indent=4)
        f.write("\n")

    logging.info(f"Experiment results saved to {filename}")




def set_seed(seed: int = 42) -> None:
    """
    Set seeds for reproducibility across various libraries and frameworks.

    Args:
        seed (int, optional): The seed value to use. Defaults to 42.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior in CUDA
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logging.info(f"Random seed set to {seed} for reproducibility.")


def setup_logging(log_level: str = "INFO") -> None:
    """
    Configure logging for the script. Logs are written to both the console and a log file.

    Args:
        log_level (str, optional): Logging level. Defaults to "INFO".
    """
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level}")

    # Create logs directory if it doesn't exist
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)

    # Create a unique log file name with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"experiment.log"
    log_filepath = os.path.join(log_dir, log_filename)

    # Configure root logger
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),  # Console handler
            logging.FileHandler(log_filepath),  # File handler
        ],
    )
    logging.info(f"Logging initialized. Log file: {log_filepath}")


def load_data(fmri_path: str, labels_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load and preprocess fMRI and label data from CSV files.

    Args:
        fmri_path (str): Path to the fMRI data CSV file.
        labels_path (str): Path to the labels CSV file.

    Returns:
        Tuple[np.ndarray, np.ndarray]: (features, labels)
    """
    logging.info(f"Loading fMRI data from {fmri_path}")
    fmri_data = pd.read_csv(fmri_path)
    logging.info(f"Loading labels data from {labels_path}")
    labels_data = pd.read_csv(labels_path)

    # Ensure 'IID' column exists
    if 'IID' not in fmri_data.columns or 'IID' not in labels_data.columns:
        raise ValueError("Both fMRI and labels data must contain an 'IID' column.")

    # Set IID as index for both datasets
    fmri_data.set_index("IID", inplace=True)
    labels_data.set_index("IID", inplace=True)

    # Filter the fMRI data to include only IIDs present in the labels data
    filtered_fmri_data = fmri_data.loc[labels_data.index]
    logging.info(f"Filtered fMRI data shape: {filtered_fmri_data.shape}")

    # Use DIA as the label
    if "DIA" not in labels_data.columns:
        raise ValueError("Labels data must contain a 'DIA' column.")
    labels = labels_data["DIA"]

    return filtered_fmri_data.values, labels.values


def prepare_tensors(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
    device: torch.device = torch.device("cpu"),
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Split data into training, validation, and testing sets and convert them to PyTorch tensors.

    Args:
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Labels.
        test_size (float, optional): Proportion of the dataset to include in the test split. Defaults to 0.2.
        val_size (float, optional): Proportion of the training set to include in the validation split. Defaults to 0.1.
        random_state (int, optional): Seed used by the random number generator. Defaults to 42.
        device (torch.device, optional): Device to which tensors are moved. Defaults to CPU.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    logging.info("Splitting data into training and testing sets")
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    logging.info("Splitting training data into training and validation sets")
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=val_size, random_state=random_state, stratify=y_train_full
    )

    # Convert to tensors and move to device
    logging.info("Converting data to PyTorch tensors")
    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    X_val = torch.tensor(X_val, dtype=torch.float32).to(device)
    X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.long).to(device)
    y_val = torch.tensor(y_val, dtype=torch.long).to(device)
    y_test = torch.tensor(y_test, dtype=torch.long).to(device)

    logging.info(f"Training set size: {X_train.shape[0]}")
    logging.info(f"Validation set size: {X_val.shape[0]}")
    logging.info(f"Testing set size: {X_test.shape[0]}")

    return X_train, X_val, X_test, y_train, y_val, y_test


class LogisticRegressionModel(nn.Module):
    """Logistic Regression model for binary classification."""

    def __init__(self, input_size: int, num_classes: int = 2):
        """
        Initialize the Logistic Regression model.

        Args:
            input_size (int): Number of input features.
            num_classes (int, optional): Number of output classes. Defaults to 2.
        """
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)
        logging.info(f"Logistic Regression architecture: {self.linear}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output logits.
        """
        out = self.linear(x)
        return out


def train(
    model: nn.Module,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: ReduceLROnPlateau,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    num_epochs: int = 10000,
    device: torch.device = torch.device("cpu"),
    checkpoint_path: str = "best_model.pth",
    log_interval: int = 1000,
    early_stopping_patience: int = 1000,
) -> None:
    """
    Train the Logistic Regression model.

    Args:
        model (nn.Module): The Logistic Regression model.
        criterion (nn.Module): Loss function.
        optimizer (optim.Optimizer): Optimizer.
        scheduler (ReduceLROnPlateau): Learning rate scheduler.
        X_train (torch.Tensor): Training features.
        y_train (torch.Tensor): Training labels.
        X_val (torch.Tensor): Validation features.
        y_val (torch.Tensor): Validation labels.
        num_epochs (int, optional): Number of training epochs. Defaults to 10000.
        device (torch.device, optional): Device for computation. Defaults to CPU.
        checkpoint_path (str, optional): Path to save the best model. Defaults to "best_model.pth".
        log_interval (int, optional): Interval (in epochs) for logging. Defaults to 1000.
        early_stopping_patience (int, optional): Number of epochs with no improvement after which training will be stopped. Defaults to 20.
    """
    best_loss = float("inf")
    patience_counter = 0
    logging.info("Starting training")

    for epoch in tqdm(range(1, num_epochs + 1), desc="Training"):
        model.train()
        optimizer.zero_grad()

        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            val_loss = criterion(val_outputs, y_val)

        # Step the scheduler based on validation loss
        scheduler.step(val_loss)

        # Check for improvement
        if val_loss.item() < best_loss:
            best_loss = val_loss.item()
            # save, if no, create a new file
            torch.save(model.state_dict(), checkpoint_path)

            
            logging.debug(f"Saved new best model with validation loss: {best_loss:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1

        # Early Stopping
        if patience_counter >= early_stopping_patience:
            logging.info(f"Early stopping triggered after {epoch} epochs.")
            break

        # Logging
        if epoch % log_interval == 0 or epoch == 1:
            current_lr = optimizer.param_groups[0]['lr']
            logging.info(
                f"Epoch [{epoch}/{num_epochs}], "
                f"Train Loss: {loss.item():.4f}, "
                f"Val Loss: {val_loss.item():.4f}, "
                f"LR: {current_lr:.6f}"
            )

    logging.info("Training completed")


def evaluate(
    model: nn.Module,
    X_test: torch.Tensor,
    y_test: torch.Tensor,
) -> dict:
    """
    Evaluate the trained model on the test set.

    Args:
        model (nn.Module): The trained Logistic Regression model.
        X_test (torch.Tensor): Test features.
        y_test (torch.Tensor): Test labels.

    Returns:
        dict: Evaluation metrics.
    """
    logging.info("Evaluating the model on the test set")
    model.eval()
    with torch.no_grad():
        outputs = model(X_test)
        _, predicted = torch.max(outputs, 1)

    # Convert predictions and labels to numpy arrays
    y_pred = predicted.cpu().numpy()
    y_true = y_test.cpu().numpy()

    # Calculate metrics
    acc = accuracy_score(y_true, y_pred)
    sensitivity = recall_score(y_true, y_pred, pos_label=1)  # Recall for class 1
    f1 = f1_score(y_true, y_pred)

    # Calculate specificity (recall for class 0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    # AUROC (binary case)
    # For AUROC, we need the probability estimates for the positive class
    # Apply softmax to get probabilities
    softmax = nn.Softmax(dim=1)
    with torch.no_grad():
        probabilities = softmax(outputs)[:, 1]
    auroc = roc_auc_score(y_true, probabilities.cpu())

    metrics = {
        "Accuracy": acc,
        "Sensitivity (Recall for class 1)": sensitivity,
        "Specificity (Recall for class 0)": specificity,
        "F1-Score": f1,
        "AUROC": auroc,
    }

    logging.info("Evaluation Metrics:")
    for metric, value in metrics.items():
        logging.info(f"{metric}: {value:.4f}")

    return metrics


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Train a Logistic Regression model for ADHD classification with reproducibility and comprehensive logging."
    )

    # Data paths
    parser.add_argument(
        "--fmri_path",
        type=str,
        # required=True,
        default='/home/songlinzhao/multi_modal_normative_modeling/data/ADHD/fMRI.csv',
        help="Path to the fMRI data CSV file.",
    )
    parser.add_argument(
        "--labels_path",
        type=str,
        # required=True,
        default='/home/songlinzhao/multi_modal_normative_modeling/data/ADHD/y.csv',
        help="Path to the labels CSV file.",
    )

    # Training parameters
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=1000,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--initial_lr",
        type=float,
        default=0.0001,
        help="Initial learning rate for the optimizer.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Number of epochs with no improvement after which learning rate will be reduced.",
    )
    parser.add_argument(
        "--factor",
        type=float,
        default=0.5,
        help="Factor by which the learning rate will be reduced.",
    )
    parser.add_argument(
        "--min_lr",
        type=float,
        default=1e-9,
        help="Minimum learning rate.",
    )
    parser.add_argument(
        "--hidden_layers",
        type=int,
        nargs='+',
        default=[116, 64, 32],
        help="List of hidden layer sizes. Example: --hidden_layers 512 256 128",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.0,
        help="Dropout rate between layers.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Batch size for training. If not set, uses full batch.",
    )

    # Checkpoint and logging
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="best_model.pth",
        help="Path to save the best model checkpoint.",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level.",
    )

    # Device configuration
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to run the training on ('cpu' or 'cuda').",
    )

    return parser.parse_args()


def main():
    # Parse command-line arguments
    args = parse_arguments()
    print('args::::::', args)

    # Setup logging
    setup_logging(args.log_level)

    # Log all arguments
    logging.info("Experiment Configuration:")
    for arg, value in vars(args).items():
        logging.info(f"{arg}: {value}")

    # Set reproducibility seeds
    set_seed(42)

    # Set device
    if args.device == "cuda" and not torch.cuda.is_available():
        logging.warning("CUDA is not available. Falling back to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    logging.info(f"Using device: {device}")

    # Validate data paths
    if not os.path.isfile(args.fmri_path):
        logging.error(f"fMRI data file not found at {args.fmri_path}")
        raise FileNotFoundError(f"fMRI data file not found at {args.fmri_path}")
    if not os.path.isfile(args.labels_path):
        logging.error(f"Labels data file not found at {args.labels_path}")
        raise FileNotFoundError(f"Labels data file not found at {args.labels_path}")

    # Load and preprocess data
    X, y = load_data(args.fmri_path, args.labels_path)

    # Prepare tensors
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_tensors(
        X, y, device=device
    )

    # Initialize the model
    input_size = X_train.shape[1]
    # model = LogisticRegressionModel(input_size=input_size, num_classes=2).to(device)
    model = MLP(input_size=input_size, hidden_layers=args.hidden_layers, dropout=args.dropout).to(device)

    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.initial_lr)

    # Define learning rate scheduler
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=args.factor,
        patience=args.patience,
        verbose=True,
        min_lr=args.min_lr,
    )

    # Train the model
    train(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        num_epochs=args.num_epochs,
        device=device,
        checkpoint_path=args.checkpoint_path,
        log_interval=1000,  # Can also be made an argument if needed
        early_stopping_patience=10000,  # Added early stopping
    )

    # Load the best model
    logging.info("Loading the best model for evaluation")
    model.load_state_dict(torch.load(args.checkpoint_path))

    # Evaluate the model
    metrics = evaluate(model, X_test, y_test)

    record_experiment(args, metrics)


    # Log and save metrics
    logging.info("Saving evaluation metrics to file.")
    metrics_path = os.path.splitext(args.checkpoint_path)[0] + "_metrics.txt"
    with open(metrics_path, 'w') as f:
        for metric, value in metrics.items():
            f.write(f"{metric}: {value:.4f}\n")
    logging.info(f"Saved evaluation metrics to {metrics_path}")


if __name__ == "__main__":
    main()
