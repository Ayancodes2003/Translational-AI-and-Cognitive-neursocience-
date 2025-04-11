"""
EEG Model Training Script

This script trains a deep learning model on EEG data for depression detection.
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pickle
import matplotlib.pyplot as plt
import logging
from tqdm import tqdm
import argparse

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.eeg_models import EEGNet, DeepConvNet, ShallowConvNet, EEGCNN, EEGLSTM, EEGTransformer
from data.eeg.preprocess_eeg_new import EEGProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_dataset(dataset_path):
    """
    Load dataset from pickle file.
    
    Args:
        dataset_path (str): Path to dataset pickle file
    
    Returns:
        dict: Dataset dictionary
    """
    logger.info(f"Loading dataset from {dataset_path}")
    
    with open(dataset_path, 'rb') as f:
        dataset = pickle.load(f)
    
    return dataset


def create_dataloaders(dataset, batch_size=32):
    """
    Create PyTorch DataLoaders from dataset.
    
    Args:
        dataset (dict): Dataset dictionary
        batch_size (int): Batch size
    
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    logger.info("Creating DataLoaders")
    
    # Convert numpy arrays to PyTorch tensors
    X_train = torch.FloatTensor(dataset['X_train'])
    y_train = torch.FloatTensor(dataset['y_train'][:, 0:1])  # Only use binary depression label
    
    X_val = torch.FloatTensor(dataset['X_val'])
    y_val = torch.FloatTensor(dataset['y_val'][:, 0:1])
    
    X_test = torch.FloatTensor(dataset['X_test'])
    y_test = torch.FloatTensor(dataset['y_test'][:, 0:1])
    
    # Create TensorDatasets
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)
    
    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    return train_loader, val_loader, test_loader


def train_model(model, train_loader, val_loader, device, num_epochs=50, learning_rate=0.001, weight_decay=1e-4):
    """
    Train model.
    
    Args:
        model (nn.Module): PyTorch model
        train_loader (DataLoader): Training data loader
        val_loader (DataLoader): Validation data loader
        device (torch.device): Device to train on
        num_epochs (int): Number of epochs
        learning_rate (float): Learning rate
        weight_decay (float): Weight decay
    
    Returns:
        tuple: (trained_model, train_losses, val_losses)
    """
    logger.info(f"Training model for {num_epochs} epochs")
    
    # Move model to device
    model = model.to(device)
    
    # Define loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Initialize lists to store losses
    train_losses = []
    val_losses = []
    
    # Training loop
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        
        for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} (Train)"):
            # Move data to device
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Update statistics
            train_loss += loss.item() * inputs.size(0)
        
        # Calculate average training loss
        train_loss = train_loss / len(train_loader.dataset)
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for inputs, targets in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} (Val)"):
                # Move data to device
                inputs = inputs.to(device)
                targets = targets.to(device)
                
                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                # Update statistics
                val_loss += loss.item() * inputs.size(0)
        
        # Calculate average validation loss
        val_loss = val_loss / len(val_loader.dataset)
        val_losses.append(val_loss)
        
        # Print statistics
        logger.info(f"Epoch {epoch+1}/{num_epochs}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
    
    logger.info("Finished training")
    
    return model, train_losses, val_losses


def evaluate_model(model, test_loader, device):
    """
    Evaluate model.
    
    Args:
        model (nn.Module): PyTorch model
        test_loader (DataLoader): Test data loader
        device (torch.device): Device to evaluate on
    
    Returns:
        dict: Evaluation metrics
    """
    logger.info("Evaluating model")
    
    # Move model to device
    model = model.to(device)
    
    # Set model to evaluation mode
    model.eval()
    
    # Initialize lists to store predictions and targets
    all_preds = []
    all_targets = []
    
    # Evaluation loop
    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc="Evaluating"):
            # Move data to device
            inputs = inputs.to(device)
            
            # Forward pass
            outputs = model(inputs)
            preds = torch.sigmoid(outputs)
            
            # Store predictions and targets
            all_preds.append(preds.cpu().numpy())
            all_targets.append(targets.numpy())
    
    # Concatenate predictions and targets
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    
    # Convert to binary predictions
    binary_preds = (all_preds > 0.5).astype(int)
    
    # Calculate metrics
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
    
    accuracy = accuracy_score(all_targets, binary_preds)
    precision = precision_score(all_targets, binary_preds)
    recall = recall_score(all_targets, binary_preds)
    f1 = f1_score(all_targets, binary_preds)
    conf_matrix = confusion_matrix(all_targets, binary_preds)
    auc = roc_auc_score(all_targets, all_preds)
    
    # Create metrics dictionary
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': conf_matrix,
        'auc': auc
    }
    
    # Print metrics
    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall: {recall:.4f}")
    logger.info(f"F1 Score: {f1:.4f}")
    logger.info(f"AUC: {auc:.4f}")
    logger.info(f"Confusion Matrix:\n{conf_matrix}")
    
    return metrics


def plot_training_curves(train_losses, val_losses, save_path=None):
    """
    Plot training curves.
    
    Args:
        train_losses (list): Training losses
        val_losses (list): Validation losses
        save_path (str, optional): Path to save plot
    """
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
        logger.info(f"Saved training curves to {save_path}")
    
    plt.close()


def save_model(model, model_path):
    """
    Save model.
    
    Args:
        model (nn.Module): PyTorch model
        model_path (str): Path to save model
    """
    logger.info(f"Saving model to {model_path}")
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    # Save model
    torch.save(model, model_path)


def main(args):
    """
    Main function.
    
    Args:
        args: Command line arguments
    """
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process dataset if needed
    if not os.path.exists(args.dataset_path):
        logger.info(f"Dataset not found at {args.dataset_path}. Processing dataset...")
        
        # Create EEG processor
        processor = EEGProcessor()
        
        # Process dataset
        if args.dataset == 'combined':
            processor.process_all_datasets()
        else:
            processor.process_dataset(args.dataset)
        
        # Create dataset splits
        processor.create_dataset_splits(args.dataset)
    
    # Load dataset
    dataset = load_dataset(args.dataset_path)
    
    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(dataset, args.batch_size)
    
    # Create model
    input_dim = dataset['X_train'].shape[1]
    
    if args.model_type == 'eegnet':
        model = EEGNet(input_dim=input_dim)
    elif args.model_type == 'deepconvnet':
        model = DeepConvNet(input_dim=input_dim)
    elif args.model_type == 'shallowconvnet':
        model = ShallowConvNet(input_dim=input_dim)
    elif args.model_type == 'eegcnn':
        model = EEGCNN(input_dim=input_dim)
    elif args.model_type == 'eeglstm':
        model = EEGLSTM(input_dim=input_dim)
    elif args.model_type == 'eegtransformer':
        model = EEGTransformer(input_dim=input_dim)
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")
    
    # Train model
    model, train_losses, val_losses = train_model(
        model, train_loader, val_loader, device,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # Evaluate model
    metrics = evaluate_model(model, test_loader, device)
    
    # Plot training curves
    plot_training_curves(
        train_losses, val_losses,
        save_path=os.path.join(args.output_dir, f'{args.model_type}_training_curves.png')
    )
    
    # Save model
    save_model(model, os.path.join(args.output_dir, f'{args.model_type}_model.pt'))
    
    # Save metrics
    metrics_path = os.path.join(args.output_dir, f'{args.model_type}_metrics.json')
    import json
    with open(metrics_path, 'w') as f:
        # Convert numpy arrays to lists
        metrics_json = {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in metrics.items()}
        json.dump(metrics_json, f, indent=4)
    
    logger.info(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train EEG model for depression detection')
    
    parser.add_argument('--dataset', type=str, default='combined', choices=['mne_sample', 'eegbci', 'combined'],
                        help='Dataset to use')
    parser.add_argument('--dataset_path', type=str, default='data/eeg/processed/combined_dataset.pkl',
                        help='Path to dataset pickle file')
    parser.add_argument('--model_type', type=str, default='eegnet',
                        choices=['eegnet', 'deepconvnet', 'shallowconvnet', 'eegcnn', 'eeglstm', 'eegtransformer'],
                        help='Model type')
    parser.add_argument('--num_epochs', type=int, default=50,
                        help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay')
    parser.add_argument('--output_dir', type=str, default='results/eeg',
                        help='Output directory')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    args = parser.parse_args()
    
    main(args)
