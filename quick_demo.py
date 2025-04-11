"""
Quick Demo Script for Mental Health AI

This script quickly trains simplified models and generates comparison metrics.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import time
import json

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Create output directory
os.makedirs('demo_results', exist_ok=True)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Generate synthetic data for quick demo
def generate_synthetic_data(n_samples=1000, n_features=100):
    """Generate synthetic data for quick demo."""
    print("Generating synthetic data...")
    
    # Generate features
    X = np.random.randn(n_samples, n_features)
    
    # Generate binary labels (0: non-depressed, 1: depressed)
    # Use a simple rule: if sum of first 10 features > 0, then depressed
    y = (np.sum(X[:, :10], axis=1) > 0).astype(float).reshape(-1, 1)
    
    # Split into train, val, test
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, random_state=42)
    
    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train)
    X_val_tensor = torch.FloatTensor(X_val)
    y_val_tensor = torch.FloatTensor(y_val)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.FloatTensor(y_test)
    
    # Create datasets
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    test_loader = DataLoader(test_dataset, batch_size=32)
    
    return {
        'train_loader': train_loader,
        'val_loader': val_loader,
        'test_loader': test_loader,
        'input_dim': n_features
    }

# Define models
class SimpleNN(nn.Module):
    """Simple neural network model."""
    def __init__(self, input_dim, hidden_dim=64, dropout_rate=0.5):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_dim, 1)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.fc2(x)
        return x

class LSTM(nn.Module):
    """LSTM model."""
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, dropout_rate=0.5):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_dim, 1)
    
    def forward(self, x):
        # Reshape input: (batch_size, input_dim) -> (batch_size, input_dim, 1)
        x = x.unsqueeze(2)
        
        # LSTM
        output, (hidden, _) = self.lstm(x)
        
        # Use last hidden state
        x = self.fc(hidden[-1])
        
        return x

class CNN(nn.Module):
    """1D CNN model."""
    def __init__(self, input_dim, dropout_rate=0.5):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Calculate output size after convolutions and pooling
        output_dim = input_dim // 4 * 32
        
        self.fc1 = nn.Linear(output_dim, 64)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(64, 1)
    
    def forward(self, x):
        # Reshape input: (batch_size, input_dim) -> (batch_size, 1, input_dim)
        x = x.unsqueeze(1)
        
        # Convolutional layers
        x = torch.relu(self.conv1(x))
        x = self.pool1(x)
        x = torch.relu(self.conv2(x))
        x = self.pool2(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.fc2(x)
        
        return x

# Train model
def train_model(model, train_loader, val_loader, device, model_name, num_epochs=5):
    """Train model."""
    print(f"Training {model_name} model...")
    
    # Move model to device
    model = model.to(device)
    
    # Define loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Initialize lists to store losses and metrics
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    
    # Training loop
    start_time = time.time()
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for inputs, targets in train_loader:
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
            
            # Calculate accuracy
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            train_correct += (predicted == targets).sum().item()
            train_total += targets.size(0)
        
        # Calculate average training loss and accuracy
        train_loss = train_loss / len(train_loader.dataset)
        train_accuracy = train_correct / train_total
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                # Move data to device
                inputs = inputs.to(device)
                targets = targets.to(device)
                
                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                # Update statistics
                val_loss += loss.item() * inputs.size(0)
                
                # Calculate accuracy
                predicted = (torch.sigmoid(outputs) > 0.5).float()
                val_correct += (predicted == targets).sum().item()
                val_total += targets.size(0)
        
        # Calculate average validation loss and accuracy
        val_loss = val_loss / len(val_loader.dataset)
        val_accuracy = val_correct / val_total
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        
        # Print statistics
        print(f"Epoch {epoch+1}/{num_epochs}: "
              f"Train Loss = {train_loss:.4f}, Train Acc = {train_accuracy:.4f}, "
              f"Val Loss = {val_loss:.4f}, Val Acc = {val_accuracy:.4f}")
    
    # Calculate training time
    training_time = time.time() - start_time
    print(f"Training time: {training_time:.2f} seconds")
    
    # Plot training curves
    plt.figure(figsize=(12, 5))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'{model_name} - Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(val_accuracies, label='Val Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(f'{model_name} - Training and Validation Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'demo_results/{model_name}_training_curves.png')
    plt.close()
    
    return model, training_time

# Evaluate model
def evaluate_model(model, test_loader, device, model_name):
    """Evaluate model."""
    print(f"Evaluating {model_name} model...")
    
    # Move model to device
    model = model.to(device)
    
    # Set model to evaluation mode
    model.eval()
    
    # Initialize lists to store predictions and targets
    all_preds = []
    all_targets = []
    
    # Evaluation loop
    with torch.no_grad():
        for inputs, targets in test_loader:
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
    accuracy = accuracy_score(all_targets, binary_preds)
    precision = precision_score(all_targets, binary_preds)
    recall = recall_score(all_targets, binary_preds)
    f1 = f1_score(all_targets, binary_preds)
    conf_matrix = confusion_matrix(all_targets, binary_preds)
    auc = roc_auc_score(all_targets, all_preds)
    
    # Create metrics dictionary
    metrics = {
        'model': model_name,
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'auc': float(auc),
        'confusion_matrix': conf_matrix.tolist()
    }
    
    # Print metrics
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"AUC: {auc:.4f}")
    print(f"Confusion Matrix:\n{conf_matrix}")
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Non-depressed', 'Depressed'],
                yticklabels=['Non-depressed', 'Depressed'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'{model_name} - Confusion Matrix')
    plt.tight_layout()
    plt.savefig(f'demo_results/{model_name}_confusion_matrix.png')
    plt.close()
    
    return metrics

# Compare models
def compare_models(metrics_list):
    """Compare models."""
    print("Comparing models...")
    
    # Create DataFrame from metrics
    df = pd.DataFrame(metrics_list)
    
    # Plot comparison
    plt.figure(figsize=(12, 8))
    
    # Plot accuracy, precision, recall, f1, auc
    metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1', 'auc']
    
    # Bar plot
    ax = df[['model'] + metrics_to_plot].set_index('model').plot(kind='bar', figsize=(12, 6))
    plt.title('Model Comparison')
    plt.ylabel('Score')
    plt.xlabel('Model')
    plt.ylim(0, 1)
    plt.grid(axis='y')
    plt.legend(loc='lower right')
    
    # Add values on top of bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f')
    
    plt.tight_layout()
    plt.savefig('demo_results/model_comparison.png')
    plt.close()
    
    # Save metrics to JSON
    with open('demo_results/metrics.json', 'w') as f:
        json.dump(metrics_list, f, indent=4)
    
    # Create comparison table
    comparison_table = df[['model'] + metrics_to_plot].set_index('model')
    print("\nModel Comparison:")
    print(comparison_table)
    
    # Save comparison table to CSV
    comparison_table.to_csv('demo_results/comparison_table.csv')
    
    return comparison_table

# Main function
def main():
    """Main function."""
    # Generate synthetic data
    data = generate_synthetic_data(n_samples=1000, n_features=100)
    
    # Define models to train
    models = {
        'SimpleNN': SimpleNN(input_dim=data['input_dim']),
        'LSTM': LSTM(input_dim=data['input_dim']),
        'CNN': CNN(input_dim=data['input_dim'])
    }
    
    # Train and evaluate models
    metrics_list = []
    training_times = {}
    
    for model_name, model in models.items():
        # Train model
        trained_model, training_time = train_model(
            model, data['train_loader'], data['val_loader'], device, model_name, num_epochs=5
        )
        
        # Store training time
        training_times[model_name] = training_time
        
        # Evaluate model
        metrics = evaluate_model(trained_model, data['test_loader'], device, model_name)
        
        # Add training time to metrics
        metrics['training_time'] = training_time
        
        # Add metrics to list
        metrics_list.append(metrics)
    
    # Compare models
    comparison_table = compare_models(metrics_list)
    
    # Print training times
    print("\nTraining Times:")
    for model_name, time in training_times.items():
        print(f"{model_name}: {time:.2f} seconds")
    
    print("\nDemo completed successfully!")
    print("Results saved to 'demo_results' directory.")

if __name__ == "__main__":
    main()
