"""
Simple Demo Script

This script demonstrates the Mental Health AI system with synthetic data.
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def generate_synthetic_data(n_samples=100, n_features=50):
    """
    Generate synthetic data for demonstration.
    
    Args:
        n_samples (int): Number of samples
        n_features (int): Number of features
    
    Returns:
        tuple: (X, y) where X is the feature matrix and y is the label vector
    """
    logger.info(f"Generating synthetic data with {n_samples} samples and {n_features} features")
    
    # Generate features
    X = np.random.randn(n_samples, n_features)
    
    # Generate labels (binary classification: depressed/non-depressed)
    # Also generate PHQ-8 scores (0-24 scale)
    binary_labels = np.random.randint(0, 2, size=(n_samples, 1))
    phq8_scores = np.zeros((n_samples, 1))
    
    # Assign PHQ-8 scores based on binary label
    for i in range(n_samples):
        if binary_labels[i, 0] == 0:  # Non-depressed
            phq8_scores[i, 0] = np.random.randint(0, 10)  # PHQ-8 < 10: non-depressed
        else:  # Depressed
            phq8_scores[i, 0] = np.random.randint(10, 25)  # PHQ-8 >= 10: depressed
    
    # Combine binary labels and PHQ-8 scores
    y = np.hstack((binary_labels, phq8_scores))
    
    return X, y


def create_dataset_splits(X, y, test_size=0.2, val_size=0.1, random_state=42):
    """
    Create train/val/test splits.
    
    Args:
        X (numpy.ndarray): Feature matrix
        y (numpy.ndarray): Label vector
        test_size (float): Proportion of data to use for testing
        val_size (float): Proportion of data to use for validation
        random_state (int): Random seed for reproducibility
    
    Returns:
        dict: Dictionary containing the data splits
    """
    from sklearn.model_selection import train_test_split
    
    logger.info(f"Creating dataset splits with test_size={test_size}, val_size={val_size}")
    
    # Split into train+val and test
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y[:, 0]
    )
    
    # Split train+val into train and val
    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_ratio, random_state=random_state, stratify=y_train_val[:, 0]
    )
    
    # Create dataset dictionary
    dataset = {
        'X_train': X_train,
        'y_train': y_train,
        'X_val': X_val,
        'y_val': y_val,
        'X_test': X_test,
        'y_test': y_test
    }
    
    logger.info(f"Created dataset splits with {len(X_train)} training, {len(X_val)} validation, and {len(X_test)} test samples")
    
    return dataset


class SimpleModel(torch.nn.Module):
    """Simple neural network model."""
    
    def __init__(self, input_dim, hidden_dims=[64, 32], num_classes=1):
        """
        Initialize the model.
        
        Args:
            input_dim (int): Input dimension
            hidden_dims (list): List of hidden dimensions
            num_classes (int): Number of output classes
        """
        super(SimpleModel, self).__init__()
        
        # Create layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(torch.nn.Linear(prev_dim, hidden_dim))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(0.2))
            prev_dim = hidden_dim
        
        self.feature_extractor = torch.nn.Sequential(*layers)
        self.classifier = torch.nn.Linear(prev_dim, num_classes)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Output tensor
        """
        features = self.feature_extractor(x)
        output = self.classifier(features)
        return output


def train_model(dataset, input_dim, num_epochs=10, batch_size=32, learning_rate=0.001):
    """
    Train a simple model.
    
    Args:
        dataset (dict): Dataset dictionary
        input_dim (int): Input dimension
        num_epochs (int): Number of epochs
        batch_size (int): Batch size
        learning_rate (float): Learning rate
    
    Returns:
        tuple: (model, device)
    """
    logger.info("Training a simple model")
    
    # Get device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create model
    model = SimpleModel(input_dim=input_dim, hidden_dims=[64, 32], num_classes=1)
    model.to(device)
    
    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Create loss function
    criterion = torch.nn.BCEWithLogitsLoss()
    
    # Convert data to PyTorch tensors
    X_train = torch.tensor(dataset['X_train'], dtype=torch.float32).to(device)
    y_train = torch.tensor(dataset['y_train'][:, 0:1], dtype=torch.float32).to(device)
    X_val = torch.tensor(dataset['X_val'], dtype=torch.float32).to(device)
    y_val = torch.tensor(dataset['y_val'][:, 0:1], dtype=torch.float32).to(device)
    
    # Train model
    n_batches = len(X_train) // batch_size
    
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        
        # Shuffle data
        indices = torch.randperm(len(X_train))
        X_train = X_train[indices]
        y_train = y_train[indices]
        
        # Train in batches
        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = start_idx + batch_size
            
            # Get batch
            X_batch = X_train[start_idx:end_idx]
            y_batch = y_train[start_idx:end_idx]
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            val_loss = criterion(val_outputs, y_val).item()
            
            # Calculate metrics
            val_preds = (torch.sigmoid(val_outputs) > 0.5).cpu().numpy()
            val_acc = accuracy_score(y_val.cpu().numpy(), val_preds)
            val_f1 = f1_score(y_val.cpu().numpy(), val_preds)
        
        # Log progress
        train_loss = epoch_loss / n_batches
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        logger.info(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")
    
    # Plot training curves
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig('training_loss.png')
    
    logger.info("Model training completed")
    
    return model, device


def evaluate_model(model, dataset, device):
    """
    Evaluate the trained model.
    
    Args:
        model (nn.Module): Trained model
        dataset (dict): Dataset dictionary
        device (torch.device): Device
    
    Returns:
        dict: Evaluation metrics
    """
    logger.info("Evaluating model")
    
    # Convert data to PyTorch tensors
    X_test = torch.tensor(dataset['X_test'], dtype=torch.float32).to(device)
    y_test = torch.tensor(dataset['y_test'][:, 0:1], dtype=torch.float32).to(device)
    
    # Evaluate model
    model.eval()
    with torch.no_grad():
        outputs = model(X_test)
        probs = torch.sigmoid(outputs).cpu().numpy()
        preds = (probs > 0.5).astype(int)
        
        # Calculate metrics
        accuracy = accuracy_score(dataset['y_test'][:, 0], preds.flatten())
        f1 = f1_score(dataset['y_test'][:, 0], preds.flatten())
        cm = confusion_matrix(dataset['y_test'][:, 0], preds.flatten())
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    
    # Create metrics dictionary
    metrics = {
        'accuracy': accuracy,
        'f1': f1,
        'confusion_matrix': cm,
        'probabilities': probs
    }
    
    logger.info(f"Evaluation metrics - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
    
    return metrics


def generate_clinical_report(model, dataset, device, sample_idx=0):
    """
    Generate a clinical report for a sample.
    
    Args:
        model (nn.Module): Trained model
        dataset (dict): Dataset dictionary
        device (torch.device): Device
        sample_idx (int): Sample index
    
    Returns:
        dict: Clinical report
    """
    logger.info(f"Generating clinical report for sample {sample_idx}")
    
    # Get sample data
    X_sample = torch.tensor(dataset['X_test'][sample_idx:sample_idx+1], dtype=torch.float32).to(device)
    y_sample = dataset['y_test'][sample_idx:sample_idx+1]
    
    # Get prediction
    model.eval()
    with torch.no_grad():
        output = model(X_sample)
        prob = torch.sigmoid(output).item()
    
    # Determine risk level
    if prob < 0.3:
        risk_level = 'Low'
    elif prob < 0.7:
        risk_level = 'Moderate'
    else:
        risk_level = 'High'
    
    # Create report
    report = {
        'sample_id': sample_idx,
        'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
        'depression_probability': prob,
        'risk_level': risk_level,
        'true_label': int(y_sample[0, 0]),
        'true_phq8_score': int(y_sample[0, 1])
    }
    
    # Add modality contributions (simulated)
    report['modality_contributions'] = {
        'eeg': 0.45,
        'audio': 0.35,
        'text': 0.20
    }
    
    # Add observations
    observations = []
    
    if prob < 0.3:
        observations.append("Low probability of depression detected.")
    elif prob < 0.7:
        observations.append("Moderate probability of depression detected.")
    else:
        observations.append("High probability of depression detected.")
    
    observations.append("EEG patterns show significant indicators of altered brain activity.")
    observations.append("Speech patterns show notable changes in vocal characteristics.")
    
    if risk_level == 'Low':
        observations.append("Overall risk assessment indicates low risk for depression. Continued monitoring is recommended.")
    elif risk_level == 'Moderate':
        observations.append("Overall risk assessment indicates moderate risk for depression. Regular monitoring is recommended.")
    else:
        observations.append("Overall risk assessment indicates high risk for depression. Professional intervention is recommended.")
    
    report['observations'] = observations
    
    # Add suggestions
    suggestions = []
    
    if risk_level == 'Low':
        suggestions.extend([
            "Continue regular self-monitoring",
            "Maintain healthy lifestyle habits",
            "Practice stress management techniques"
        ])
    elif risk_level == 'Moderate':
        suggestions.extend([
            "Consider consulting a mental health professional",
            "Increase self-care activities",
            "Monitor mood changes",
            "Practice mindfulness and relaxation techniques"
        ])
    else:
        suggestions.extend([
            "Urgent consultation with a mental health professional",
            "Consider therapy or counseling",
            "Establish a support network",
            "Monitor symptoms closely",
            "Develop a safety plan if needed"
        ])
    
    report['suggestions'] = suggestions
    
    # Save report
    os.makedirs('results/clinical_reports', exist_ok=True)
    with open(f'results/clinical_reports/clinical_report_{sample_idx}.json', 'w') as f:
        import json
        json.dump(report, f, indent=4)
    
    logger.info(f"Saved clinical report to results/clinical_reports/clinical_report_{sample_idx}.json")
    
    return report


def main():
    """Main function."""
    # Create output directories
    os.makedirs('results', exist_ok=True)
    os.makedirs('results/clinical_reports', exist_ok=True)
    
    # Generate synthetic data
    X, y = generate_synthetic_data(n_samples=1000, n_features=50)
    
    # Create dataset splits
    dataset = create_dataset_splits(X, y)
    
    # Train model
    model, device = train_model(dataset, input_dim=X.shape[1], num_epochs=10)
    
    # Evaluate model
    metrics = evaluate_model(model, dataset, device)
    
    # Generate clinical report
    report = generate_clinical_report(model, dataset, device)
    
    # Print report
    print("\nClinical Report:")
    print(f"Sample ID: {report['sample_id']}")
    print(f"Depression Probability: {report['depression_probability']:.1%}")
    print(f"Risk Level: {report['risk_level']}")
    print(f"True Label: {'Depressed' if report['true_label'] == 1 else 'Not Depressed'}")
    print(f"True PHQ-8 Score: {report['true_phq8_score']}")
    
    print("\nModality Contributions:")
    for modality, contribution in report['modality_contributions'].items():
        print(f"- {modality.upper()}: {contribution:.1%}")
    
    print("\nObservations:")
    for observation in report['observations']:
        print(f"- {observation}")
    
    print("\nRecommendations:")
    for suggestion in report['suggestions']:
        print(f"- {suggestion}")
    
    logger.info("Demo completed successfully")


if __name__ == "__main__":
    main()
