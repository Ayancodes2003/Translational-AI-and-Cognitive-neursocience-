"""
Demo Script

This script demonstrates the Mental Health AI system by loading datasets,
training a simple model, and generating a clinical report.
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

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.dataset_loader import DatasetLoader
from models.fusion_models import EarlyFusionModel
from clinical_insights.risk_assessment import RiskAssessor
from clinical_insights.report_generator import ClinicalReportGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_datasets():
    """
    Load datasets using the DatasetLoader.
    
    Returns:
        tuple: (eeg_dataset, audio_dataset, text_dataset, fusion_dataset)
    """
    logger.info("Loading datasets")
    
    # Create dataset loader
    loader = DatasetLoader(output_dir='data')
    
    # Load and process all datasets
    eeg_dataset, audio_dataset, text_dataset, fusion_dataset = loader.load_and_process_all_data()
    
    logger.info("Datasets loaded successfully")
    
    return eeg_dataset, audio_dataset, text_dataset, fusion_dataset


def train_simple_model(fusion_dataset):
    """
    Train a simple fusion model.
    
    Args:
        fusion_dataset (dict): Fusion dataset
    
    Returns:
        tuple: (model, device)
    """
    logger.info("Training a simple fusion model")
    
    # Get device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Get input dimensions
    input_dims = fusion_dataset['feature_dims']
    
    # Create model
    model = EarlyFusionModel(input_dims=input_dims, hidden_dims=[128, 64], num_classes=1)
    model.to(device)
    
    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Create loss function
    criterion = torch.nn.BCEWithLogitsLoss()
    
    # Convert data to PyTorch tensors
    X_train = torch.tensor(fusion_dataset['X_train'], dtype=torch.float32).to(device)
    y_train = torch.tensor(fusion_dataset['y_train'][:, 0:1], dtype=torch.float32).to(device)
    X_val = torch.tensor(fusion_dataset['X_val'], dtype=torch.float32).to(device)
    y_val = torch.tensor(fusion_dataset['y_val'][:, 0:1], dtype=torch.float32).to(device)
    
    # Train model
    num_epochs = 10
    batch_size = 32
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


def evaluate_model(model, fusion_dataset, device):
    """
    Evaluate the trained model.
    
    Args:
        model (nn.Module): Trained model
        fusion_dataset (dict): Fusion dataset
        device (torch.device): Device
    
    Returns:
        dict: Evaluation metrics
    """
    logger.info("Evaluating model")
    
    # Convert data to PyTorch tensors
    X_test = torch.tensor(fusion_dataset['X_test'], dtype=torch.float32).to(device)
    y_test = torch.tensor(fusion_dataset['y_test'][:, 0:1], dtype=torch.float32).to(device)
    
    # Evaluate model
    model.eval()
    with torch.no_grad():
        outputs = model(X_test)
        probs = torch.sigmoid(outputs).cpu().numpy()
        preds = (probs > 0.5).astype(int)
        
        # Calculate metrics
        accuracy = accuracy_score(fusion_dataset['y_test'][:, 0], preds.flatten())
        f1 = f1_score(fusion_dataset['y_test'][:, 0], preds.flatten())
        cm = confusion_matrix(fusion_dataset['y_test'][:, 0], preds.flatten())
    
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


def generate_clinical_report(model, fusion_dataset, device):
    """
    Generate a clinical report for a sample.
    
    Args:
        model (nn.Module): Trained model
        fusion_dataset (dict): Fusion dataset
        device (torch.device): Device
    
    Returns:
        dict: Clinical report
    """
    logger.info("Generating clinical report")
    
    # Create risk assessor
    risk_assessor = RiskAssessor(threshold_low=0.3, threshold_high=0.7)
    risk_assessor.model = model
    risk_assessor.device = device
    
    # Create report generator
    report_generator = ClinicalReportGenerator(risk_assessor=risk_assessor)
    
    # Get a sample from the test set
    sample_idx = 0
    sample_data = torch.tensor(fusion_dataset['X_test'][sample_idx:sample_idx+1], dtype=torch.float32).to(device)
    sample_label = torch.tensor(fusion_dataset['y_test'][sample_idx:sample_idx+1], dtype=torch.float32).to(device)
    
    # Generate report
    report = report_generator.generate_report(sample_data, sample_label, sample_id=sample_idx)
    
    # Save report
    report_generator.save_report(report)
    
    logger.info("Clinical report generated")
    
    return report


def main():
    """Main function."""
    # Create output directories
    os.makedirs('data', exist_ok=True)
    os.makedirs('data/eeg/processed', exist_ok=True)
    os.makedirs('data/audio/processed', exist_ok=True)
    os.makedirs('data/text/processed', exist_ok=True)
    os.makedirs('data/fusion/processed', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    os.makedirs('results/clinical_reports', exist_ok=True)
    
    # Load datasets
    eeg_dataset, audio_dataset, text_dataset, fusion_dataset = load_datasets()
    
    # Train a simple model
    model, device = train_simple_model(fusion_dataset)
    
    # Evaluate model
    metrics = evaluate_model(model, fusion_dataset, device)
    
    # Generate clinical report
    report = generate_clinical_report(model, fusion_dataset, device)
    
    # Print report
    print("\nClinical Report:")
    print(f"Sample ID: {report['sample_id']}")
    print(f"Depression Probability: {report['depression_probability']:.1%}")
    print(f"Risk Level: {report['risk_level']}")
    
    if 'observations' in report:
        print("\nObservations:")
        for observation in report['observations']:
            print(f"- {observation}")
    
    if 'suggestions' in report:
        print("\nRecommendations:")
        for suggestion in report['suggestions']:
            print(f"- {suggestion}")
    
    logger.info("Demo completed successfully")


if __name__ == "__main__":
    main()
