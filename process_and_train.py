"""
Process and Train Script

This script processes the real datasets and trains models for the Mental Health AI project.
"""

import os
import sys
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
import logging
import argparse
import pickle

# Add mental_health_ai directory to path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mental_health_ai'))

# Import necessary modules
from mental_health_ai.data.dataset_loader import DatasetLoader
from mental_health_ai.data.eeg.preprocess_eeg_new import EEGProcessor
from mental_health_ai.data.audio.preprocess_audio import AudioProcessor
from mental_health_ai.data.text.preprocess_text import TextProcessor
from mental_health_ai.models.eeg_models import EEGNet, DeepConvNet, ShallowConvNet, EEGCNN, EEGLSTM, EEGTransformer
from mental_health_ai.models.audio_models import AudioCNN, AudioLSTM, AudioCRNN, AudioResNet
from mental_health_ai.models.text_models import TextCNN, TextLSTM, TextBiLSTM, TextTransformer
from mental_health_ai.models.fusion_models import EarlyFusionModel, LateFusionModel, HierarchicalFusionModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Create output directories
os.makedirs('mental_health_ai/data/eeg/processed', exist_ok=True)
os.makedirs('mental_health_ai/data/audio/processed', exist_ok=True)
os.makedirs('mental_health_ai/data/text/processed', exist_ok=True)
os.makedirs('mental_health_ai/data/fusion/processed', exist_ok=True)
os.makedirs('results', exist_ok=True)
os.makedirs('results/eeg', exist_ok=True)
os.makedirs('results/audio', exist_ok=True)
os.makedirs('results/text', exist_ok=True)
os.makedirs('results/fusion', exist_ok=True)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")

# Process datasets
def process_datasets():
    """Process all datasets."""
    logger.info("Processing datasets...")
    
    # Process EEG data
    logger.info("Processing EEG data...")
    eeg_processor = EEGProcessor(
        data_path='mental_health_ai/data/eeg/raw',
        output_path='mental_health_ai/data/eeg/processed'
    )
    eeg_processor.process_all_datasets()
    eeg_processor.create_dataset_splits('combined')
    
    # Process audio data
    logger.info("Processing audio data...")
    dataset_loader = DatasetLoader(output_dir='mental_health_ai/data')
    audio_data, audio_labels = dataset_loader.load_audio_data()
    
    # Save audio data
    np.save('mental_health_ai/data/audio/processed/audio_data.npy', audio_data)
    np.save('mental_health_ai/data/audio/processed/audio_labels.npy', audio_labels)
    
    # Create audio dataset splits
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        audio_data, audio_labels, test_size=0.2, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.2, random_state=42
    )
    
    audio_dataset = {
        'X_train': X_train,
        'y_train': y_train,
        'X_val': X_val,
        'y_val': y_val,
        'X_test': X_test,
        'y_test': y_test
    }
    
    with open('mental_health_ai/data/audio/processed/audio_dataset.pkl', 'wb') as f:
        pickle.dump(audio_dataset, f)
    
    # Process text data
    logger.info("Processing text data...")
    text_data, text_labels = dataset_loader.load_text_data()
    
    # Save text data
    with open('mental_health_ai/data/text/processed/text_data.pkl', 'wb') as f:
        pickle.dump(text_data, f)
    np.save('mental_health_ai/data/text/processed/text_labels.npy', text_labels)
    
    # Extract text features
    text_processor = TextProcessor(None, 'mental_health_ai/data/text/processed')
    text_features = []
    
    for text in text_data:
        features = text_processor.extract_linguistic_features([text])
        feature_list = [
            features.get('token_count', 0),
            features.get('unique_token_count', 0),
            features.get('lexical_diversity', 0),
            features.get('sentence_count', 0),
            features.get('avg_sentence_length', 0),
            features.get('depression_keyword_count', 0),
            features.get('pronoun_count', 0),
            features.get('negative_word_count', 0),
            features.get('first_person_pronoun_count', 0)
        ]
        text_features.append(feature_list)
    
    text_features = np.array(text_features)
    np.save('mental_health_ai/data/text/processed/text_features.npy', text_features)
    
    # Create text dataset splits
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        text_features, text_labels, test_size=0.2, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.2, random_state=42
    )
    
    text_dataset = {
        'X_train': X_train,
        'y_train': y_train,
        'X_val': X_val,
        'y_val': y_val,
        'X_test': X_test,
        'y_test': y_test
    }
    
    with open('mental_health_ai/data/text/processed/text_dataset.pkl', 'wb') as f:
        pickle.dump(text_dataset, f)
    
    logger.info("Finished processing datasets")

# Load dataset
def load_dataset(modality):
    """Load dataset for the specified modality."""
    logger.info(f"Loading {modality} dataset...")
    
    if modality == 'eeg':
        # Load EEG dataset
        with open('mental_health_ai/data/eeg/processed/combined_dataset.pkl', 'rb') as f:
            dataset = pickle.load(f)
    elif modality == 'audio':
        # Load audio dataset
        with open('mental_health_ai/data/audio/processed/audio_dataset.pkl', 'rb') as f:
            dataset = pickle.load(f)
    elif modality == 'text':
        # Load text dataset
        with open('mental_health_ai/data/text/processed/text_dataset.pkl', 'rb') as f:
            dataset = pickle.load(f)
    else:  # fusion
        # Load all datasets
        with open('mental_health_ai/data/eeg/processed/combined_dataset.pkl', 'rb') as f:
            eeg_dataset = pickle.load(f)
        
        with open('mental_health_ai/data/audio/processed/audio_dataset.pkl', 'rb') as f:
            audio_dataset = pickle.load(f)
        
        with open('mental_health_ai/data/text/processed/text_dataset.pkl', 'rb') as f:
            text_dataset = pickle.load(f)
        
        # Ensure all datasets have the same number of samples
        min_train_samples = min(
            len(eeg_dataset['X_train']),
            len(audio_dataset['X_train']),
            len(text_dataset['X_train'])
        )
        
        min_val_samples = min(
            len(eeg_dataset['X_val']),
            len(audio_dataset['X_val']),
            len(text_dataset['X_val'])
        )
        
        min_test_samples = min(
            len(eeg_dataset['X_test']),
            len(audio_dataset['X_test']),
            len(text_dataset['X_test'])
        )
        
        # Combine datasets
        X_train = np.hstack((
            eeg_dataset['X_train'][:min_train_samples],
            audio_dataset['X_train'][:min_train_samples],
            text_dataset['X_train'][:min_train_samples]
        ))
        
        y_train = eeg_dataset['y_train'][:min_train_samples]
        
        X_val = np.hstack((
            eeg_dataset['X_val'][:min_val_samples],
            audio_dataset['X_val'][:min_val_samples],
            text_dataset['X_val'][:min_val_samples]
        ))
        
        y_val = eeg_dataset['y_val'][:min_val_samples]
        
        X_test = np.hstack((
            eeg_dataset['X_test'][:min_test_samples],
            audio_dataset['X_test'][:min_test_samples],
            text_dataset['X_test'][:min_test_samples]
        ))
        
        y_test = eeg_dataset['y_test'][:min_test_samples]
        
        dataset = {
            'X_train': X_train,
            'y_train': y_train,
            'X_val': X_val,
            'y_val': y_val,
            'X_test': X_test,
            'y_test': y_test
        }
    
    # Create PyTorch dataloaders
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
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    test_loader = DataLoader(test_dataset, batch_size=32)
    
    # Get input dimension
    input_dim = dataset['X_train'].shape[1]
    
    logger.info(f"Loaded {modality} dataset with input dimension {input_dim}")
    
    return {
        'train_loader': train_loader,
        'val_loader': val_loader,
        'test_loader': test_loader,
        'input_dim': input_dim
    }

# Train model
def train_model(model, train_loader, val_loader, device, model_name, num_epochs=10):
    """Train model."""
    logger.info(f"Training {model_name} model...")
    
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
        logger.info(f"Epoch {epoch+1}/{num_epochs}: "
              f"Train Loss = {train_loss:.4f}, Train Acc = {train_accuracy:.4f}, "
              f"Val Loss = {val_loss:.4f}, Val Acc = {val_accuracy:.4f}")
    
    # Calculate training time
    training_time = time.time() - start_time
    logger.info(f"Training time: {training_time:.2f} seconds")
    
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
    plt.savefig(f'results/{model_name}_training_curves.png')
    plt.close()
    
    return model, training_time

# Evaluate model
def evaluate_model(model, test_loader, device, model_name):
    """Evaluate model."""
    logger.info(f"Evaluating {model_name} model...")
    
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
    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall: {recall:.4f}")
    logger.info(f"F1 Score: {f1:.4f}")
    logger.info(f"AUC: {auc:.4f}")
    logger.info(f"Confusion Matrix:\n{conf_matrix}")
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Non-depressed', 'Depressed'],
                yticklabels=['Non-depressed', 'Depressed'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'{model_name} - Confusion Matrix')
    plt.tight_layout()
    plt.savefig(f'results/{model_name}_confusion_matrix.png')
    plt.close()
    
    return metrics

# Compare models
def compare_models(metrics_list, modality):
    """Compare models."""
    logger.info(f"Comparing {modality} models...")
    
    # Create DataFrame from metrics
    df = pd.DataFrame(metrics_list)
    
    # Plot comparison
    plt.figure(figsize=(12, 8))
    
    # Plot accuracy, precision, recall, f1, auc
    metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1', 'auc']
    
    # Bar plot
    ax = df[['model'] + metrics_to_plot].set_index('model').plot(kind='bar', figsize=(12, 6))
    plt.title(f'{modality.upper()} Model Comparison')
    plt.ylabel('Score')
    plt.xlabel('Model')
    plt.ylim(0, 1)
    plt.grid(axis='y')
    plt.legend(loc='lower right')
    
    # Add values on top of bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f')
    
    plt.tight_layout()
    plt.savefig(f'results/{modality}_model_comparison.png')
    plt.close()
    
    # Save metrics to JSON
    with open(f'results/{modality}_metrics.json', 'w') as f:
        json.dump(metrics_list, f, indent=4)
    
    # Create comparison table
    comparison_table = df[['model'] + metrics_to_plot].set_index('model')
    logger.info(f"\n{modality.upper()} Model Comparison:")
    logger.info(f"\n{comparison_table}")
    
    # Save comparison table to CSV
    comparison_table.to_csv(f'results/{modality}_comparison_table.csv')
    
    return comparison_table

# Train models for a specific modality
def train_modality_models(modality, num_epochs=10):
    """Train models for a specific modality."""
    logger.info(f"Training {modality} models...")
    
    # Load dataset
    data = load_dataset(modality)
    
    # Define models to train
    if modality == 'eeg':
        models = {
            'EEGNet': EEGNet(input_dim=data['input_dim']),
            'DeepConvNet': DeepConvNet(input_dim=data['input_dim']),
            'ShallowConvNet': ShallowConvNet(input_dim=data['input_dim']),
            'EEGCNN': EEGCNN(input_dim=data['input_dim']),
            'EEGLSTM': EEGLSTM(input_dim=data['input_dim'])
        }
    elif modality == 'audio':
        models = {
            'AudioCNN': AudioCNN(input_dim=data['input_dim']),
            'AudioLSTM': AudioLSTM(input_dim=data['input_dim']),
            'AudioCRNN': AudioCRNN(input_dim=data['input_dim'])
        }
    elif modality == 'text':
        models = {
            'TextCNN': TextCNN(input_dim=data['input_dim']),
            'TextLSTM': TextLSTM(input_dim=data['input_dim']),
            'TextBiLSTM': TextBiLSTM(input_dim=data['input_dim'])
        }
    else:  # fusion
        # Get dimensions for each modality
        eeg_data = load_dataset('eeg')
        audio_data = load_dataset('audio')
        text_data = load_dataset('text')
        
        eeg_dim = eeg_data['input_dim']
        audio_dim = audio_data['input_dim']
        text_dim = text_data['input_dim']
        
        models = {
            'EarlyFusion': EarlyFusionModel(eeg_dim=eeg_dim, audio_dim=audio_dim, text_dim=text_dim),
            'LateFusion': LateFusionModel(eeg_dim=eeg_dim, audio_dim=audio_dim, text_dim=text_dim),
            'HierarchicalFusion': HierarchicalFusionModel(eeg_dim=eeg_dim, audio_dim=audio_dim, text_dim=text_dim)
        }
    
    # Train and evaluate models
    metrics_list = []
    training_times = {}
    
    for model_name, model in models.items():
        # Train model
        trained_model, training_time = train_model(
            model, data['train_loader'], data['val_loader'], device, model_name, num_epochs=num_epochs
        )
        
        # Store training time
        training_times[model_name] = training_time
        
        # Evaluate model
        metrics = evaluate_model(trained_model, data['test_loader'], device, model_name)
        
        # Add training time to metrics
        metrics['training_time'] = training_time
        
        # Add metrics to list
        metrics_list.append(metrics)
        
        # Save model
        os.makedirs(f'results/{modality}', exist_ok=True)
        torch.save(trained_model, f'results/{modality}/{model_name}.pt')
    
    # Compare models
    comparison_table = compare_models(metrics_list, modality)
    
    # Print training times
    logger.info(f"\n{modality.upper()} Training Times:")
    for model_name, time in training_times.items():
        logger.info(f"{model_name}: {time:.2f} seconds")
    
    return metrics_list

# Main function
def main(args):
    """Main function."""
    logger.info("Starting processing and training...")
    
    # Process datasets if needed
    if args.process_data:
        process_datasets()
    
    # Train models for each modality
    all_metrics = {}
    
    if args.modality == 'all' or args.modality == 'eeg':
        logger.info("Training EEG models...")
        all_metrics['eeg'] = train_modality_models('eeg', num_epochs=args.num_epochs)
    
    if args.modality == 'all' or args.modality == 'audio':
        logger.info("Training audio models...")
        all_metrics['audio'] = train_modality_models('audio', num_epochs=args.num_epochs)
    
    if args.modality == 'all' or args.modality == 'text':
        logger.info("Training text models...")
        all_metrics['text'] = train_modality_models('text', num_epochs=args.num_epochs)
    
    if args.modality == 'all' or args.modality == 'fusion':
        logger.info("Training fusion models...")
        all_metrics['fusion'] = train_modality_models('fusion', num_epochs=args.num_epochs)
    
    # Compare across modalities
    if args.modality == 'all':
        logger.info("Comparing across modalities...")
        
        # Get best model from each modality
        best_models = []
        
        for modality, metrics_list in all_metrics.items():
            # Find best model based on F1 score
            best_model = max(metrics_list, key=lambda x: x['f1'])
            best_model['modality'] = modality
            best_models.append(best_model)
        
        # Create DataFrame from best models
        df = pd.DataFrame(best_models)
        
        # Plot comparison
        plt.figure(figsize=(12, 8))
        
        # Plot accuracy, precision, recall, f1, auc
        metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1', 'auc']
        
        # Bar plot
        ax = df[['modality'] + metrics_to_plot].set_index('modality').plot(kind='bar', figsize=(12, 6))
        plt.title('Cross-Modality Comparison (Best Models)')
        plt.ylabel('Score')
        plt.xlabel('Modality')
        plt.ylim(0, 1)
        plt.grid(axis='y')
        plt.legend(loc='lower right')
        
        # Add values on top of bars
        for container in ax.containers:
            ax.bar_label(container, fmt='%.2f')
        
        plt.tight_layout()
        plt.savefig('results/cross_modality_comparison.png')
        plt.close()
        
        # Save metrics to JSON
        with open('results/cross_modality_metrics.json', 'w') as f:
            json.dump(best_models, f, indent=4)
        
        # Create comparison table
        comparison_table = df[['modality'] + metrics_to_plot].set_index('modality')
        logger.info("\nCross-Modality Comparison (Best Models):")
        logger.info(f"\n{comparison_table}")
        
        # Save comparison table to CSV
        comparison_table.to_csv('results/cross_modality_comparison_table.csv')
    
    logger.info("Processing and training completed successfully!")
    logger.info("Results saved to 'results' directory.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process and train models for Mental Health AI')
    
    parser.add_argument('--modality', type=str, default='all', choices=['all', 'eeg', 'audio', 'text', 'fusion'],
                        help='Modality to train')
    parser.add_argument('--num_epochs', type=int, default=20,
                        help='Number of epochs')
    parser.add_argument('--process_data', action='store_true',
                        help='Process datasets before training')
    
    args = parser.parse_args()
    
    main(args)
