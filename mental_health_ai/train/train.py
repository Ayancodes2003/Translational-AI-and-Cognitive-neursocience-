"""
Training Module

This module handles the training and evaluation of models.
"""

import os
import time
import argparse
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pickle
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.eeg_models import EEGCNN, EEGLSTM, EEGBiLSTMAttention, EEG1DCNN, EEG1DCNNGRU, EEGTransformer
from models.audio_models import AudioCNN, AudioLSTM, AudioBiLSTMAttention, Audio2DCNN, Audio1DCNNGRU, AudioTransformer
from models.text_models import TextCNN, TextLSTM, TextBiLSTMAttention, TextCNN1D, BERTClassifier, TextTransformer
from models.fusion_models import EarlyFusionModel, LateFusionModel, IntermediateFusionModel, CrossModalAttentionFusion, HierarchicalFusionModel, EnsembleModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Trainer:
    """Class for training and evaluating models."""
    
    def __init__(self, model, train_loader, val_loader, test_loader=None, 
                 criterion=None, optimizer=None, scheduler=None, device=None,
                 config=None):
        """
        Initialize the trainer.
        
        Args:
            model (nn.Module): Model to train
            train_loader (DataLoader): Training data loader
            val_loader (DataLoader): Validation data loader
            test_loader (DataLoader, optional): Test data loader
            criterion (nn.Module, optional): Loss function
            optimizer (optim.Optimizer, optional): Optimizer
            scheduler (optim.lr_scheduler._LRScheduler, optional): Learning rate scheduler
            device (torch.device, optional): Device to use for training
            config (dict, optional): Configuration dictionary
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        
        # Set default criterion if not provided
        self.criterion = criterion or nn.BCEWithLogitsLoss()
        
        # Set default optimizer if not provided
        self.optimizer = optimizer or optim.Adam(model.parameters(), lr=0.001)
        
        # Set default scheduler if not provided
        self.scheduler = scheduler
        
        # Set default device if not provided
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Set default config if not provided
        self.config = config or {
            'num_epochs': 50,
            'patience': 10,
            'model_save_path': 'models/saved',
            'log_interval': 10,
            'early_stopping': True
        }
        
        # Move model to device
        self.model.to(self.device)
        
        # Create model save directory if it doesn't exist
        os.makedirs(self.config['model_save_path'], exist_ok=True)
        
        # Initialize training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'train_f1': [],
            'val_f1': []
        }
    
    def train_epoch(self):
        """
        Train the model for one epoch.
        
        Returns:
            tuple: (train_loss, train_acc, train_f1)
        """
        self.model.train()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        for batch_idx, (data, target) in enumerate(tqdm(self.train_loader, desc='Training')):
            # Move data to device
            if isinstance(data, dict):
                # For multimodal data
                data = {k: v.to(self.device) for k, v in data.items()}
            else:
                # For single modality data
                data = data.to(self.device)
            
            target = target.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            output = self.model(data)
            
            # Calculate loss
            if target.shape[1] > 1:
                # Multi-task learning (e.g., binary classification + regression)
                loss = self.criterion(output, target[:, 0].unsqueeze(1))
            else:
                # Single task
                loss = self.criterion(output, target)
            
            # Backward pass
            loss.backward()
            
            # Update weights
            self.optimizer.step()
            
            # Accumulate loss
            total_loss += loss.item()
            
            # Store predictions and labels for metrics calculation
            all_preds.append(torch.sigmoid(output).cpu().detach().numpy())
            all_labels.append(target.cpu().numpy())
            
            # Log progress
            if batch_idx % self.config['log_interval'] == 0:
                logger.info(f'Train Batch: {batch_idx}/{len(self.train_loader)} Loss: {loss.item():.6f}')
        
        # Calculate average loss
        avg_loss = total_loss / len(self.train_loader)
        
        # Calculate metrics
        all_preds = np.vstack(all_preds)
        all_labels = np.vstack(all_labels)
        
        # Convert predictions to binary
        binary_preds = (all_preds > 0.5).astype(int)
        
        # Calculate accuracy and F1 score
        accuracy = accuracy_score(all_labels[:, 0], binary_preds[:, 0])
        f1 = f1_score(all_labels[:, 0], binary_preds[:, 0])
        
        return avg_loss, accuracy, f1
    
    def validate(self, loader=None):
        """
        Validate the model.
        
        Args:
            loader (DataLoader, optional): Data loader to use for validation
                                          (defaults to self.val_loader)
        
        Returns:
            tuple: (val_loss, val_acc, val_f1)
        """
        loader = loader or self.val_loader
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for data, target in tqdm(loader, desc='Validation'):
                # Move data to device
                if isinstance(data, dict):
                    # For multimodal data
                    data = {k: v.to(self.device) for k, v in data.items()}
                else:
                    # For single modality data
                    data = data.to(self.device)
                
                target = target.to(self.device)
                
                # Forward pass
                output = self.model(data)
                
                # Calculate loss
                if target.shape[1] > 1:
                    # Multi-task learning (e.g., binary classification + regression)
                    loss = self.criterion(output, target[:, 0].unsqueeze(1))
                else:
                    # Single task
                    loss = self.criterion(output, target)
                
                # Accumulate loss
                total_loss += loss.item()
                
                # Store predictions and labels for metrics calculation
                all_preds.append(torch.sigmoid(output).cpu().numpy())
                all_labels.append(target.cpu().numpy())
        
        # Calculate average loss
        avg_loss = total_loss / len(loader)
        
        # Calculate metrics
        all_preds = np.vstack(all_preds)
        all_labels = np.vstack(all_labels)
        
        # Convert predictions to binary
        binary_preds = (all_preds > 0.5).astype(int)
        
        # Calculate accuracy and F1 score
        accuracy = accuracy_score(all_labels[:, 0], binary_preds[:, 0])
        f1 = f1_score(all_labels[:, 0], binary_preds[:, 0])
        
        return avg_loss, accuracy, f1
    
    def train(self):
        """
        Train the model for multiple epochs.
        
        Returns:
            dict: Training history
        """
        logger.info(f"Training on device: {self.device}")
        logger.info(f"Model: {type(self.model).__name__}")
        logger.info(f"Optimizer: {type(self.optimizer).__name__}")
        logger.info(f"Criterion: {type(self.criterion).__name__}")
        logger.info(f"Number of epochs: {self.config['num_epochs']}")
        
        best_val_loss = float('inf')
        best_epoch = 0
        patience_counter = 0
        
        for epoch in range(self.config['num_epochs']):
            logger.info(f"Epoch {epoch+1}/{self.config['num_epochs']}")
            
            # Train for one epoch
            train_loss, train_acc, train_f1 = self.train_epoch()
            
            # Validate
            val_loss, val_acc, val_f1 = self.validate()
            
            # Update learning rate
            if self.scheduler is not None:
                self.scheduler.step(val_loss)
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            self.history['train_f1'].append(train_f1)
            self.history['val_f1'].append(val_f1)
            
            # Log metrics
            logger.info(f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
            logger.info(f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")
            logger.info(f"Train F1: {train_f1:.4f} | Val F1: {val_f1:.4f}")
            
            # Check if this is the best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                patience_counter = 0
                
                # Save best model
                self.save_model(os.path.join(self.config['model_save_path'], 'best_model.pt'))
                logger.info(f"New best model saved at epoch {epoch+1}")
            else:
                patience_counter += 1
                logger.info(f"No improvement for {patience_counter} epochs")
            
            # Early stopping
            if self.config['early_stopping'] and patience_counter >= self.config['patience']:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
        
        logger.info(f"Training completed. Best model at epoch {best_epoch+1}")
        
        # Load best model
        self.load_model(os.path.join(self.config['model_save_path'], 'best_model.pt'))
        
        # Evaluate on test set if available
        if self.test_loader is not None:
            test_loss, test_acc, test_f1 = self.validate(self.test_loader)
            logger.info(f"Test Loss: {test_loss:.6f} | Test Acc: {test_acc:.4f} | Test F1: {test_f1:.4f}")
            
            # Add test metrics to history
            self.history['test_loss'] = test_loss
            self.history['test_acc'] = test_acc
            self.history['test_f1'] = test_f1
        
        # Save training history
        self.save_history(os.path.join(self.config['model_save_path'], 'history.pkl'))
        
        # Plot training history
        self.plot_history(os.path.join(self.config['model_save_path'], 'history.png'))
        
        return self.history
    
    def save_model(self, path):
        """
        Save the model.
        
        Args:
            path (str): Path to save the model
        """
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'config': self.config
        }, path)
    
    def load_model(self, path):
        """
        Load the model.
        
        Args:
            path (str): Path to load the model from
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.config = checkpoint['config']
    
    def save_history(self, path):
        """
        Save the training history.
        
        Args:
            path (str): Path to save the history
        """
        with open(path, 'wb') as f:
            pickle.dump(self.history, f)
    
    def load_history(self, path):
        """
        Load the training history.
        
        Args:
            path (str): Path to load the history from
        """
        with open(path, 'rb') as f:
            self.history = pickle.load(f)
    
    def plot_history(self, path=None):
        """
        Plot the training history.
        
        Args:
            path (str, optional): Path to save the plot
        """
        plt.figure(figsize=(12, 8))
        
        # Plot loss
        plt.subplot(2, 2, 1)
        plt.plot(self.history['train_loss'], label='Train Loss')
        plt.plot(self.history['val_loss'], label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss')
        plt.legend()
        
        # Plot accuracy
        plt.subplot(2, 2, 2)
        plt.plot(self.history['train_acc'], label='Train Acc')
        plt.plot(self.history['val_acc'], label='Val Acc')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Accuracy')
        plt.legend()
        
        # Plot F1 score
        plt.subplot(2, 2, 3)
        plt.plot(self.history['train_f1'], label='Train F1')
        plt.plot(self.history['val_f1'], label='Val F1')
        plt.xlabel('Epoch')
        plt.ylabel('F1 Score')
        plt.title('F1 Score')
        plt.legend()
        
        # Plot learning rate if available
        if self.scheduler:
            plt.subplot(2, 2, 4)
            plt.plot([group['lr'] for group in self.optimizer.param_groups])
            plt.xlabel('Epoch')
            plt.ylabel('Learning Rate')
            plt.title('Learning Rate')
        
        plt.tight_layout()
        
        if path:
            plt.savefig(path)
        else:
            plt.show()
    
    def evaluate(self, loader=None, detailed=True):
        """
        Evaluate the model.
        
        Args:
            loader (DataLoader, optional): Data loader to use for evaluation
                                          (defaults to self.test_loader)
            detailed (bool): Whether to compute detailed metrics
        
        Returns:
            dict: Evaluation metrics
        """
        loader = loader or self.test_loader
        if loader is None:
            raise ValueError("No test loader provided")
        
        self.model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for data, target in tqdm(loader, desc='Evaluation'):
                # Move data to device
                if isinstance(data, dict):
                    # For multimodal data
                    data = {k: v.to(self.device) for k, v in data.items()}
                else:
                    # For single modality data
                    data = data.to(self.device)
                
                target = target.to(self.device)
                
                # Forward pass
                output = self.model(data)
                
                # Store predictions and labels
                all_preds.append(torch.sigmoid(output).cpu().numpy())
                all_labels.append(target.cpu().numpy())
        
        # Concatenate predictions and labels
        all_preds = np.vstack(all_preds)
        all_labels = np.vstack(all_labels)
        
        # Convert predictions to binary
        binary_preds = (all_preds > 0.5).astype(int)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(all_labels[:, 0], binary_preds[:, 0]),
            'precision': precision_score(all_labels[:, 0], binary_preds[:, 0]),
            'recall': recall_score(all_labels[:, 0], binary_preds[:, 0]),
            'f1': f1_score(all_labels[:, 0], binary_preds[:, 0]),
            'auc': roc_auc_score(all_labels[:, 0], all_preds[:, 0])
        }
        
        # Compute confusion matrix
        cm = confusion_matrix(all_labels[:, 0], binary_preds[:, 0])
        metrics['confusion_matrix'] = cm
        
        # Log metrics
        logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"Precision: {metrics['precision']:.4f}")
        logger.info(f"Recall: {metrics['recall']:.4f}")
        logger.info(f"F1 Score: {metrics['f1']:.4f}")
        logger.info(f"AUC: {metrics['auc']:.4f}")
        logger.info(f"Confusion Matrix:\n{cm}")
        
        # Plot confusion matrix
        if detailed:
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.title('Confusion Matrix')
            plt.savefig(os.path.join(self.config['model_save_path'], 'confusion_matrix.png'))
        
        return metrics


def create_model(modality, model_type, input_dim, num_classes=1):
    """
    Create a model for the specified modality and type.
    
    Args:
        modality (str): Modality ('eeg', 'audio', 'text', or 'fusion')
        model_type (str): Model type (e.g., 'cnn', 'lstm', 'transformer')
        input_dim (int or dict): Input dimension(s)
        num_classes (int): Number of output classes
    
    Returns:
        nn.Module: Model
    """
    if modality == 'eeg':
        if model_type == 'cnn':
            return EEGCNN(input_dim=input_dim, num_classes=num_classes)
        elif model_type == 'lstm':
            return EEGLSTM(input_dim=input_dim, num_classes=num_classes)
        elif model_type == 'bilstm_attention':
            return EEGBiLSTMAttention(input_dim=input_dim, num_classes=num_classes)
        elif model_type == '1dcnn':
            return EEG1DCNN(input_dim=input_dim, num_classes=num_classes)
        elif model_type == '1dcnn_gru':
            return EEG1DCNNGRU(input_dim=input_dim, num_classes=num_classes)
        elif model_type == 'transformer':
            return EEGTransformer(input_dim=input_dim, num_classes=num_classes)
        else:
            raise ValueError(f"Unknown model type for EEG: {model_type}")
    
    elif modality == 'audio':
        if model_type == 'cnn':
            return AudioCNN(input_dim=input_dim, num_classes=num_classes)
        elif model_type == 'lstm':
            return AudioLSTM(input_dim=input_dim, num_classes=num_classes)
        elif model_type == 'bilstm_attention':
            return AudioBiLSTMAttention(input_dim=input_dim, num_classes=num_classes)
        elif model_type == '2dcnn':
            return Audio2DCNN(num_classes=num_classes)
        elif model_type == '1dcnn_gru':
            return Audio1DCNNGRU(input_dim=input_dim, num_classes=num_classes)
        elif model_type == 'transformer':
            return AudioTransformer(input_dim=input_dim, num_classes=num_classes)
        else:
            raise ValueError(f"Unknown model type for audio: {model_type}")
    
    elif modality == 'text':
        if model_type == 'cnn':
            return TextCNN(input_dim=input_dim, num_classes=num_classes)
        elif model_type == 'lstm':
            return TextLSTM(vocab_size=input_dim, num_classes=num_classes)
        elif model_type == 'bilstm_attention':
            return TextBiLSTMAttention(vocab_size=input_dim, num_classes=num_classes)
        elif model_type == '1dcnn':
            return TextCNN1D(vocab_size=input_dim, num_classes=num_classes)
        elif model_type == 'bert':
            return BERTClassifier(num_classes=num_classes)
        elif model_type == 'transformer':
            return TextTransformer(vocab_size=input_dim, num_classes=num_classes)
        else:
            raise ValueError(f"Unknown model type for text: {model_type}")
    
    elif modality == 'fusion':
        if model_type == 'early':
            return EarlyFusionModel(input_dims=input_dim, num_classes=num_classes)
        elif model_type == 'late':
            return LateFusionModel(input_dims=input_dim, num_classes=num_classes)
        elif model_type == 'intermediate':
            return IntermediateFusionModel(input_dims=input_dim, num_classes=num_classes)
        elif model_type == 'cross_attention':
            return CrossModalAttentionFusion(input_dims=input_dim, num_classes=num_classes)
        elif model_type == 'hierarchical':
            return HierarchicalFusionModel(input_dims=input_dim, num_classes=num_classes)
        elif model_type == 'ensemble':
            return EnsembleModel(input_dims=input_dim, num_classes=num_classes)
        else:
            raise ValueError(f"Unknown model type for fusion: {model_type}")
    
    else:
        raise ValueError(f"Unknown modality: {modality}")


def create_dataloaders(data_path, modality, batch_size=32):
    """
    Create data loaders for the specified modality.
    
    Args:
        data_path (str): Path to the data
        modality (str): Modality ('eeg', 'audio', 'text', or 'fusion')
        batch_size (int): Batch size
    
    Returns:
        tuple: (train_loader, val_loader, test_loader, input_dim)
    """
    if modality in ['eeg', 'audio', 'text']:
        # Load dataset
        with open(os.path.join(data_path, f'{modality}_dataset.pkl'), 'rb') as f:
            dataset = pickle.load(f)
        
        # Create PyTorch datasets
        train_dataset = TensorDataset(
            torch.tensor(dataset['X_train'], dtype=torch.float32),
            torch.tensor(dataset['y_train'], dtype=torch.float32)
        )
        val_dataset = TensorDataset(
            torch.tensor(dataset['X_val'], dtype=torch.float32),
            torch.tensor(dataset['y_val'], dtype=torch.float32)
        )
        test_dataset = TensorDataset(
            torch.tensor(dataset['X_test'], dtype=torch.float32),
            torch.tensor(dataset['y_test'], dtype=torch.float32)
        )
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        
        # Get input dimension
        input_dim = dataset['X_train'].shape[1]
    
    elif modality == 'fusion':
        # Load fusion dataset
        fusion_type = 'early'  # Default fusion type
        
        # Check if fusion type is specified in the data path
        if 'late' in data_path:
            fusion_type = 'late'
        elif 'intermediate' in data_path:
            fusion_type = 'intermediate'
        
        with open(os.path.join(data_path, f'fusion_{fusion_type}_dataset.pkl'), 'rb') as f:
            dataset = pickle.load(f)
        
        # Get input dimensions
        input_dim = dataset['feature_dims']
        
        if fusion_type == 'early':
            # Create PyTorch datasets
            train_dataset = TensorDataset(
                torch.tensor(dataset['X_train'], dtype=torch.float32),
                torch.tensor(dataset['y_train'], dtype=torch.float32)
            )
            val_dataset = TensorDataset(
                torch.tensor(dataset['X_val'], dtype=torch.float32),
                torch.tensor(dataset['y_val'], dtype=torch.float32)
            )
            test_dataset = TensorDataset(
                torch.tensor(dataset['X_test'], dtype=torch.float32),
                torch.tensor(dataset['y_test'], dtype=torch.float32)
            )
            
            # Create data loaders
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size)
            test_loader = DataLoader(test_dataset, batch_size=batch_size)
        
        else:  # 'late' or 'intermediate' fusion
            # Create custom datasets for multimodal data
            from data.fusion_dataset import MultimodalDataset
            
            train_dataset = MultimodalDataset(
                dataset['eeg_train'], dataset['audio_train'], dataset['text_train'], dataset['y_train']
            )
            val_dataset = MultimodalDataset(
                dataset['eeg_val'], dataset['audio_val'], dataset['text_val'], dataset['y_val']
            )
            test_dataset = MultimodalDataset(
                dataset['eeg_test'], dataset['audio_test'], dataset['text_test'], dataset['y_test']
            )
            
            # Create data loaders
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size)
            test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    else:
        raise ValueError(f"Unknown modality: {modality}")
    
    return train_loader, val_loader, test_loader, input_dim


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Train a model')
    
    # Data arguments
    parser.add_argument('--data_path', type=str, default='data/processed',
                        help='Path to the processed data')
    
    # Model arguments
    parser.add_argument('--modality', type=str, required=True,
                        choices=['eeg', 'audio', 'text', 'fusion'],
                        help='Modality to use')
    parser.add_argument('--model', type=str, required=True,
                        help='Model type to use')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=50,
                        help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--patience', type=int, default=10,
                        help='Patience for early stopping')
    parser.add_argument('--model_save_path', type=str, default='models/saved',
                        help='Path to save the model')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Create data loaders
    train_loader, val_loader, test_loader, input_dim = create_dataloaders(
        args.data_path, args.modality, args.batch_size
    )
    
    # Create model
    model = create_model(args.modality, args.model, input_dim)
    
    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Create scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=5, verbose=True
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        config={
            'num_epochs': args.num_epochs,
            'patience': args.patience,
            'model_save_path': os.path.join(args.model_save_path, f"{args.modality}_{args.model}"),
            'log_interval': 10,
            'early_stopping': True
        }
    )
    
    # Train model
    trainer.train()
    
    # Evaluate model
    metrics = trainer.evaluate()
    
    # Save metrics
    metrics_df = pd.DataFrame({
        'Metric': list(metrics.keys())[:-1],  # Exclude confusion matrix
        'Value': list(metrics.values())[:-1]
    })
    metrics_df.to_csv(os.path.join(trainer.config['model_save_path'], 'metrics.csv'), index=False)
    
    logger.info(f"Metrics saved to {os.path.join(trainer.config['model_save_path'], 'metrics.csv')}")


if __name__ == '__main__':
    main()
