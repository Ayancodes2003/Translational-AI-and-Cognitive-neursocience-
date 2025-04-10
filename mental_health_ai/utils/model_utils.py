"""
Model Utility Functions

This module contains utility functions for model creation, training, and evaluation.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import logging
import time
import pickle
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_optimizer(model, optimizer_type='adam', lr=0.001, weight_decay=0.0001):
    """
    Create an optimizer for the model.
    
    Args:
        model (nn.Module): Model to optimize
        optimizer_type (str): Type of optimizer ('adam', 'sgd', or 'rmsprop')
        lr (float): Learning rate
        weight_decay (float): Weight decay
    
    Returns:
        optim.Optimizer: Optimizer
    """
    logger.info(f"Creating {optimizer_type} optimizer with lr={lr}, weight_decay={weight_decay}")
    
    if optimizer_type.lower() == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_type.lower() == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    elif optimizer_type.lower() == 'rmsprop':
        optimizer = optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")
    
    return optimizer


def create_scheduler(optimizer, scheduler_type='plateau', patience=5, factor=0.1):
    """
    Create a learning rate scheduler.
    
    Args:
        optimizer (optim.Optimizer): Optimizer
        scheduler_type (str): Type of scheduler ('plateau', 'step', or 'cosine')
        patience (int): Patience for ReduceLROnPlateau
        factor (float): Factor for ReduceLROnPlateau
    
    Returns:
        optim.lr_scheduler._LRScheduler: Scheduler
    """
    logger.info(f"Creating {scheduler_type} scheduler")
    
    if scheduler_type.lower() == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=factor, patience=patience, verbose=True
        )
    elif scheduler_type.lower() == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=10, gamma=0.1
        )
    elif scheduler_type.lower() == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=10
        )
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")
    
    return scheduler


def create_criterion(criterion_type='bce', pos_weight=None):
    """
    Create a loss criterion.
    
    Args:
        criterion_type (str): Type of criterion ('bce', 'mse', or 'ce')
        pos_weight (torch.Tensor, optional): Positive weight for BCEWithLogitsLoss
    
    Returns:
        nn.Module: Loss criterion
    """
    logger.info(f"Creating {criterion_type} criterion")
    
    if criterion_type.lower() == 'bce':
        if pos_weight is not None:
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            criterion = nn.BCEWithLogitsLoss()
    elif criterion_type.lower() == 'mse':
        criterion = nn.MSELoss()
    elif criterion_type.lower() == 'ce':
        criterion = nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Unknown criterion type: {criterion_type}")
    
    return criterion


def save_model(model, optimizer, scheduler, config, path):
    """
    Save a model checkpoint.
    
    Args:
        model (nn.Module): Model to save
        optimizer (optim.Optimizer): Optimizer
        scheduler (optim.lr_scheduler._LRScheduler): Scheduler
        config (dict): Configuration
        path (str): Path to save the checkpoint
    """
    logger.info(f"Saving model to {path}")
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # Save checkpoint
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'config': config
    }, path)


def load_model(path, model, optimizer=None, scheduler=None, device=None):
    """
    Load a model checkpoint.
    
    Args:
        path (str): Path to the checkpoint
        model (nn.Module): Model to load
        optimizer (optim.Optimizer, optional): Optimizer to load
        scheduler (optim.lr_scheduler._LRScheduler, optional): Scheduler to load
        device (torch.device, optional): Device to load the model on
    
    Returns:
        tuple: (model, optimizer, scheduler, config)
    """
    logger.info(f"Loading model from {path}")
    
    # Set device
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load checkpoint
    checkpoint = torch.load(path, map_location=device)
    
    # Load model state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load optimizer state dict
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Load scheduler state dict
    if scheduler and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    # Get config
    config = checkpoint.get('config', {})
    
    return model, optimizer, scheduler, config


def train_epoch(model, train_loader, criterion, optimizer, device, log_interval=10):
    """
    Train the model for one epoch.
    
    Args:
        model (nn.Module): Model to train
        train_loader (DataLoader): Training data loader
        criterion (nn.Module): Loss criterion
        optimizer (optim.Optimizer): Optimizer
        device (torch.device): Device to use for training
        log_interval (int): Interval for logging
    
    Returns:
        tuple: (train_loss, train_acc, train_f1)
    """
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    for batch_idx, (data, target) in enumerate(tqdm(train_loader, desc='Training')):
        # Move data to device
        if isinstance(data, dict):
            # For multimodal data
            data = {k: v.to(device) for k, v in data.items()}
        else:
            # For single modality data
            data = data.to(device)
        
        target = target.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        output = model(data)
        
        # Calculate loss
        if target.shape[1] > 1:
            # Multi-task learning (e.g., binary classification + regression)
            loss = criterion(output, target[:, 0].unsqueeze(1))
        else:
            # Single task
            loss = criterion(output, target)
        
        # Backward pass
        loss.backward()
        
        # Update weights
        optimizer.step()
        
        # Accumulate loss
        total_loss += loss.item()
        
        # Store predictions and labels for metrics calculation
        all_preds.append(torch.sigmoid(output).cpu().detach().numpy())
        all_labels.append(target.cpu().numpy())
        
        # Log progress
        if batch_idx % log_interval == 0:
            logger.info(f'Train Batch: {batch_idx}/{len(train_loader)} Loss: {loss.item():.6f}')
    
    # Calculate average loss
    avg_loss = total_loss / len(train_loader)
    
    # Calculate metrics
    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)
    
    # Convert predictions to binary
    binary_preds = (all_preds > 0.5).astype(int)
    
    # Calculate accuracy and F1 score
    accuracy = accuracy_score(all_labels[:, 0], binary_preds[:, 0])
    f1 = f1_score(all_labels[:, 0], binary_preds[:, 0])
    
    return avg_loss, accuracy, f1


def validate(model, val_loader, criterion, device):
    """
    Validate the model.
    
    Args:
        model (nn.Module): Model to validate
        val_loader (DataLoader): Validation data loader
        criterion (nn.Module): Loss criterion
        device (torch.device): Device to use for validation
    
    Returns:
        tuple: (val_loss, val_acc, val_f1)
    """
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for data, target in tqdm(val_loader, desc='Validation'):
            # Move data to device
            if isinstance(data, dict):
                # For multimodal data
                data = {k: v.to(device) for k, v in data.items()}
            else:
                # For single modality data
                data = data.to(device)
            
            target = target.to(device)
            
            # Forward pass
            output = model(data)
            
            # Calculate loss
            if target.shape[1] > 1:
                # Multi-task learning (e.g., binary classification + regression)
                loss = criterion(output, target[:, 0].unsqueeze(1))
            else:
                # Single task
                loss = criterion(output, target)
            
            # Accumulate loss
            total_loss += loss.item()
            
            # Store predictions and labels for metrics calculation
            all_preds.append(torch.sigmoid(output).cpu().numpy())
            all_labels.append(target.cpu().numpy())
    
    # Calculate average loss
    avg_loss = total_loss / len(val_loader)
    
    # Calculate metrics
    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)
    
    # Convert predictions to binary
    binary_preds = (all_preds > 0.5).astype(int)
    
    # Calculate accuracy and F1 score
    accuracy = accuracy_score(all_labels[:, 0], binary_preds[:, 0])
    f1 = f1_score(all_labels[:, 0], binary_preds[:, 0])
    
    return avg_loss, accuracy, f1


def evaluate(model, test_loader, device, detailed=True):
    """
    Evaluate the model.
    
    Args:
        model (nn.Module): Model to evaluate
        test_loader (DataLoader): Test data loader
        device (torch.device): Device to use for evaluation
        detailed (bool): Whether to compute detailed metrics
    
    Returns:
        dict: Evaluation metrics
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    inference_times = []
    
    with torch.no_grad():
        for data, target in tqdm(test_loader, desc='Evaluation'):
            # Move data to device
            if isinstance(data, dict):
                # For multimodal data
                data = {k: v.to(device) for k, v in data.items()}
            else:
                # For single modality data
                data = data.to(device)
            
            target = target.to(device)
            
            # Measure inference time
            start_time = time.time()
            
            # Forward pass
            output = model(data)
            
            # Calculate inference time
            inference_time = time.time() - start_time
            inference_times.append(inference_time)
            
            # Get probabilities
            probs = torch.sigmoid(output).cpu().numpy()
            
            # Store predictions and labels
            all_probs.append(probs)
            all_preds.append((probs > 0.5).astype(int))
            all_labels.append(target.cpu().numpy())
    
    # Concatenate predictions and labels
    all_probs = np.vstack(all_probs)
    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(all_labels[:, 0], all_preds[:, 0]),
        'precision': precision_score(all_labels[:, 0], all_preds[:, 0]),
        'recall': recall_score(all_labels[:, 0], all_preds[:, 0]),
        'f1': f1_score(all_labels[:, 0], all_preds[:, 0]),
        'auc': roc_auc_score(all_labels[:, 0], all_probs[:, 0]),
        'avg_inference_time': np.mean(inference_times),
        'std_inference_time': np.std(inference_times)
    }
    
    # Compute confusion matrix
    cm = confusion_matrix(all_labels[:, 0], all_preds[:, 0])
    metrics['confusion_matrix'] = cm
    
    # Log metrics
    logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"Precision: {metrics['precision']:.4f}")
    logger.info(f"Recall: {metrics['recall']:.4f}")
    logger.info(f"F1 Score: {metrics['f1']:.4f}")
    logger.info(f"AUC: {metrics['auc']:.4f}")
    logger.info(f"Average Inference Time: {metrics['avg_inference_time']:.4f} seconds")
    logger.info(f"Confusion Matrix:\n{cm}")
    
    return metrics


def save_metrics(metrics, path):
    """
    Save evaluation metrics.
    
    Args:
        metrics (dict): Evaluation metrics
        path (str): Path to save the metrics
    """
    logger.info(f"Saving metrics to {path}")
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # Save metrics
    with open(path, 'wb') as f:
        pickle.dump(metrics, f)


def load_metrics(path):
    """
    Load evaluation metrics.
    
    Args:
        path (str): Path to the metrics
    
    Returns:
        dict: Evaluation metrics
    """
    logger.info(f"Loading metrics from {path}")
    
    # Load metrics
    with open(path, 'rb') as f:
        metrics = pickle.load(f)
    
    return metrics


def count_parameters(model):
    """
    Count the number of trainable parameters in a model.
    
    Args:
        model (nn.Module): Model
    
    Returns:
        int: Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_size(model):
    """
    Get the size of a model in MB.
    
    Args:
        model (nn.Module): Model
    
    Returns:
        float: Model size in MB
    """
    # Save model to temporary file
    temp_path = 'temp_model.pt'
    torch.save(model.state_dict(), temp_path)
    
    # Get file size
    size_bytes = os.path.getsize(temp_path)
    size_mb = size_bytes / (1024 * 1024)
    
    # Remove temporary file
    os.remove(temp_path)
    
    return size_mb


def get_model_summary(model, input_size=None):
    """
    Get a summary of a model.
    
    Args:
        model (nn.Module): Model
        input_size (tuple, optional): Input size for the model
    
    Returns:
        str: Model summary
    """
    from torchsummary import summary
    
    if input_size:
        return summary(model, input_size)
    else:
        return str(model)
