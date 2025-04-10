"""
Data Utility Functions

This module contains utility functions for data processing and manipulation.
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
import pickle
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_dataset(data_path, modality):
    """
    Load a dataset for a specific modality.
    
    Args:
        data_path (str): Path to the data
        modality (str): Modality ('eeg', 'audio', 'text', or 'fusion')
    
    Returns:
        dict: Dataset dictionary
    """
    logger.info(f"Loading {modality} dataset from {data_path}")
    
    # Check if dataset exists
    dataset_path = os.path.join(data_path, f'{modality}_dataset.pkl')
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    
    # Load dataset
    with open(dataset_path, 'rb') as f:
        dataset = pickle.load(f)
    
    logger.info(f"Loaded {modality} dataset with {len(dataset['X_train'])} training samples")
    
    return dataset


def create_dataloaders(dataset, batch_size=32):
    """
    Create PyTorch DataLoaders from a dataset.
    
    Args:
        dataset (dict): Dataset dictionary
        batch_size (int): Batch size
    
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    logger.info(f"Creating DataLoaders with batch size {batch_size}")
    
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
    
    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    logger.info(f"Created DataLoaders with {len(train_loader)} training batches")
    
    return train_loader, val_loader, test_loader


def create_multimodal_dataloaders(eeg_dataset, audio_dataset, text_dataset, batch_size=32):
    """
    Create PyTorch DataLoaders for multimodal data.
    
    Args:
        eeg_dataset (dict): EEG dataset dictionary
        audio_dataset (dict): Audio dataset dictionary
        text_dataset (dict): Text dataset dictionary
        batch_size (int): Batch size
    
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    logger.info(f"Creating multimodal DataLoaders with batch size {batch_size}")
    
    # Create PyTorch datasets
    train_dataset = MultimodalDataset(
        eeg_dataset['X_train'], audio_dataset['X_train'], text_dataset['X_train'], eeg_dataset['y_train']
    )
    val_dataset = MultimodalDataset(
        eeg_dataset['X_val'], audio_dataset['X_val'], text_dataset['X_val'], eeg_dataset['y_val']
    )
    test_dataset = MultimodalDataset(
        eeg_dataset['X_test'], audio_dataset['X_test'], text_dataset['X_test'], eeg_dataset['y_test']
    )
    
    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    logger.info(f"Created multimodal DataLoaders with {len(train_loader)} training batches")
    
    return train_loader, val_loader, test_loader


class MultimodalDataset(Dataset):
    """Dataset class for multimodal data."""
    
    def __init__(self, eeg_features, audio_features, text_features, labels):
        """
        Initialize the multimodal dataset.
        
        Args:
            eeg_features (numpy.ndarray): EEG features
            audio_features (numpy.ndarray): Audio features
            text_features (numpy.ndarray): Text features
            labels (numpy.ndarray): Labels
        """
        self.eeg_features = torch.tensor(eeg_features, dtype=torch.float32)
        self.audio_features = torch.tensor(audio_features, dtype=torch.float32)
        self.text_features = torch.tensor(text_features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return {
            'eeg': self.eeg_features[idx],
            'audio': self.audio_features[idx],
            'text': self.text_features[idx]
        }, self.labels[idx]


def create_dataset_splits(features, labels, test_size=0.2, val_size=0.1, random_state=42):
    """
    Create train/val/test splits from features and labels.
    
    Args:
        features (numpy.ndarray): Feature matrix
        labels (numpy.ndarray): Label vector
        test_size (float): Proportion of data to use for testing
        val_size (float): Proportion of data to use for validation
        random_state (int): Random seed for reproducibility
    
    Returns:
        dict: Dictionary containing the data splits
    """
    logger.info(f"Creating dataset splits with test_size={test_size}, val_size={val_size}")
    
    # Split into train+val and test
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        features, labels, test_size=test_size, random_state=random_state, stratify=labels[:, 0]
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


def normalize_features(features, scaler=None):
    """
    Normalize features using StandardScaler.
    
    Args:
        features (numpy.ndarray): Feature matrix
        scaler (sklearn.preprocessing.StandardScaler, optional): Scaler to use
    
    Returns:
        tuple: (normalized_features, scaler)
    """
    logger.info(f"Normalizing features of shape {features.shape}")
    
    # Create scaler if not provided
    if scaler is None:
        scaler = StandardScaler()
        scaler.fit(features)
    
    # Normalize features
    normalized_features = scaler.transform(features)
    
    return normalized_features, scaler


def save_dataset(dataset, output_path, name):
    """
    Save a dataset to disk.
    
    Args:
        dataset (dict): Dataset dictionary
        output_path (str): Path to save the dataset
        name (str): Name of the dataset
    """
    logger.info(f"Saving {name} dataset to {output_path}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    
    # Save dataset
    with open(os.path.join(output_path, f'{name}_dataset.pkl'), 'wb') as f:
        pickle.dump(dataset, f)
    
    # Save features and labels separately
    np.save(os.path.join(output_path, f'{name}_features.npy'), np.vstack([dataset['X_train'], dataset['X_val'], dataset['X_test']]))
    np.save(os.path.join(output_path, f'{name}_labels.npy'), np.vstack([dataset['y_train'], dataset['y_val'], dataset['y_test']]))
    
    logger.info(f"Saved {name} dataset to {output_path}")


def load_features_and_labels(data_path, modality):
    """
    Load features and labels for a specific modality.
    
    Args:
        data_path (str): Path to the data
        modality (str): Modality ('eeg', 'audio', 'text', or 'fusion')
    
    Returns:
        tuple: (features, labels)
    """
    logger.info(f"Loading {modality} features and labels from {data_path}")
    
    # Check if features and labels exist
    features_path = os.path.join(data_path, f'{modality}_features.npy')
    labels_path = os.path.join(data_path, f'{modality}_labels.npy')
    
    if not os.path.exists(features_path) or not os.path.exists(labels_path):
        raise FileNotFoundError(f"Features or labels not found: {features_path}, {labels_path}")
    
    # Load features and labels
    features = np.load(features_path)
    labels = np.load(labels_path)
    
    logger.info(f"Loaded {modality} features of shape {features.shape} and labels of shape {labels.shape}")
    
    return features, labels
