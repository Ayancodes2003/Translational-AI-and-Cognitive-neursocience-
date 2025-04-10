"""
Multimodal Fusion Dataset Module

This module handles the creation of multimodal datasets by combining EEG, audio, and text data.
It provides functionality for different fusion strategies.
"""

import os
import numpy as np
import pandas as pd
import pickle
import logging
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


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
            'text': self.text_features[idx],
            'label': self.labels[idx]
        }


class MultimodalFusionDataset:
    """Class for creating multimodal fusion datasets."""
    
    def __init__(self, eeg_path, audio_path, text_path, output_path=None):
        """
        Initialize the multimodal fusion dataset.
        
        Args:
            eeg_path (str): Path to the processed EEG data
            audio_path (str): Path to the processed audio data
            text_path (str): Path to the processed text data
            output_path (str, optional): Path to save the fused dataset
        """
        self.eeg_path = eeg_path
        self.audio_path = audio_path
        self.text_path = text_path
        self.output_path = output_path or 'data/fusion/processed'
        os.makedirs(self.output_path, exist_ok=True)
    
    def load_modality_data(self, modality):
        """
        Load data for a specific modality.
        
        Args:
            modality (str): Modality name ('eeg', 'audio', or 'text')
            
        Returns:
            tuple: (features, labels) for the specified modality
        """
        if modality == 'eeg':
            path = self.eeg_path
        elif modality == 'audio':
            path = self.audio_path
        elif modality == 'text':
            path = self.text_path
        else:
            raise ValueError(f"Unknown modality: {modality}")
        
        # Load features and labels
        features = np.load(os.path.join(path, f'{modality}_features.npy'))
        labels = np.load(os.path.join(path, f'{modality}_labels.npy'))
        
        return features, labels
    
    def create_subject_mapping(self):
        """
        Create a mapping between subjects across modalities.
        
        This is a placeholder function. In a real-world scenario, you would need
        to implement a way to map subjects across different datasets.
        
        Returns:
            dict: Mapping between subjects across modalities
        """
        # In a real-world scenario, you would need to implement this based on your data
        # For now, we'll assume a one-to-one mapping for demonstration purposes
        return {i: i for i in range(100)}
    
    def create_fused_dataset(self, fusion_type='early', test_size=0.2, val_size=0.1, random_state=42):
        """
        Create a fused dataset from the individual modalities.
        
        Args:
            fusion_type (str): Type of fusion ('early', 'late', or 'intermediate')
            test_size (float): Proportion of data to use for testing
            val_size (float): Proportion of data to use for validation
            random_state (int): Random seed for reproducibility
            
        Returns:
            dict: Dictionary containing the fused dataset splits
        """
        logger.info(f"Creating fused dataset with {fusion_type} fusion")
        
        # Load data for each modality
        eeg_features, eeg_labels = self.load_modality_data('eeg')
        audio_features, audio_labels = self.load_modality_data('audio')
        text_features, text_labels = self.load_modality_data('text')
        
        # For demonstration purposes, we'll use a subset of the data
        # In a real-world scenario, you would need to properly align the data
        min_samples = min(len(eeg_labels), len(audio_labels), len(text_labels))
        eeg_features = eeg_features[:min_samples]
        eeg_labels = eeg_labels[:min_samples]
        audio_features = audio_features[:min_samples]
        audio_labels = audio_labels[:min_samples]
        text_features = text_features[:min_samples]
        text_labels = text_labels[:min_samples]
        
        # Ensure labels are consistent
        # In a real-world scenario, you would need to properly align the labels
        labels = eeg_labels  # Assuming all labels are the same
        
        # Create fused dataset based on fusion type
        if fusion_type == 'early':
            # Early fusion: concatenate features
            fused_features = np.hstack([eeg_features, audio_features, text_features])
            
            # Split into train/val/test
            X_train_val, X_test, y_train_val, y_test = train_test_split(
                fused_features, labels, test_size=test_size, random_state=random_state, stratify=labels[:, 0]
            )
            
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
                'y_test': y_test,
                'feature_dims': {
                    'eeg': eeg_features.shape[1],
                    'audio': audio_features.shape[1],
                    'text': text_features.shape[1]
                }
            }
        else:  # 'late' or 'intermediate' fusion
            # For late and intermediate fusion, keep features separate
            # Split each modality into train/val/test
            eeg_train_val, eeg_test, audio_train_val, audio_test, text_train_val, text_test, y_train_val, y_test = train_test_split(
                eeg_features, audio_features, text_features, labels, 
                test_size=test_size, random_state=random_state, stratify=labels[:, 0]
            )
            
            val_ratio = val_size / (1 - test_size)
            eeg_train, eeg_val, audio_train, audio_val, text_train, text_val, y_train, y_val = train_test_split(
                eeg_train_val, audio_train_val, text_train_val, y_train_val,
                test_size=val_ratio, random_state=random_state, stratify=y_train_val[:, 0]
            )
            
            # Create dataset dictionary
            dataset = {
                'eeg_train': eeg_train,
                'eeg_val': eeg_val,
                'eeg_test': eeg_test,
                'audio_train': audio_train,
                'audio_val': audio_val,
                'audio_test': audio_test,
                'text_train': text_train,
                'text_val': text_val,
                'text_test': text_test,
                'y_train': y_train,
                'y_val': y_val,
                'y_test': y_test,
                'feature_dims': {
                    'eeg': eeg_features.shape[1],
                    'audio': audio_features.shape[1],
                    'text': text_features.shape[1]
                }
            }
        
        # Save dataset
        with open(os.path.join(self.output_path, f'fusion_{fusion_type}_dataset.pkl'), 'wb') as f:
            pickle.dump(dataset, f)
        
        logger.info(f"Finished creating fused dataset with {fusion_type} fusion")
        
        return dataset
    
    def create_dataloaders(self, fusion_type='early', batch_size=32):
        """
        Create PyTorch DataLoaders for the fused dataset.
        
        Args:
            fusion_type (str): Type of fusion ('early', 'late', or 'intermediate')
            batch_size (int): Batch size for the DataLoaders
            
        Returns:
            dict: Dictionary containing the DataLoaders
        """
        logger.info(f"Creating DataLoaders for {fusion_type} fusion")
        
        # Load fused dataset
        with open(os.path.join(self.output_path, f'fusion_{fusion_type}_dataset.pkl'), 'rb') as f:
            dataset = pickle.load(f)
        
        if fusion_type == 'early':
            # Create PyTorch datasets
            train_dataset = torch.utils.data.TensorDataset(
                torch.tensor(dataset['X_train'], dtype=torch.float32),
                torch.tensor(dataset['y_train'], dtype=torch.float32)
            )
            val_dataset = torch.utils.data.TensorDataset(
                torch.tensor(dataset['X_val'], dtype=torch.float32),
                torch.tensor(dataset['y_val'], dtype=torch.float32)
            )
            test_dataset = torch.utils.data.TensorDataset(
                torch.tensor(dataset['X_test'], dtype=torch.float32),
                torch.tensor(dataset['y_test'], dtype=torch.float32)
            )
            
            # Create DataLoaders
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size)
            test_loader = DataLoader(test_dataset, batch_size=batch_size)
            
            dataloaders = {
                'train': train_loader,
                'val': val_loader,
                'test': test_loader
            }
        else:  # 'late' or 'intermediate' fusion
            # Create PyTorch datasets
            train_dataset = MultimodalDataset(
                dataset['eeg_train'], dataset['audio_train'], dataset['text_train'], dataset['y_train']
            )
            val_dataset = MultimodalDataset(
                dataset['eeg_val'], dataset['audio_val'], dataset['text_val'], dataset['y_val']
            )
            test_dataset = MultimodalDataset(
                dataset['eeg_test'], dataset['audio_test'], dataset['text_test'], dataset['y_test']
            )
            
            # Create DataLoaders
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size)
            test_loader = DataLoader(test_dataset, batch_size=batch_size)
            
            dataloaders = {
                'train': train_loader,
                'val': val_loader,
                'test': test_loader
            }
        
        logger.info(f"Finished creating DataLoaders for {fusion_type} fusion")
        
        return dataloaders


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Create multimodal fusion dataset')
    parser.add_argument('--eeg_path', type=str, default='data/eeg/processed',
                        help='Path to the processed EEG data')
    parser.add_argument('--audio_path', type=str, default='data/audio/processed',
                        help='Path to the processed audio data')
    parser.add_argument('--text_path', type=str, default='data/text/processed',
                        help='Path to the processed text data')
    parser.add_argument('--output_path', type=str, default='data/fusion/processed',
                        help='Path to save the fused dataset')
    parser.add_argument('--fusion_type', type=str, default='early', choices=['early', 'late', 'intermediate'],
                        help='Type of fusion to perform')
    
    args = parser.parse_args()
    
    # Create fusion dataset
    fusion_dataset = MultimodalFusionDataset(
        args.eeg_path, args.audio_path, args.text_path, args.output_path
    )
    
    # Create fused dataset
    fusion_dataset.create_fused_dataset(args.fusion_type)
