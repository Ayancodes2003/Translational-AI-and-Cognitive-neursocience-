"""
Generate Synthetic Data

This script generates synthetic data for the Mental Health AI project.
"""

import os
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set random seed for reproducibility
np.random.seed(42)

# Create output directories
os.makedirs('data/eeg/processed', exist_ok=True)
os.makedirs('data/audio/processed', exist_ok=True)
os.makedirs('data/text/processed', exist_ok=True)
os.makedirs('data/fusion/processed', exist_ok=True)

def generate_eeg_data(n_samples=1000, n_features=100):
    """Generate synthetic EEG data."""
    logger.info(f"Generating synthetic EEG data with {n_samples} samples and {n_features} features")
    
    # Generate features
    X = np.random.randn(n_samples, n_features)
    
    # Generate binary labels (0: non-depressed, 1: depressed)
    # Use a simple rule: if sum of first 10 features > 0, then depressed
    binary_labels = (np.sum(X[:, :10], axis=1) > 0).astype(float).reshape(-1, 1)
    
    # Generate PHQ-8 scores based on binary labels
    phq8_scores = np.zeros((n_samples, 1))
    for i in range(n_samples):
        if binary_labels[i, 0] == 0:  # Non-depressed
            phq8_scores[i, 0] = np.random.randint(0, 10)  # PHQ-8 < 10: non-depressed
        else:  # Depressed
            phq8_scores[i, 0] = np.random.randint(10, 25)  # PHQ-8 >= 10: depressed
    
    # Combine binary labels and PHQ-8 scores
    y = np.hstack((binary_labels, phq8_scores))
    
    # Split into train, val, test
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, random_state=42)
    
    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    # Create dataset dictionary
    dataset = {
        'X_train': X_train,
        'y_train': y_train,
        'X_val': X_val,
        'y_val': y_val,
        'X_test': X_test,
        'y_test': y_test
    }
    
    # Save dataset
    with open('data/eeg/processed/combined_dataset.pkl', 'wb') as f:
        pickle.dump(dataset, f)
    
    logger.info(f"Saved synthetic EEG dataset to data/eeg/processed/combined_dataset.pkl")
    
    return dataset

def generate_audio_data(n_samples=1000, n_features=80):
    """Generate synthetic audio data."""
    logger.info(f"Generating synthetic audio data with {n_samples} samples and {n_features} features")
    
    # Generate features
    X = np.random.randn(n_samples, n_features)
    
    # Generate binary labels (0: non-depressed, 1: depressed)
    # Use a simple rule: if sum of first 10 features > 0, then depressed
    binary_labels = (np.sum(X[:, :10], axis=1) > 0).astype(float).reshape(-1, 1)
    
    # Generate PHQ-8 scores based on binary labels
    phq8_scores = np.zeros((n_samples, 1))
    for i in range(n_samples):
        if binary_labels[i, 0] == 0:  # Non-depressed
            phq8_scores[i, 0] = np.random.randint(0, 10)  # PHQ-8 < 10: non-depressed
        else:  # Depressed
            phq8_scores[i, 0] = np.random.randint(10, 25)  # PHQ-8 >= 10: depressed
    
    # Combine binary labels and PHQ-8 scores
    y = np.hstack((binary_labels, phq8_scores))
    
    # Split into train, val, test
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, random_state=42)
    
    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    # Create dataset dictionary
    dataset = {
        'X_train': X_train,
        'y_train': y_train,
        'X_val': X_val,
        'y_val': y_val,
        'X_test': X_test,
        'y_test': y_test
    }
    
    # Save dataset
    with open('data/audio/processed/audio_dataset.pkl', 'wb') as f:
        pickle.dump(dataset, f)
    
    logger.info(f"Saved synthetic audio dataset to data/audio/processed/audio_dataset.pkl")
    
    return dataset

def generate_text_data(n_samples=1000, n_features=50):
    """Generate synthetic text data."""
    logger.info(f"Generating synthetic text data with {n_samples} samples and {n_features} features")
    
    # Generate features
    X = np.random.randn(n_samples, n_features)
    
    # Generate binary labels (0: non-depressed, 1: depressed)
    # Use a simple rule: if sum of first 10 features > 0, then depressed
    binary_labels = (np.sum(X[:, :10], axis=1) > 0).astype(float).reshape(-1, 1)
    
    # Generate PHQ-8 scores based on binary labels
    phq8_scores = np.zeros((n_samples, 1))
    for i in range(n_samples):
        if binary_labels[i, 0] == 0:  # Non-depressed
            phq8_scores[i, 0] = np.random.randint(0, 10)  # PHQ-8 < 10: non-depressed
        else:  # Depressed
            phq8_scores[i, 0] = np.random.randint(10, 25)  # PHQ-8 >= 10: depressed
    
    # Combine binary labels and PHQ-8 scores
    y = np.hstack((binary_labels, phq8_scores))
    
    # Split into train, val, test
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, random_state=42)
    
    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    # Create dataset dictionary
    dataset = {
        'X_train': X_train,
        'y_train': y_train,
        'X_val': X_val,
        'y_val': y_val,
        'X_test': X_test,
        'y_test': y_test
    }
    
    # Save dataset
    with open('data/text/processed/text_dataset.pkl', 'wb') as f:
        pickle.dump(dataset, f)
    
    logger.info(f"Saved synthetic text dataset to data/text/processed/text_dataset.pkl")
    
    return dataset

def generate_fusion_data():
    """Generate synthetic fusion data."""
    logger.info("Generating synthetic fusion data")
    
    # Load individual datasets
    with open('data/eeg/processed/combined_dataset.pkl', 'rb') as f:
        eeg_dataset = pickle.load(f)
    
    with open('data/audio/processed/audio_dataset.pkl', 'rb') as f:
        audio_dataset = pickle.load(f)
    
    with open('data/text/processed/text_dataset.pkl', 'rb') as f:
        text_dataset = pickle.load(f)
    
    # Get minimum number of samples in each split
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
    
    # Combine features
    X_train = np.hstack((
        eeg_dataset['X_train'][:min_train_samples],
        audio_dataset['X_train'][:min_train_samples],
        text_dataset['X_train'][:min_train_samples]
    ))
    
    X_val = np.hstack((
        eeg_dataset['X_val'][:min_val_samples],
        audio_dataset['X_val'][:min_val_samples],
        text_dataset['X_val'][:min_val_samples]
    ))
    
    X_test = np.hstack((
        eeg_dataset['X_test'][:min_test_samples],
        audio_dataset['X_test'][:min_test_samples],
        text_dataset['X_test'][:min_test_samples]
    ))
    
    # Use labels from EEG dataset
    y_train = eeg_dataset['y_train'][:min_train_samples]
    y_val = eeg_dataset['y_val'][:min_val_samples]
    y_test = eeg_dataset['y_test'][:min_test_samples]
    
    # Create dataset dictionary
    dataset = {
        'X_train': X_train,
        'y_train': y_train,
        'X_val': X_val,
        'y_val': y_val,
        'X_test': X_test,
        'y_test': y_test,
        'feature_dims': {
            'eeg': eeg_dataset['X_train'].shape[1],
            'audio': audio_dataset['X_train'].shape[1],
            'text': text_dataset['X_train'].shape[1]
        }
    }
    
    # Save dataset
    with open('data/fusion/processed/fusion_early_dataset.pkl', 'wb') as f:
        pickle.dump(dataset, f)
    
    logger.info(f"Saved synthetic fusion dataset to data/fusion/processed/fusion_early_dataset.pkl")
    
    return dataset

def main():
    """Main function."""
    logger.info("Generating synthetic data for Mental Health AI")
    
    # Generate data for each modality
    generate_eeg_data()
    generate_audio_data()
    generate_text_data()
    generate_fusion_data()
    
    logger.info("Finished generating synthetic data")

if __name__ == "__main__":
    main()
