"""
Dataset Loader Module

This module handles loading datasets from public repositories like Hugging Face Datasets,
TensorFlow Datasets, or other public sources.
"""

import os
import numpy as np
import pandas as pd
import torch
import logging
from datasets import load_dataset
import tensorflow_datasets as tfds
import mne
from mne.datasets import sample
from sklearn.model_selection import train_test_split
import pickle
import librosa

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DatasetLoader:
    """Class for loading datasets from public repositories."""
    
    def __init__(self, output_dir='data'):
        """
        Initialize the dataset loader.
        
        Args:
            output_dir (str): Directory to save processed data
        """
        self.output_dir = output_dir
        
        # Create output directories
        os.makedirs(os.path.join(output_dir, 'eeg', 'processed'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'audio', 'processed'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'text', 'processed'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'fusion', 'processed'), exist_ok=True)
    
    def load_eeg_data(self):
        """
        Load EEG data from MNE sample dataset.
        
        Since DEAP requires manual download, we'll use MNE sample data as a substitute.
        
        Returns:
            tuple: (eeg_data, labels)
        """
        logger.info("Loading EEG data from MNE sample dataset")
        
        # Load MNE sample data
        data_path = sample.data_path()
        raw = mne.io.read_raw_fif(os.path.join(data_path, 'MEG', 'sample', 'sample_audvis_raw.fif'), preload=True)
        
        # Extract EEG channels
        picks = mne.pick_types(raw.info, meg=False, eeg=True, eog=False, stim=False, exclude='bads')
        eeg_data = raw.get_data(picks=picks)
        
        # Create synthetic labels for demonstration (0: non-depressed, 1: depressed)
        n_samples = 100  # Create 100 synthetic samples
        n_channels, n_times = eeg_data.shape
        
        # Segment the continuous data into epochs
        epoch_duration = 1000  # 1000 time points per epoch
        n_epochs = min(n_samples, n_times // epoch_duration)
        
        # Create epochs
        epochs_data = np.zeros((n_epochs, n_channels, epoch_duration))
        for i in range(n_epochs):
            start = i * epoch_duration
            end = start + epoch_duration
            epochs_data[i] = eeg_data[:, start:end]
        
        # Create synthetic labels (binary classification: depressed/non-depressed)
        # Also create PHQ-8 scores (0-24 scale)
        binary_labels = np.random.randint(0, 2, size=(n_epochs, 1))
        phq8_scores = np.zeros((n_epochs, 1))
        
        # Assign PHQ-8 scores based on binary label
        for i in range(n_epochs):
            if binary_labels[i, 0] == 0:  # Non-depressed
                phq8_scores[i, 0] = np.random.randint(0, 10)  # PHQ-8 < 10: non-depressed
            else:  # Depressed
                phq8_scores[i, 0] = np.random.randint(10, 25)  # PHQ-8 >= 10: depressed
        
        # Combine binary labels and PHQ-8 scores
        labels = np.hstack((binary_labels, phq8_scores))
        
        logger.info(f"Loaded EEG data with shape {epochs_data.shape} and labels with shape {labels.shape}")
        
        return epochs_data, labels
    
    def load_audio_data(self):
        """
        Load audio data from RAVDESS dataset via Hugging Face.
        
        Returns:
            tuple: (audio_data, labels)
        """
        logger.info("Loading audio data from RAVDESS dataset")
        
        # Load RAVDESS dataset from Hugging Face
        dataset = load_dataset("jonatasgrosman/ravdess", split="train")
        
        # Extract audio data and emotion labels
        audio_data = []
        emotion_labels = []
        
        for sample in dataset:
            # Get audio array
            audio = sample["audio"]["array"]
            
            # Resample to 16kHz if needed
            if sample["audio"]["sampling_rate"] != 16000:
                audio = librosa.resample(
                    audio, 
                    orig_sr=sample["audio"]["sampling_rate"], 
                    target_sr=16000
                )
            
            # Trim to fixed length (5 seconds)
            max_length = 16000 * 5  # 5 seconds at 16kHz
            if len(audio) > max_length:
                audio = audio[:max_length]
            else:
                # Pad with zeros
                audio = np.pad(audio, (0, max_length - len(audio)))
            
            audio_data.append(audio)
            
            # Map emotion to binary depression label (for demonstration)
            # In RAVDESS: 1=neutral, 2=calm, 3=happy, 4=sad, 5=angry, 6=fearful, 7=disgust, 8=surprised
            # We'll map sad (4) and fearful (6) to "depressed" (1), others to "non-depressed" (0)
            emotion = sample["emotion"]
            binary_label = 1 if emotion in [4, 6] else 0
            
            # Create synthetic PHQ-8 score based on binary label
            if binary_label == 0:  # Non-depressed
                phq8_score = np.random.randint(0, 10)  # PHQ-8 < 10: non-depressed
            else:  # Depressed
                phq8_score = np.random.randint(10, 25)  # PHQ-8 >= 10: depressed
            
            emotion_labels.append([binary_label, phq8_score])
        
        # Convert to numpy arrays
        audio_data = np.array(audio_data)
        emotion_labels = np.array(emotion_labels)
        
        logger.info(f"Loaded audio data with shape {audio_data.shape} and labels with shape {emotion_labels.shape}")
        
        return audio_data, emotion_labels
    
    def load_text_data(self):
        """
        Load text data from Hugging Face datasets.
        
        We'll use the "dair-ai/emotion" dataset which contains text with emotion labels.
        
        Returns:
            tuple: (text_data, labels)
        """
        logger.info("Loading text data from emotion dataset")
        
        # Load emotion dataset from Hugging Face
        dataset = load_dataset("dair-ai/emotion", split="train")
        
        # Extract text data and emotion labels
        text_data = []
        emotion_labels = []
        
        for sample in dataset:
            text = sample["text"]
            emotion = sample["label"]
            
            text_data.append(text)
            
            # Map emotion to binary depression label (for demonstration)
            # In this dataset: 0=sadness, 1=joy, 2=love, 3=anger, 4=fear, 5=surprise
            # We'll map sadness (0) and fear (4) to "depressed" (1), others to "non-depressed" (0)
            binary_label = 1 if emotion in [0, 4] else 0
            
            # Create synthetic PHQ-8 score based on binary label
            if binary_label == 0:  # Non-depressed
                phq8_score = np.random.randint(0, 10)  # PHQ-8 < 10: non-depressed
            else:  # Depressed
                phq8_score = np.random.randint(10, 25)  # PHQ-8 >= 10: depressed
            
            emotion_labels.append([binary_label, phq8_score])
        
        # Convert to numpy arrays
        emotion_labels = np.array(emotion_labels)
        
        logger.info(f"Loaded text data with {len(text_data)} samples and labels with shape {emotion_labels.shape}")
        
        return text_data, emotion_labels
    
    def create_dataset_splits(self, features, labels, test_size=0.2, val_size=0.1, random_state=42):
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
    
    def save_dataset(self, dataset, modality):
        """
        Save dataset to disk.
        
        Args:
            dataset (dict): Dataset dictionary
            modality (str): Modality name ('eeg', 'audio', or 'text')
        """
        logger.info(f"Saving {modality} dataset")
        
        # Create output directory
        output_dir = os.path.join(self.output_dir, modality, 'processed')
        os.makedirs(output_dir, exist_ok=True)
        
        # Save dataset
        with open(os.path.join(output_dir, f'{modality}_dataset.pkl'), 'wb') as f:
            pickle.dump(dataset, f)
        
        # Save features and labels separately
        np.save(os.path.join(output_dir, f'{modality}_features.npy'), 
                np.vstack([dataset['X_train'], dataset['X_val'], dataset['X_test']]))
        np.save(os.path.join(output_dir, f'{modality}_labels.npy'), 
                np.vstack([dataset['y_train'], dataset['y_val'], dataset['y_test']]))
        
        logger.info(f"Saved {modality} dataset to {output_dir}")
    
    def create_fusion_dataset(self, eeg_dataset, audio_dataset, text_dataset, fusion_type='early'):
        """
        Create fusion dataset from individual modality datasets.
        
        Args:
            eeg_dataset (dict): EEG dataset dictionary
            audio_dataset (dict): Audio dataset dictionary
            text_dataset (dict): Text dataset dictionary
            fusion_type (str): Fusion type ('early', 'late', or 'intermediate')
        
        Returns:
            dict: Fusion dataset dictionary
        """
        logger.info(f"Creating fusion dataset with {fusion_type} fusion")
        
        # Ensure all datasets have the same number of samples
        min_train_samples = min(len(eeg_dataset['X_train']), len(audio_dataset['X_train']), len(text_dataset['X_train']))
        min_val_samples = min(len(eeg_dataset['X_val']), len(audio_dataset['X_val']), len(text_dataset['X_val']))
        min_test_samples = min(len(eeg_dataset['X_test']), len(audio_dataset['X_test']), len(text_dataset['X_test']))
        
        # Truncate datasets to the same size
        eeg_dataset = {
            'X_train': eeg_dataset['X_train'][:min_train_samples],
            'y_train': eeg_dataset['y_train'][:min_train_samples],
            'X_val': eeg_dataset['X_val'][:min_val_samples],
            'y_val': eeg_dataset['y_val'][:min_val_samples],
            'X_test': eeg_dataset['X_test'][:min_test_samples],
            'y_test': eeg_dataset['y_test'][:min_test_samples]
        }
        
        audio_dataset = {
            'X_train': audio_dataset['X_train'][:min_train_samples],
            'y_train': audio_dataset['y_train'][:min_train_samples],
            'X_val': audio_dataset['X_val'][:min_val_samples],
            'y_val': audio_dataset['y_val'][:min_val_samples],
            'X_test': audio_dataset['X_test'][:min_test_samples],
            'y_test': audio_dataset['y_test'][:min_test_samples]
        }
        
        text_dataset = {
            'X_train': text_dataset['X_train'][:min_train_samples],
            'y_train': text_dataset['y_train'][:min_train_samples],
            'X_val': text_dataset['X_val'][:min_val_samples],
            'y_val': text_dataset['y_val'][:min_val_samples],
            'X_test': text_dataset['X_test'][:min_test_samples],
            'y_test': text_dataset['y_test'][:min_test_samples]
        }
        
        # Create fusion dataset based on fusion type
        if fusion_type == 'early':
            # Early fusion: concatenate features
            fusion_dataset = {
                'X_train': np.hstack([eeg_dataset['X_train'], audio_dataset['X_train'], text_dataset['X_train']]),
                'y_train': eeg_dataset['y_train'],  # All labels should be the same
                'X_val': np.hstack([eeg_dataset['X_val'], audio_dataset['X_val'], text_dataset['X_val']]),
                'y_val': eeg_dataset['y_val'],
                'X_test': np.hstack([eeg_dataset['X_test'], audio_dataset['X_test'], text_dataset['X_test']]),
                'y_test': eeg_dataset['y_test'],
                'feature_dims': {
                    'eeg': eeg_dataset['X_train'].shape[1],
                    'audio': audio_dataset['X_train'].shape[1],
                    'text': text_dataset['X_train'].shape[1]
                }
            }
        else:  # 'late' or 'intermediate' fusion
            # For late and intermediate fusion, keep features separate
            fusion_dataset = {
                'eeg_train': eeg_dataset['X_train'],
                'audio_train': audio_dataset['X_train'],
                'text_train': text_dataset['X_train'],
                'y_train': eeg_dataset['y_train'],
                'eeg_val': eeg_dataset['X_val'],
                'audio_val': audio_dataset['X_val'],
                'text_val': text_dataset['X_val'],
                'y_val': eeg_dataset['y_val'],
                'eeg_test': eeg_dataset['X_test'],
                'audio_test': audio_dataset['X_test'],
                'text_test': text_dataset['X_test'],
                'y_test': eeg_dataset['y_test'],
                'feature_dims': {
                    'eeg': eeg_dataset['X_train'].shape[1],
                    'audio': audio_dataset['X_train'].shape[1],
                    'text': text_dataset['X_train'].shape[1]
                }
            }
        
        # Save fusion dataset
        output_dir = os.path.join(self.output_dir, 'fusion', 'processed')
        os.makedirs(output_dir, exist_ok=True)
        
        with open(os.path.join(output_dir, f'fusion_{fusion_type}_dataset.pkl'), 'wb') as f:
            pickle.dump(fusion_dataset, f)
        
        logger.info(f"Created and saved fusion dataset with {fusion_type} fusion")
        
        return fusion_dataset
    
    def load_and_process_all_data(self):
        """
        Load and process all datasets.
        
        Returns:
            tuple: (eeg_dataset, audio_dataset, text_dataset, fusion_dataset)
        """
        logger.info("Loading and processing all datasets")
        
        # Load EEG data
        eeg_data, eeg_labels = self.load_eeg_data()
        
        # Extract EEG features
        from data.eeg.preprocess_eeg import EEGProcessor
        eeg_processor = EEGProcessor(None, os.path.join(self.output_dir, 'eeg', 'processed'))
        eeg_features = eeg_processor.extract_features(eeg_data)
        
        # Create EEG dataset splits
        eeg_dataset = self.create_dataset_splits(eeg_features, eeg_labels)
        self.save_dataset(eeg_dataset, 'eeg')
        
        # Load audio data
        audio_data, audio_labels = self.load_audio_data()
        
        # Extract audio features
        from data.audio.preprocess_audio import AudioProcessor
        audio_processor = AudioProcessor(None, os.path.join(self.output_dir, 'audio', 'processed'))
        audio_features = np.array([audio_processor.extract_features(audio) for audio in audio_data])
        
        # Create audio dataset splits
        audio_dataset = self.create_dataset_splits(audio_features, audio_labels)
        self.save_dataset(audio_dataset, 'audio')
        
        # Load text data
        text_data, text_labels = self.load_text_data()
        
        # Extract text features
        from data.text.preprocess_text import TextProcessor
        text_processor = TextProcessor(None, os.path.join(self.output_dir, 'text', 'processed'))
        
        # Use TF-IDF for text features
        from sklearn.feature_extraction.text import TfidfVectorizer
        vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
        text_features = vectorizer.fit_transform(text_data).toarray()
        
        # Add some linguistic features
        linguistic_features = np.array([list(text_processor.extract_linguistic_features([text]).values()) for text in text_data])
        text_features = np.hstack([text_features, linguistic_features])
        
        # Create text dataset splits
        text_dataset = self.create_dataset_splits(text_features, text_labels)
        self.save_dataset(text_dataset, 'text')
        
        # Create fusion datasets
        early_fusion_dataset = self.create_fusion_dataset(eeg_dataset, audio_dataset, text_dataset, 'early')
        late_fusion_dataset = self.create_fusion_dataset(eeg_dataset, audio_dataset, text_dataset, 'late')
        intermediate_fusion_dataset = self.create_fusion_dataset(eeg_dataset, audio_dataset, text_dataset, 'intermediate')
        
        logger.info("Finished loading and processing all datasets")
        
        return eeg_dataset, audio_dataset, text_dataset, early_fusion_dataset


if __name__ == "__main__":
    # Create dataset loader
    loader = DatasetLoader()
    
    # Load and process all datasets
    eeg_dataset, audio_dataset, text_dataset, fusion_dataset = loader.load_and_process_all_data()
