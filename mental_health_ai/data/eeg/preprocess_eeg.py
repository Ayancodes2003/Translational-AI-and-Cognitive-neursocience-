"""
EEG Data Preprocessing Module

This module handles the preprocessing of EEG data from the DEAP dataset.
It includes functions for loading, filtering, normalization, and feature extraction.
"""

import os
import numpy as np
import pandas as pd
import mne
from scipy import signal
import pickle
import antropy as ant
import pywt
from sklearn.preprocessing import StandardScaler
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define constants
SAMPLE_RATE = 128  # Hz
EEG_CHANNELS = 32
FREQ_BANDS = {
    'delta': (1, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta': (13, 30),
    'gamma': (30, 45)
}

class EEGProcessor:
    """Class for processing EEG data from the DEAP dataset."""

    def __init__(self, data_path, output_path=None):
        """
        Initialize the EEG processor.

        Args:
            data_path (str): Path to the DEAP dataset
            output_path (str, optional): Path to save processed data
        """
        self.data_path = data_path
        self.output_path = output_path or os.path.join(data_path, 'processed')
        os.makedirs(self.output_path, exist_ok=True)

    def load_data(self, subject_id):
        """
        Load EEG data for a specific subject.

        Args:
            subject_id (int): Subject ID (1-32)

        Returns:
            tuple: (data, labels) where data is a numpy array of shape (40, 40, 8064)
                  and labels is a numpy array of shape (40, 4)
        """
        logger.info(f"Loading data for subject {subject_id}")

        # Format subject ID with leading zeros
        subject_str = f"s{subject_id:02d}"

        # Path to the data file
        data_file = os.path.join(self.data_path, f"{subject_str}.dat")

        # Check if file exists
        if not os.path.exists(data_file):
            raise FileNotFoundError(f"Data file not found: {data_file}")

        # Load data
        with open(data_file, 'rb') as f:
            data = pickle.load(f, encoding='latin1')

        return data['data'], data['labels']

    def preprocess_eeg(self, eeg_data):
        """
        Preprocess EEG data.

        Args:
            eeg_data (numpy.ndarray): EEG data of shape (trials, channels, samples)

        Returns:
            numpy.ndarray: Preprocessed EEG data
        """
        logger.info("Preprocessing EEG data")

        n_trials, n_channels, n_samples = eeg_data.shape
        preprocessed_data = np.zeros_like(eeg_data)

        for trial in range(n_trials):
            for channel in range(n_channels):
                # Get channel data
                channel_data = eeg_data[trial, channel, :]

                # Apply bandpass filter (4-45 Hz)
                b, a = signal.butter(4, [4/SAMPLE_RATE*2, 45/SAMPLE_RATE*2], btype='bandpass')
                filtered_data = signal.filtfilt(b, a, channel_data)

                # Normalize
                normalized_data = (filtered_data - np.mean(filtered_data)) / np.std(filtered_data)

                # Store preprocessed data
                preprocessed_data[trial, channel, :] = normalized_data

        return preprocessed_data

    def extract_features(self, eeg_data):
        """
        Extract features from EEG data.

        Args:
            eeg_data (numpy.ndarray): EEG data of shape (trials, channels, samples)
                                      or (channels, samples) for a single trial

        Returns:
            numpy.ndarray: Feature matrix of shape (trials, n_features)
                          or (1, n_features) for a single trial
        """
        logger.info("Extracting features from EEG data")

        # Check if input is a single trial or multiple trials
        if len(eeg_data.shape) == 2:
            # Single trial: reshape to (1, channels, samples)
            eeg_data = np.expand_dims(eeg_data, axis=0)

        n_trials, n_channels, n_samples = eeg_data.shape

        # Define feature extraction functions
        feature_funcs = {
            'mean': np.mean,
            'std': np.std,
            'ptp': np.ptp,  # Peak-to-peak amplitude
            'skew': lambda x: np.mean((x - np.mean(x))**3) / (np.std(x)**3) if np.std(x) > 0 else 0,
            'kurtosis': lambda x: np.mean((x - np.mean(x))**4) / (np.std(x)**4) if np.std(x) > 0 else 0
        }

        # For the demo, we'll use a simplified feature set to avoid dependencies
        # Calculate number of features
        n_bands = len(FREQ_BANDS)
        n_features_per_channel = len(feature_funcs) + n_bands  # Basic features + band power
        n_features = n_channels * n_features_per_channel

        # Initialize feature matrix
        features = np.zeros((n_trials, n_features))

        for trial in range(n_trials):
            feature_idx = 0

            for channel in range(n_channels):
                # Get channel data
                channel_data = eeg_data[trial, channel, :]

                # Extract basic features
                for func_name, func in feature_funcs.items():
                    try:
                        features[trial, feature_idx] = func(channel_data)
                    except Exception as e:
                        features[trial, feature_idx] = 0
                        logger.warning(f"Error calculating {func_name}: {e}")
                    feature_idx += 1

                # Extract frequency band features
                for band_name, (low_freq, high_freq) in FREQ_BANDS.items():
                    try:
                        # Apply bandpass filter
                        b, a = signal.butter(4, [low_freq/SAMPLE_RATE*2, high_freq/SAMPLE_RATE*2], btype='bandpass')
                        band_data = signal.filtfilt(b, a, channel_data)

                        # Calculate band power (variance of filtered signal)
                        band_power = np.var(band_data)
                        features[trial, feature_idx] = band_power
                    except Exception as e:
                        features[trial, feature_idx] = 0
                        logger.warning(f"Error calculating band power for {band_name}: {e}")
                    feature_idx += 1

        # Normalize features
        scaler = StandardScaler()
        features = scaler.fit_transform(features)

        return features

    def process_all_subjects(self):
        """
        Process all subjects in the dataset.

        Returns:
            tuple: (X, y) where X is the feature matrix and y is the label vector
        """
        logger.info("Processing all subjects")

        all_features = []
        all_labels = []

        for subject_id in range(1, 33):  # DEAP has 32 subjects
            try:
                # Load data
                data, labels = self.load_data(subject_id)

                # Extract EEG data (first 32 channels)
                eeg_data = data[:, :EEG_CHANNELS, :]

                # Preprocess EEG data
                preprocessed_data = self.preprocess_eeg(eeg_data)

                # Extract features
                features = self.extract_features(preprocessed_data)

                # Store features and labels
                all_features.append(features)
                all_labels.append(labels)

                logger.info(f"Processed subject {subject_id}")
            except Exception as e:
                logger.error(f"Error processing subject {subject_id}: {e}")

        # Concatenate features and labels
        X = np.vstack(all_features)
        y = np.vstack(all_labels)

        # Save processed data
        np.save(os.path.join(self.output_path, 'eeg_features.npy'), X)
        np.save(os.path.join(self.output_path, 'eeg_labels.npy'), y)

        logger.info("Finished processing all subjects")

        return X, y

    def create_dataset_splits(self, test_size=0.2, val_size=0.1, random_state=42):
        """
        Create train/val/test splits from the processed data.

        Args:
            test_size (float): Proportion of data to use for testing
            val_size (float): Proportion of data to use for validation
            random_state (int): Random seed for reproducibility

        Returns:
            dict: Dictionary containing the data splits
        """
        from sklearn.model_selection import train_test_split

        logger.info("Creating dataset splits")

        # Load processed data
        X = np.load(os.path.join(self.output_path, 'eeg_features.npy'))
        y = np.load(os.path.join(self.output_path, 'eeg_labels.npy'))

        # Split into train+val and test
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        # Split train+val into train and val
        val_ratio = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=val_ratio, random_state=random_state
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

        # Save dataset splits
        with open(os.path.join(self.output_path, 'eeg_dataset.pkl'), 'wb') as f:
            pickle.dump(dataset, f)

        logger.info("Finished creating dataset splits")

        return dataset


def download_deap_dataset(output_dir):
    """
    Function to guide users through downloading the DEAP dataset.

    Args:
        output_dir (str): Directory to save instructions
    """
    instructions = """
    # DEAP Dataset Download Instructions

    The DEAP dataset is not directly downloadable without registration. Follow these steps to obtain it:

    1. Visit the DEAP dataset website: https://www.eecs.qmul.ac.uk/mmv/datasets/deap/download.html

    2. Fill out the End User License Agreement (EULA) form on the website.

    3. You will receive credentials to download the dataset via email.

    4. Download the following files:
       - data_preprocessed_python.zip (Preprocessed data in Python format)

    5. Extract the downloaded files to the 'data/eeg/raw' directory in this project.

    6. Run the preprocessing script:
       ```
       python -m data.eeg.preprocess_eeg
       ```
    """

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Write instructions to file
    with open(os.path.join(output_dir, 'deap_download_instructions.md'), 'w') as f:
        f.write(instructions)

    print(f"Instructions for downloading the DEAP dataset have been saved to {output_dir}/deap_download_instructions.md")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Process EEG data from the DEAP dataset')
    parser.add_argument('--data_path', type=str, default='data/eeg/raw',
                        help='Path to the DEAP dataset')
    parser.add_argument('--output_path', type=str, default='data/eeg/processed',
                        help='Path to save processed data')
    parser.add_argument('--download_instructions', action='store_true',
                        help='Generate download instructions for the DEAP dataset')

    args = parser.parse_args()

    if args.download_instructions:
        download_deap_dataset(os.path.dirname(args.data_path))
    else:
        # Check if data exists
        if not os.path.exists(args.data_path):
            print(f"Data path {args.data_path} does not exist. Generating download instructions...")
            download_deap_dataset(os.path.dirname(args.data_path))
        else:
            # Process data
            processor = EEGProcessor(args.data_path, args.output_path)
            processor.process_all_subjects()
            processor.create_dataset_splits()
