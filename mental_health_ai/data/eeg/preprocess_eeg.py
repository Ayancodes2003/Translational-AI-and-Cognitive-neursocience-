"""
EEG Data Preprocessing Module

This module handles the preprocessing of EEG data from the MNE Sample Dataset and EEG Motor Movement/Imagery Dataset.
It includes functions for loading, filtering, normalization, and feature extraction.
"""

import os
import numpy as np
import pandas as pd
import mne
from mne.datasets import sample
from mne.datasets import eegbci
from mne.io import read_raw_edf
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
    """Class for processing EEG data from MNE Sample Dataset and EEG Motor Movement/Imagery Dataset."""

    def __init__(self, data_path=None, output_path=None):
        """
        Initialize the EEG processor.

        Args:
            data_path (str, optional): Path to save downloaded data
            output_path (str, optional): Path to save processed data
        """
        self.data_path = data_path or os.path.join('data', 'eeg', 'raw')
        self.output_path = output_path or os.path.join('data', 'eeg', 'processed')
        os.makedirs(self.data_path, exist_ok=True)
        os.makedirs(self.output_path, exist_ok=True)

        # Set MNE logging level
        mne.set_log_level('WARNING')

    def load_mne_sample_data(self):
        """
        Load EEG data from MNE sample dataset.

        Returns:
            tuple: (data, labels) where data is a numpy array of shape (n_epochs, n_channels, n_times)
                  and labels is a numpy array of shape (n_epochs, 2)
        """
        logger.info("Loading data from MNE sample dataset")

        # Download MNE sample data if needed
        data_path = sample.data_path()

        # Load raw data
        raw_fname = os.path.join(data_path, 'MEG', 'sample', 'sample_audvis_raw.fif')
        raw = mne.io.read_raw_fif(raw_fname, preload=True)

        # Extract events
        events = mne.find_events(raw, stim_channel='STI 014')

        # Define event IDs
        event_id = {'auditory/left': 1, 'auditory/right': 2, 'visual/left': 3, 'visual/right': 4}

        # Extract epochs
        epochs = mne.Epochs(raw, events, event_id, tmin=-0.2, tmax=0.5, proj=True,
                          picks='eeg', baseline=(None, 0), preload=True)

        # Get data and labels
        X = epochs.get_data()  # shape: (n_epochs, n_channels, n_times)

        # Create binary labels (0: auditory, 1: visual)
        y = np.zeros((len(epochs), 2))
        for i, event in enumerate(epochs.events[:, 2]):
            if event in [1, 2]:  # auditory
                y[i, 0] = 0
            else:  # visual
                y[i, 0] = 1

            # Add a synthetic PHQ-8 score (for demonstration)
            if y[i, 0] == 0:  # auditory (non-depressed)
                y[i, 1] = np.random.randint(0, 10)  # PHQ-8 < 10: non-depressed
            else:  # visual (depressed)
                y[i, 1] = np.random.randint(10, 25)  # PHQ-8 >= 10: depressed

        logger.info(f"Loaded MNE sample data with shape {X.shape} and labels with shape {y.shape}")

        return X, y

    def load_eegbci_data(self, subjects=range(1, 5), runs=[6, 10, 14], force_download=False):
        """
        Load EEG data from EEG Motor Movement/Imagery Dataset.

        Args:
            subjects (list): List of subject IDs to load
            runs (list): List of runs to load
            force_download (bool): Whether to force download even if files exist

        Returns:
            tuple: (data, labels) where data is a numpy array of shape (n_epochs, n_channels, n_times)
                  and labels is a numpy array of shape (n_epochs, 2)
        """
        logger.info("Loading data from EEG Motor Movement/Imagery Dataset")

        all_epochs = []
        all_labels = []

        for subject in subjects:
            # Download data if needed
            eegbci.load_data(subject, runs, path=self.data_path, force_update=force_download)

            # Get file paths for the runs
            fnames = [eegbci.load_data(subject, run, path=self.data_path)[0] for run in runs]

            # Load and concatenate the runs
            raw_list = []
            for fname in fnames:
                raw = read_raw_edf(fname, preload=True)
                raw_list.append(raw)

            raw = mne.concatenate_raws(raw_list)

            # Set montage
            eegbci.standardize(raw)
            montage = mne.channels.make_standard_montage('standard_1005')
            raw.set_montage(montage)

            # Extract events
            events, _ = mne.events_from_annotations(raw)

            # Define event IDs
            # Event IDs: T0=rest, T1=left hand, T2=right hand
            event_id = {'T0': 0, 'T1': 1, 'T2': 2}

            # Extract epochs
            tmin, tmax = 0, 4  # 4 seconds of data
            epochs = mne.Epochs(raw, events, event_id, tmin, tmax, proj=True,
                              picks='eeg', baseline=None, preload=True)

            # Get data
            X = epochs.get_data()

            # Create binary labels (0: rest, 1: movement)
            y = np.zeros((len(epochs), 2))
            for i, event in enumerate(epochs.events[:, 2]):
                if event == 0:  # rest
                    y[i, 0] = 0
                else:  # movement
                    y[i, 0] = 1

                # Add a synthetic PHQ-8 score (for demonstration)
                if y[i, 0] == 0:  # rest (non-depressed)
                    y[i, 1] = np.random.randint(0, 10)  # PHQ-8 < 10: non-depressed
                else:  # movement (depressed)
                    y[i, 1] = np.random.randint(10, 25)  # PHQ-8 >= 10: depressed

            all_epochs.append(X)
            all_labels.append(y)

        # Concatenate data from all subjects
        X = np.vstack(all_epochs)
        y = np.vstack(all_labels)

        logger.info(f"Loaded EEG Motor Movement/Imagery data with shape {X.shape} and labels with shape {y.shape}")

        return X, y

    def load_data(self, dataset='mne_sample'):
        """
        Load EEG data from the specified dataset.

        Args:
            dataset (str): Dataset to load ('mne_sample' or 'eegbci')

        Returns:
            tuple: (data, labels) where data is a numpy array and labels is a numpy array
        """
        if dataset == 'mne_sample':
            return self.load_mne_sample_data()
        elif dataset == 'eegbci':
            return self.load_eegbci_data()
        else:
            raise ValueError(f"Unknown dataset: {dataset}")

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

    def process_dataset(self, dataset='mne_sample'):
        """
        Process the specified dataset.

        Args:
            dataset (str): Dataset to process ('mne_sample' or 'eegbci')

        Returns:
            tuple: (X, y) where X is the feature matrix and y is the label vector
        """
        logger.info(f"Processing {dataset} dataset")

        try:
            # Load data
            eeg_data, labels = self.load_data(dataset)

            # Preprocess EEG data
            preprocessed_data = self.preprocess_eeg(eeg_data)

            # Extract features
            features = self.extract_features(preprocessed_data)

            # Save processed data
            np.save(os.path.join(self.output_path, f'{dataset}_features.npy'), features)
            np.save(os.path.join(self.output_path, f'{dataset}_labels.npy'), labels)

            logger.info(f"Finished processing {dataset} dataset")

            return features, labels
        except Exception as e:
            logger.error(f"Error processing {dataset} dataset: {e}")
            raise

    def process_all_datasets(self):
        """
        Process all available datasets.

        Returns:
            dict: Dictionary containing features and labels for each dataset
        """
        logger.info("Processing all datasets")

        results = {}

        # Process MNE sample dataset
        try:
            features, labels = self.process_dataset('mne_sample')
            results['mne_sample'] = {'features': features, 'labels': labels}
        except Exception as e:
            logger.error(f"Error processing MNE sample dataset: {e}")

        # Process EEG Motor Movement/Imagery dataset
        try:
            features, labels = self.process_dataset('eegbci')
            results['eegbci'] = {'features': features, 'labels': labels}
        except Exception as e:
            logger.error(f"Error processing EEG Motor Movement/Imagery dataset: {e}")

        # Combine datasets if both are available
        if 'mne_sample' in results and 'eegbci' in results:
            combined_features = np.vstack([results['mne_sample']['features'], results['eegbci']['features']])
            combined_labels = np.vstack([results['mne_sample']['labels'], results['eegbci']['labels']])

            results['combined'] = {'features': combined_features, 'labels': combined_labels}

            # Save combined data
            np.save(os.path.join(self.output_path, 'combined_features.npy'), combined_features)
            np.save(os.path.join(self.output_path, 'combined_labels.npy'), combined_labels)

            logger.info("Combined datasets successfully")

        logger.info("Finished processing all datasets")

        return results

    def create_dataset_splits(self, dataset='combined', test_size=0.2, val_size=0.1, random_state=42):
        """
        Create train/val/test splits from the processed data.

        Args:
            dataset (str): Dataset to use ('mne_sample', 'eegbci', or 'combined')
            test_size (float): Proportion of data to use for testing
            val_size (float): Proportion of data to use for validation
            random_state (int): Random seed for reproducibility

        Returns:
            dict: Dictionary containing the data splits
        """
        from sklearn.model_selection import train_test_split

        logger.info(f"Creating dataset splits for {dataset} dataset")

        # Load processed data
        try:
            X = np.load(os.path.join(self.output_path, f'{dataset}_features.npy'))
            y = np.load(os.path.join(self.output_path, f'{dataset}_labels.npy'))
        except FileNotFoundError:
            logger.error(f"Processed data for {dataset} not found. Processing dataset first.")
            X, y = self.process_dataset(dataset)

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
        dataset_dict = {
            'X_train': X_train,
            'y_train': y_train,
            'X_val': X_val,
            'y_val': y_val,
            'X_test': X_test,
            'y_test': y_test
        }

        # Save dataset splits
        with open(os.path.join(self.output_path, f'{dataset}_dataset.pkl'), 'wb') as f:
            pickle.dump(dataset_dict, f)

        logger.info(f"Finished creating dataset splits for {dataset} dataset")

        return dataset_dict


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
            processor.process_all_datasets()
            processor.create_dataset_splits()
