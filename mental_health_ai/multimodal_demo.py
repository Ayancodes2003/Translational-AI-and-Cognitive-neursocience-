"""
Multimodal Dataset Demo

This script demonstrates how to use the multimodal datasets for mental health analysis.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.dataset_loader import DatasetLoader


def visualize_eeg_data(eeg_data, n_channels=5, n_samples=1):
    """
    Visualize EEG data.
    
    Args:
        eeg_data (numpy.ndarray): EEG data of shape (samples, channels, time)
        n_channels (int): Number of channels to visualize
        n_samples (int): Number of samples to visualize
    """
    logger.info(f"Visualizing EEG data for {n_samples} samples and {n_channels} channels")
    
    # Create figure
    fig, axes = plt.subplots(n_samples, 1, figsize=(12, 4 * n_samples))
    if n_samples == 1:
        axes = [axes]
    
    # Plot EEG data
    for i in range(n_samples):
        sample = eeg_data[i]
        
        for j in range(min(n_channels, sample.shape[0])):
            axes[i].plot(sample[j], label=f'Channel {j+1}')
        
        axes[i].set_xlabel('Time (samples)')
        axes[i].set_ylabel('Amplitude')
        axes[i].set_title(f'EEG Signal - Sample {i+1}')
        axes[i].legend()
    
    plt.tight_layout()
    plt.savefig('eeg_visualization.png')
    logger.info("Saved EEG visualization to eeg_visualization.png")


def visualize_audio_data(audio_data, n_samples=1):
    """
    Visualize audio data.
    
    Args:
        audio_data (numpy.ndarray): Audio data of shape (samples, time)
        n_samples (int): Number of samples to visualize
    """
    logger.info(f"Visualizing audio data for {n_samples} samples")
    
    # Create figure
    fig, axes = plt.subplots(n_samples, 1, figsize=(12, 4 * n_samples))
    if n_samples == 1:
        axes = [axes]
    
    # Plot audio data
    for i in range(n_samples):
        sample = audio_data[i]
        
        axes[i].plot(sample)
        axes[i].set_xlabel('Time (samples)')
        axes[i].set_ylabel('Amplitude')
        axes[i].set_title(f'Audio Signal - Sample {i+1}')
    
    plt.tight_layout()
    plt.savefig('audio_visualization.png')
    logger.info("Saved audio visualization to audio_visualization.png")


def visualize_text_data(text_data, labels, n_samples=5):
    """
    Visualize text data.
    
    Args:
        text_data (list): List of text strings
        labels (numpy.ndarray): Labels of shape (samples, 2)
        n_samples (int): Number of samples to visualize
    """
    logger.info(f"Visualizing text data for {n_samples} samples")
    
    # Create dataframe
    df = pd.DataFrame({
        'Text': text_data[:n_samples],
        'Depression': labels[:n_samples, 0],
        'PHQ-8 Score': labels[:n_samples, 1]
    })
    
    # Print dataframe
    print("\nText Data Samples:")
    print(df)
    
    # Save to CSV
    df.to_csv('text_samples.csv', index=False)
    logger.info("Saved text samples to text_samples.csv")
    
    # Create visualization
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Depression', y='PHQ-8 Score', data=df)
    plt.title('PHQ-8 Scores by Depression Label')
    plt.savefig('text_visualization.png')
    logger.info("Saved text visualization to text_visualization.png")


def visualize_label_distribution(labels):
    """
    Visualize label distribution.
    
    Args:
        labels (numpy.ndarray): Labels of shape (samples, 2)
    """
    logger.info("Visualizing label distribution")
    
    # Count depression labels
    depression_counts = np.bincount(labels[:, 0].astype(int))
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot depression label distribution
    ax1.bar(['Non-depressed', 'Depressed'], depression_counts)
    ax1.set_xlabel('Depression Label')
    ax1.set_ylabel('Count')
    ax1.set_title('Depression Label Distribution')
    
    # Plot PHQ-8 score distribution
    ax2.hist(labels[:, 1], bins=24, alpha=0.7)
    ax2.set_xlabel('PHQ-8 Score')
    ax2.set_ylabel('Count')
    ax2.set_title('PHQ-8 Score Distribution')
    
    plt.tight_layout()
    plt.savefig('label_distribution.png')
    logger.info("Saved label distribution to label_distribution.png")


def main():
    """Main function."""
    logger.info("Starting multimodal dataset demo")
    
    # Create output directory
    os.makedirs('visualizations', exist_ok=True)
    
    # Create dataset loader
    loader = DatasetLoader(output_dir='data')
    
    # Load EEG data
    logger.info("Loading EEG data")
    eeg_data, eeg_labels = loader.load_eeg_data()
    logger.info(f"Loaded EEG data with shape {eeg_data.shape} and labels with shape {eeg_labels.shape}")
    
    # Load audio data
    logger.info("Loading audio data")
    audio_data, audio_labels = loader.load_audio_data()
    logger.info(f"Loaded audio data with shape {audio_data.shape} and labels with shape {audio_labels.shape}")
    
    # Load text data
    logger.info("Loading text data")
    text_data, text_labels = loader.load_text_data()
    logger.info(f"Loaded text data with {len(text_data)} samples and labels with shape {text_labels.shape}")
    
    # Visualize EEG data
    visualize_eeg_data(eeg_data, n_channels=5, n_samples=2)
    
    # Visualize audio data
    visualize_audio_data(audio_data, n_samples=2)
    
    # Visualize text data
    visualize_text_data(text_data, text_labels, n_samples=5)
    
    # Visualize label distribution
    visualize_label_distribution(eeg_labels)
    
    logger.info("Multimodal dataset demo completed")


if __name__ == "__main__":
    main()
