"""
Preprocess All Data Script

This script preprocesses all data for the Mental Health AI project.
"""

import os
import sys
import subprocess
import logging
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def preprocess_eeg_data(args):
    """
    Preprocess EEG data.
    
    Args:
        args: Command line arguments
    """
    logger.info("Preprocessing EEG data")
    
    # Create output directory
    os.makedirs(os.path.join(args.output_dir, 'eeg', 'processed'), exist_ok=True)
    
    # Build command
    cmd = [
        'python', 'preprocess_eeg_data.py',
        '--dataset', args.eeg_dataset,
        '--data_dir', os.path.join(args.output_dir, 'eeg', 'raw'),
        '--output_dir', os.path.join(args.output_dir, 'eeg', 'processed')
    ]
    
    # Run command
    try:
        subprocess.run(cmd, check=True)
        logger.info("Successfully preprocessed EEG data")
    except subprocess.CalledProcessError as e:
        logger.error(f"Error preprocessing EEG data: {e}")


def preprocess_audio_data(args):
    """
    Preprocess audio data.
    
    Args:
        args: Command line arguments
    """
    logger.info("Preprocessing audio data")
    
    # Create output directory
    os.makedirs(os.path.join(args.output_dir, 'audio', 'processed'), exist_ok=True)
    
    # Build command
    cmd = [
        'python', 'preprocess_audio_data.py',
        '--output_dir', os.path.join(args.output_dir, 'audio', 'processed')
    ]
    
    # Run command
    try:
        subprocess.run(cmd, check=True)
        logger.info("Successfully preprocessed audio data")
    except subprocess.CalledProcessError as e:
        logger.error(f"Error preprocessing audio data: {e}")


def preprocess_text_data(args):
    """
    Preprocess text data.
    
    Args:
        args: Command line arguments
    """
    logger.info("Preprocessing text data")
    
    # Create output directory
    os.makedirs(os.path.join(args.output_dir, 'text', 'processed'), exist_ok=True)
    
    # Build command
    cmd = [
        'python', 'preprocess_text_data.py',
        '--output_dir', os.path.join(args.output_dir, 'text', 'processed')
    ]
    
    # Run command
    try:
        subprocess.run(cmd, check=True)
        logger.info("Successfully preprocessed text data")
    except subprocess.CalledProcessError as e:
        logger.error(f"Error preprocessing text data: {e}")


def main(args):
    """
    Main function.
    
    Args:
        args: Command line arguments
    """
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Preprocess data
    if args.modality == 'all' or args.modality == 'eeg':
        preprocess_eeg_data(args)
    
    if args.modality == 'all' or args.modality == 'audio':
        preprocess_audio_data(args)
    
    if args.modality == 'all' or args.modality == 'text':
        preprocess_text_data(args)
    
    logger.info("Finished preprocessing all data")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Preprocess all data for Mental Health AI')
    
    parser.add_argument('--modality', type=str, default='all', choices=['all', 'eeg', 'audio', 'text'],
                        help='Modality to preprocess')
    parser.add_argument('--eeg_dataset', type=str, default='combined',
                        choices=['mne_sample', 'eegbci', 'combined'],
                        help='EEG dataset to use')
    parser.add_argument('--output_dir', type=str, default='data',
                        help='Output directory')
    
    args = parser.parse_args()
    
    main(args)
