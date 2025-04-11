"""
Run Pipeline Script

This script runs the entire Mental Health AI pipeline.
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


def preprocess_data(args):
    """
    Preprocess data.
    
    Args:
        args: Command line arguments
    """
    logger.info("Preprocessing data")
    
    # Build command
    cmd = [
        'python', 'preprocess_all_data.py',
        '--modality', args.modality,
        '--eeg_dataset', args.eeg_dataset,
        '--output_dir', args.data_dir
    ]
    
    # Run command
    try:
        subprocess.run(cmd, check=True)
        logger.info("Successfully preprocessed data")
    except subprocess.CalledProcessError as e:
        logger.error(f"Error preprocessing data: {e}")


def train_models(args):
    """
    Train models.
    
    Args:
        args: Command line arguments
    """
    logger.info("Training models")
    
    # Build command
    cmd = [
        'python', 'train_all_models.py',
        '--modality', args.modality,
        '--eeg_dataset', args.eeg_dataset,
        '--audio_dataset', args.audio_dataset,
        '--text_dataset', args.text_dataset,
        '--num_epochs', str(args.num_epochs),
        '--batch_size', str(args.batch_size),
        '--output_dir', args.output_dir
    ]
    
    # Run command
    try:
        subprocess.run(cmd, check=True)
        logger.info("Successfully trained models")
    except subprocess.CalledProcessError as e:
        logger.error(f"Error training models: {e}")


def run_chatbot(args):
    """
    Run chatbot.
    
    Args:
        args: Command line arguments
    """
    logger.info("Running chatbot")
    
    # Build command
    cmd = [
        'streamlit', 'run', 'mental_health_ai/trained_chatbot.py'
    ]
    
    # Run command
    try:
        subprocess.run(cmd, check=True)
        logger.info("Successfully ran chatbot")
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running chatbot: {e}")


def main(args):
    """
    Main function.
    
    Args:
        args: Command line arguments
    """
    # Create directories
    os.makedirs(args.data_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run pipeline
    if args.step == 'all' or args.step == 'preprocess':
        preprocess_data(args)
    
    if args.step == 'all' or args.step == 'train':
        train_models(args)
    
    if args.step == 'all' or args.step == 'chatbot':
        run_chatbot(args)
    
    logger.info("Finished running pipeline")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run Mental Health AI pipeline')
    
    parser.add_argument('--step', type=str, default='all', choices=['all', 'preprocess', 'train', 'chatbot'],
                        help='Pipeline step to run')
    parser.add_argument('--modality', type=str, default='all', choices=['all', 'eeg', 'audio', 'text', 'fusion'],
                        help='Modality to process')
    parser.add_argument('--eeg_dataset', type=str, default='combined',
                        choices=['mne_sample', 'eegbci', 'combined'],
                        help='EEG dataset to use')
    parser.add_argument('--audio_dataset', type=str, default='combined',
                        choices=['ravdess', 'cremad', 'combined'],
                        help='Audio dataset to use')
    parser.add_argument('--text_dataset', type=str, default='combined',
                        choices=['emotion', 'go_emotions', 'tweet_eval', 'combined'],
                        help='Text dataset to use')
    parser.add_argument('--num_epochs', type=int, default=10,
                        help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--data_dir', type=str, default='data',
                        help='Data directory')
    parser.add_argument('--output_dir', type=str, default='results',
                        help='Output directory')
    
    args = parser.parse_args()
    
    main(args)
