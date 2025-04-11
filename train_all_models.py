"""
Train All Models Script

This script trains all models for the Mental Health AI project.
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


def train_eeg_models(args):
    """
    Train EEG models.
    
    Args:
        args: Command line arguments
    """
    logger.info("Training EEG models")
    
    # Create output directory
    os.makedirs(os.path.join(args.output_dir, 'eeg'), exist_ok=True)
    
    # Define model types
    model_types = ['eegnet', 'eegcnn', 'eeglstm']
    
    # Train each model type
    for model_type in model_types:
        logger.info(f"Training {model_type} model")
        
        # Build command
        cmd = [
            'python', 'train/train_eeg_model.py',
            '--dataset', args.eeg_dataset,
            '--model_type', model_type,
            '--num_epochs', str(args.num_epochs),
            '--batch_size', str(args.batch_size),
            '--output_dir', os.path.join(args.output_dir, 'eeg')
        ]
        
        # Run command
        try:
            subprocess.run(cmd, check=True)
            logger.info(f"Successfully trained {model_type} model")
        except subprocess.CalledProcessError as e:
            logger.error(f"Error training {model_type} model: {e}")


def train_audio_models(args):
    """
    Train audio models.
    
    Args:
        args: Command line arguments
    """
    logger.info("Training audio models")
    
    # Create output directory
    os.makedirs(os.path.join(args.output_dir, 'audio'), exist_ok=True)
    
    # Define model types
    model_types = ['audiocnn', 'audiolstm', 'audiocrnn']
    
    # Train each model type
    for model_type in model_types:
        logger.info(f"Training {model_type} model")
        
        # Build command
        cmd = [
            'python', 'train/train_audio_model.py',
            '--dataset', args.audio_dataset,
            '--model_type', model_type,
            '--num_epochs', str(args.num_epochs),
            '--batch_size', str(args.batch_size),
            '--output_dir', os.path.join(args.output_dir, 'audio')
        ]
        
        # Run command
        try:
            subprocess.run(cmd, check=True)
            logger.info(f"Successfully trained {model_type} model")
        except subprocess.CalledProcessError as e:
            logger.error(f"Error training {model_type} model: {e}")


def train_text_models(args):
    """
    Train text models.
    
    Args:
        args: Command line arguments
    """
    logger.info("Training text models")
    
    # Create output directory
    os.makedirs(os.path.join(args.output_dir, 'text'), exist_ok=True)
    
    # Define model types
    model_types = ['textcnn', 'textlstm', 'textbilstm']
    
    # Train each model type
    for model_type in model_types:
        logger.info(f"Training {model_type} model")
        
        # Build command
        cmd = [
            'python', 'train/train_text_model.py',
            '--dataset', args.text_dataset,
            '--model_type', model_type,
            '--num_epochs', str(args.num_epochs),
            '--batch_size', str(args.batch_size),
            '--output_dir', os.path.join(args.output_dir, 'text')
        ]
        
        # Run command
        try:
            subprocess.run(cmd, check=True)
            logger.info(f"Successfully trained {model_type} model")
        except subprocess.CalledProcessError as e:
            logger.error(f"Error training {model_type} model: {e}")


def train_fusion_models(args):
    """
    Train fusion models.
    
    Args:
        args: Command line arguments
    """
    logger.info("Training fusion models")
    
    # Create output directory
    os.makedirs(os.path.join(args.output_dir, 'fusion'), exist_ok=True)
    
    # Define model types
    model_types = ['earlyfusion', 'latefusion', 'hierarchicalfusion']
    
    # Train each model type
    for model_type in model_types:
        logger.info(f"Training {model_type} model")
        
        # Build command
        cmd = [
            'python', 'train/train_fusion_model.py',
            '--eeg_model', os.path.join(args.output_dir, 'eeg', 'eegnet_model.pt'),
            '--audio_model', os.path.join(args.output_dir, 'audio', 'audiocnn_model.pt'),
            '--text_model', os.path.join(args.output_dir, 'text', 'textcnn_model.pt'),
            '--model_type', model_type,
            '--num_epochs', str(args.num_epochs),
            '--batch_size', str(args.batch_size),
            '--output_dir', os.path.join(args.output_dir, 'fusion')
        ]
        
        # Run command
        try:
            subprocess.run(cmd, check=True)
            logger.info(f"Successfully trained {model_type} model")
        except subprocess.CalledProcessError as e:
            logger.error(f"Error training {model_type} model: {e}")


def main(args):
    """
    Main function.
    
    Args:
        args: Command line arguments
    """
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Train models
    if args.modality == 'all' or args.modality == 'eeg':
        train_eeg_models(args)
    
    if args.modality == 'all' or args.modality == 'audio':
        train_audio_models(args)
    
    if args.modality == 'all' or args.modality == 'text':
        train_text_models(args)
    
    if args.modality == 'all' or args.modality == 'fusion':
        train_fusion_models(args)
    
    logger.info("Finished training all models")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train all models for Mental Health AI')
    
    parser.add_argument('--modality', type=str, default='all', choices=['all', 'eeg', 'audio', 'text', 'fusion'],
                        help='Modality to train')
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
    parser.add_argument('--output_dir', type=str, default='results',
                        help='Output directory')
    
    args = parser.parse_args()
    
    main(args)
