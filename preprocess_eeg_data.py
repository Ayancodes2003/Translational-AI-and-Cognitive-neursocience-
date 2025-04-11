"""
Preprocess EEG Data Script

This script preprocesses EEG data for the Mental Health AI project.
"""

import os
import sys
import logging
import argparse

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.eeg.preprocess_eeg_new import EEGProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main(args):
    """
    Main function.
    
    Args:
        args: Command line arguments
    """
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create EEG processor
    processor = EEGProcessor(args.data_dir, args.output_dir)
    
    # Process dataset
    if args.dataset == 'combined':
        logger.info("Processing all EEG datasets")
        results = processor.process_all_datasets()
        
        # Create dataset splits for combined dataset
        if 'combined' in results:
            processor.create_dataset_splits('combined')
    else:
        logger.info(f"Processing {args.dataset} dataset")
        processor.process_dataset(args.dataset)
        
        # Create dataset splits
        processor.create_dataset_splits(args.dataset)
    
    logger.info("Finished preprocessing EEG data")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Preprocess EEG data for Mental Health AI')
    
    parser.add_argument('--dataset', type=str, default='combined',
                        choices=['mne_sample', 'eegbci', 'combined'],
                        help='Dataset to process')
    parser.add_argument('--data_dir', type=str, default='data/eeg/raw',
                        help='Directory to save downloaded data')
    parser.add_argument('--output_dir', type=str, default='data/eeg/processed',
                        help='Directory to save processed data')
    
    args = parser.parse_args()
    
    main(args)
