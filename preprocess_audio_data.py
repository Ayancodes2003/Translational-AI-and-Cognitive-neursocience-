"""
Preprocess Audio Data Script

This script preprocesses audio data for the Mental Health AI project.
"""

import os
import sys
import logging
import argparse

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Add mental_health_ai directory to path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mental_health_ai'))

from mental_health_ai.data.dataset_loader import DatasetLoader

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

    # Create dataset loader
    loader = DatasetLoader(args.output_dir)

    # Load audio data
    logger.info("Loading audio data")
    audio_data, audio_labels = loader.load_audio_data()

    # Save audio data
    logger.info("Saving audio data")
    import numpy as np
    np.save(os.path.join(args.output_dir, 'audio_data.npy'), audio_data)
    np.save(os.path.join(args.output_dir, 'audio_labels.npy'), audio_labels)

    # Create dataset splits
    logger.info("Creating dataset splits")
    from sklearn.model_selection import train_test_split

    # Split into train+val and test
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        audio_data, audio_labels, test_size=0.2, random_state=42
    )

    # Split train+val into train and val
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.1/0.8, random_state=42
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
    import pickle
    with open(os.path.join(args.output_dir, 'audio_dataset.pkl'), 'wb') as f:
        pickle.dump(dataset, f)

    logger.info("Finished preprocessing audio data")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Preprocess audio data for Mental Health AI')

    parser.add_argument('--output_dir', type=str, default='data/audio/processed',
                        help='Directory to save processed data')

    args = parser.parse_args()

    main(args)
