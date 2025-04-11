"""
Preprocess Text Data Script

This script preprocesses text data for the Mental Health AI project.
"""

import os
import sys
import logging
import argparse
import numpy as np

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Add mental_health_ai directory to path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mental_health_ai'))

from mental_health_ai.data.dataset_loader import DatasetLoader
from mental_health_ai.data.text.preprocess_text import TextProcessor

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

    # Load text data
    logger.info("Loading text data")
    text_data, text_labels = loader.load_text_data()

    # Create text processor
    processor = TextProcessor(None, args.output_dir)

    # Preprocess text data
    logger.info("Preprocessing text data")
    preprocessed_texts = processor.preprocess_dataset(text_data)

    # Extract features
    logger.info("Extracting features")
    features = []

    for text in preprocessed_texts:
        # Extract linguistic features
        text_features = processor.extract_linguistic_features([text])

        # Convert to list
        feature_list = [
            text_features.get('token_count', 0),
            text_features.get('unique_token_count', 0),
            text_features.get('lexical_diversity', 0),
            text_features.get('sentence_count', 0),
            text_features.get('avg_sentence_length', 0),
            text_features.get('depression_keyword_count', 0),
            text_features.get('pronoun_count', 0),
            text_features.get('negative_word_count', 0),
            text_features.get('first_person_pronoun_count', 0)
        ]

        features.append(feature_list)

    # Convert to numpy array
    features = np.array(features)

    # Save text data
    logger.info("Saving text data")
    np.save(os.path.join(args.output_dir, 'text_features.npy'), features)
    np.save(os.path.join(args.output_dir, 'text_labels.npy'), text_labels)

    # Create dataset splits
    logger.info("Creating dataset splits")
    from sklearn.model_selection import train_test_split

    # Split into train+val and test
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        features, text_labels, test_size=0.2, random_state=42
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
    with open(os.path.join(args.output_dir, 'text_dataset.pkl'), 'wb') as f:
        pickle.dump(dataset, f)

    logger.info("Finished preprocessing text data")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Preprocess text data for Mental Health AI')

    parser.add_argument('--output_dir', type=str, default='data/text/processed',
                        help='Directory to save processed data')

    args = parser.parse_args()

    main(args)
