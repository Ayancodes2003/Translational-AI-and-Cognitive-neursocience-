"""pipeline for the Mental Health AI project.
"""

import os
import argparse
import logging
import torch
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.eeg.preprocess_eeg import EEGProcessor
from data.audio.preprocess_audio import AudioProcessor
from data.text.preprocess_text import TextProcessor
from data.fusion_dataset import MultimodalFusionDataset
from train.train import Trainer
from train.evaluate import Evaluator
from train.config import Config
from clinical_insights.risk_assessment import RiskAssessor
from clinical_insights.modality_contribution import ModalityContributionAnalyzer
from clinical_insights.report_generator import ClinicalReportGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """
    Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description='Mental Health AI Pipeline')

    # Pipeline stages
    parser.add_argument('--preprocess', action='store_true',
                        help='Run preprocessing stage')
    parser.add_argument('--train', action='store_true',
                        help='Run training stage')
    parser.add_argument('--evaluate', action='store_true',
                        help='Run evaluation stage')
    parser.add_argument('--clinical_insights', action='store_true',
                        help='Run clinical insights stage')
    parser.add_argument('--all', action='store_true',
                        help='Run all stages')

    # Configuration
    parser.add_argument('--config', type=str, default=None,
                        help='Path to configuration file')

    # Data paths
    parser.add_argument('--output_dir', type=str, default='data',
                        help='Directory to save all processed data')

    # Model paths
    parser.add_argument('--model_save_path', type=str, default='models/saved',
                        help='Path to save trained models')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to trained model for evaluation and clinical insights')

    # Training parameters
    parser.add_argument('--modality', type=str, default='fusion',
                        choices=['eeg', 'audio', 'text', 'fusion'],
                        help='Modality to train')
    parser.add_argument('--model_type', type=str, default='early',
                        help='Model type to train')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=50,
                        help='Number of epochs for training')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate for training')

    # Evaluation parameters
    parser.add_argument('--results_dir', type=str, default='results',
                        help='Directory to save evaluation results')

    # Clinical insights parameters
    parser.add_argument('--num_reports', type=int, default=10,
                        help='Number of clinical reports to generate')

    return parser.parse_args()


def preprocess_data(args):
    """
    Preprocess data for all modalities using public datasets.

    Args:
        args (argparse.Namespace): Command line arguments
    """
    logger.info("Starting preprocessing stage using public datasets")

    # Import dataset loader
    from data.dataset_loader import DatasetLoader

    # Create dataset loader
    loader = DatasetLoader(output_dir=args.output_dir)

    # Load and process all datasets
    logger.info("Loading and processing all datasets")
    eeg_dataset, audio_dataset, text_dataset, fusion_dataset = loader.load_and_process_all_data()

    logger.info("Preprocessing stage completed")


def train_models(args):
    """
    Train models for the specified modality.

    Args:
        args (argparse.Namespace): Command line arguments
    """
    logger.info(f"Starting training stage for {args.modality} modality with {args.model_type} model")

    # Load configuration
    config = Config(args.config)

    # Update configuration from command line arguments
    config.config['training']['batch_size'] = args.batch_size
    config.config['training']['num_epochs'] = args.num_epochs
    config.config['training']['learning_rate'] = args.learning_rate
    config.config['training']['model_save_path'] = args.model_save_path
    config.config['model'][args.modality]['type'] = args.model_type

    # Create model save directory if it doesn't exist
    os.makedirs(args.model_save_path, exist_ok=True)

    # Save configuration
    config.save_config(os.path.join(args.model_save_path, 'config.yaml'))

    # Import necessary modules
    from train.train import main as train_main

    # Set command line arguments for training
    sys.argv = [
        'train.py',
        f'--modality={args.modality}',
        f'--model={args.model_type}',
        f'--batch_size={args.batch_size}',
        f'--num_epochs={args.num_epochs}',
        f'--lr={args.learning_rate}',
        f'--model_save_path={args.model_save_path}'
    ]

    # Run training
    train_main()

    logger.info("Training stage completed")


def evaluate_models(args):
    """
    Evaluate trained models.

    Args:
        args (argparse.Namespace): Command line arguments
    """
    logger.info("Starting evaluation stage")

    # Import necessary modules
    from train.evaluate import main as evaluate_main

    # Set model path
    model_path = args.model_path
    if model_path is None:
        # Find the latest model
        model_dirs = [d for d in os.listdir(args.model_save_path) if os.path.isdir(os.path.join(args.model_save_path, d))]
        if not model_dirs:
            logger.error("No trained models found")
            return

        latest_model_dir = max(model_dirs, key=lambda d: os.path.getmtime(os.path.join(args.model_save_path, d)))
        model_path = os.path.join(args.model_save_path, latest_model_dir, 'best_model.pt')

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Set command line arguments for evaluation
    sys.argv = [
        'evaluate.py',
        f'--model_path={model_path}',
        f'--modality={args.modality}',
        f'--output_dir={args.output_dir}',
        '--detailed',
        '--risk_levels',
        '--clinical_report'
    ]

    # Run evaluation
    evaluate_main()

    logger.info("Evaluation stage completed")


def generate_clinical_insights(args):
    """
    Generate clinical insights from trained models.

    Args:
        args (argparse.Namespace): Command line arguments
    """
    logger.info("Starting clinical insights stage")

    # Import necessary modules
    from clinical_insights.report_generator import main as report_main

    # Set model path
    model_path = args.model_path
    if model_path is None:
        # Find the latest model
        model_dirs = [d for d in os.listdir(args.model_save_path) if os.path.isdir(os.path.join(args.model_save_path, d))]
        if not model_dirs:
            logger.error("No trained models found")
            return

        latest_model_dir = max(model_dirs, key=lambda d: os.path.getmtime(os.path.join(args.model_save_path, d)))
        model_path = os.path.join(args.model_save_path, latest_model_dir, 'best_model.pt')

    # Create output directory if it doesn't exist
    clinical_output_dir = os.path.join(args.output_dir, 'clinical_insights')
    os.makedirs(clinical_output_dir, exist_ok=True)

    # Set command line arguments for clinical insights
    sys.argv = [
        'report_generator.py',
        f'--model_path={model_path}',
        f'--data_path={args.fusion_output}',
        f'--num_samples={args.num_reports}',
        f'--output_dir={clinical_output_dir}'
    ]

    # Run clinical insights
    report_main()

    logger.info("Clinical insights stage completed")


def main():
    """Main function."""
    # Parse command line arguments
    args = parse_args()

    # Run all stages if --all is specified
    if args.all:
        args.preprocess = True
        args.train = True
        args.evaluate = True
        args.clinical_insights = True

    # Run preprocessing stage
    if args.preprocess:
        preprocess_data(args)

    # Run training stage
    if args.train:
        train_models(args)

    # Run evaluation stage
    if args.evaluate:
        evaluate_models(args)

    # Run clinical insights stage
    if args.clinical_insights:
        generate_clinical_insights(args)

    logger.info("Pipeline completed successfully")


if __name__ == "__main__":
    main()
