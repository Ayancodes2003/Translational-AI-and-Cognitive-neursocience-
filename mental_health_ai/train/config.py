"""
Configuration Module

This module contains configuration settings for training and evaluation.
"""

import os
import yaml
import argparse
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Config:
    """Configuration class for training and evaluation."""
    
    def __init__(self, config_path=None):
        """
        Initialize the configuration.
        
        Args:
            config_path (str, optional): Path to the configuration file
        """
        # Default configuration
        self.config = {
            # Data settings
            'data': {
                'eeg_path': 'data/eeg/processed',
                'audio_path': 'data/audio/processed',
                'text_path': 'data/text/processed',
                'fusion_path': 'data/fusion/processed'
            },
            
            # Model settings
            'model': {
                'eeg': {
                    'type': 'cnn',
                    'hidden_dims': [128, 64],
                    'dropout_rate': 0.5
                },
                'audio': {
                    'type': 'cnn',
                    'hidden_dims': [128, 64],
                    'dropout_rate': 0.5
                },
                'text': {
                    'type': 'cnn',
                    'hidden_dims': [128, 64],
                    'dropout_rate': 0.5
                },
                'fusion': {
                    'type': 'early',
                    'hidden_dims': [256, 128, 64],
                    'dropout_rate': 0.5
                }
            },
            
            # Training settings
            'training': {
                'batch_size': 32,
                'num_epochs': 50,
                'learning_rate': 0.001,
                'weight_decay': 0.0001,
                'patience': 10,
                'early_stopping': True,
                'model_save_path': 'models/saved'
            },
            
            # Evaluation settings
            'evaluation': {
                'batch_size': 32,
                'detailed': True,
                'risk_levels': True,
                'clinical_report': True,
                'output_dir': 'results'
            }
        }
        
        # Load configuration from file if provided
        if config_path and os.path.exists(config_path):
            self.load_config(config_path)
    
    def load_config(self, config_path):
        """
        Load configuration from a YAML file.
        
        Args:
            config_path (str): Path to the configuration file
        """
        try:
            with open(config_path, 'r') as f:
                loaded_config = yaml.safe_load(f)
            
            # Update configuration
            self._update_config(self.config, loaded_config)
            
            logger.info(f"Loaded configuration from {config_path}")
        except Exception as e:
            logger.error(f"Error loading configuration from {config_path}: {e}")
    
    def _update_config(self, config, updates):
        """
        Recursively update configuration.
        
        Args:
            config (dict): Configuration to update
            updates (dict): Updates to apply
        """
        for key, value in updates.items():
            if key in config and isinstance(config[key], dict) and isinstance(value, dict):
                self._update_config(config[key], value)
            else:
                config[key] = value
    
    def save_config(self, config_path):
        """
        Save configuration to a YAML file.
        
        Args:
            config_path (str): Path to save the configuration
        """
        try:
            with open(config_path, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False)
            
            logger.info(f"Saved configuration to {config_path}")
        except Exception as e:
            logger.error(f"Error saving configuration to {config_path}: {e}")
    
    def get_config(self):
        """
        Get the configuration.
        
        Returns:
            dict: Configuration
        """
        return self.config
    
    def get_data_config(self):
        """
        Get the data configuration.
        
        Returns:
            dict: Data configuration
        """
        return self.config['data']
    
    def get_model_config(self, modality):
        """
        Get the model configuration for a specific modality.
        
        Args:
            modality (str): Modality ('eeg', 'audio', 'text', or 'fusion')
        
        Returns:
            dict: Model configuration
        """
        return self.config['model'][modality]
    
    def get_training_config(self):
        """
        Get the training configuration.
        
        Returns:
            dict: Training configuration
        """
        return self.config['training']
    
    def get_evaluation_config(self):
        """
        Get the evaluation configuration.
        
        Returns:
            dict: Evaluation configuration
        """
        return self.config['evaluation']


def parse_args():
    """
    Parse command line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description='Configuration for training and evaluation')
    
    # Configuration file
    parser.add_argument('--config', type=str, default=None,
                        help='Path to the configuration file')
    
    # Data arguments
    parser.add_argument('--eeg_path', type=str, default=None,
                        help='Path to the processed EEG data')
    parser.add_argument('--audio_path', type=str, default=None,
                        help='Path to the processed audio data')
    parser.add_argument('--text_path', type=str, default=None,
                        help='Path to the processed text data')
    parser.add_argument('--fusion_path', type=str, default=None,
                        help='Path to the processed fusion data')
    
    # Model arguments
    parser.add_argument('--modality', type=str, default=None,
                        choices=['eeg', 'audio', 'text', 'fusion'],
                        help='Modality to use')
    parser.add_argument('--model_type', type=str, default=None,
                        help='Model type to use')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=None,
                        help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=None,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=None,
                        help='Weight decay')
    parser.add_argument('--patience', type=int, default=None,
                        help='Patience for early stopping')
    parser.add_argument('--early_stopping', action='store_true',
                        help='Whether to use early stopping')
    parser.add_argument('--model_save_path', type=str, default=None,
                        help='Path to save the model')
    
    # Evaluation arguments
    parser.add_argument('--eval_batch_size', type=int, default=None,
                        help='Batch size for evaluation')
    parser.add_argument('--detailed', action='store_true',
                        help='Whether to compute detailed metrics and plots')
    parser.add_argument('--risk_levels', action='store_true',
                        help='Whether to evaluate risk levels')
    parser.add_argument('--clinical_report', action='store_true',
                        help='Whether to generate a clinical report')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory to save evaluation results')
    
    return parser.parse_args()


def update_config_from_args(config, args):
    """
    Update configuration from command line arguments.
    
    Args:
        config (Config): Configuration object
        args (argparse.Namespace): Parsed arguments
    
    Returns:
        Config: Updated configuration object
    """
    # Update data configuration
    if args.eeg_path:
        config.config['data']['eeg_path'] = args.eeg_path
    if args.audio_path:
        config.config['data']['audio_path'] = args.audio_path
    if args.text_path:
        config.config['data']['text_path'] = args.text_path
    if args.fusion_path:
        config.config['data']['fusion_path'] = args.fusion_path
    
    # Update model configuration
    if args.modality and args.model_type:
        config.config['model'][args.modality]['type'] = args.model_type
    
    # Update training configuration
    if args.batch_size:
        config.config['training']['batch_size'] = args.batch_size
    if args.num_epochs:
        config.config['training']['num_epochs'] = args.num_epochs
    if args.learning_rate:
        config.config['training']['learning_rate'] = args.learning_rate
    if args.weight_decay:
        config.config['training']['weight_decay'] = args.weight_decay
    if args.patience:
        config.config['training']['patience'] = args.patience
    if args.early_stopping:
        config.config['training']['early_stopping'] = args.early_stopping
    if args.model_save_path:
        config.config['training']['model_save_path'] = args.model_save_path
    
    # Update evaluation configuration
    if args.eval_batch_size:
        config.config['evaluation']['batch_size'] = args.eval_batch_size
    if args.detailed:
        config.config['evaluation']['detailed'] = args.detailed
    if args.risk_levels:
        config.config['evaluation']['risk_levels'] = args.risk_levels
    if args.clinical_report:
        config.config['evaluation']['clinical_report'] = args.clinical_report
    if args.output_dir:
        config.config['evaluation']['output_dir'] = args.output_dir
    
    return config


def get_config():
    """
    Get configuration from file and command line arguments.
    
    Returns:
        Config: Configuration object
    """
    # Parse command line arguments
    args = parse_args()
    
    # Create configuration object
    config = Config(args.config)
    
    # Update configuration from command line arguments
    config = update_config_from_args(config, args)
    
    return config


if __name__ == '__main__':
    # Get configuration
    config = get_config()
    
    # Print configuration
    import json
    print(json.dumps(config.get_config(), indent=4))
