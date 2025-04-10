"""
Evaluation Module

This module handles the evaluation of trained models.
"""

import os
import argparse
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pickle
import sys
import time

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.eeg_models import EEGCNN, EEGLSTM, EEGBiLSTMAttention, EEG1DCNN, EEG1DCNNGRU, EEGTransformer
from models.audio_models import AudioCNN, AudioLSTM, AudioBiLSTMAttention, Audio2DCNN, Audio1DCNNGRU, AudioTransformer
from models.text_models import TextCNN, TextLSTM, TextBiLSTMAttention, TextCNN1D, BERTClassifier, TextTransformer
from models.fusion_models import EarlyFusionModel, LateFusionModel, IntermediateFusionModel, CrossModalAttentionFusion, HierarchicalFusionModel, EnsembleModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Evaluator:
    """Class for evaluating trained models."""
    
    def __init__(self, model, test_loader, device=None, output_dir=None):
        """
        Initialize the evaluator.
        
        Args:
            model (nn.Module): Trained model
            test_loader (DataLoader): Test data loader
            device (torch.device, optional): Device to use for evaluation
            output_dir (str, optional): Directory to save evaluation results
        """
        self.model = model
        self.test_loader = test_loader
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.output_dir = output_dir or 'results'
        
        # Move model to device
        self.model.to(self.device)
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
    
    def evaluate(self, detailed=True):
        """
        Evaluate the model.
        
        Args:
            detailed (bool): Whether to compute detailed metrics and plots
        
        Returns:
            dict: Evaluation metrics
        """
        logger.info(f"Evaluating model on device: {self.device}")
        
        self.model.eval()
        all_preds = []
        all_labels = []
        all_probs = []
        inference_times = []
        
        with torch.no_grad():
            for data, target in tqdm(self.test_loader, desc='Evaluation'):
                # Move data to device
                if isinstance(data, dict):
                    # For multimodal data
                    data = {k: v.to(self.device) for k, v in data.items()}
                else:
                    # For single modality data
                    data = data.to(self.device)
                
                target = target.to(self.device)
                
                # Measure inference time
                start_time = time.time()
                
                # Forward pass
                output = self.model(data)
                
                # Calculate inference time
                inference_time = time.time() - start_time
                inference_times.append(inference_time)
                
                # Get probabilities
                probs = torch.sigmoid(output).cpu().numpy()
                
                # Store predictions and labels
                all_probs.append(probs)
                all_preds.append((probs > 0.5).astype(int))
                all_labels.append(target.cpu().numpy())
        
        # Concatenate predictions and labels
        all_probs = np.vstack(all_probs)
        all_preds = np.vstack(all_preds)
        all_labels = np.vstack(all_labels)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(all_labels[:, 0], all_preds[:, 0]),
            'precision': precision_score(all_labels[:, 0], all_preds[:, 0]),
            'recall': recall_score(all_labels[:, 0], all_preds[:, 0]),
            'f1': f1_score(all_labels[:, 0], all_preds[:, 0]),
            'auc': roc_auc_score(all_labels[:, 0], all_probs[:, 0]),
            'avg_inference_time': np.mean(inference_times),
            'std_inference_time': np.std(inference_times)
        }
        
        # Compute confusion matrix
        cm = confusion_matrix(all_labels[:, 0], all_preds[:, 0])
        metrics['confusion_matrix'] = cm
        
        # Log metrics
        logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"Precision: {metrics['precision']:.4f}")
        logger.info(f"Recall: {metrics['recall']:.4f}")
        logger.info(f"F1 Score: {metrics['f1']:.4f}")
        logger.info(f"AUC: {metrics['auc']:.4f}")
        logger.info(f"Average Inference Time: {metrics['avg_inference_time']:.4f} seconds")
        logger.info(f"Confusion Matrix:\n{cm}")
        
        # Save metrics to CSV
        metrics_df = pd.DataFrame({
            'Metric': list(metrics.keys())[:-1],  # Exclude confusion matrix
            'Value': list(metrics.values())[:-1]
        })
        metrics_df.to_csv(os.path.join(self.output_dir, 'metrics.csv'), index=False)
        
        if detailed:
            # Plot confusion matrix
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.title('Confusion Matrix')
            plt.savefig(os.path.join(self.output_dir, 'confusion_matrix.png'))
            plt.close()
            
            # Plot ROC curve
            plt.figure(figsize=(8, 6))
            fpr, tpr, _ = roc_curve(all_labels[:, 0], all_probs[:, 0])
            plt.plot(fpr, tpr, label=f'AUC = {metrics["auc"]:.4f}')
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve')
            plt.legend()
            plt.savefig(os.path.join(self.output_dir, 'roc_curve.png'))
            plt.close()
            
            # Plot Precision-Recall curve
            plt.figure(figsize=(8, 6))
            precision, recall, _ = precision_recall_curve(all_labels[:, 0], all_probs[:, 0])
            plt.plot(recall, precision, label=f'F1 = {metrics["f1"]:.4f}')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Precision-Recall Curve')
            plt.legend()
            plt.savefig(os.path.join(self.output_dir, 'pr_curve.png'))
            plt.close()
            
            # Plot inference time distribution
            plt.figure(figsize=(8, 6))
            plt.hist(inference_times, bins=20)
            plt.axvline(metrics['avg_inference_time'], color='r', linestyle='--', 
                        label=f'Mean: {metrics["avg_inference_time"]:.4f}s')
            plt.xlabel('Inference Time (seconds)')
            plt.ylabel('Frequency')
            plt.title('Inference Time Distribution')
            plt.legend()
            plt.savefig(os.path.join(self.output_dir, 'inference_time.png'))
            plt.close()
        
        return metrics
    
    def evaluate_risk_levels(self, threshold_low=0.3, threshold_high=0.7):
        """
        Evaluate the model for risk level assessment.
        
        Args:
            threshold_low (float): Threshold for low risk
            threshold_high (float): Threshold for high risk
        
        Returns:
            dict: Risk level metrics
        """
        logger.info(f"Evaluating risk levels with thresholds: low={threshold_low}, high={threshold_high}")
        
        self.model.eval()
        all_probs = []
        all_labels = []
        
        with torch.no_grad():
            for data, target in tqdm(self.test_loader, desc='Risk Level Evaluation'):
                # Move data to device
                if isinstance(data, dict):
                    # For multimodal data
                    data = {k: v.to(self.device) for k, v in data.items()}
                else:
                    # For single modality data
                    data = data.to(self.device)
                
                target = target.to(self.device)
                
                # Forward pass
                output = self.model(data)
                
                # Get probabilities
                probs = torch.sigmoid(output).cpu().numpy()
                
                # Store probabilities and labels
                all_probs.append(probs)
                all_labels.append(target.cpu().numpy())
        
        # Concatenate probabilities and labels
        all_probs = np.vstack(all_probs)
        all_labels = np.vstack(all_labels)
        
        # Convert probabilities to risk levels
        risk_levels = np.zeros(len(all_probs))
        risk_levels[(all_probs[:, 0] >= threshold_low) & (all_probs[:, 0] < threshold_high)] = 1  # Moderate risk
        risk_levels[all_probs[:, 0] >= threshold_high] = 2  # High risk
        
        # Count samples in each risk level
        risk_counts = {
            'low': np.sum(risk_levels == 0),
            'moderate': np.sum(risk_levels == 1),
            'high': np.sum(risk_levels == 2)
        }
        
        # Calculate percentage of samples in each risk level
        risk_percentages = {
            'low': risk_counts['low'] / len(risk_levels) * 100,
            'moderate': risk_counts['moderate'] / len(risk_levels) * 100,
            'high': risk_counts['high'] / len(risk_levels) * 100
        }
        
        # Calculate average probability for each risk level
        avg_probs = {
            'low': np.mean(all_probs[risk_levels == 0, 0]) if risk_counts['low'] > 0 else 0,
            'moderate': np.mean(all_probs[risk_levels == 1, 0]) if risk_counts['moderate'] > 0 else 0,
            'high': np.mean(all_probs[risk_levels == 2, 0]) if risk_counts['high'] > 0 else 0
        }
        
        # Calculate average PHQ-8 score for each risk level (if available)
        if all_labels.shape[1] > 1:
            avg_phq8 = {
                'low': np.mean(all_labels[risk_levels == 0, 1]) if risk_counts['low'] > 0 else 0,
                'moderate': np.mean(all_labels[risk_levels == 1, 1]) if risk_counts['moderate'] > 0 else 0,
                'high': np.mean(all_labels[risk_levels == 2, 1]) if risk_counts['high'] > 0 else 0
            }
        else:
            avg_phq8 = None
        
        # Create risk level metrics
        risk_metrics = {
            'counts': risk_counts,
            'percentages': risk_percentages,
            'avg_probs': avg_probs,
            'avg_phq8': avg_phq8
        }
        
        # Log risk level metrics
        logger.info(f"Risk Level Counts: {risk_counts}")
        logger.info(f"Risk Level Percentages: {risk_percentages}")
        logger.info(f"Average Probabilities: {avg_probs}")
        if avg_phq8:
            logger.info(f"Average PHQ-8 Scores: {avg_phq8}")
        
        # Plot risk level distribution
        plt.figure(figsize=(10, 6))
        
        # Bar plot
        plt.subplot(1, 2, 1)
        plt.bar(['Low', 'Moderate', 'High'], 
                [risk_counts['low'], risk_counts['moderate'], risk_counts['high']])
        plt.xlabel('Risk Level')
        plt.ylabel('Count')
        plt.title('Risk Level Distribution')
        
        # Pie chart
        plt.subplot(1, 2, 2)
        plt.pie([risk_percentages['low'], risk_percentages['moderate'], risk_percentages['high']],
                labels=['Low', 'Moderate', 'High'],
                autopct='%1.1f%%')
        plt.title('Risk Level Percentages')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'risk_levels.png'))
        plt.close()
        
        # If PHQ-8 scores are available, plot average PHQ-8 score by risk level
        if avg_phq8:
            plt.figure(figsize=(8, 6))
            plt.bar(['Low', 'Moderate', 'High'], 
                    [avg_phq8['low'], avg_phq8['moderate'], avg_phq8['high']])
            plt.xlabel('Risk Level')
            plt.ylabel('Average PHQ-8 Score')
            plt.title('Average PHQ-8 Score by Risk Level')
            plt.savefig(os.path.join(self.output_dir, 'phq8_by_risk.png'))
            plt.close()
        
        return risk_metrics
    
    def generate_clinical_report(self, sample_idx=0):
        """
        Generate a clinical report for a sample.
        
        Args:
            sample_idx (int): Index of the sample to generate a report for
        
        Returns:
            dict: Clinical report
        """
        logger.info(f"Generating clinical report for sample {sample_idx}")
        
        self.model.eval()
        
        # Get sample
        data, target = next(iter(self.test_loader))
        
        # If batch size > 1, select the specified sample
        if isinstance(data, dict):
            # For multimodal data
            sample_data = {k: v[sample_idx:sample_idx+1].to(self.device) for k, v in data.items()}
        else:
            # For single modality data
            sample_data = data[sample_idx:sample_idx+1].to(self.device)
        
        sample_target = target[sample_idx:sample_idx+1].to(self.device)
        
        # Forward pass
        with torch.no_grad():
            output = self.model(sample_data)
            prob = torch.sigmoid(output).cpu().numpy()[0, 0]
        
        # Determine risk level
        if prob < 0.3:
            risk_level = 'Low'
            suggestions = [
                "Regular mental health check-ups",
                "Maintain healthy lifestyle habits",
                "Practice stress management techniques"
            ]
        elif prob < 0.7:
            risk_level = 'Moderate'
            suggestions = [
                "Consider consulting a mental health professional",
                "Increase self-care activities",
                "Monitor mood changes",
                "Practice mindfulness and relaxation techniques"
            ]
        else:
            risk_level = 'High'
            suggestions = [
                "Urgent consultation with a mental health professional",
                "Consider therapy or counseling",
                "Establish a support network",
                "Monitor symptoms closely",
                "Develop a safety plan if needed"
            ]
        
        # Create clinical report
        report = {
            'sample_id': sample_idx,
            'depression_probability': float(prob),
            'risk_level': risk_level,
            'suggestions': suggestions,
            'true_label': int(sample_target.cpu().numpy()[0, 0])
        }
        
        # If PHQ-8 score is available
        if sample_target.shape[1] > 1:
            report['phq8_score'] = float(sample_target.cpu().numpy()[0, 1])
        
        # Log report
        logger.info(f"Clinical Report:")
        logger.info(f"Sample ID: {report['sample_id']}")
        logger.info(f"Depression Probability: {report['depression_probability']:.4f}")
        logger.info(f"Risk Level: {report['risk_level']}")
        logger.info(f"Suggestions: {report['suggestions']}")
        logger.info(f"True Label: {report['true_label']}")
        if 'phq8_score' in report:
            logger.info(f"PHQ-8 Score: {report['phq8_score']}")
        
        # Save report to JSON
        import json
        with open(os.path.join(self.output_dir, f'clinical_report_sample_{sample_idx}.json'), 'w') as f:
            json.dump(report, f, indent=4)
        
        return report


def load_model(model_path, device=None):
    """
    Load a trained model.
    
    Args:
        model_path (str): Path to the model checkpoint
        device (torch.device, optional): Device to load the model on
    
    Returns:
        nn.Module: Loaded model
    """
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Extract model type from path
    model_type = os.path.basename(os.path.dirname(model_path))
    
    # Create model based on type
    if 'eeg_cnn' in model_type:
        model = EEGCNN(input_dim=100)  # Placeholder input_dim
    elif 'eeg_lstm' in model_type:
        model = EEGLSTM(input_dim=100)
    elif 'eeg_bilstm_attention' in model_type:
        model = EEGBiLSTMAttention(input_dim=100)
    elif 'eeg_1dcnn' in model_type:
        model = EEG1DCNN(input_dim=100)
    elif 'eeg_1dcnn_gru' in model_type:
        model = EEG1DCNNGRU(input_dim=100)
    elif 'eeg_transformer' in model_type:
        model = EEGTransformer(input_dim=100)
    elif 'audio_cnn' in model_type:
        model = AudioCNN(input_dim=100)
    elif 'audio_lstm' in model_type:
        model = AudioLSTM(input_dim=100)
    elif 'audio_bilstm_attention' in model_type:
        model = AudioBiLSTMAttention(input_dim=100)
    elif 'audio_2dcnn' in model_type:
        model = Audio2DCNN()
    elif 'audio_1dcnn_gru' in model_type:
        model = Audio1DCNNGRU(input_dim=100)
    elif 'audio_transformer' in model_type:
        model = AudioTransformer(input_dim=100)
    elif 'text_cnn' in model_type:
        model = TextCNN(input_dim=100)
    elif 'text_lstm' in model_type:
        model = TextLSTM(vocab_size=10000)
    elif 'text_bilstm_attention' in model_type:
        model = TextBiLSTMAttention(vocab_size=10000)
    elif 'text_1dcnn' in model_type:
        model = TextCNN1D(vocab_size=10000)
    elif 'text_bert' in model_type:
        model = BERTClassifier()
    elif 'text_transformer' in model_type:
        model = TextTransformer(vocab_size=10000)
    elif 'fusion_early' in model_type:
        model = EarlyFusionModel(input_dims={'eeg': 100, 'audio': 100, 'text': 100})
    elif 'fusion_late' in model_type:
        model = LateFusionModel(input_dims={'eeg': 100, 'audio': 100, 'text': 100})
    elif 'fusion_intermediate' in model_type:
        model = IntermediateFusionModel(input_dims={'eeg': 100, 'audio': 100, 'text': 100})
    elif 'fusion_cross_attention' in model_type:
        model = CrossModalAttentionFusion(input_dims={'eeg': 100, 'audio': 100, 'text': 100})
    elif 'fusion_hierarchical' in model_type:
        model = HierarchicalFusionModel(input_dims={'eeg': 100, 'audio': 100, 'text': 100})
    elif 'fusion_ensemble' in model_type:
        model = EnsembleModel(input_dims={'eeg': 100, 'audio': 100, 'text': 100})
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Load model state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Move model to device
    model.to(device)
    
    return model


def create_test_loader(data_path, modality, batch_size=32):
    """
    Create a test data loader.
    
    Args:
        data_path (str): Path to the data
        modality (str): Modality ('eeg', 'audio', 'text', or 'fusion')
        batch_size (int): Batch size
    
    Returns:
        DataLoader: Test data loader
    """
    if modality in ['eeg', 'audio', 'text']:
        # Load dataset
        with open(os.path.join(data_path, f'{modality}_dataset.pkl'), 'rb') as f:
            dataset = pickle.load(f)
        
        # Create PyTorch dataset
        test_dataset = TensorDataset(
            torch.tensor(dataset['X_test'], dtype=torch.float32),
            torch.tensor(dataset['y_test'], dtype=torch.float32)
        )
        
        # Create data loader
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    elif modality == 'fusion':
        # Load fusion dataset
        fusion_type = 'early'  # Default fusion type
        
        # Check if fusion type is specified in the data path
        if 'late' in data_path:
            fusion_type = 'late'
        elif 'intermediate' in data_path:
            fusion_type = 'intermediate'
        
        with open(os.path.join(data_path, f'fusion_{fusion_type}_dataset.pkl'), 'rb') as f:
            dataset = pickle.load(f)
        
        if fusion_type == 'early':
            # Create PyTorch dataset
            test_dataset = TensorDataset(
                torch.tensor(dataset['X_test'], dtype=torch.float32),
                torch.tensor(dataset['y_test'], dtype=torch.float32)
            )
            
            # Create data loader
            test_loader = DataLoader(test_dataset, batch_size=batch_size)
        
        else:  # 'late' or 'intermediate' fusion
            # Create custom dataset for multimodal data
            from data.fusion_dataset import MultimodalDataset
            
            test_dataset = MultimodalDataset(
                dataset['eeg_test'], dataset['audio_test'], dataset['text_test'], dataset['y_test']
            )
            
            # Create data loader
            test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    else:
        raise ValueError(f"Unknown modality: {modality}")
    
    return test_loader


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Evaluate a trained model')
    
    # Model arguments
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the trained model checkpoint')
    
    # Data arguments
    parser.add_argument('--data_path', type=str, default='data/processed',
                        help='Path to the processed data')
    parser.add_argument('--modality', type=str, required=True,
                        choices=['eeg', 'audio', 'text', 'fusion'],
                        help='Modality to use')
    
    # Evaluation arguments
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--output_dir', type=str, default='results',
                        help='Directory to save evaluation results')
    parser.add_argument('--detailed', action='store_true',
                        help='Whether to compute detailed metrics and plots')
    parser.add_argument('--risk_levels', action='store_true',
                        help='Whether to evaluate risk levels')
    parser.add_argument('--clinical_report', action='store_true',
                        help='Whether to generate a clinical report')
    parser.add_argument('--sample_idx', type=int, default=0,
                        help='Index of the sample to generate a clinical report for')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    model = load_model(args.model_path)
    
    # Create test loader
    test_loader = create_test_loader(args.data_path, args.modality, args.batch_size)
    
    # Create evaluator
    evaluator = Evaluator(model, test_loader, output_dir=args.output_dir)
    
    # Evaluate model
    metrics = evaluator.evaluate(detailed=args.detailed)
    
    # Evaluate risk levels
    if args.risk_levels:
        risk_metrics = evaluator.evaluate_risk_levels()
    
    # Generate clinical report
    if args.clinical_report:
        report = evaluator.generate_clinical_report(sample_idx=args.sample_idx)


if __name__ == '__main__':
    main()
