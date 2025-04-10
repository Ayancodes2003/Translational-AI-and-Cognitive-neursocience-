"""
Risk Assessment Module

This module handles the assessment of mental health risk levels based on model predictions.
"""

import os
import numpy as np
import pandas as pd
import torch
import logging
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RiskAssessor:
    """Class for assessing mental health risk levels."""
    
    def __init__(self, threshold_low=0.3, threshold_high=0.7, output_dir=None):
        """
        Initialize the risk assessor.
        
        Args:
            threshold_low (float): Threshold for low risk
            threshold_high (float): Threshold for high risk
            output_dir (str, optional): Directory to save assessment results
        """
        self.threshold_low = threshold_low
        self.threshold_high = threshold_high
        self.output_dir = output_dir or 'results/risk_assessment'
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
    
    def assess_risk(self, probabilities):
        """
        Assess risk levels based on probabilities.
        
        Args:
            probabilities (numpy.ndarray): Predicted probabilities
        
        Returns:
            numpy.ndarray: Risk levels (0: low, 1: moderate, 2: high)
        """
        risk_levels = np.zeros(len(probabilities))
        risk_levels[(probabilities >= self.threshold_low) & (probabilities < self.threshold_high)] = 1  # Moderate risk
        risk_levels[probabilities >= self.threshold_high] = 2  # High risk
        
        return risk_levels
    
    def get_risk_distribution(self, risk_levels):
        """
        Get the distribution of risk levels.
        
        Args:
            risk_levels (numpy.ndarray): Risk levels
        
        Returns:
            dict: Risk level counts and percentages
        """
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
        
        return {
            'counts': risk_counts,
            'percentages': risk_percentages
        }
    
    def plot_risk_distribution(self, risk_distribution, path=None):
        """
        Plot the distribution of risk levels.
        
        Args:
            risk_distribution (dict): Risk level distribution
            path (str, optional): Path to save the plot
        """
        risk_counts = risk_distribution['counts']
        risk_percentages = risk_distribution['percentages']
        
        plt.figure(figsize=(12, 5))
        
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
        
        if path:
            plt.savefig(path)
            logger.info(f"Saved risk distribution plot to {path}")
        else:
            plt.show()
    
    def evaluate_risk_assessment(self, model, test_loader, device):
        """
        Evaluate risk assessment on a test set.
        
        Args:
            model (nn.Module): Trained model
            test_loader (DataLoader): Test data loader
            device (torch.device): Device to use for evaluation
        
        Returns:
            dict: Risk assessment metrics
        """
        logger.info("Evaluating risk assessment")
        
        model.eval()
        all_probs = []
        all_labels = []
        
        with torch.no_grad():
            for data, target in test_loader:
                # Move data to device
                if isinstance(data, dict):
                    # For multimodal data
                    data = {k: v.to(device) for k, v in data.items()}
                else:
                    # For single modality data
                    data = data.to(device)
                
                target = target.to(device)
                
                # Forward pass
                output = model(data)
                
                # Get probabilities
                probs = torch.sigmoid(output).cpu().numpy()
                
                # Store probabilities and labels
                all_probs.append(probs)
                all_labels.append(target.cpu().numpy())
        
        # Concatenate probabilities and labels
        all_probs = np.vstack(all_probs)[:, 0]
        all_labels = np.vstack(all_labels)[:, 0]
        
        # Assess risk levels
        risk_levels = self.assess_risk(all_probs)
        
        # Get risk distribution
        risk_distribution = self.get_risk_distribution(risk_levels)
        
        # Calculate average probability for each risk level
        avg_probs = {
            'low': np.mean(all_probs[risk_levels == 0]) if risk_distribution['counts']['low'] > 0 else 0,
            'moderate': np.mean(all_probs[risk_levels == 1]) if risk_distribution['counts']['moderate'] > 0 else 0,
            'high': np.mean(all_probs[risk_levels == 2]) if risk_distribution['counts']['high'] > 0 else 0
        }
        
        # Calculate true positive rate for each risk level
        tpr = {
            'low': np.mean(all_labels[risk_levels == 0] == 0) if risk_distribution['counts']['low'] > 0 else 0,
            'moderate': np.mean(all_labels[risk_levels == 1] == 1) if risk_distribution['counts']['moderate'] > 0 else 0,
            'high': np.mean(all_labels[risk_levels == 2] == 1) if risk_distribution['counts']['high'] > 0 else 0
        }
        
        # Create risk assessment metrics
        metrics = {
            'distribution': risk_distribution,
            'avg_probs': avg_probs,
            'tpr': tpr
        }
        
        # Log metrics
        logger.info(f"Risk Level Distribution: {risk_distribution['counts']}")
        logger.info(f"Risk Level Percentages: {risk_distribution['percentages']}")
        logger.info(f"Average Probabilities: {avg_probs}")
        logger.info(f"True Positive Rates: {tpr}")
        
        # Plot risk distribution
        self.plot_risk_distribution(
            risk_distribution, 
            path=os.path.join(self.output_dir, 'risk_distribution.png')
        )
        
        # Save metrics to JSON
        with open(os.path.join(self.output_dir, 'risk_metrics.json'), 'w') as f:
            json.dump({
                'distribution': {
                    'counts': {k: int(v) for k, v in risk_distribution['counts'].items()},
                    'percentages': {k: float(v) for k, v in risk_distribution['percentages'].items()}
                },
                'avg_probs': {k: float(v) for k, v in avg_probs.items()},
                'tpr': {k: float(v) for k, v in tpr.items()}
            }, f, indent=4)
        
        return metrics
    
    def generate_risk_report(self, probability, phq8_score=None):
        """
        Generate a risk report for a single sample.
        
        Args:
            probability (float): Predicted probability
            phq8_score (float, optional): PHQ-8 score
        
        Returns:
            dict: Risk report
        """
        # Assess risk level
        if probability < self.threshold_low:
            risk_level = 'Low'
            suggestions = [
                "Regular mental health check-ups",
                "Maintain healthy lifestyle habits",
                "Practice stress management techniques"
            ]
        elif probability < self.threshold_high:
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
        
        # Create risk report
        report = {
            'probability': float(probability),
            'risk_level': risk_level,
            'suggestions': suggestions
        }
        
        # Add PHQ-8 score if available
        if phq8_score is not None:
            report['phq8_score'] = float(phq8_score)
            
            # Add PHQ-8 interpretation
            if phq8_score < 5:
                report['phq8_interpretation'] = "Minimal or no depression"
            elif phq8_score < 10:
                report['phq8_interpretation'] = "Mild depression"
            elif phq8_score < 15:
                report['phq8_interpretation'] = "Moderate depression"
            elif phq8_score < 20:
                report['phq8_interpretation'] = "Moderately severe depression"
            else:
                report['phq8_interpretation'] = "Severe depression"
        
        return report
    
    def batch_generate_risk_reports(self, probabilities, phq8_scores=None):
        """
        Generate risk reports for multiple samples.
        
        Args:
            probabilities (numpy.ndarray): Predicted probabilities
            phq8_scores (numpy.ndarray, optional): PHQ-8 scores
        
        Returns:
            list: Risk reports
        """
        reports = []
        
        for i in range(len(probabilities)):
            if phq8_scores is not None:
                report = self.generate_risk_report(probabilities[i], phq8_scores[i])
            else:
                report = self.generate_risk_report(probabilities[i])
            
            reports.append(report)
        
        return reports
    
    def save_risk_reports(self, reports, path):
        """
        Save risk reports to a file.
        
        Args:
            reports (list): Risk reports
            path (str): Path to save the reports
        """
        with open(path, 'w') as f:
            json.dump(reports, f, indent=4)
        
        logger.info(f"Saved risk reports to {path}")


def main():
    """Main function."""
    import argparse
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    import pickle
    import sys
    
    # Add parent directory to path
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    parser = argparse.ArgumentParser(description='Risk assessment')
    
    # Model arguments
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the trained model checkpoint')
    
    # Data arguments
    parser.add_argument('--data_path', type=str, default='data/processed',
                        help='Path to the processed data')
    parser.add_argument('--modality', type=str, required=True,
                        choices=['eeg', 'audio', 'text', 'fusion'],
                        help='Modality to use')
    
    # Risk assessment arguments
    parser.add_argument('--threshold_low', type=float, default=0.3,
                        help='Threshold for low risk')
    parser.add_argument('--threshold_high', type=float, default=0.7,
                        help='Threshold for high risk')
    parser.add_argument('--output_dir', type=str, default='results/risk_assessment',
                        help='Directory to save assessment results')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(args.model_path, map_location=device)
    
    # TODO: Create model based on modality and load state dict
    
    # Load test data
    with open(os.path.join(args.data_path, f'{args.modality}_dataset.pkl'), 'rb') as f:
        dataset = pickle.load(f)
    
    # Create test loader
    test_dataset = TensorDataset(
        torch.tensor(dataset['X_test'], dtype=torch.float32),
        torch.tensor(dataset['y_test'], dtype=torch.float32)
    )
    test_loader = DataLoader(test_dataset, batch_size=32)
    
    # Create risk assessor
    risk_assessor = RiskAssessor(
        threshold_low=args.threshold_low,
        threshold_high=args.threshold_high,
        output_dir=args.output_dir
    )
    
    # Evaluate risk assessment
    metrics = risk_assessor.evaluate_risk_assessment(model, test_loader, device)
    
    # Generate and save risk reports
    model.eval()
    with torch.no_grad():
        data = torch.tensor(dataset['X_test'], dtype=torch.float32).to(device)
        output = model(data)
        probabilities = torch.sigmoid(output).cpu().numpy()[:, 0]
    
    if dataset['y_test'].shape[1] > 1:
        phq8_scores = dataset['y_test'][:, 1]
        reports = risk_assessor.batch_generate_risk_reports(probabilities, phq8_scores)
    else:
        reports = risk_assessor.batch_generate_risk_reports(probabilities)
    
    risk_assessor.save_risk_reports(reports, os.path.join(args.output_dir, 'risk_reports.json'))


if __name__ == '__main__':
    main()
