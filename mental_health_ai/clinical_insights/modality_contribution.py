"""
Modality Contribution Module

This module analyzes the contribution of each modality to the model's predictions.
"""

import os
import numpy as np
import pandas as pd
import torch
import logging
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ModalityContributionAnalyzer:
    """Class for analyzing the contribution of each modality to the model's predictions."""
    
    def __init__(self, model, device=None, output_dir=None):
        """
        Initialize the modality contribution analyzer.
        
        Args:
            model (nn.Module): Trained fusion model
            device (torch.device, optional): Device to use for analysis
            output_dir (str, optional): Directory to save analysis results
        """
        self.model = model
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.output_dir = output_dir or 'results/modality_contribution'
        
        # Move model to device
        self.model.to(self.device)
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
    
    def analyze_ablation(self, test_loader):
        """
        Analyze modality contribution using ablation.
        
        This method evaluates the model's performance when each modality is removed.
        
        Args:
            test_loader (DataLoader): Test data loader
        
        Returns:
            dict: Modality contributions
        """
        logger.info("Analyzing modality contribution using ablation")
        
        self.model.eval()
        
        # Get modalities from the first batch
        data, _ = next(iter(test_loader))
        modalities = list(data.keys()) if isinstance(data, dict) else ['all']
        
        # Evaluate full model
        full_metrics = self._evaluate_model(test_loader)
        logger.info(f"Full model metrics: {full_metrics}")
        
        # Evaluate with each modality ablated
        ablation_metrics = {}
        
        for modality in modalities:
            logger.info(f"Ablating modality: {modality}")
            ablation_metrics[modality] = self._evaluate_model(test_loader, ablated_modality=modality)
            logger.info(f"Metrics without {modality}: {ablation_metrics[modality]}")
        
        # Calculate contribution based on performance drop
        contributions = {}
        
        for modality in modalities:
            # Calculate contribution as the drop in F1 score when the modality is ablated
            contribution = full_metrics['f1'] - ablation_metrics[modality]['f1']
            contributions[modality] = max(0, contribution)  # Ensure non-negative contribution
        
        # Normalize contributions to sum to 1
        total_contribution = sum(contributions.values())
        if total_contribution > 0:
            for modality in modalities:
                contributions[modality] /= total_contribution
        
        logger.info(f"Modality contributions: {contributions}")
        
        # Plot contributions
        self._plot_contributions(contributions, 'ablation')
        
        # Save contributions to JSON
        with open(os.path.join(self.output_dir, 'ablation_contributions.json'), 'w') as f:
            json.dump({k: float(v) for k, v in contributions.items()}, f, indent=4)
        
        return contributions
    
    def analyze_permutation(self, test_loader, n_repeats=5):
        """
        Analyze modality contribution using permutation importance.
        
        This method evaluates the model's performance when each modality is permuted.
        
        Args:
            test_loader (DataLoader): Test data loader
            n_repeats (int): Number of permutation repeats
        
        Returns:
            dict: Modality contributions
        """
        logger.info(f"Analyzing modality contribution using permutation importance with {n_repeats} repeats")
        
        self.model.eval()
        
        # Get modalities from the first batch
        data, _ = next(iter(test_loader))
        modalities = list(data.keys()) if isinstance(data, dict) else ['all']
        
        # Evaluate full model
        full_metrics = self._evaluate_model(test_loader)
        logger.info(f"Full model metrics: {full_metrics}")
        
        # Evaluate with each modality permuted
        permutation_metrics = {}
        
        for modality in modalities:
            logger.info(f"Permuting modality: {modality}")
            modality_metrics = []
            
            for i in range(n_repeats):
                metrics = self._evaluate_model(test_loader, permuted_modality=modality)
                modality_metrics.append(metrics)
            
            # Average metrics across repeats
            permutation_metrics[modality] = {
                'accuracy': np.mean([m['accuracy'] for m in modality_metrics]),
                'f1': np.mean([m['f1'] for m in modality_metrics])
            }
            
            logger.info(f"Metrics with {modality} permuted: {permutation_metrics[modality]}")
        
        # Calculate contribution based on performance drop
        contributions = {}
        
        for modality in modalities:
            # Calculate contribution as the drop in F1 score when the modality is permuted
            contribution = full_metrics['f1'] - permutation_metrics[modality]['f1']
            contributions[modality] = max(0, contribution)  # Ensure non-negative contribution
        
        # Normalize contributions to sum to 1
        total_contribution = sum(contributions.values())
        if total_contribution > 0:
            for modality in modalities:
                contributions[modality] /= total_contribution
        
        logger.info(f"Modality contributions: {contributions}")
        
        # Plot contributions
        self._plot_contributions(contributions, 'permutation')
        
        # Save contributions to JSON
        with open(os.path.join(self.output_dir, 'permutation_contributions.json'), 'w') as f:
            json.dump({k: float(v) for k, v in contributions.items()}, f, indent=4)
        
        return contributions
    
    def analyze_attention(self, test_loader):
        """
        Analyze modality contribution using attention weights.
        
        This method is only applicable for models with attention mechanisms.
        
        Args:
            test_loader (DataLoader): Test data loader
        
        Returns:
            dict: Modality contributions
        """
        logger.info("Analyzing modality contribution using attention weights")
        
        # Check if model has attention mechanism
        if not hasattr(self.model, 'attention') and not hasattr(self.model, 'cross_attention'):
            logger.warning("Model does not have attention mechanism. Skipping attention analysis.")
            return None
        
        self.model.eval()
        
        # Get modalities from the first batch
        data, _ = next(iter(test_loader))
        modalities = list(data.keys()) if isinstance(data, dict) else ['all']
        
        # Collect attention weights
        attention_weights = []
        
        with torch.no_grad():
            for data, _ in test_loader:
                # Move data to device
                if isinstance(data, dict):
                    data = {k: v.to(self.device) for k, v in data.items()}
                else:
                    data = data.to(self.device)
                
                # Forward pass
                _ = self.model(data)
                
                # Get attention weights
                if hasattr(self.model, 'attention'):
                    # For models with self-attention
                    weights = self.model.attention.cpu().numpy()
                elif hasattr(self.model, 'cross_attention'):
                    # For models with cross-attention
                    weights = {}
                    for modality in modalities:
                        if hasattr(self.model.cross_attention, modality):
                            weights[modality] = self.model.cross_attention[modality].cpu().numpy()
                
                attention_weights.append(weights)
        
        # Average attention weights across batches
        avg_attention_weights = {}
        
        if isinstance(attention_weights[0], dict):
            # For cross-attention
            for modality in modalities:
                avg_attention_weights[modality] = np.mean([w[modality] for w in attention_weights if modality in w])
        else:
            # For self-attention
            avg_attention_weights = np.mean(attention_weights, axis=0)
        
        # Calculate contributions based on attention weights
        contributions = {}
        
        if isinstance(avg_attention_weights, dict):
            # For cross-attention
            for modality in modalities:
                contributions[modality] = np.mean(avg_attention_weights[modality])
        else:
            # For self-attention
            for i, modality in enumerate(modalities):
                contributions[modality] = np.mean(avg_attention_weights[:, i])
        
        # Normalize contributions to sum to 1
        total_contribution = sum(contributions.values())
        if total_contribution > 0:
            for modality in modalities:
                contributions[modality] /= total_contribution
        
        logger.info(f"Modality contributions: {contributions}")
        
        # Plot contributions
        self._plot_contributions(contributions, 'attention')
        
        # Save contributions to JSON
        with open(os.path.join(self.output_dir, 'attention_contributions.json'), 'w') as f:
            json.dump({k: float(v) for k, v in contributions.items()}, f, indent=4)
        
        return contributions
    
    def analyze_gradient(self, test_loader):
        """
        Analyze modality contribution using gradient-based methods.
        
        This method calculates the gradient of the output with respect to each modality's input.
        
        Args:
            test_loader (DataLoader): Test data loader
        
        Returns:
            dict: Modality contributions
        """
        logger.info("Analyzing modality contribution using gradients")
        
        # Get modalities from the first batch
        data, _ = next(iter(test_loader))
        modalities = list(data.keys()) if isinstance(data, dict) else ['all']
        
        # Collect gradients
        gradients = {modality: [] for modality in modalities}
        
        for data, target in test_loader:
            # Move data to device
            if isinstance(data, dict):
                data = {k: v.to(self.device).requires_grad_(True) for k, v in data.items()}
            else:
                data = data.to(self.device).requires_grad_(True)
            
            target = target.to(self.device)
            
            # Forward pass
            self.model.zero_grad()
            output = self.model(data)
            
            # Backward pass
            output.backward(torch.ones_like(output))
            
            # Collect gradients
            if isinstance(data, dict):
                for modality in modalities:
                    grad = data[modality].grad.abs().mean().item()
                    gradients[modality].append(grad)
            else:
                grad = data.grad.abs().mean().item()
                gradients['all'].append(grad)
        
        # Calculate average gradients
        avg_gradients = {modality: np.mean(grads) for modality, grads in gradients.items()}
        
        # Calculate contributions based on gradients
        contributions = {}
        
        for modality in modalities:
            contributions[modality] = avg_gradients[modality]
        
        # Normalize contributions to sum to 1
        total_contribution = sum(contributions.values())
        if total_contribution > 0:
            for modality in modalities:
                contributions[modality] /= total_contribution
        
        logger.info(f"Modality contributions: {contributions}")
        
        # Plot contributions
        self._plot_contributions(contributions, 'gradient')
        
        # Save contributions to JSON
        with open(os.path.join(self.output_dir, 'gradient_contributions.json'), 'w') as f:
            json.dump({k: float(v) for k, v in contributions.items()}, f, indent=4)
        
        return contributions
    
    def _evaluate_model(self, test_loader, ablated_modality=None, permuted_modality=None):
        """
        Evaluate the model on the test set.
        
        Args:
            test_loader (DataLoader): Test data loader
            ablated_modality (str, optional): Modality to ablate (set to zeros)
            permuted_modality (str, optional): Modality to permute (shuffle across samples)
        
        Returns:
            dict: Evaluation metrics
        """
        self.model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for data, target in test_loader:
                # Move data to device
                if isinstance(data, dict):
                    # For multimodal data
                    data = {k: v.to(self.device) for k, v in data.items()}
                    
                    # Ablate modality if specified
                    if ablated_modality is not None:
                        data[ablated_modality] = torch.zeros_like(data[ablated_modality])
                    
                    # Permute modality if specified
                    if permuted_modality is not None:
                        idx = torch.randperm(data[permuted_modality].size(0))
                        data[permuted_modality] = data[permuted_modality][idx]
                else:
                    # For single modality data
                    data = data.to(self.device)
                
                target = target.to(self.device)
                
                # Forward pass
                output = self.model(data)
                
                # Store predictions and labels
                all_preds.append((torch.sigmoid(output) > 0.5).cpu().numpy())
                all_labels.append(target.cpu().numpy())
        
        # Concatenate predictions and labels
        all_preds = np.vstack(all_preds)
        all_labels = np.vstack(all_labels)
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels[:, 0], all_preds[:, 0])
        f1 = f1_score(all_labels[:, 0], all_preds[:, 0])
        
        return {
            'accuracy': accuracy,
            'f1': f1
        }
    
    def _plot_contributions(self, contributions, method):
        """
        Plot modality contributions.
        
        Args:
            contributions (dict): Modality contributions
            method (str): Analysis method
        """
        modalities = list(contributions.keys())
        values = list(contributions.values())
        
        plt.figure(figsize=(12, 5))
        
        # Bar plot
        plt.subplot(1, 2, 1)
        plt.bar(modalities, values)
        plt.xlabel('Modality')
        plt.ylabel('Contribution')
        plt.title(f'Modality Contribution - {method.capitalize()}')
        
        # Pie chart
        plt.subplot(1, 2, 2)
        plt.pie(values, labels=modalities, autopct='%1.1f%%')
        plt.title(f'Modality Contribution - {method.capitalize()}')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f'{method}_contributions.png'))
        plt.close()
    
    def generate_contribution_report(self, sample_data, sample_label=None):
        """
        Generate a contribution report for a single sample.
        
        Args:
            sample_data (dict): Sample data for each modality
            sample_label (torch.Tensor, optional): Sample label
        
        Returns:
            dict: Contribution report
        """
        logger.info("Generating contribution report for a single sample")
        
        self.model.eval()
        
        # Move data to device
        if isinstance(sample_data, dict):
            sample_data = {k: v.to(self.device) for k, v in sample_data.items()}
        else:
            sample_data = sample_data.to(self.device)
        
        if sample_label is not None:
            sample_label = sample_label.to(self.device)
        
        # Get modalities
        modalities = list(sample_data.keys()) if isinstance(sample_data, dict) else ['all']
        
        # Forward pass
        with torch.no_grad():
            output = self.model(sample_data)
            probability = torch.sigmoid(output).cpu().numpy()[0, 0]
        
        # Calculate contribution for each modality
        contributions = {}
        
        for modality in modalities:
            # Create a copy of the data with the current modality ablated
            ablated_data = {k: v.clone() for k, v in sample_data.items()}
            ablated_data[modality] = torch.zeros_like(ablated_data[modality])
            
            # Forward pass with ablated data
            with torch.no_grad():
                ablated_output = self.model(ablated_data)
                ablated_probability = torch.sigmoid(ablated_output).cpu().numpy()[0, 0]
            
            # Calculate contribution as the drop in probability when the modality is ablated
            contribution = probability - ablated_probability
            contributions[modality] = max(0, contribution)  # Ensure non-negative contribution
        
        # Normalize contributions to sum to 1
        total_contribution = sum(contributions.values())
        if total_contribution > 0:
            for modality in modalities:
                contributions[modality] /= total_contribution
        
        # Create contribution report
        report = {
            'probability': float(probability),
            'contributions': {k: float(v) for k, v in contributions.items()}
        }
        
        # Add true label if available
        if sample_label is not None:
            report['true_label'] = int(sample_label.cpu().numpy()[0, 0])
        
        return report
    
    def batch_generate_contribution_reports(self, test_loader, num_samples=10):
        """
        Generate contribution reports for multiple samples.
        
        Args:
            test_loader (DataLoader): Test data loader
            num_samples (int): Number of samples to generate reports for
        
        Returns:
            list: Contribution reports
        """
        logger.info(f"Generating contribution reports for {num_samples} samples")
        
        reports = []
        count = 0
        
        for data, target in test_loader:
            # Move data to device
            if isinstance(data, dict):
                data = {k: v.to(self.device) for k, v in data.items()}
            else:
                data = data.to(self.device)
            
            target = target.to(self.device)
            
            # Generate reports for each sample in the batch
            for i in range(len(target)):
                if count >= num_samples:
                    break
                
                # Extract single sample
                if isinstance(data, dict):
                    sample_data = {k: v[i:i+1] for k, v in data.items()}
                else:
                    sample_data = data[i:i+1]
                
                sample_target = target[i:i+1]
                
                # Generate report
                report = self.generate_contribution_report(sample_data, sample_target)
                reports.append(report)
                
                count += 1
            
            if count >= num_samples:
                break
        
        # Save reports to JSON
        with open(os.path.join(self.output_dir, 'contribution_reports.json'), 'w') as f:
            json.dump(reports, f, indent=4)
        
        return reports


def main():
    """Main function."""
    import argparse
    import torch
    from torch.utils.data import DataLoader
    import pickle
    import sys
    
    # Add parent directory to path
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    parser = argparse.ArgumentParser(description='Modality contribution analysis')
    
    # Model arguments
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the trained model checkpoint')
    
    # Data arguments
    parser.add_argument('--data_path', type=str, default='data/fusion/processed',
                        help='Path to the processed fusion data')
    
    # Analysis arguments
    parser.add_argument('--method', type=str, default='all',
                        choices=['ablation', 'permutation', 'attention', 'gradient', 'all'],
                        help='Analysis method')
    parser.add_argument('--output_dir', type=str, default='results/modality_contribution',
                        help='Directory to save analysis results')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(args.model_path, map_location=device)
    
    # TODO: Create model based on checkpoint and load state dict
    
    # Load test data
    with open(os.path.join(args.data_path, 'fusion_dataset.pkl'), 'rb') as f:
        dataset = pickle.load(f)
    
    # Create test loader
    # TODO: Create test loader based on dataset
    
    # Create modality contribution analyzer
    analyzer = ModalityContributionAnalyzer(model, device, args.output_dir)
    
    # Run analysis
    if args.method == 'ablation' or args.method == 'all':
        analyzer.analyze_ablation(test_loader)
    
    if args.method == 'permutation' or args.method == 'all':
        analyzer.analyze_permutation(test_loader)
    
    if args.method == 'attention' or args.method == 'all':
        analyzer.analyze_attention(test_loader)
    
    if args.method == 'gradient' or args.method == 'all':
        analyzer.analyze_gradient(test_loader)
    
    # Generate contribution reports
    analyzer.batch_generate_contribution_reports(test_loader)


if __name__ == '__main__':
    main()
