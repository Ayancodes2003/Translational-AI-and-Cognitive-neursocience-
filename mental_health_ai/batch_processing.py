"""
Batch Processing with Mental Health AI

This script demonstrates how to use the Mental Health AI system for batch processing.
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import json
from datetime import datetime
import argparse
from tqdm import tqdm

# Import from simple_demo
from simple_demo import SimpleModel, generate_synthetic_data, create_dataset_splits, train_model, evaluate_model


def process_batch(model, data, device, batch_size=32):
    """
    Process a batch of data.
    
    Args:
        model (nn.Module): Trained model
        data (numpy.ndarray): Data to process
        device (torch.device): Device to use for processing
        batch_size (int): Batch size
    
    Returns:
        numpy.ndarray: Predictions
    """
    # Convert data to tensor
    data_tensor = torch.tensor(data, dtype=torch.float32).to(device)
    
    # Initialize predictions
    predictions = []
    
    # Process in batches
    model.eval()
    with torch.no_grad():
        for i in range(0, len(data), batch_size):
            # Get batch
            batch = data_tensor[i:i+batch_size]
            
            # Make predictions
            outputs = model(batch)
            probs = torch.sigmoid(outputs).cpu().numpy()
            
            # Add to predictions
            predictions.append(probs)
    
    # Concatenate predictions
    predictions = np.concatenate(predictions)
    
    return predictions


def generate_batch_reports(predictions, risk_threshold_low=0.3, risk_threshold_high=0.7):
    """
    Generate reports for a batch of predictions.
    
    Args:
        predictions (numpy.ndarray): Predictions
        risk_threshold_low (float): Threshold for low risk
        risk_threshold_high (float): Threshold for high risk
    
    Returns:
        list: List of reports
    """
    # Initialize reports
    reports = []
    
    # Generate reports
    for i, prob in enumerate(predictions):
        # Determine risk level
        if prob < risk_threshold_low:
            risk_level = 'Low'
        elif prob < risk_threshold_high:
            risk_level = 'Moderate'
        else:
            risk_level = 'High'
        
        # Create report
        report = {
            'sample_id': i,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'depression_probability': float(prob),
            'risk_level': risk_level
        }
        
        # Add observations
        observations = []
        
        if prob < risk_threshold_low:
            observations.append("Low probability of depression detected.")
        elif prob < risk_threshold_high:
            observations.append("Moderate probability of depression detected.")
        else:
            observations.append("High probability of depression detected.")
        
        if risk_level == 'Low':
            observations.append("Overall risk assessment indicates low risk for depression. Continued monitoring is recommended.")
        elif risk_level == 'Moderate':
            observations.append("Overall risk assessment indicates moderate risk for depression. Regular monitoring is recommended.")
        else:
            observations.append("Overall risk assessment indicates high risk for depression. Professional intervention is recommended.")
        
        report['observations'] = observations
        
        # Add suggestions
        suggestions = []
        
        if risk_level == 'Low':
            suggestions.extend([
                "Continue regular self-monitoring",
                "Maintain healthy lifestyle habits",
                "Practice stress management techniques"
            ])
        elif risk_level == 'Moderate':
            suggestions.extend([
                "Consider consulting a mental health professional",
                "Increase self-care activities",
                "Monitor mood changes",
                "Practice mindfulness and relaxation techniques"
            ])
        else:
            suggestions.extend([
                "Urgent consultation with a mental health professional",
                "Consider therapy or counseling",
                "Establish a support network",
                "Monitor symptoms closely",
                "Develop a safety plan if needed"
            ])
        
        report['suggestions'] = suggestions
        
        # Add to reports
        reports.append(report)
    
    return reports


def save_batch_reports(reports, output_dir):
    """
    Save batch reports to files.
    
    Args:
        reports (list): List of reports
        output_dir (str): Directory to save reports
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save reports
    for report in reports:
        # Create filename
        filename = f"report_{report['sample_id']}.json"
        filepath = os.path.join(output_dir, filename)
        
        # Save report
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=4)
    
    # Save summary
    summary = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'num_reports': len(reports),
        'risk_levels': {
            'Low': sum(1 for report in reports if report['risk_level'] == 'Low'),
            'Moderate': sum(1 for report in reports if report['risk_level'] == 'Moderate'),
            'High': sum(1 for report in reports if report['risk_level'] == 'High')
        },
        'average_probability': sum(report['depression_probability'] for report in reports) / len(reports)
    }
    
    # Save summary
    with open(os.path.join(output_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=4)
    
    return summary


def main():
    """Main function."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Batch Processing with Mental Health AI')
    parser.add_argument('--input', type=str, help='Path to input data file (CSV or NPY)')
    parser.add_argument('--model', type=str, help='Path to trained model')
    parser.add_argument('--output_dir', type=str, default='results/batch_reports', help='Directory to save reports')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--num_samples', type=int, default=100, help='Number of samples to generate (if no input file)')
    
    args = parser.parse_args()
    
    print("Mental Health AI - Batch Processing")
    print("=" * 50)
    
    # Step 1: Load or generate data
    print("\nStep 1: Load or generate data")
    
    if args.input and os.path.exists(args.input):
        # Load data from file
        print(f"Loading data from {args.input}")
        
        if args.input.endswith('.csv'):
            # Load CSV file
            df = pd.read_csv(args.input)
            data = df.values
        elif args.input.endswith('.npy'):
            # Load NPY file
            data = np.load(args.input)
        else:
            print(f"Unsupported file format: {args.input}")
            print("Using synthetic data instead")
            data, _ = generate_synthetic_data(n_samples=args.num_samples, n_features=50)
    else:
        # Generate synthetic data
        print(f"Generating synthetic data with {args.num_samples} samples")
        data, _ = generate_synthetic_data(n_samples=args.num_samples, n_features=50)
    
    print(f"Data shape: {data.shape}")
    
    # Step 2: Load or create model
    print("\nStep 2: Load or create model")
    
    # Get device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if args.model and os.path.exists(args.model):
        # Load model from file
        print(f"Loading model from {args.model}")
        model = torch.load(args.model, map_location=device)
    else:
        # Create and train model
        print("Creating and training new model")
        
        # Generate labels for training
        X, y = generate_synthetic_data(n_samples=1000, n_features=data.shape[1])
        
        # Create dataset splits
        dataset = create_dataset_splits(X, y)
        
        # Train model
        model, _ = train_model(dataset, input_dim=data.shape[1], num_epochs=10, batch_size=args.batch_size)
    
    # Step 3: Process batch
    print("\nStep 3: Process batch")
    
    # Process data in batches
    predictions = process_batch(model, data, device, batch_size=args.batch_size)
    
    print(f"Processed {len(predictions)} samples")
    print(f"Average depression probability: {np.mean(predictions):.4f}")
    print(f"Min depression probability: {np.min(predictions):.4f}")
    print(f"Max depression probability: {np.max(predictions):.4f}")
    
    # Step 4: Generate reports
    print("\nStep 4: Generate reports")
    
    # Generate reports
    reports = generate_batch_reports(predictions)
    
    print(f"Generated {len(reports)} reports")
    
    # Count risk levels
    risk_counts = {
        'Low': sum(1 for report in reports if report['risk_level'] == 'Low'),
        'Moderate': sum(1 for report in reports if report['risk_level'] == 'Moderate'),
        'High': sum(1 for report in reports if report['risk_level'] == 'High')
    }
    
    print(f"Risk level distribution:")
    for risk_level, count in risk_counts.items():
        print(f"- {risk_level}: {count} ({count/len(reports):.1%})")
    
    # Step 5: Save reports
    print("\nStep 5: Save reports")
    
    # Save reports
    summary = save_batch_reports(reports, args.output_dir)
    
    print(f"Saved {len(reports)} reports to {args.output_dir}")
    print(f"Summary saved to {os.path.join(args.output_dir, 'summary.json')}")
    
    # Step 6: Visualize results
    print("\nStep 6: Visualize results")
    
    # Create output directory for plots
    plots_dir = os.path.join(args.output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Plot histogram of depression probabilities
    plt.figure(figsize=(10, 6))
    plt.hist(predictions, bins=20, alpha=0.7)
    plt.axvline(x=0.3, color='g', linestyle='--', label='Low Risk Threshold')
    plt.axvline(x=0.7, color='r', linestyle='--', label='High Risk Threshold')
    plt.xlabel('Depression Probability')
    plt.ylabel('Count')
    plt.title('Distribution of Depression Probabilities')
    plt.legend()
    plt.savefig(os.path.join(plots_dir, 'probability_histogram.png'))
    
    # Plot pie chart of risk levels
    plt.figure(figsize=(8, 8))
    plt.pie(risk_counts.values(), labels=risk_counts.keys(), autopct='%1.1f%%')
    plt.title('Risk Level Distribution')
    plt.savefig(os.path.join(plots_dir, 'risk_level_pie.png'))
    
    print(f"Saved plots to {plots_dir}")
    
    print("\nBatch processing completed successfully!")


if __name__ == "__main__":
    main()
