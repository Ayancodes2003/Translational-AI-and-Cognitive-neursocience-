"""
Example Usage of Mental Health AI

This script demonstrates how to use the Mental Health AI system programmatically.
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

# Import from simple_demo
from simple_demo import generate_synthetic_data, create_dataset_splits, SimpleModel, train_model, evaluate_model, generate_clinical_report


def main():
    """Main function."""
    print("Mental Health AI - Example Usage")
    print("=" * 50)
    
    # Step 1: Generate synthetic data
    print("\nStep 1: Generate synthetic data")
    X, y = generate_synthetic_data(n_samples=1000, n_features=50)
    print(f"Generated synthetic data with shape: {X.shape}")
    print(f"Generated labels with shape: {y.shape}")
    
    # Print class distribution
    class_counts = np.bincount(y[:, 0].astype(int))
    print(f"Class distribution: {class_counts[0]} non-depressed, {class_counts[1]} depressed")
    
    # Step 2: Create dataset splits
    print("\nStep 2: Create dataset splits")
    dataset = create_dataset_splits(X, y)
    print(f"Training set: {dataset['X_train'].shape}")
    print(f"Validation set: {dataset['X_val'].shape}")
    print(f"Test set: {dataset['X_test'].shape}")
    
    # Step 3: Train model
    print("\nStep 3: Train model")
    model, device = train_model(dataset, input_dim=X.shape[1], num_epochs=10, batch_size=32, learning_rate=0.001)
    print(f"Model trained on device: {device}")
    
    # Step 4: Evaluate model
    print("\nStep 4: Evaluate model")
    metrics = evaluate_model(model, dataset, device)
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")
    print(f"Confusion Matrix:\n{metrics['confusion_matrix']}")
    
    # Step 5: Generate clinical report
    print("\nStep 5: Generate clinical report")
    report = generate_clinical_report(model, dataset, device, sample_idx=0)
    
    print("\nClinical Report:")
    print(f"Sample ID: {report['sample_id']}")
    print(f"Depression Probability: {report['depression_probability']:.1%}")
    print(f"Risk Level: {report['risk_level']}")
    print(f"True Label: {'Depressed' if report['true_label'] == 1 else 'Not Depressed'}")
    print(f"True PHQ-8 Score: {report['true_phq8_score']}")
    
    print("\nModality Contributions:")
    for modality, contribution in report['modality_contributions'].items():
        print(f"- {modality.upper()}: {contribution:.1%}")
    
    print("\nObservations:")
    for observation in report['observations']:
        print(f"- {observation}")
    
    print("\nRecommendations:")
    for suggestion in report['suggestions']:
        print(f"- {suggestion}")
    
    # Step 6: Save report to file
    print("\nStep 6: Save report to file")
    os.makedirs('results/clinical_reports', exist_ok=True)
    report_path = f"results/clinical_reports/example_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=4)
    print(f"Report saved to: {report_path}")
    
    print("\nExample usage completed successfully!")


if __name__ == "__main__":
    main()
