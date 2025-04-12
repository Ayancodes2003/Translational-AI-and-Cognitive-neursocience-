"""
Compare All Modalities

This script compares the best models from each modality.
"""

import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create output directory
os.makedirs('results/comparison', exist_ok=True)

def load_metrics(modality):
    """Load metrics for a modality."""
    try:
        with open(f'results/{modality}/metrics.json', 'r') as f:
            metrics = json.load(f)
        return metrics
    except Exception as e:
        logger.error(f"Error loading metrics for {modality}: {e}")
        return []

def get_best_model(metrics_list, metric='f1'):
    """Get the best model based on a metric."""
    if not metrics_list:
        return None
    
    return max(metrics_list, key=lambda x: x[metric])

def compare_modalities():
    """Compare modalities."""
    logger.info("Comparing modalities")
    
    # Load metrics for each modality
    eeg_metrics = load_metrics('eeg')
    audio_metrics = load_metrics('audio')
    text_metrics = load_metrics('text')
    
    # Get best model for each modality
    best_eeg = get_best_model(eeg_metrics)
    best_audio = get_best_model(audio_metrics)
    best_text = get_best_model(text_metrics)
    
    if not all([best_eeg, best_audio, best_text]):
        logger.error("Could not find best models for all modalities")
        return
    
    # Add modality to each model
    best_eeg['modality'] = 'EEG'
    best_audio['modality'] = 'Audio'
    best_text['modality'] = 'Text'
    
    # Create DataFrame
    best_models = [best_eeg, best_audio, best_text]
    df = pd.DataFrame(best_models)
    
    # Plot comparison
    plt.figure(figsize=(12, 8))
    
    # Plot accuracy, precision, recall, f1, auc
    metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1', 'auc']
    
    # Bar plot
    ax = df[['modality'] + metrics_to_plot].set_index('modality').plot(kind='bar', figsize=(12, 6))
    plt.title('Cross-Modality Comparison (Best Models)')
    plt.ylabel('Score')
    plt.xlabel('Modality')
    plt.ylim(0, 1)
    plt.grid(axis='y')
    plt.legend(loc='lower right')
    
    # Add values on top of bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f')
    
    plt.tight_layout()
    plt.savefig('results/comparison/cross_modality_comparison.png')
    plt.close()
    
    # Save metrics to JSON
    with open('results/comparison/cross_modality_metrics.json', 'w') as f:
        json.dump(best_models, f, indent=4)
    
    # Create comparison table
    comparison_table = df[['modality'] + metrics_to_plot].set_index('modality')
    logger.info(f"\nCross-Modality Comparison (Best Models):")
    logger.info(f"\n{comparison_table}")
    
    # Save comparison table to CSV
    comparison_table.to_csv('results/comparison/cross_modality_comparison_table.csv')
    
    return comparison_table

def main():
    """Main function."""
    logger.info("Starting comparison of all modalities")
    
    # Compare modalities
    compare_modalities()
    
    logger.info("Comparison completed successfully!")

if __name__ == "__main__":
    main()
