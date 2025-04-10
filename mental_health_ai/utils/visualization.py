"""
Visualization Utility Functions

This module contains utility functions for data and model visualization.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, auc
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import torch
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def plot_training_history(history, path=None):
    """
    Plot training history.
    
    Args:
        history (dict): Training history
        path (str, optional): Path to save the plot
    """
    plt.figure(figsize=(12, 8))
    
    # Plot loss
    plt.subplot(2, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss')
    plt.legend()
    
    # Plot accuracy
    plt.subplot(2, 2, 2)
    plt.plot(history['train_acc'], label='Train Acc')
    plt.plot(history['val_acc'], label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy')
    plt.legend()
    
    # Plot F1 score
    plt.subplot(2, 2, 3)
    plt.plot(history['train_f1'], label='Train F1')
    plt.plot(history['val_f1'], label='Val F1')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.title('F1 Score')
    plt.legend()
    
    # Plot learning rate if available
    if 'lr' in history:
        plt.subplot(2, 2, 4)
        plt.plot(history['lr'])
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate')
    
    plt.tight_layout()
    
    if path:
        plt.savefig(path)
        logger.info(f"Saved training history plot to {path}")
    else:
        plt.show()


def plot_confusion_matrix(y_true, y_pred, path=None):
    """
    Plot confusion matrix.
    
    Args:
        y_true (numpy.ndarray): True labels
        y_pred (numpy.ndarray): Predicted labels
        path (str, optional): Path to save the plot
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    
    if path:
        plt.savefig(path)
        logger.info(f"Saved confusion matrix plot to {path}")
    else:
        plt.show()


def plot_roc_curve(y_true, y_score, path=None):
    """
    Plot ROC curve.
    
    Args:
        y_true (numpy.ndarray): True labels
        y_score (numpy.ndarray): Predicted scores
        path (str, optional): Path to save the plot
    """
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.4f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    
    if path:
        plt.savefig(path)
        logger.info(f"Saved ROC curve plot to {path}")
    else:
        plt.show()


def plot_precision_recall_curve(y_true, y_score, path=None):
    """
    Plot Precision-Recall curve.
    
    Args:
        y_true (numpy.ndarray): True labels
        y_score (numpy.ndarray): Predicted scores
        path (str, optional): Path to save the plot
    """
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    pr_auc = auc(recall, precision)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f'AUC = {pr_auc:.4f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    
    if path:
        plt.savefig(path)
        logger.info(f"Saved Precision-Recall curve plot to {path}")
    else:
        plt.show()


def plot_feature_importance(feature_names, importances, path=None):
    """
    Plot feature importance.
    
    Args:
        feature_names (list): Feature names
        importances (numpy.ndarray): Feature importances
        path (str, optional): Path to save the plot
    """
    # Sort features by importance
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(importances)), importances[indices])
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=90)
    plt.xlabel('Feature')
    plt.ylabel('Importance')
    plt.title('Feature Importance')
    plt.tight_layout()
    
    if path:
        plt.savefig(path)
        logger.info(f"Saved feature importance plot to {path}")
    else:
        plt.show()


def plot_tsne(features, labels, path=None, perplexity=30, n_iter=1000):
    """
    Plot t-SNE visualization of features.
    
    Args:
        features (numpy.ndarray): Feature matrix
        labels (numpy.ndarray): Label vector
        path (str, optional): Path to save the plot
        perplexity (int): Perplexity parameter for t-SNE
        n_iter (int): Number of iterations for t-SNE
    """
    logger.info(f"Computing t-SNE with perplexity={perplexity}, n_iter={n_iter}")
    
    # Compute t-SNE
    tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, random_state=42)
    features_tsne = tsne.fit_transform(features)
    
    # Plot t-SNE
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(features_tsne[:, 0], features_tsne[:, 1], c=labels, cmap='viridis', alpha=0.7)
    plt.colorbar(scatter, label='Label')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.title('t-SNE Visualization')
    
    if path:
        plt.savefig(path)
        logger.info(f"Saved t-SNE plot to {path}")
    else:
        plt.show()


def plot_pca(features, labels, path=None, n_components=2):
    """
    Plot PCA visualization of features.
    
    Args:
        features (numpy.ndarray): Feature matrix
        labels (numpy.ndarray): Label vector
        path (str, optional): Path to save the plot
        n_components (int): Number of PCA components
    """
    logger.info(f"Computing PCA with n_components={n_components}")
    
    # Compute PCA
    pca = PCA(n_components=n_components)
    features_pca = pca.fit_transform(features)
    
    # Plot PCA
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(features_pca[:, 0], features_pca[:, 1], c=labels, cmap='viridis', alpha=0.7)
    plt.colorbar(scatter, label='Label')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
    plt.title('PCA Visualization')
    
    if path:
        plt.savefig(path)
        logger.info(f"Saved PCA plot to {path}")
    else:
        plt.show()


def plot_eeg_signals(eeg_data, channel_names=None, sampling_rate=128, path=None):
    """
    Plot EEG signals.
    
    Args:
        eeg_data (numpy.ndarray): EEG data of shape (channels, samples)
        channel_names (list, optional): Channel names
        sampling_rate (int): Sampling rate in Hz
        path (str, optional): Path to save the plot
    """
    n_channels, n_samples = eeg_data.shape
    
    # Create time axis
    time = np.arange(n_samples) / sampling_rate
    
    # Create channel names if not provided
    if channel_names is None:
        channel_names = [f'Channel {i+1}' for i in range(n_channels)]
    
    # Plot EEG signals
    plt.figure(figsize=(12, 8))
    for i in range(n_channels):
        plt.subplot(n_channels, 1, i+1)
        plt.plot(time, eeg_data[i])
        plt.ylabel(channel_names[i])
        plt.xlim(time[0], time[-1])
        
        # Only show x-axis for the bottom subplot
        if i < n_channels - 1:
            plt.xticks([])
    
    plt.xlabel('Time (s)')
    plt.tight_layout()
    
    if path:
        plt.savefig(path)
        logger.info(f"Saved EEG signals plot to {path}")
    else:
        plt.show()


def plot_spectrogram(audio_data, sampling_rate=16000, path=None):
    """
    Plot spectrogram of audio data.
    
    Args:
        audio_data (numpy.ndarray): Audio data
        sampling_rate (int): Sampling rate in Hz
        path (str, optional): Path to save the plot
    """
    plt.figure(figsize=(10, 6))
    plt.specgram(audio_data, Fs=sampling_rate, cmap='viridis')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.title('Spectrogram')
    plt.colorbar(label='Intensity (dB)')
    
    if path:
        plt.savefig(path)
        logger.info(f"Saved spectrogram plot to {path}")
    else:
        plt.show()


def plot_attention_weights(attention_weights, input_tokens=None, output_tokens=None, path=None):
    """
    Plot attention weights.
    
    Args:
        attention_weights (numpy.ndarray): Attention weights of shape (output_len, input_len)
        input_tokens (list, optional): Input tokens
        output_tokens (list, optional): Output tokens
        path (str, optional): Path to save the plot
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(attention_weights, cmap='viridis', xticklabels=input_tokens, yticklabels=output_tokens)
    plt.xlabel('Input')
    plt.ylabel('Output')
    plt.title('Attention Weights')
    
    if path:
        plt.savefig(path)
        logger.info(f"Saved attention weights plot to {path}")
    else:
        plt.show()


def plot_model_comparison(model_names, metrics, metric_name='accuracy', path=None):
    """
    Plot model comparison.
    
    Args:
        model_names (list): Model names
        metrics (list): Metrics for each model
        metric_name (str): Name of the metric
        path (str, optional): Path to save the plot
    """
    plt.figure(figsize=(10, 6))
    plt.bar(model_names, metrics)
    plt.xlabel('Model')
    plt.ylabel(metric_name.capitalize())
    plt.title(f'Model Comparison - {metric_name.capitalize()}')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    if path:
        plt.savefig(path)
        logger.info(f"Saved model comparison plot to {path}")
    else:
        plt.show()


def plot_modality_contribution(contributions, path=None):
    """
    Plot modality contribution.
    
    Args:
        contributions (dict): Modality contributions
        path (str, optional): Path to save the plot
    """
    modalities = list(contributions.keys())
    values = list(contributions.values())
    
    plt.figure(figsize=(10, 6))
    
    # Bar plot
    plt.subplot(1, 2, 1)
    plt.bar(modalities, values)
    plt.xlabel('Modality')
    plt.ylabel('Contribution')
    plt.title('Modality Contribution - Bar Plot')
    
    # Pie chart
    plt.subplot(1, 2, 2)
    plt.pie(values, labels=modalities, autopct='%1.1f%%')
    plt.title('Modality Contribution - Pie Chart')
    
    plt.tight_layout()
    
    if path:
        plt.savefig(path)
        logger.info(f"Saved modality contribution plot to {path}")
    else:
        plt.show()


def plot_risk_levels(risk_counts, path=None):
    """
    Plot risk level distribution.
    
    Args:
        risk_counts (dict): Risk level counts
        path (str, optional): Path to save the plot
    """
    risk_levels = list(risk_counts.keys())
    counts = list(risk_counts.values())
    
    plt.figure(figsize=(10, 6))
    
    # Bar plot
    plt.subplot(1, 2, 1)
    plt.bar(risk_levels, counts)
    plt.xlabel('Risk Level')
    plt.ylabel('Count')
    plt.title('Risk Level Distribution - Bar Plot')
    
    # Pie chart
    plt.subplot(1, 2, 2)
    plt.pie(counts, labels=risk_levels, autopct='%1.1f%%')
    plt.title('Risk Level Distribution - Pie Chart')
    
    plt.tight_layout()
    
    if path:
        plt.savefig(path)
        logger.info(f"Saved risk level plot to {path}")
    else:
        plt.show()


def plot_feature_correlation(features, feature_names=None, path=None):
    """
    Plot feature correlation matrix.
    
    Args:
        features (numpy.ndarray): Feature matrix
        feature_names (list, optional): Feature names
        path (str, optional): Path to save the plot
    """
    # Calculate correlation matrix
    corr_matrix = np.corrcoef(features.T)
    
    # Create feature names if not provided
    if feature_names is None:
        feature_names = [f'Feature {i+1}' for i in range(features.shape[1])]
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', xticklabels=feature_names, yticklabels=feature_names)
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    
    if path:
        plt.savefig(path)
        logger.info(f"Saved feature correlation plot to {path}")
    else:
        plt.show()


def plot_class_distribution(labels, path=None):
    """
    Plot class distribution.
    
    Args:
        labels (numpy.ndarray): Label vector
        path (str, optional): Path to save the plot
    """
    # Count class occurrences
    unique, counts = np.unique(labels, return_counts=True)
    
    plt.figure(figsize=(8, 6))
    plt.bar(['Class ' + str(int(u)) for u in unique], counts)
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.title('Class Distribution')
    
    if path:
        plt.savefig(path)
        logger.info(f"Saved class distribution plot to {path}")
    else:
        plt.show()


def plot_learning_curves(train_scores, val_scores, train_sizes=None, path=None):
    """
    Plot learning curves.
    
    Args:
        train_scores (list): Training scores
        val_scores (list): Validation scores
        train_sizes (list, optional): Training sizes
        path (str, optional): Path to save the plot
    """
    if train_sizes is None:
        train_sizes = np.linspace(0.1, 1.0, len(train_scores))
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_scores, 'o-', label='Training Score')
    plt.plot(train_sizes, val_scores, 'o-', label='Validation Score')
    plt.xlabel('Training Size')
    plt.ylabel('Score')
    plt.title('Learning Curves')
    plt.legend()
    plt.grid(True)
    
    if path:
        plt.savefig(path)
        logger.info(f"Saved learning curves plot to {path}")
    else:
        plt.show()
