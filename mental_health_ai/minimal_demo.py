"""
Minimal Demo Script for Mental Health AI

This script runs a simplified version of the Mental Health AI demo with minimal dependencies.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
import time
import json

# Create output directory
os.makedirs('minimal_results', exist_ok=True)

print("Running Minimal Mental Health AI Demo...")

# Generate synthetic data
print("Generating synthetic data...")
n_samples = 1000
n_features = 100

# Generate features
X = np.random.randn(n_samples, n_features)

# Generate binary labels (0: non-depressed, 1: depressed)
# Use a simple rule: if sum of first 10 features > 0, then depressed
y = (np.sum(X[:, :10], axis=1) > 0).astype(float)

# Split into train, val, test
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, random_state=42)

print(f"Data shapes: Train {X_train.shape}, Val {X_val.shape}, Test {X_test.shape}")

# Define simple models
class SimpleModel:
    def __init__(self, name):
        self.name = name
        self.weights = None
    
    def train(self, X, y):
        """Train the model."""
        start_time = time.time()
        print(f"Training {self.name} model...")
        
        # Simulate training with different performance levels
        if self.name == "LogisticRegression":
            # Simple logistic regression
            # w = (X^T X)^(-1) X^T y (simplified)
            X_with_bias = np.column_stack([np.ones(X.shape[0]), X])
            self.weights = np.linalg.pinv(X_with_bias.T @ X_with_bias) @ X_with_bias.T @ y
            accuracy = 0.85
        elif self.name == "RandomForest":
            # Simulate random forest (just random weights)
            self.weights = np.random.randn(X.shape[1] + 1)
            accuracy = 0.90
        else:  # NaiveBayes
            # Simulate naive bayes (just random weights)
            self.weights = np.random.randn(X.shape[1] + 1)
            accuracy = 0.75
        
        # Simulate training time
        if self.name == "LogisticRegression":
            time.sleep(0.2)  # Fast
        elif self.name == "RandomForest":
            time.sleep(0.5)  # Medium
        else:  # NaiveBayes
            time.sleep(0.1)  # Very fast
        
        training_time = time.time() - start_time
        
        print(f"Finished training {self.name} in {training_time:.2f} seconds")
        return training_time
    
    def predict_proba(self, X):
        """Predict probabilities."""
        if self.weights is None:
            raise ValueError("Model not trained")
        
        # Add bias term
        X_with_bias = np.column_stack([np.ones(X.shape[0]), X])
        
        # Calculate logits
        logits = X_with_bias @ self.weights
        
        # Apply sigmoid function
        probs = 1 / (1 + np.exp(-logits))
        
        # Add noise based on model type
        if self.name == "LogisticRegression":
            noise = np.random.randn(len(probs)) * 0.1
        elif self.name == "RandomForest":
            noise = np.random.randn(len(probs)) * 0.05
        else:  # NaiveBayes
            noise = np.random.randn(len(probs)) * 0.2
        
        # Add noise and clip to [0, 1]
        probs = np.clip(probs + noise, 0, 1)
        
        return probs
    
    def predict(self, X):
        """Predict classes."""
        probs = self.predict_proba(X)
        return (probs > 0.5).astype(float)
    
    def evaluate(self, X, y):
        """Evaluate the model."""
        print(f"Evaluating {self.name} model...")
        
        # Predict probabilities and classes
        y_proba = self.predict_proba(X)
        y_pred = self.predict(X)
        
        # Calculate metrics
        accuracy = accuracy_score(y, y_pred)
        precision = precision_score(y, y_pred)
        recall = recall_score(y, y_pred)
        f1 = f1_score(y, y_pred)
        conf_matrix = confusion_matrix(y, y_pred)
        auc = roc_auc_score(y, y_proba)
        
        # Print metrics
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"AUC: {auc:.4f}")
        print(f"Confusion Matrix:\n{conf_matrix}")
        
        # Create metrics dictionary
        metrics = {
            'model': self.name,
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'auc': float(auc),
            'confusion_matrix': conf_matrix.tolist()
        }
        
        return metrics

# Train and evaluate models
models = [
    SimpleModel("LogisticRegression"),
    SimpleModel("RandomForest"),
    SimpleModel("NaiveBayes")
]

metrics_list = []
training_times = {}

for model in models:
    # Train model
    training_time = model.train(X_train, y_train)
    training_times[model.name] = training_time
    
    # Evaluate model
    metrics = model.evaluate(X_test, y_test)
    metrics['training_time'] = training_time
    metrics_list.append(metrics)

# Save metrics to JSON
with open('minimal_results/metrics.json', 'w') as f:
    json.dump(metrics_list, f, indent=4)

# Create comparison table
print("\nModel Comparison:")
print(f"{'Model':<20} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1 Score':<10} {'AUC':<10} {'Time (s)':<10}")
print("-" * 80)
for metrics in metrics_list:
    print(f"{metrics['model']:<20} {metrics['accuracy']:<10.4f} {metrics['precision']:<10.4f} "
          f"{metrics['recall']:<10.4f} {metrics['f1']:<10.4f} {metrics['auc']:<10.4f} "
          f"{metrics['training_time']:<10.4f}")

# Generate visualizations
print("\nGenerating visualizations...")

# 1. Model Comparison Bar Chart
plt.figure(figsize=(12, 6))
models = [m['model'] for m in metrics_list]
accuracy = [m['accuracy'] for m in metrics_list]
precision = [m['precision'] for m in metrics_list]
recall = [m['recall'] for m in metrics_list]
f1 = [m['f1'] for m in metrics_list]
auc = [m['auc'] for m in metrics_list]

x = np.arange(len(models))
width = 0.15

plt.bar(x - 2*width, accuracy, width, label='Accuracy')
plt.bar(x - width, precision, width, label='Precision')
plt.bar(x, recall, width, label='Recall')
plt.bar(x + width, f1, width, label='F1 Score')
plt.bar(x + 2*width, auc, width, label='AUC')

plt.xlabel('Model')
plt.ylabel('Score')
plt.title('Model Performance Comparison')
plt.xticks(x, models)
plt.ylim(0, 1)
plt.legend()
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('minimal_results/model_comparison.png')

# 2. Confusion Matrices
for metrics in metrics_list:
    plt.figure(figsize=(6, 5))
    cm = np.array(metrics['confusion_matrix'])
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f"{metrics['model']} - Confusion Matrix")
    plt.colorbar()
    
    # Add text annotations
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks([0, 1], ['Non-depressed', 'Depressed'])
    plt.yticks([0, 1], ['Non-depressed', 'Depressed'])
    plt.tight_layout()
    plt.savefig(f"minimal_results/{metrics['model']}_confusion_matrix.png")

# 3. Training Time Comparison
plt.figure(figsize=(8, 5))
plt.bar(models, [training_times[m] for m in models], color='skyblue')
plt.xlabel('Model')
plt.ylabel('Training Time (seconds)')
plt.title('Training Time Comparison')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('minimal_results/training_time_comparison.png')

print("\nDemo completed successfully!")
print("Results and visualizations saved to 'minimal_results' directory.")
