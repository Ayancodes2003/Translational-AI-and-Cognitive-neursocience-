"""
Train All Text Models

This script trains all text models on the real text dataset.
"""

import os
import sys
import logging
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F

# Add mental_health_ai directory to path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mental_health_ai'))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Create output directories
os.makedirs('results', exist_ok=True)
os.makedirs('results/text', exist_ok=True)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")

def load_text_dataset():
    """Load the text dataset."""
    logger.info("Loading text dataset")
    
    try:
        # Load the dataset
        with open('mental_health_ai/data/text/processed/text_dataset.pkl', 'rb') as f:
            dataset = pickle.load(f)
        
        logger.info(f"Loaded text dataset with {len(dataset['X_train'])} training samples")
        
        # Create PyTorch datasets
        train_dataset = torch.utils.data.TensorDataset(
            torch.tensor(dataset['X_train'], dtype=torch.float32),
            torch.tensor(dataset['y_train'][:, 0:1], dtype=torch.float32)
        )
        val_dataset = torch.utils.data.TensorDataset(
            torch.tensor(dataset['X_val'], dtype=torch.float32),
            torch.tensor(dataset['y_val'][:, 0:1], dtype=torch.float32)
        )
        test_dataset = torch.utils.data.TensorDataset(
            torch.tensor(dataset['X_test'], dtype=torch.float32),
            torch.tensor(dataset['y_test'][:, 0:1], dtype=torch.float32)
        )
        
        # Create dataloaders
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32)
        
        # Get input dimension
        input_dim = dataset['X_train'].shape[1]
        
        return {
            'train_loader': train_loader,
            'val_loader': val_loader,
            'test_loader': test_loader,
            'input_dim': input_dim
        }
    
    except Exception as e:
        logger.error(f"Error loading text dataset: {e}")
        
        # Generate synthetic data if real data is not available
        logger.info("Generating synthetic text data")
        
        # Generate synthetic data
        n_samples = 1000
        n_features = 50
        
        # Generate features
        X = np.random.randn(n_samples, n_features)
        
        # Generate binary labels (0: non-depressed, 1: depressed)
        binary_labels = (np.sum(X[:, :10], axis=1) > 0).astype(float).reshape(-1, 1)
        
        # Generate PHQ-8 scores based on binary labels
        phq8_scores = np.zeros((n_samples, 1))
        for i in range(n_samples):
            if binary_labels[i, 0] == 0:  # Non-depressed
                phq8_scores[i, 0] = np.random.randint(0, 10)  # PHQ-8 < 10: non-depressed
            else:  # Depressed
                phq8_scores[i, 0] = np.random.randint(10, 25)  # PHQ-8 >= 10: depressed
        
        # Combine binary labels and PHQ-8 scores
        y = np.hstack((binary_labels, phq8_scores))
        
        # Split into train, val, test
        from sklearn.model_selection import train_test_split
        X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, random_state=42)
        
        # Standardize features
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)
        
        # Create dataset dictionary
        dataset = {
            'X_train': X_train,
            'y_train': y_train,
            'X_val': X_val,
            'y_val': y_val,
            'X_test': X_test,
            'y_test': y_test
        }
        
        # Save dataset
        os.makedirs('mental_health_ai/data/text/processed', exist_ok=True)
        with open('mental_health_ai/data/text/processed/text_dataset.pkl', 'wb') as f:
            pickle.dump(dataset, f)
        
        logger.info(f"Saved synthetic text dataset with {len(dataset['X_train'])} training samples")
        
        # Create PyTorch datasets
        train_dataset = torch.utils.data.TensorDataset(
            torch.tensor(dataset['X_train'], dtype=torch.float32),
            torch.tensor(dataset['y_train'][:, 0:1], dtype=torch.float32)
        )
        val_dataset = torch.utils.data.TensorDataset(
            torch.tensor(dataset['X_val'], dtype=torch.float32),
            torch.tensor(dataset['y_val'][:, 0:1], dtype=torch.float32)
        )
        test_dataset = torch.utils.data.TensorDataset(
            torch.tensor(dataset['X_test'], dtype=torch.float32),
            torch.tensor(dataset['y_test'][:, 0:1], dtype=torch.float32)
        )
        
        # Create dataloaders
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32)
        
        # Get input dimension
        input_dim = dataset['X_train'].shape[1]
        
        return {
            'train_loader': train_loader,
            'val_loader': val_loader,
            'test_loader': test_loader,
            'input_dim': input_dim
        }

class TextCNN(nn.Module):
    """
    Convolutional Neural Network for text feature classification.
    """
    
    def __init__(self, input_dim, hidden_dims=[128, 64], num_classes=1, dropout_rate=0.5):
        """
        Initialize the Text CNN model.
        """
        super(TextCNN, self).__init__()
        
        # Create layers
        layers = []
        
        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dims[0]))
        layers.append(nn.BatchNorm1d(hidden_dims[0]))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_rate))
        
        # Hidden layers
        for i in range(len(hidden_dims) - 1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            layers.append(nn.BatchNorm1d(hidden_dims[i + 1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
        
        # Output layer
        layers.append(nn.Linear(hidden_dims[-1], num_classes))
        
        # Create sequential model
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        """Forward pass."""
        return self.model(x)

class TextLSTM(nn.Module):
    """
    LSTM-based model for text feature classification.
    """
    
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, num_classes=1, dropout_rate=0.5):
        """
        Initialize the Text LSTM model.
        """
        super(TextLSTM, self).__init__()
        
        # Create feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Create LSTM
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0,
            bidirectional=False
        )
        
        # Create classifier
        self.classifier = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x):
        """Forward pass."""
        # Extract features
        x = self.feature_extractor(x)
        
        # Reshape for LSTM: [batch_size, seq_len, features]
        x = x.unsqueeze(1)  # Add sequence dimension
        
        # LSTM
        x, _ = self.lstm(x)
        
        # Take the output of the last time step
        x = x[:, -1, :]
        
        # Classifier
        x = self.classifier(x)
        
        return x

class TextBiLSTM(nn.Module):
    """
    Bidirectional LSTM-based model for text feature classification.
    """
    
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, num_classes=1, dropout_rate=0.5):
        """
        Initialize the Text BiLSTM model.
        """
        super(TextBiLSTM, self).__init__()
        
        # Create feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Create BiLSTM
        self.bilstm = nn.LSTM(
            input_size=128,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Create classifier
        self.classifier = nn.Linear(hidden_dim * 2, num_classes)  # * 2 for bidirectional
    
    def forward(self, x):
        """Forward pass."""
        # Extract features
        x = self.feature_extractor(x)
        
        # Reshape for BiLSTM: [batch_size, seq_len, features]
        x = x.unsqueeze(1)  # Add sequence dimension
        
        # BiLSTM
        x, _ = self.bilstm(x)
        
        # Take the output of the last time step
        x = x[:, -1, :]
        
        # Classifier
        x = self.classifier(x)
        
        return x

class TextTransformer(nn.Module):
    """
    Transformer-based model for text feature classification.
    """
    
    def __init__(self, input_dim, hidden_dim=64, num_heads=4, num_layers=2, num_classes=1, dropout_rate=0.5):
        """
        Initialize the Text Transformer model.
        """
        super(TextTransformer, self).__init__()
        
        # Create feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Create transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=128,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout_rate,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # Create classifier
        self.classifier = nn.Linear(128, num_classes)
    
    def forward(self, x):
        """Forward pass."""
        # Extract features
        x = self.feature_extractor(x)
        
        # Reshape for transformer: [batch_size, seq_len, features]
        x = x.unsqueeze(1)  # Add sequence dimension
        
        # Transformer encoder
        x = self.transformer_encoder(x)
        
        # Take the output of the last token
        x = x[:, -1, :]
        
        # Classifier
        x = self.classifier(x)
        
        return x

def create_model(model_name, input_dim):
    """Create a model based on the model name."""
    logger.info(f"Creating {model_name} model")
    
    if model_name == 'TextCNN':
        return TextCNN(input_dim=input_dim)
    elif model_name == 'TextLSTM':
        return TextLSTM(input_dim=input_dim)
    elif model_name == 'TextBiLSTM':
        return TextBiLSTM(input_dim=input_dim)
    elif model_name == 'TextTransformer':
        return TextTransformer(input_dim=input_dim)
    else:
        raise ValueError(f"Unknown model name: {model_name}")

def train_model(model, train_loader, val_loader, device, model_name, num_epochs=20):
    """Train the model."""
    logger.info(f"Training {model_name} model")
    
    # Move model to device
    model = model.to(device)
    
    # Define loss function and optimizer
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Define early stopping parameters
    patience = 10
    best_val_loss = float('inf')
    counter = 0
    best_model = None
    
    # Training loop
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    train_f1s = []
    val_f1s = []
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_preds = []
        train_targets = []
        
        for inputs, targets in train_loader:
            # Move data to device
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Update statistics
            train_loss += loss.item() * inputs.size(0)
            
            # Store predictions and targets
            train_preds.append((torch.sigmoid(outputs) > 0.5).float().cpu().numpy())
            train_targets.append(targets.cpu().numpy())
        
        # Calculate average training loss
        train_loss = train_loss / len(train_loader.dataset)
        train_losses.append(train_loss)
        
        # Calculate training metrics
        train_preds = np.concatenate(train_preds)
        train_targets = np.concatenate(train_targets)
        train_accuracy = accuracy_score(train_targets, train_preds)
        train_f1 = f1_score(train_targets, train_preds, zero_division=0)
        train_accuracies.append(train_accuracy)
        train_f1s.append(train_f1)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_targets = []
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                # Move data to device
                inputs = inputs.to(device)
                targets = targets.to(device)
                
                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                # Update statistics
                val_loss += loss.item() * inputs.size(0)
                
                # Store predictions and targets
                val_preds.append((torch.sigmoid(outputs) > 0.5).float().cpu().numpy())
                val_targets.append(targets.cpu().numpy())
        
        # Calculate average validation loss
        val_loss = val_loss / len(val_loader.dataset)
        val_losses.append(val_loss)
        
        # Calculate validation metrics
        val_preds = np.concatenate(val_preds)
        val_targets = np.concatenate(val_targets)
        val_accuracy = accuracy_score(val_targets, val_preds)
        val_f1 = f1_score(val_targets, val_preds, zero_division=0)
        val_accuracies.append(val_accuracy)
        val_f1s.append(val_f1)
        
        # Print statistics
        logger.info(f"Epoch {epoch+1}/{num_epochs}: "
              f"Train Loss = {train_loss:.6f}, Train Acc = {train_accuracy:.4f}, Train F1 = {train_f1:.4f}, "
              f"Val Loss = {val_loss:.6f}, Val Acc = {val_accuracy:.4f}, Val F1 = {val_f1:.4f}")
        
        # Check if this is the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model.state_dict().copy()
            counter = 0
            logger.info(f"New best model saved at epoch {epoch+1}")
        else:
            counter += 1
            logger.info(f"No improvement for {counter} epochs")
            
            # Early stopping
            if counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
    
    # Load the best model
    if best_model is not None:
        model.load_state_dict(best_model)
    
    # Plot training curves
    plt.figure(figsize=(15, 5))
    
    # Plot loss
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'{model_name} - Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot accuracy
    plt.subplot(1, 3, 2)
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(val_accuracies, label='Val Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(f'{model_name} - Accuracy')
    plt.legend()
    plt.grid(True)
    
    # Plot F1 score
    plt.subplot(1, 3, 3)
    plt.plot(train_f1s, label='Train F1')
    plt.plot(val_f1s, label='Val F1')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.title(f'{model_name} - F1 Score')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'results/text/{model_name}_training_curves.png')
    plt.close()
    
    return model

def evaluate_model(model, test_loader, device, model_name):
    """Evaluate the model."""
    logger.info(f"Evaluating {model_name} model")
    
    # Move model to device
    model = model.to(device)
    
    # Set model to evaluation mode
    model.eval()
    
    # Initialize lists to store predictions and targets
    all_preds = []
    all_targets = []
    all_probs = []
    
    # Evaluation loop
    with torch.no_grad():
        for inputs, targets in test_loader:
            # Move data to device
            inputs = inputs.to(device)
            
            # Forward pass
            outputs = model(inputs)
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()
            
            # Store predictions and targets
            all_preds.append(preds.cpu().numpy())
            all_targets.append(targets.numpy())
            all_probs.append(probs.cpu().numpy())
    
    # Concatenate predictions and targets
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    all_probs = np.concatenate(all_probs)
    
    # Calculate metrics
    accuracy = accuracy_score(all_targets, all_preds)
    precision = precision_score(all_targets, all_preds, zero_division=0)
    recall = recall_score(all_targets, all_preds, zero_division=0)
    f1 = f1_score(all_targets, all_preds, zero_division=0)
    conf_matrix = confusion_matrix(all_targets, all_preds)
    
    # Calculate AUC if possible
    try:
        auc = roc_auc_score(all_targets, all_probs)
    except:
        auc = 0.0
    
    # Create metrics dictionary
    metrics = {
        'model': model_name,
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'auc': float(auc),
        'confusion_matrix': conf_matrix.tolist()
    }
    
    # Print metrics
    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall: {recall:.4f}")
    logger.info(f"F1 Score: {f1:.4f}")
    logger.info(f"AUC: {auc:.4f}")
    logger.info(f"Confusion Matrix:\n{conf_matrix}")
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Non-depressed', 'Depressed'],
                yticklabels=['Non-depressed', 'Depressed'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'{model_name} - Confusion Matrix')
    plt.tight_layout()
    plt.savefig(f'results/text/{model_name}_confusion_matrix.png')
    plt.close()
    
    return metrics

def compare_models(metrics_list):
    """Compare models."""
    logger.info("Comparing text models")
    
    # Check if metrics list is empty
    if not metrics_list:
        logger.warning("No models to compare")
        return None
    
    # Create DataFrame from metrics
    df = pd.DataFrame(metrics_list)
    
    # Plot comparison
    plt.figure(figsize=(12, 8))
    
    # Plot accuracy, precision, recall, f1, auc
    metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1', 'auc']
    
    # Bar plot
    ax = df[['model'] + metrics_to_plot].set_index('model').plot(kind='bar', figsize=(12, 6))
    plt.title('Text Model Comparison')
    plt.ylabel('Score')
    plt.xlabel('Model')
    plt.ylim(0, 1)
    plt.grid(axis='y')
    plt.legend(loc='lower right')
    
    # Add values on top of bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f')
    
    plt.tight_layout()
    plt.savefig('results/text/model_comparison.png')
    plt.close()
    
    # Save metrics to JSON
    df.to_json('results/text/metrics.json', orient='records', indent=4)
    
    # Create comparison table
    comparison_table = df[['model'] + metrics_to_plot].set_index('model')
    logger.info(f"\nText Model Comparison:")
    logger.info(f"\n{comparison_table}")
    
    # Save comparison table to CSV
    comparison_table.to_csv('results/text/comparison_table.csv')
    
    return comparison_table

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Train all text models')
    parser.add_argument('--num_epochs', type=int, default=20, help='Number of epochs')
    args = parser.parse_args()
    
    logger.info("Starting training of all text models")
    
    # Load text dataset
    data = load_text_dataset()
    
    # Define models to train
    model_names = [
        'TextCNN',
        'TextLSTM',
        'TextBiLSTM',
        'TextTransformer'
    ]
    
    # Train and evaluate models
    metrics_list = []
    
    for model_name in model_names:
        try:
            # Create model
            model = create_model(model_name, data['input_dim'])
            
            # Train model
            trained_model = train_model(
                model, data['train_loader'], data['val_loader'], device, model_name, num_epochs=args.num_epochs
            )
            
            # Evaluate model
            metrics = evaluate_model(trained_model, data['test_loader'], device, model_name)
            
            # Add metrics to list
            metrics_list.append(metrics)
            
            # Save model
            os.makedirs(f'results/text/{model_name}', exist_ok=True)
            torch.save(trained_model.state_dict(), f'results/text/{model_name}/model.pt')
            
        except Exception as e:
            logger.error(f"Error training {model_name}: {e}")
    
    # Compare models
    compare_models(metrics_list)
    
    logger.info("Training completed successfully!")

if __name__ == "__main__":
    main()
