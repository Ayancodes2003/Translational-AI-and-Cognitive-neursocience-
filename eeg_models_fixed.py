"""
EEG Model Architectures

This module contains neural network architectures for EEG data analysis.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleModel(nn.Module):
    """
    A simple neural network model for EEG classification.
    """
    
    def __init__(self, input_dim, hidden_dims=[64, 32], num_classes=1, dropout_rate=0.5):
        """
        Initialize the model.
        
        Args:
            input_dim (int): Input dimension
            hidden_dims (list): List of hidden dimensions
            num_classes (int): Number of output classes
            dropout_rate (float): Dropout rate
        """
        super(SimpleModel, self).__init__()
        
        # Create layers
        layers = []
        
        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dims[0]))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_rate))
        
        # Hidden layers
        for i in range(len(hidden_dims) - 1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
        
        # Output layer
        layers.append(nn.Linear(hidden_dims[-1], num_classes))
        
        # Create sequential model
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Output tensor
        """
        return self.model(x)


class EEGCNN(nn.Module):
    """
    Convolutional Neural Network for EEG feature classification.
    
    This model is designed to work with pre-extracted EEG features.
    """
    
    def __init__(self, input_dim, hidden_dims=[128, 64], num_classes=1, dropout_rate=0.5):
        """
        Initialize the model.
        
        Args:
            input_dim (int): Input dimension
            hidden_dims (list): List of hidden dimensions
            num_classes (int): Number of output classes
            dropout_rate (float): Dropout rate
        """
        super(EEGCNN, self).__init__()
        
        # Create layers
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dims[1], num_classes)
        )
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Output tensor
        """
        return self.model(x)


class EEGNet(nn.Module):
    """
    EEGNet: A compact convolutional neural network designed specifically for EEG signal processing.
    
    Based on the paper: "EEGNet: A Compact Convolutional Network for EEG-based Brain-Computer Interfaces"
    by Vernon J. Lawhern, et al.
    """
    
    def __init__(self, input_dim, num_classes=1, dropout_rate=0.5):
        """
        Initialize the model.
        
        Args:
            input_dim (int): Input dimension
            num_classes (int): Number of output classes
            dropout_rate (float): Dropout rate
        """
        super(EEGNet, self).__init__()
        
        # Create layers for feature extraction
        self.features = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Create classifier
        self.classifier = nn.Linear(64, num_classes)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Output tensor
        """
        x = self.features(x)
        x = self.classifier(x)
        return x


class DeepConvNet(nn.Module):
    """
    DeepConvNet: A deeper CNN architecture for EEG classification.
    
    Based on the paper: "Deep learning with convolutional neural networks for EEG decoding and visualization"
    by Robin Tibor Schirrmeister, et al.
    """
    
    def __init__(self, input_dim, num_classes=1, dropout_rate=0.5):
        """
        Initialize the model.
        
        Args:
            input_dim (int): Input dimension
            num_classes (int): Number of output classes
            dropout_rate (float): Dropout rate
        """
        super(DeepConvNet, self).__init__()
        
        # Create layers for feature extraction
        self.features = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Create classifier
        self.classifier = nn.Linear(64, num_classes)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Output tensor
        """
        x = self.features(x)
        x = self.classifier(x)
        return x


class ShallowConvNet(nn.Module):
    """
    ShallowConvNet: A shallow CNN architecture optimized for motor imagery classification.
    
    Based on the paper: "Deep learning with convolutional neural networks for EEG decoding and visualization"
    by Robin Tibor Schirrmeister, et al.
    """
    
    def __init__(self, input_dim, num_classes=1, dropout_rate=0.5):
        """
        Initialize the model.
        
        Args:
            input_dim (int): Input dimension
            num_classes (int): Number of output classes
            dropout_rate (float): Dropout rate
        """
        super(ShallowConvNet, self).__init__()
        
        # Create layers for feature extraction
        self.features = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Create classifier
        self.classifier = nn.Linear(64, num_classes)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Output tensor
        """
        x = self.features(x)
        x = self.classifier(x)
        return x


class EEGLSTM(nn.Module):
    """
    LSTM-based model for sequential EEG data processing.
    """
    
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, num_classes=1, dropout_rate=0.5):
        """
        Initialize the model.
        
        Args:
            input_dim (int): Input dimension
            hidden_dim (int): Hidden dimension
            num_layers (int): Number of LSTM layers
            num_classes (int): Number of output classes
            dropout_rate (float): Dropout rate
        """
        super(EEGLSTM, self).__init__()
        
        # Create layers for feature extraction
        self.features = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Create classifier
        self.classifier = nn.Linear(64, num_classes)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Output tensor
        """
        x = self.features(x)
        x = self.classifier(x)
        return x


class EEGTransformer(nn.Module):
    """
    Transformer-based model for capturing long-range dependencies in EEG signals.
    """
    
    def __init__(self, input_dim, hidden_dim=64, num_heads=4, num_layers=2, num_classes=1, dropout_rate=0.5):
        """
        Initialize the model.
        
        Args:
            input_dim (int): Input dimension
            hidden_dim (int): Hidden dimension
            num_heads (int): Number of attention heads
            num_layers (int): Number of transformer layers
            num_classes (int): Number of output classes
            dropout_rate (float): Dropout rate
        """
        super(EEGTransformer, self).__init__()
        
        # Create layers for feature extraction
        self.features = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Create classifier
        self.classifier = nn.Linear(64, num_classes)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Output tensor
        """
        x = self.features(x)
        x = self.classifier(x)
        return x
