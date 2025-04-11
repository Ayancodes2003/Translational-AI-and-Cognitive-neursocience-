"""
EEG Models Module

This module contains PyTorch models for EEG data.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class EEGNet(nn.Module):
    """
    EEGNet model for EEG classification.
    
    Based on the paper:
    Lawhern, V. J., Solon, A. J., Waytowich, N. R., Gordon, S. M., Hung, C. P., & Lance, B. J. (2018).
    EEGNet: A compact convolutional neural network for EEG-based brain-computer interfaces.
    Journal of Neural Engineering, 15(5), 056013.
    """
    
    def __init__(self, input_dim, num_classes=1, dropout_rate=0.5):
        """
        Initialize EEGNet model.
        
        Args:
            input_dim (int): Input dimension
            num_classes (int): Number of output classes
            dropout_rate (float): Dropout rate
        """
        super(EEGNet, self).__init__()
        
        # Reshape input to 2D for convolutional layers
        self.input_dim = input_dim
        
        # First block
        self.conv1 = nn.Conv1d(1, 16, kernel_size=64, stride=1, padding=32, bias=False)
        self.batchnorm1 = nn.BatchNorm1d(16)
        self.pool1 = nn.MaxPool1d(kernel_size=4, stride=4)
        self.dropout1 = nn.Dropout(dropout_rate)
        
        # Second block
        self.conv2 = nn.Conv1d(16, 32, kernel_size=16, stride=1, padding=8, bias=False)
        self.batchnorm2 = nn.BatchNorm1d(32)
        self.pool2 = nn.MaxPool1d(kernel_size=4, stride=4)
        self.dropout2 = nn.Dropout(dropout_rate)
        
        # Fully connected layer
        # Calculate output size of convolutional layers
        self.fc_input_dim = self._get_fc_input_dim()
        self.fc = nn.Linear(self.fc_input_dim, num_classes)
    
    def _get_fc_input_dim(self):
        """
        Calculate input dimension for fully connected layer.
        
        Returns:
            int: Input dimension for fully connected layer
        """
        # Reshape input to match expected input of conv1
        x = torch.randn(1, 1, self.input_dim)
        
        # Forward pass through convolutional layers
        x = self.pool1(self.batchnorm1(self.conv1(x)))
        x = self.pool2(self.batchnorm2(self.conv2(x)))
        
        # Flatten output
        return x.numel()
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim)
        
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_classes)
        """
        # Reshape input to match expected input of conv1
        x = x.unsqueeze(1)  # (batch_size, 1, input_dim)
        
        # First block
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = F.elu(x)
        x = self.pool1(x)
        x = self.dropout1(x)
        
        # Second block
        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = F.elu(x)
        x = self.pool2(x)
        x = self.dropout2(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layer
        x = self.fc(x)
        
        return x


class DeepConvNet(nn.Module):
    """
    DeepConvNet model for EEG classification.
    
    Based on the paper:
    Schirrmeister, R. T., Springenberg, J. T., Fiederer, L. D. J., Glasstetter, M., Eggensperger, K.,
    Tangermann, M., ... & Ball, T. (2017). Deep learning with convolutional neural networks for EEG
    decoding and visualization. Human Brain Mapping, 38(11), 5391-5420.
    """
    
    def __init__(self, input_dim, num_classes=1, dropout_rate=0.5):
        """
        Initialize DeepConvNet model.
        
        Args:
            input_dim (int): Input dimension
            num_classes (int): Number of output classes
            dropout_rate (float): Dropout rate
        """
        super(DeepConvNet, self).__init__()
        
        # Reshape input to 2D for convolutional layers
        self.input_dim = input_dim
        
        # First block
        self.conv1 = nn.Conv1d(1, 25, kernel_size=10, stride=1, padding=0)
        self.batchnorm1 = nn.BatchNorm1d(25)
        self.pool1 = nn.MaxPool1d(kernel_size=3, stride=3)
        self.dropout1 = nn.Dropout(dropout_rate)
        
        # Second block
        self.conv2 = nn.Conv1d(25, 50, kernel_size=10, stride=1, padding=0)
        self.batchnorm2 = nn.BatchNorm1d(50)
        self.pool2 = nn.MaxPool1d(kernel_size=3, stride=3)
        self.dropout2 = nn.Dropout(dropout_rate)
        
        # Third block
        self.conv3 = nn.Conv1d(50, 100, kernel_size=10, stride=1, padding=0)
        self.batchnorm3 = nn.BatchNorm1d(100)
        self.pool3 = nn.MaxPool1d(kernel_size=3, stride=3)
        self.dropout3 = nn.Dropout(dropout_rate)
        
        # Fourth block
        self.conv4 = nn.Conv1d(100, 200, kernel_size=10, stride=1, padding=0)
        self.batchnorm4 = nn.BatchNorm1d(200)
        self.pool4 = nn.MaxPool1d(kernel_size=3, stride=3)
        self.dropout4 = nn.Dropout(dropout_rate)
        
        # Fully connected layer
        # Calculate output size of convolutional layers
        self.fc_input_dim = self._get_fc_input_dim()
        self.fc = nn.Linear(self.fc_input_dim, num_classes)
    
    def _get_fc_input_dim(self):
        """
        Calculate input dimension for fully connected layer.
        
        Returns:
            int: Input dimension for fully connected layer
        """
        # Reshape input to match expected input of conv1
        x = torch.randn(1, 1, self.input_dim)
        
        # Forward pass through convolutional layers
        x = self.pool1(self.batchnorm1(self.conv1(x)))
        x = self.pool2(self.batchnorm2(self.conv2(x)))
        x = self.pool3(self.batchnorm3(self.conv3(x)))
        
        try:
            x = self.pool4(self.batchnorm4(self.conv4(x)))
        except:
            # If the input is too small for the fourth block, skip it
            pass
        
        # Flatten output
        return max(x.numel(), 1)  # Ensure at least 1 dimension
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim)
        
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_classes)
        """
        # Reshape input to match expected input of conv1
        x = x.unsqueeze(1)  # (batch_size, 1, input_dim)
        
        # First block
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = F.elu(x)
        x = self.pool1(x)
        x = self.dropout1(x)
        
        # Second block
        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = F.elu(x)
        x = self.pool2(x)
        x = self.dropout2(x)
        
        # Third block
        x = self.conv3(x)
        x = self.batchnorm3(x)
        x = F.elu(x)
        x = self.pool3(x)
        x = self.dropout3(x)
        
        try:
            # Fourth block
            x = self.conv4(x)
            x = self.batchnorm4(x)
            x = F.elu(x)
            x = self.pool4(x)
            x = self.dropout4(x)
        except:
            # If the input is too small for the fourth block, skip it
            pass
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layer
        x = self.fc(x)
        
        return x


class ShallowConvNet(nn.Module):
    """
    ShallowConvNet model for EEG classification.
    
    Based on the paper:
    Schirrmeister, R. T., Springenberg, J. T., Fiederer, L. D. J., Glasstetter, M., Eggensperger, K.,
    Tangermann, M., ... & Ball, T. (2017). Deep learning with convolutional neural networks for EEG
    decoding and visualization. Human Brain Mapping, 38(11), 5391-5420.
    """
    
    def __init__(self, input_dim, num_classes=1, dropout_rate=0.5):
        """
        Initialize ShallowConvNet model.
        
        Args:
            input_dim (int): Input dimension
            num_classes (int): Number of output classes
            dropout_rate (float): Dropout rate
        """
        super(ShallowConvNet, self).__init__()
        
        # Reshape input to 2D for convolutional layers
        self.input_dim = input_dim
        
        # First block
        self.conv1 = nn.Conv1d(1, 40, kernel_size=25, stride=1, padding=0)
        self.conv2 = nn.Conv1d(40, 40, kernel_size=1, stride=1, padding=0)
        self.batchnorm = nn.BatchNorm1d(40)
        self.pool = nn.AvgPool1d(kernel_size=75, stride=15)
        self.dropout = nn.Dropout(dropout_rate)
        
        # Fully connected layer
        # Calculate output size of convolutional layers
        self.fc_input_dim = self._get_fc_input_dim()
        self.fc = nn.Linear(self.fc_input_dim, num_classes)
    
    def _get_fc_input_dim(self):
        """
        Calculate input dimension for fully connected layer.
        
        Returns:
            int: Input dimension for fully connected layer
        """
        # Reshape input to match expected input of conv1
        x = torch.randn(1, 1, self.input_dim)
        
        # Forward pass through convolutional layers
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.batchnorm(x)
        x = torch.square(x)
        x = self.pool(x)
        x = torch.log(torch.clamp(x, min=1e-6))
        
        # Flatten output
        return max(x.numel(), 1)  # Ensure at least 1 dimension
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim)
        
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_classes)
        """
        # Reshape input to match expected input of conv1
        x = x.unsqueeze(1)  # (batch_size, 1, input_dim)
        
        # First block
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.batchnorm(x)
        x = torch.square(x)
        x = self.pool(x)
        x = torch.log(torch.clamp(x, min=1e-6))
        x = self.dropout(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layer
        x = self.fc(x)
        
        return x


class EEGCNN(nn.Module):
    """
    1D CNN model for EEG classification.
    """
    
    def __init__(self, input_dim, num_classes=1, dropout_rate=0.5):
        """
        Initialize EEGCNN model.
        
        Args:
            input_dim (int): Input dimension
            num_classes (int): Number of output classes
            dropout_rate (float): Dropout rate
        """
        super(EEGCNN, self).__init__()
        
        # Reshape input to 2D for convolutional layers
        self.input_dim = input_dim
        
        # Convolutional layers
        self.conv1 = nn.Conv1d(1, 16, kernel_size=3, stride=1, padding=1)
        self.batchnorm1 = nn.BatchNorm1d(16)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1)
        self.batchnorm2 = nn.BatchNorm1d(32)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        self.conv3 = nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1)
        self.batchnorm3 = nn.BatchNorm1d(64)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Fully connected layers
        # Calculate output size of convolutional layers
        self.fc_input_dim = self._get_fc_input_dim()
        
        self.fc1 = nn.Linear(self.fc_input_dim, 128)
        self.dropout1 = nn.Dropout(dropout_rate)
        
        self.fc2 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(dropout_rate)
        
        self.fc3 = nn.Linear(64, num_classes)
    
    def _get_fc_input_dim(self):
        """
        Calculate input dimension for fully connected layer.
        
        Returns:
            int: Input dimension for fully connected layer
        """
        # Reshape input to match expected input of conv1
        x = torch.randn(1, 1, self.input_dim)
        
        # Forward pass through convolutional layers
        x = self.pool1(self.batchnorm1(self.conv1(x)))
        x = self.pool2(self.batchnorm2(self.conv2(x)))
        x = self.pool3(self.batchnorm3(self.conv3(x)))
        
        # Flatten output
        return max(x.numel(), 1)  # Ensure at least 1 dimension
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim)
        
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_classes)
        """
        # Reshape input to match expected input of conv1
        x = x.unsqueeze(1)  # (batch_size, 1, input_dim)
        
        # Convolutional layers
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = F.relu(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = F.relu(x)
        x = self.pool2(x)
        
        x = self.conv3(x)
        x = self.batchnorm3(x)
        x = F.relu(x)
        x = self.pool3(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        
        x = self.fc3(x)
        
        return x


class EEGLSTM(nn.Module):
    """
    LSTM model for EEG classification.
    """
    
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, num_classes=1, dropout_rate=0.5, bidirectional=True):
        """
        Initialize EEGLSTM model.
        
        Args:
            input_dim (int): Input dimension
            hidden_dim (int): Hidden dimension
            num_layers (int): Number of LSTM layers
            num_classes (int): Number of output classes
            dropout_rate (float): Dropout rate
            bidirectional (bool): Whether to use bidirectional LSTM
        """
        super(EEGLSTM, self).__init__()
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Fully connected layers
        fc_input_dim = hidden_dim * 2 if bidirectional else hidden_dim
        
        self.fc1 = nn.Linear(fc_input_dim, 64)
        self.dropout1 = nn.Dropout(dropout_rate)
        
        self.fc2 = nn.Linear(64, num_classes)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim)
        
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_classes)
        """
        # Reshape input to match expected input of LSTM
        x = x.unsqueeze(2)  # (batch_size, input_dim, 1)
        
        # LSTM layer
        output, (hidden, cell) = self.lstm(x)
        
        # Get the last output
        if self.lstm.bidirectional:
            # Concatenate the last output from both directions
            hidden_forward = hidden[-2, :, :]
            hidden_backward = hidden[-1, :, :]
            hidden_cat = torch.cat((hidden_forward, hidden_backward), dim=1)
        else:
            # Get the last output
            hidden_cat = hidden[-1, :, :]
        
        # Fully connected layers
        x = self.fc1(hidden_cat)
        x = F.relu(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        
        return x


class EEGTransformer(nn.Module):
    """
    Transformer model for EEG classification.
    """
    
    def __init__(self, input_dim, d_model=64, nhead=8, num_layers=2, dim_feedforward=128, num_classes=1, dropout_rate=0.1):
        """
        Initialize EEGTransformer model.
        
        Args:
            input_dim (int): Input dimension
            d_model (int): Model dimension
            nhead (int): Number of attention heads
            num_layers (int): Number of transformer layers
            dim_feedforward (int): Dimension of feedforward network
            num_classes (int): Number of output classes
            dropout_rate (float): Dropout rate
        """
        super(EEGTransformer, self).__init__()
        
        # Embedding layer
        self.embedding = nn.Linear(1, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout_rate)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout_rate, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Fully connected layers
        self.fc1 = nn.Linear(d_model, 64)
        self.dropout1 = nn.Dropout(dropout_rate)
        
        self.fc2 = nn.Linear(64, num_classes)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim)
        
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_classes)
        """
        # Reshape input to match expected input of embedding
        x = x.unsqueeze(2)  # (batch_size, input_dim, 1)
        
        # Embedding
        x = self.embedding(x)  # (batch_size, input_dim, d_model)
        
        # Positional encoding
        x = self.pos_encoder(x)
        
        # Transformer encoder
        x = self.transformer_encoder(x)
        
        # Global average pooling
        x = torch.mean(x, dim=1)
        
        # Fully connected layers
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        
        return x


class PositionalEncoding(nn.Module):
    """
    Positional encoding for transformer models.
    """
    
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        """
        Initialize positional encoding.
        
        Args:
            d_model (int): Model dimension
            dropout (float): Dropout rate
            max_len (int): Maximum sequence length
        """
        super(PositionalEncoding, self).__init__()
        
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)
        
        # Register buffer (not a parameter, but should be saved and loaded with the model)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model)
        
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)
