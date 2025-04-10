"""
Audio Model Architectures

This module contains neural network architectures for audio data analysis.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AudioCNN(nn.Module):
    """
    Convolutional Neural Network for audio feature classification.
    
    This model is designed to work with pre-extracted audio features.
    """
    
    def __init__(self, input_dim, hidden_dims=[128, 64], num_classes=1, dropout_rate=0.5):
        """
        Initialize the Audio CNN model.
        
        Args:
            input_dim (int): Dimensionality of input features
            hidden_dims (list): List of hidden layer dimensions
            num_classes (int): Number of output classes (1 for binary classification)
            dropout_rate (float): Dropout rate for regularization
        """
        super(AudioCNN, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.num_classes = num_classes
        
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
        
        # Combine layers
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_classes)
        """
        return self.model(x)


class AudioLSTM(nn.Module):
    """
    Long Short-Term Memory network for audio sequence classification.
    
    This model is designed to work with raw audio sequences or spectrograms.
    """
    
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, num_classes=1, dropout_rate=0.5, bidirectional=False):
        """
        Initialize the Audio LSTM model.
        
        Args:
            input_dim (int): Dimensionality of input features
            hidden_dim (int): Dimensionality of LSTM hidden state
            num_layers (int): Number of LSTM layers
            num_classes (int): Number of output classes (1 for binary classification)
            dropout_rate (float): Dropout rate for regularization
            bidirectional (bool): Whether to use bidirectional LSTM
        """
        super(AudioLSTM, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.bidirectional = bidirectional
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Output layer
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.fc = nn.Linear(lstm_output_dim, num_classes)
        
        # Dropout
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_length, input_dim)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_classes)
        """
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)
        
        # Take the output of the last time step
        lstm_out = lstm_out[:, -1, :]
        
        # Apply dropout
        lstm_out = self.dropout(lstm_out)
        
        # Linear layer
        output = self.fc(lstm_out)
        
        return output


class AudioBiLSTMAttention(nn.Module):
    """
    Bidirectional LSTM with Attention for audio sequence classification.
    
    This model is designed to work with raw audio sequences or spectrograms and uses
    an attention mechanism to focus on the most relevant parts of the sequence.
    """
    
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, num_classes=1, dropout_rate=0.5):
        """
        Initialize the Audio BiLSTM with Attention model.
        
        Args:
            input_dim (int): Dimensionality of input features
            hidden_dim (int): Dimensionality of LSTM hidden state
            num_layers (int): Number of LSTM layers
            num_classes (int): Number of output classes (1 for binary classification)
            dropout_rate (float): Dropout rate for regularization
        """
        super(AudioBiLSTMAttention, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Attention layer
        self.attention = nn.Linear(hidden_dim * 2, 1)
        
        # Output layer
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        
        # Dropout
        self.dropout = nn.Dropout(dropout_rate)
    
    def attention_net(self, lstm_output):
        """
        Attention mechanism to focus on relevant parts of the sequence.
        
        Args:
            lstm_output (torch.Tensor): Output from LSTM of shape (batch_size, seq_length, hidden_dim*2)
            
        Returns:
            torch.Tensor: Context vector of shape (batch_size, hidden_dim*2)
        """
        # Calculate attention weights
        attn_weights = self.attention(lstm_output).squeeze(-1)
        soft_attn_weights = F.softmax(attn_weights, dim=1)
        
        # Apply attention weights to LSTM output
        context = torch.bmm(soft_attn_weights.unsqueeze(1), lstm_output).squeeze(1)
        
        return context
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_length, input_dim)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_classes)
        """
        # LSTM forward pass
        lstm_output, _ = self.lstm(x)
        
        # Apply attention
        attn_output = self.attention_net(lstm_output)
        
        # Apply dropout
        attn_output = self.dropout(attn_output)
        
        # Linear layer
        output = self.fc(attn_output)
        
        return output


class Audio2DCNN(nn.Module):
    """
    2D Convolutional Neural Network for audio spectrogram classification.
    
    This model is designed to work with spectrograms or mel-spectrograms.
    """
    
    def __init__(self, input_channels=1, num_classes=1, dropout_rate=0.5):
        """
        Initialize the Audio 2D CNN model.
        
        Args:
            input_channels (int): Number of input channels (1 for mono, 2 for stereo)
            num_classes (int): Number of output classes (1 for binary classification)
            dropout_rate (float): Dropout rate for regularization
        """
        super(Audio2DCNN, self).__init__()
        
        self.input_channels = input_channels
        self.num_classes = num_classes
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Adaptive pooling to handle variable input sizes
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Fully connected layers
        self.fc1 = nn.Linear(256, 128)
        self.bn5 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, num_classes)
        
        # Dropout
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_channels, height, width)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_classes)
        """
        # Convolutional layers
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool4(x)
        
        # Adaptive pooling
        x = self.adaptive_pool(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = F.relu(self.bn5(self.fc1(x)))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


class Audio1DCNNGRU(nn.Module):
    """
    1D Convolutional Neural Network with GRU for audio sequence classification.
    
    This model combines CNNs for feature extraction with GRUs for sequence modeling.
    """
    
    def __init__(self, input_dim, hidden_dim=128, num_classes=1, dropout_rate=0.5):
        """
        Initialize the Audio 1D CNN + GRU model.
        
        Args:
            input_dim (int): Dimensionality of input features
            hidden_dim (int): Dimensionality of GRU hidden state
            num_classes (int): Number of output classes (1 for binary classification)
            dropout_rate (float): Dropout rate for regularization
        """
        super(Audio1DCNNGRU, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        
        # Convolutional layers
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Calculate size after convolutions and pooling
        self.cnn_output_dim = input_dim // 4
        
        # GRU layer
        self.gru = nn.GRU(
            input_size=64,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=dropout_rate,
            bidirectional=True
        )
        
        # Output layer
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        
        # Dropout
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 1, seq_length)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_classes)
        """
        # Convolutional layers
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        
        # Reshape for GRU: (batch_size, channels, seq_length) -> (batch_size, seq_length, channels)
        x = x.permute(0, 2, 1)
        
        # GRU forward pass
        gru_out, _ = self.gru(x)
        
        # Take the output of the last time step
        gru_out = gru_out[:, -1, :]
        
        # Apply dropout
        gru_out = self.dropout(gru_out)
        
        # Linear layer
        output = self.fc(gru_out)
        
        return output


class AudioTransformer(nn.Module):
    """
    Transformer model for audio sequence classification.
    
    This model uses a transformer encoder to process audio sequences.
    """
    
    def __init__(self, input_dim, d_model=128, nhead=8, num_layers=4, 
                 dim_feedforward=512, num_classes=1, dropout_rate=0.1):
        """
        Initialize the Audio Transformer model.
        
        Args:
            input_dim (int): Dimensionality of input features
            d_model (int): Dimensionality of the transformer model
            nhead (int): Number of attention heads
            num_layers (int): Number of transformer encoder layers
            dim_feedforward (int): Dimensionality of the feedforward network
            num_classes (int): Number of output classes (1 for binary classification)
            dropout_rate (float): Dropout rate for regularization
        """
        super(AudioTransformer, self).__init__()
        
        self.input_dim = input_dim
        self.d_model = d_model
        self.num_classes = num_classes
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout_rate)
        
        # Transformer encoder
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout_rate,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        # Output layer
        self.fc = nn.Linear(d_model, num_classes)
        
        # Dropout
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_length, input_dim)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_classes)
        """
        # Project input to d_model dimensions
        x = self.input_projection(x)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Transformer encoder
        x = self.transformer_encoder(x)
        
        # Global average pooling
        x = torch.mean(x, dim=1)
        
        # Apply dropout
        x = self.dropout(x)
        
        # Linear layer
        output = self.fc(x)
        
        return output


class PositionalEncoding(nn.Module):
    """
    Positional encoding for the transformer model.
    """
    
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        """
        Initialize the positional encoding.
        
        Args:
            d_model (int): Dimensionality of the model
            dropout (float): Dropout rate
            max_len (int): Maximum sequence length
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_length, d_model)
            
        Returns:
            torch.Tensor: Output tensor with positional encoding added
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)
