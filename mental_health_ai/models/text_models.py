"""
Text Model Architectures

This module contains neural network architectures for text data analysis.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertConfig


class TextCNN(nn.Module):
    """
    Convolutional Neural Network for text feature classification.
    
    This model is designed to work with pre-extracted text features.
    """
    
    def __init__(self, input_dim, hidden_dims=[128, 64], num_classes=1, dropout_rate=0.5):
        """
        Initialize the Text CNN model.
        
        Args:
            input_dim (int): Dimensionality of input features
            hidden_dims (list): List of hidden layer dimensions
            num_classes (int): Number of output classes (1 for binary classification)
            dropout_rate (float): Dropout rate for regularization
        """
        super(TextCNN, self).__init__()
        
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


class TextLSTM(nn.Module):
    """
    Long Short-Term Memory network for text sequence classification.
    
    This model is designed to work with text sequences.
    """
    
    def __init__(self, vocab_size, embedding_dim=300, hidden_dim=128, num_layers=2, 
                 num_classes=1, dropout_rate=0.5, bidirectional=False, pretrained_embeddings=None):
        """
        Initialize the Text LSTM model.
        
        Args:
            vocab_size (int): Size of the vocabulary
            embedding_dim (int): Dimensionality of word embeddings
            hidden_dim (int): Dimensionality of LSTM hidden state
            num_layers (int): Number of LSTM layers
            num_classes (int): Number of output classes (1 for binary classification)
            dropout_rate (float): Dropout rate for regularization
            bidirectional (bool): Whether to use bidirectional LSTM
            pretrained_embeddings (torch.Tensor, optional): Pretrained word embeddings
        """
        super(TextLSTM, self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.bidirectional = bidirectional
        
        # Embedding layer
        if pretrained_embeddings is not None:
            self.embedding = nn.Embedding.from_pretrained(
                pretrained_embeddings, freeze=False
            )
        else:
            self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
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
            x (torch.Tensor): Input tensor of shape (batch_size, seq_length)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_classes)
        """
        # Embedding layer
        x = self.embedding(x)
        
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)
        
        # Take the output of the last time step
        lstm_out = lstm_out[:, -1, :]
        
        # Apply dropout
        lstm_out = self.dropout(lstm_out)
        
        # Linear layer
        output = self.fc(lstm_out)
        
        return output


class TextBiLSTMAttention(nn.Module):
    """
    Bidirectional LSTM with Attention for text sequence classification.
    
    This model is designed to work with text sequences and uses an attention
    mechanism to focus on the most relevant parts of the sequence.
    """
    
    def __init__(self, vocab_size, embedding_dim=300, hidden_dim=128, num_layers=2, 
                 num_classes=1, dropout_rate=0.5, pretrained_embeddings=None):
        """
        Initialize the Text BiLSTM with Attention model.
        
        Args:
            vocab_size (int): Size of the vocabulary
            embedding_dim (int): Dimensionality of word embeddings
            hidden_dim (int): Dimensionality of LSTM hidden state
            num_layers (int): Number of LSTM layers
            num_classes (int): Number of output classes (1 for binary classification)
            dropout_rate (float): Dropout rate for regularization
            pretrained_embeddings (torch.Tensor, optional): Pretrained word embeddings
        """
        super(TextBiLSTMAttention, self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes
        
        # Embedding layer
        if pretrained_embeddings is not None:
            self.embedding = nn.Embedding.from_pretrained(
                pretrained_embeddings, freeze=False
            )
        else:
            self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
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
            x (torch.Tensor): Input tensor of shape (batch_size, seq_length)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_classes)
        """
        # Embedding layer
        x = self.embedding(x)
        
        # LSTM forward pass
        lstm_output, _ = self.lstm(x)
        
        # Apply attention
        attn_output = self.attention_net(lstm_output)
        
        # Apply dropout
        attn_output = self.dropout(attn_output)
        
        # Linear layer
        output = self.fc(attn_output)
        
        return output


class TextCNN1D(nn.Module):
    """
    1D Convolutional Neural Network for text sequence classification.
    
    This model uses 1D convolutions to extract n-gram features from text.
    """
    
    def __init__(self, vocab_size, embedding_dim=300, num_filters=100, filter_sizes=[3, 4, 5], 
                 num_classes=1, dropout_rate=0.5, pretrained_embeddings=None):
        """
        Initialize the Text 1D CNN model.
        
        Args:
            vocab_size (int): Size of the vocabulary
            embedding_dim (int): Dimensionality of word embeddings
            num_filters (int): Number of filters per filter size
            filter_sizes (list): List of filter sizes
            num_classes (int): Number of output classes (1 for binary classification)
            dropout_rate (float): Dropout rate for regularization
            pretrained_embeddings (torch.Tensor, optional): Pretrained word embeddings
        """
        super(TextCNN1D, self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.num_filters = num_filters
        self.filter_sizes = filter_sizes
        self.num_classes = num_classes
        
        # Embedding layer
        if pretrained_embeddings is not None:
            self.embedding = nn.Embedding.from_pretrained(
                pretrained_embeddings, freeze=False
            )
        else:
            self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Convolutional layers
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embedding_dim, out_channels=num_filters, kernel_size=fs)
            for fs in filter_sizes
        ])
        
        # Output layer
        self.fc = nn.Linear(len(filter_sizes) * num_filters, num_classes)
        
        # Dropout
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_length)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_classes)
        """
        # Embedding layer
        x = self.embedding(x)  # (batch_size, seq_length, embedding_dim)
        
        # Transpose for convolution
        x = x.permute(0, 2, 1)  # (batch_size, embedding_dim, seq_length)
        
        # Apply convolutions and max-pooling
        conv_outputs = []
        for conv in self.convs:
            conv_out = F.relu(conv(x))  # (batch_size, num_filters, seq_length - filter_size + 1)
            conv_out = F.max_pool1d(conv_out, conv_out.size(2))  # (batch_size, num_filters, 1)
            conv_outputs.append(conv_out.squeeze(2))  # (batch_size, num_filters)
        
        # Concatenate outputs from different filter sizes
        x = torch.cat(conv_outputs, dim=1)  # (batch_size, len(filter_sizes) * num_filters)
        
        # Apply dropout
        x = self.dropout(x)
        
        # Linear layer
        output = self.fc(x)
        
        return output


class BERTClassifier(nn.Module):
    """
    BERT-based model for text classification.
    
    This model uses a pretrained BERT model for feature extraction.
    """
    
    def __init__(self, bert_model_name='bert-base-uncased', num_classes=1, dropout_rate=0.1, freeze_bert=False):
        """
        Initialize the BERT Classifier model.
        
        Args:
            bert_model_name (str): Name of the pretrained BERT model
            num_classes (int): Number of output classes (1 for binary classification)
            dropout_rate (float): Dropout rate for regularization
            freeze_bert (bool): Whether to freeze the BERT parameters
        """
        super(BERTClassifier, self).__init__()
        
        self.bert_model_name = bert_model_name
        self.num_classes = num_classes
        
        # Load pretrained BERT model
        self.bert = BertModel.from_pretrained(bert_model_name)
        
        # Freeze BERT parameters if specified
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
        
        # Get BERT configuration
        config = BertConfig.from_pretrained(bert_model_name)
        
        # Output layer
        self.fc = nn.Linear(config.hidden_size, num_classes)
        
        # Dropout
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        """
        Forward pass through the network.
        
        Args:
            input_ids (torch.Tensor): Input token IDs of shape (batch_size, seq_length)
            attention_mask (torch.Tensor, optional): Attention mask of shape (batch_size, seq_length)
            token_type_ids (torch.Tensor, optional): Token type IDs of shape (batch_size, seq_length)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_classes)
        """
        # BERT forward pass
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        # Get the [CLS] token representation
        pooled_output = outputs.pooler_output
        
        # Apply dropout
        pooled_output = self.dropout(pooled_output)
        
        # Linear layer
        output = self.fc(pooled_output)
        
        return output


class TextTransformer(nn.Module):
    """
    Transformer model for text sequence classification.
    
    This model uses a transformer encoder to process text sequences.
    """
    
    def __init__(self, vocab_size, embedding_dim=300, d_model=512, nhead=8, num_layers=4, 
                 dim_feedforward=2048, num_classes=1, dropout_rate=0.1, pretrained_embeddings=None):
        """
        Initialize the Text Transformer model.
        
        Args:
            vocab_size (int): Size of the vocabulary
            embedding_dim (int): Dimensionality of word embeddings
            d_model (int): Dimensionality of the transformer model
            nhead (int): Number of attention heads
            num_layers (int): Number of transformer encoder layers
            dim_feedforward (int): Dimensionality of the feedforward network
            num_classes (int): Number of output classes (1 for binary classification)
            dropout_rate (float): Dropout rate for regularization
            pretrained_embeddings (torch.Tensor, optional): Pretrained word embeddings
        """
        super(TextTransformer, self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.d_model = d_model
        self.num_classes = num_classes
        
        # Embedding layer
        if pretrained_embeddings is not None:
            self.embedding = nn.Embedding.from_pretrained(
                pretrained_embeddings, freeze=False
            )
        else:
            self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Projection layer if embedding_dim != d_model
        if embedding_dim != d_model:
            self.projection = nn.Linear(embedding_dim, d_model)
        else:
            self.projection = nn.Identity()
        
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
    
    def forward(self, x, mask=None):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_length)
            mask (torch.Tensor, optional): Mask tensor of shape (batch_size, seq_length)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_classes)
        """
        # Embedding layer
        x = self.embedding(x)  # (batch_size, seq_length, embedding_dim)
        
        # Project to d_model dimensions if needed
        x = self.projection(x)  # (batch_size, seq_length, d_model)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Transformer encoder
        if mask is not None:
            x = self.transformer_encoder(x, src_key_padding_mask=mask)
        else:
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
