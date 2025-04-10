"""
Multimodal Fusion Model Architectures

This module contains neural network architectures for multimodal fusion of EEG, audio, and text data.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class EarlyFusionModel(nn.Module):
    """
    Early Fusion Model for multimodal data.
    
    This model concatenates features from different modalities at the input level
    and processes them through a shared network.
    """
    
    def __init__(self, input_dims, hidden_dims=[256, 128, 64], num_classes=1, dropout_rate=0.5):
        """
        Initialize the Early Fusion model.
        
        Args:
            input_dims (dict): Dictionary of input dimensions for each modality
                               (e.g., {'eeg': 100, 'audio': 200, 'text': 300})
            hidden_dims (list): List of hidden layer dimensions
            num_classes (int): Number of output classes (1 for binary classification)
            dropout_rate (float): Dropout rate for regularization
        """
        super(EarlyFusionModel, self).__init__()
        
        self.input_dims = input_dims
        self.hidden_dims = hidden_dims
        self.num_classes = num_classes
        
        # Calculate total input dimension
        self.total_input_dim = sum(input_dims.values())
        
        # Create layers
        layers = []
        
        # Input layer
        layers.append(nn.Linear(self.total_input_dim, hidden_dims[0]))
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
            x (torch.Tensor): Input tensor of shape (batch_size, total_input_dim)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_classes)
        """
        return self.model(x)


class LateFusionModel(nn.Module):
    """
    Late Fusion Model for multimodal data.
    
    This model processes each modality through separate networks and combines
    their outputs at the decision level.
    """
    
    def __init__(self, input_dims, hidden_dims=[128, 64], fusion_method='concat', 
                 num_classes=1, dropout_rate=0.5):
        """
        Initialize the Late Fusion model.
        
        Args:
            input_dims (dict): Dictionary of input dimensions for each modality
                               (e.g., {'eeg': 100, 'audio': 200, 'text': 300})
            hidden_dims (list): List of hidden layer dimensions for each modality
            fusion_method (str): Method for fusion ('concat', 'average', or 'weighted')
            num_classes (int): Number of output classes (1 for binary classification)
            dropout_rate (float): Dropout rate for regularization
        """
        super(LateFusionModel, self).__init__()
        
        self.input_dims = input_dims
        self.hidden_dims = hidden_dims
        self.fusion_method = fusion_method
        self.num_classes = num_classes
        
        # Create separate networks for each modality
        self.modality_networks = nn.ModuleDict()
        
        for modality, input_dim in input_dims.items():
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
            
            # Output layer for this modality
            layers.append(nn.Linear(hidden_dims[-1], num_classes))
            
            # Combine layers
            self.modality_networks[modality] = nn.Sequential(*layers)
        
        # For weighted fusion
        if fusion_method == 'weighted':
            self.fusion_weights = nn.Parameter(torch.ones(len(input_dims)) / len(input_dims))
    
    def forward(self, x_dict):
        """
        Forward pass through the network.
        
        Args:
            x_dict (dict): Dictionary of input tensors for each modality
                          (e.g., {'eeg': eeg_tensor, 'audio': audio_tensor, 'text': text_tensor})
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_classes)
        """
        # Process each modality
        outputs = {}
        for modality, network in self.modality_networks.items():
            outputs[modality] = network(x_dict[modality])
        
        # Fusion
        if self.fusion_method == 'concat':
            # Concatenate outputs
            fused_output = torch.cat(list(outputs.values()), dim=1)
            # Additional linear layer to reduce dimensionality
            fused_output = nn.Linear(fused_output.size(1), self.num_classes)(fused_output)
        
        elif self.fusion_method == 'average':
            # Average outputs
            fused_output = torch.mean(torch.stack(list(outputs.values())), dim=0)
        
        elif self.fusion_method == 'weighted':
            # Weighted average
            normalized_weights = F.softmax(self.fusion_weights, dim=0)
            fused_output = torch.zeros_like(outputs[list(outputs.keys())[0]])
            for i, modality in enumerate(outputs.keys()):
                fused_output += normalized_weights[i] * outputs[modality]
        
        else:
            raise ValueError(f"Unknown fusion method: {self.fusion_method}")
        
        return fused_output


class IntermediateFusionModel(nn.Module):
    """
    Intermediate Fusion Model for multimodal data.
    
    This model processes each modality through separate networks up to an intermediate
    layer, then combines their features and processes them through a shared network.
    """
    
    def __init__(self, input_dims, modality_hidden_dims=[128], shared_hidden_dims=[128, 64], 
                 fusion_method='concat', num_classes=1, dropout_rate=0.5):
        """
        Initialize the Intermediate Fusion model.
        
        Args:
            input_dims (dict): Dictionary of input dimensions for each modality
                               (e.g., {'eeg': 100, 'audio': 200, 'text': 300})
            modality_hidden_dims (list): List of hidden layer dimensions for each modality
            shared_hidden_dims (list): List of hidden layer dimensions for the shared network
            fusion_method (str): Method for fusion ('concat', 'attention', or 'gated')
            num_classes (int): Number of output classes (1 for binary classification)
            dropout_rate (float): Dropout rate for regularization
        """
        super(IntermediateFusionModel, self).__init__()
        
        self.input_dims = input_dims
        self.modality_hidden_dims = modality_hidden_dims
        self.shared_hidden_dims = shared_hidden_dims
        self.fusion_method = fusion_method
        self.num_classes = num_classes
        
        # Create separate networks for each modality
        self.modality_encoders = nn.ModuleDict()
        
        for modality, input_dim in input_dims.items():
            layers = []
            
            # Input layer
            layers.append(nn.Linear(input_dim, modality_hidden_dims[0]))
            layers.append(nn.BatchNorm1d(modality_hidden_dims[0]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            
            # Hidden layers
            for i in range(len(modality_hidden_dims) - 1):
                layers.append(nn.Linear(modality_hidden_dims[i], modality_hidden_dims[i + 1]))
                layers.append(nn.BatchNorm1d(modality_hidden_dims[i + 1]))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout_rate))
            
            # Combine layers
            self.modality_encoders[modality] = nn.Sequential(*layers)
        
        # Fusion mechanism
        if fusion_method == 'concat':
            # Concatenation
            fusion_output_dim = len(input_dims) * modality_hidden_dims[-1]
        
        elif fusion_method == 'attention':
            # Attention-based fusion
            fusion_output_dim = modality_hidden_dims[-1]
            self.attention = nn.ModuleDict()
            for modality in input_dims.keys():
                self.attention[modality] = nn.Linear(modality_hidden_dims[-1], 1)
        
        elif fusion_method == 'gated':
            # Gated fusion
            fusion_output_dim = modality_hidden_dims[-1]
            self.gates = nn.ModuleDict()
            for modality in input_dims.keys():
                self.gates[modality] = nn.Sequential(
                    nn.Linear(modality_hidden_dims[-1], modality_hidden_dims[-1]),
                    nn.Sigmoid()
                )
        
        else:
            raise ValueError(f"Unknown fusion method: {fusion_method}")
        
        # Shared network after fusion
        shared_layers = []
        
        # Input layer
        shared_layers.append(nn.Linear(fusion_output_dim, shared_hidden_dims[0]))
        shared_layers.append(nn.BatchNorm1d(shared_hidden_dims[0]))
        shared_layers.append(nn.ReLU())
        shared_layers.append(nn.Dropout(dropout_rate))
        
        # Hidden layers
        for i in range(len(shared_hidden_dims) - 1):
            shared_layers.append(nn.Linear(shared_hidden_dims[i], shared_hidden_dims[i + 1]))
            shared_layers.append(nn.BatchNorm1d(shared_hidden_dims[i + 1]))
            shared_layers.append(nn.ReLU())
            shared_layers.append(nn.Dropout(dropout_rate))
        
        # Output layer
        shared_layers.append(nn.Linear(shared_hidden_dims[-1], num_classes))
        
        # Combine layers
        self.shared_network = nn.Sequential(*shared_layers)
    
    def forward(self, x_dict):
        """
        Forward pass through the network.
        
        Args:
            x_dict (dict): Dictionary of input tensors for each modality
                          (e.g., {'eeg': eeg_tensor, 'audio': audio_tensor, 'text': text_tensor})
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_classes)
        """
        # Process each modality
        encoded_features = {}
        for modality, encoder in self.modality_encoders.items():
            encoded_features[modality] = encoder(x_dict[modality])
        
        # Fusion
        if self.fusion_method == 'concat':
            # Concatenate features
            fused_features = torch.cat(list(encoded_features.values()), dim=1)
        
        elif self.fusion_method == 'attention':
            # Attention-based fusion
            attention_weights = {}
            for modality in encoded_features.keys():
                attention_weights[modality] = self.attention[modality](encoded_features[modality])
            
            # Normalize attention weights
            attention_weights_cat = torch.cat(list(attention_weights.values()), dim=1)
            attention_weights_norm = F.softmax(attention_weights_cat, dim=1)
            
            # Apply attention weights
            fused_features = torch.zeros_like(encoded_features[list(encoded_features.keys())[0]])
            for i, modality in enumerate(encoded_features.keys()):
                fused_features += attention_weights_norm[:, i:i+1] * encoded_features[modality]
        
        elif self.fusion_method == 'gated':
            # Gated fusion
            gated_features = {}
            for modality in encoded_features.keys():
                gate = self.gates[modality](encoded_features[modality])
                gated_features[modality] = gate * encoded_features[modality]
            
            # Sum gated features
            fused_features = torch.zeros_like(encoded_features[list(encoded_features.keys())[0]])
            for modality in gated_features.keys():
                fused_features += gated_features[modality]
        
        # Process through shared network
        output = self.shared_network(fused_features)
        
        return output


class CrossModalAttentionFusion(nn.Module):
    """
    Cross-Modal Attention Fusion Model for multimodal data.
    
    This model uses cross-modal attention to capture interactions between modalities.
    """
    
    def __init__(self, input_dims, modality_hidden_dims=[128], shared_hidden_dims=[128, 64], 
                 num_classes=1, dropout_rate=0.5):
        """
        Initialize the Cross-Modal Attention Fusion model.
        
        Args:
            input_dims (dict): Dictionary of input dimensions for each modality
                               (e.g., {'eeg': 100, 'audio': 200, 'text': 300})
            modality_hidden_dims (list): List of hidden layer dimensions for each modality
            shared_hidden_dims (list): List of hidden layer dimensions for the shared network
            num_classes (int): Number of output classes (1 for binary classification)
            dropout_rate (float): Dropout rate for regularization
        """
        super(CrossModalAttentionFusion, self).__init__()
        
        self.input_dims = input_dims
        self.modality_hidden_dims = modality_hidden_dims
        self.shared_hidden_dims = shared_hidden_dims
        self.num_classes = num_classes
        
        # Create separate encoders for each modality
        self.modality_encoders = nn.ModuleDict()
        
        for modality, input_dim in input_dims.items():
            layers = []
            
            # Input layer
            layers.append(nn.Linear(input_dim, modality_hidden_dims[0]))
            layers.append(nn.BatchNorm1d(modality_hidden_dims[0]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            
            # Hidden layers
            for i in range(len(modality_hidden_dims) - 1):
                layers.append(nn.Linear(modality_hidden_dims[i], modality_hidden_dims[i + 1]))
                layers.append(nn.BatchNorm1d(modality_hidden_dims[i + 1]))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout_rate))
            
            # Combine layers
            self.modality_encoders[modality] = nn.Sequential(*layers)
        
        # Cross-modal attention
        self.cross_attention = nn.ModuleDict()
        modalities = list(input_dims.keys())
        
        for i, mod_i in enumerate(modalities):
            self.cross_attention[mod_i] = nn.ModuleDict()
            for j, mod_j in enumerate(modalities):
                if i != j:
                    # Attention from mod_i to mod_j
                    self.cross_attention[mod_i][mod_j] = nn.Sequential(
                        nn.Linear(modality_hidden_dims[-1] * 2, modality_hidden_dims[-1]),
                        nn.Tanh(),
                        nn.Linear(modality_hidden_dims[-1], 1)
                    )
        
        # Fusion layer
        self.fusion_layer = nn.Linear(len(input_dims) * modality_hidden_dims[-1], shared_hidden_dims[0])
        
        # Shared network after fusion
        shared_layers = []
        
        # First hidden layer is already created as fusion_layer
        shared_layers.append(nn.BatchNorm1d(shared_hidden_dims[0]))
        shared_layers.append(nn.ReLU())
        shared_layers.append(nn.Dropout(dropout_rate))
        
        # Hidden layers
        for i in range(len(shared_hidden_dims) - 1):
            shared_layers.append(nn.Linear(shared_hidden_dims[i], shared_hidden_dims[i + 1]))
            shared_layers.append(nn.BatchNorm1d(shared_hidden_dims[i + 1]))
            shared_layers.append(nn.ReLU())
            shared_layers.append(nn.Dropout(dropout_rate))
        
        # Output layer
        shared_layers.append(nn.Linear(shared_hidden_dims[-1], num_classes))
        
        # Combine layers
        self.shared_network = nn.Sequential(*shared_layers)
    
    def forward(self, x_dict):
        """
        Forward pass through the network.
        
        Args:
            x_dict (dict): Dictionary of input tensors for each modality
                          (e.g., {'eeg': eeg_tensor, 'audio': audio_tensor, 'text': text_tensor})
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_classes)
        """
        # Process each modality
        encoded_features = {}
        for modality, encoder in self.modality_encoders.items():
            encoded_features[modality] = encoder(x_dict[modality])
        
        # Cross-modal attention
        attended_features = {}
        modalities = list(x_dict.keys())
        
        for i, mod_i in enumerate(modalities):
            # Start with the original features
            attended_features[mod_i] = encoded_features[mod_i]
            
            # Apply attention from other modalities
            for j, mod_j in enumerate(modalities):
                if i != j:
                    # Concatenate features from both modalities
                    concat_features = torch.cat([encoded_features[mod_i], encoded_features[mod_j]], dim=1)
                    
                    # Calculate attention weights
                    attention_weights = self.cross_attention[mod_i][mod_j](concat_features)
                    attention_weights = torch.sigmoid(attention_weights)
                    
                    # Apply attention
                    attended_features[mod_i] = attended_features[mod_i] + attention_weights * encoded_features[mod_j]
        
        # Concatenate attended features
        fused_features = torch.cat(list(attended_features.values()), dim=1)
        
        # Process through fusion layer and shared network
        fused_features = self.fusion_layer(fused_features)
        output = self.shared_network(fused_features)
        
        return output


class HierarchicalFusionModel(nn.Module):
    """
    Hierarchical Fusion Model for multimodal data.
    
    This model fuses modalities in a hierarchical manner, first combining pairs
    of modalities and then combining the results.
    """
    
    def __init__(self, input_dims, modality_hidden_dims=[128], fusion_hidden_dims=[128], 
                 shared_hidden_dims=[128, 64], num_classes=1, dropout_rate=0.5):
        """
        Initialize the Hierarchical Fusion model.
        
        Args:
            input_dims (dict): Dictionary of input dimensions for each modality
                               (e.g., {'eeg': 100, 'audio': 200, 'text': 300})
            modality_hidden_dims (list): List of hidden layer dimensions for each modality
            fusion_hidden_dims (list): List of hidden layer dimensions for fusion networks
            shared_hidden_dims (list): List of hidden layer dimensions for the shared network
            num_classes (int): Number of output classes (1 for binary classification)
            dropout_rate (float): Dropout rate for regularization
        """
        super(HierarchicalFusionModel, self).__init__()
        
        self.input_dims = input_dims
        self.modality_hidden_dims = modality_hidden_dims
        self.fusion_hidden_dims = fusion_hidden_dims
        self.shared_hidden_dims = shared_hidden_dims
        self.num_classes = num_classes
        
        # Create separate encoders for each modality
        self.modality_encoders = nn.ModuleDict()
        
        for modality, input_dim in input_dims.items():
            layers = []
            
            # Input layer
            layers.append(nn.Linear(input_dim, modality_hidden_dims[0]))
            layers.append(nn.BatchNorm1d(modality_hidden_dims[0]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            
            # Hidden layers
            for i in range(len(modality_hidden_dims) - 1):
                layers.append(nn.Linear(modality_hidden_dims[i], modality_hidden_dims[i + 1]))
                layers.append(nn.BatchNorm1d(modality_hidden_dims[i + 1]))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout_rate))
            
            # Combine layers
            self.modality_encoders[modality] = nn.Sequential(*layers)
        
        # Define fusion hierarchy
        modalities = list(input_dims.keys())
        
        # First level fusion: EEG + Audio
        self.fusion_eeg_audio = nn.Sequential(
            nn.Linear(modality_hidden_dims[-1] * 2, fusion_hidden_dims[0]),
            nn.BatchNorm1d(fusion_hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Second level fusion: (EEG + Audio) + Text
        self.fusion_final = nn.Sequential(
            nn.Linear(fusion_hidden_dims[0] + modality_hidden_dims[-1], shared_hidden_dims[0]),
            nn.BatchNorm1d(shared_hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Shared network after fusion
        shared_layers = []
        
        # Hidden layers
        for i in range(len(shared_hidden_dims) - 1):
            shared_layers.append(nn.Linear(shared_hidden_dims[i], shared_hidden_dims[i + 1]))
            shared_layers.append(nn.BatchNorm1d(shared_hidden_dims[i + 1]))
            shared_layers.append(nn.ReLU())
            shared_layers.append(nn.Dropout(dropout_rate))
        
        # Output layer
        shared_layers.append(nn.Linear(shared_hidden_dims[-1], num_classes))
        
        # Combine layers
        self.shared_network = nn.Sequential(*shared_layers)
    
    def forward(self, x_dict):
        """
        Forward pass through the network.
        
        Args:
            x_dict (dict): Dictionary of input tensors for each modality
                          (e.g., {'eeg': eeg_tensor, 'audio': audio_tensor, 'text': text_tensor})
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_classes)
        """
        # Process each modality
        encoded_features = {}
        for modality, encoder in self.modality_encoders.items():
            encoded_features[modality] = encoder(x_dict[modality])
        
        # First level fusion: EEG + Audio
        eeg_audio_concat = torch.cat([encoded_features['eeg'], encoded_features['audio']], dim=1)
        eeg_audio_fused = self.fusion_eeg_audio(eeg_audio_concat)
        
        # Second level fusion: (EEG + Audio) + Text
        final_concat = torch.cat([eeg_audio_fused, encoded_features['text']], dim=1)
        final_fused = self.fusion_final(final_concat)
        
        # Process through shared network
        output = self.shared_network(final_fused)
        
        return output


class EnsembleModel(nn.Module):
    """
    Ensemble Model for multimodal data.
    
    This model trains separate models for each modality and combines their predictions.
    """
    
    def __init__(self, input_dims, hidden_dims=[128, 64], ensemble_method='voting', 
                 num_classes=1, dropout_rate=0.5):
        """
        Initialize the Ensemble model.
        
        Args:
            input_dims (dict): Dictionary of input dimensions for each modality
                               (e.g., {'eeg': 100, 'audio': 200, 'text': 300})
            hidden_dims (list): List of hidden layer dimensions for each modality
            ensemble_method (str): Method for ensemble ('voting', 'averaging', or 'stacking')
            num_classes (int): Number of output classes (1 for binary classification)
            dropout_rate (float): Dropout rate for regularization
        """
        super(EnsembleModel, self).__init__()
        
        self.input_dims = input_dims
        self.hidden_dims = hidden_dims
        self.ensemble_method = ensemble_method
        self.num_classes = num_classes
        
        # Create separate models for each modality
        self.modality_models = nn.ModuleDict()
        
        for modality, input_dim in input_dims.items():
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
            self.modality_models[modality] = nn.Sequential(*layers)
        
        # For stacking ensemble
        if ensemble_method == 'stacking':
            self.stacking_layer = nn.Linear(len(input_dims) * num_classes, num_classes)
    
    def forward(self, x_dict):
        """
        Forward pass through the network.
        
        Args:
            x_dict (dict): Dictionary of input tensors for each modality
                          (e.g., {'eeg': eeg_tensor, 'audio': audio_tensor, 'text': text_tensor})
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_classes)
        """
        # Process each modality
        outputs = {}
        for modality, model in self.modality_models.items():
            outputs[modality] = model(x_dict[modality])
        
        # Ensemble
        if self.ensemble_method == 'voting':
            # Hard voting (for classification)
            if self.num_classes == 1:
                # Binary classification
                votes = torch.cat([torch.sign(output) for output in outputs.values()], dim=1)
                ensemble_output = torch.sign(torch.sum(votes, dim=1, keepdim=True))
            else:
                # Multi-class classification
                predictions = [torch.argmax(output, dim=1) for output in outputs.values()]
                predictions = torch.stack(predictions, dim=1)
                ensemble_output = torch.zeros(predictions.size(0), self.num_classes)
                for i in range(predictions.size(0)):
                    # Count votes for each class
                    for j in range(predictions.size(1)):
                        ensemble_output[i, predictions[i, j]] += 1
        
        elif self.ensemble_method == 'averaging':
            # Soft voting / averaging
            ensemble_output = torch.mean(torch.stack(list(outputs.values())), dim=0)
        
        elif self.ensemble_method == 'stacking':
            # Stacking
            stacking_input = torch.cat(list(outputs.values()), dim=1)
            ensemble_output = self.stacking_layer(stacking_input)
        
        else:
            raise ValueError(f"Unknown ensemble method: {self.ensemble_method}")
        
        return ensemble_output
