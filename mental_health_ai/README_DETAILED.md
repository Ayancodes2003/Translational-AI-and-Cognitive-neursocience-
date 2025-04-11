# Mental Health AI: Multimodal Depression Detection

## Project Overview

This project implements an AI-enabled mental healthcare system using multimodal deep learning to detect and assess depression. The system analyzes three types of data:

1. **EEG Data**: Brain activity patterns associated with depression
2. **Audio Data**: Speech characteristics that may indicate depression
3. **Text Data**: Linguistic patterns and content that correlate with depressive symptoms

By combining these modalities, the system provides a more comprehensive assessment than single-modality approaches.

## Key Features

- **Multimodal Analysis**: Integrates EEG, audio, and text data for holistic assessment
- **Multiple Model Architectures**: Implements and compares various deep learning models
- **Real-time Processing**: Processes user input in real-time for immediate feedback
- **Interactive Interface**: Provides a chatbot-like interface for user interaction
- **Clinical Insights**: Generates reports with observations and suggestions

## Data Sources

### EEG Data
- **MNE Sample Dataset**: Standard neuroimaging dataset with labeled EEG recordings
- **EEG Motor Movement/Imagery Dataset**: Contains EEG recordings during motor tasks

### Audio Data
- **RAVDESS**: Ryerson Audio-Visual Database of Emotional Speech and Song
- **CREMA-D**: Crowd-sourced Emotional Multimodal Actors Dataset

### Text Data
- **Emotion Dataset**: Text samples labeled with emotional content
- **Go Emotions**: Fine-grained emotion classification dataset
- **Tweet Eval**: Sentiment analysis dataset from Twitter

## Model Architectures

### EEG Models

1. **EEGNet**
   - Compact CNN designed specifically for EEG data
   - Efficient spatial and temporal filtering
   - Low parameter count for faster training

2. **DeepConvNet**
   - Deep convolutional network with multiple layers
   - Extracts hierarchical features from EEG signals
   - Higher capacity for complex pattern recognition

3. **ShallowConvNet**
   - Simpler convolutional architecture
   - Focuses on frequency-domain features
   - Computationally efficient

4. **EEGCNN**
   - Custom CNN architecture for EEG classification
   - Multiple convolutional and pooling layers
   - Fully connected layers for classification

5. **EEGLSTM**
   - Recurrent neural network for temporal EEG patterns
   - Captures long-term dependencies in brain activity
   - Bidirectional processing for context awareness

6. **EEGTransformer**
   - Attention-based architecture for EEG analysis
   - Self-attention mechanisms for feature importance
   - Positional encoding for temporal information

### Audio Models

1. **AudioCNN**
   - Convolutional network for spectral audio features
   - Processes mel-spectrograms as 2D images
   - Extracts frequency and temporal patterns

2. **AudioLSTM**
   - Recurrent network for sequential audio processing
   - Captures temporal dynamics in speech
   - Effective for prosodic feature analysis

3. **AudioCRNN**
   - Combined CNN-RNN architecture
   - CNN layers extract local features
   - LSTM layers model temporal relationships

### Text Models

1. **TextCNN**
   - Convolutional network for text classification
   - Multiple filter sizes for n-gram processing
   - Parallel convolutional operations

2. **TextLSTM**
   - LSTM network for sequential text processing
   - Word embedding input layer
   - Captures long-range dependencies

3. **TextBiLSTM**
   - Bidirectional LSTM for text analysis
   - Processes text in both directions
   - Enhanced context awareness

### Fusion Models

1. **EarlyFusion**
   - Combines raw features from all modalities
   - Joint feature learning
   - Single classification head

2. **LateFusion**
   - Trains separate models for each modality
   - Combines predictions at decision level
   - Weighted averaging of model outputs

3. **HierarchicalFusion**
   - Multi-stage fusion approach
   - Intermediate feature fusion
   - Attention mechanisms for modality importance

## Implementation Details

### Data Preprocessing

#### EEG Preprocessing
```python
# EEG preprocessing pipeline
1. Load raw EEG data
2. Apply bandpass filtering (4-45 Hz)
3. Remove artifacts and noise
4. Extract frequency band features (delta, theta, alpha, beta, gamma)
5. Compute statistical features (mean, std, entropy)
6. Normalize features
```

#### Audio Preprocessing
```python
# Audio preprocessing pipeline
1. Load audio files
2. Extract mel-spectrograms
3. Compute MFCCs (Mel-Frequency Cepstral Coefficients)
4. Extract prosodic features (pitch, energy, speaking rate)
5. Normalize features
```

#### Text Preprocessing
```python
# Text preprocessing pipeline
1. Tokenize text
2. Remove stop words and punctuation
3. Apply stemming or lemmatization
4. Extract linguistic features (lexical diversity, sentiment)
5. Generate word embeddings
```

### Model Training

The training process follows these steps:

1. **Data Loading**: Load preprocessed data from each modality
2. **Dataset Creation**: Create PyTorch datasets and dataloaders
3. **Model Initialization**: Initialize model architecture with appropriate parameters
4. **Training Loop**: Train model with backpropagation and optimization
5. **Validation**: Monitor performance on validation set
6. **Early Stopping**: Prevent overfitting by stopping when validation loss plateaus
7. **Model Saving**: Save best model based on validation performance

Example training code:
```python
# Train model
def train_model(model, train_loader, val_loader, device, num_epochs=50):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{num_epochs}: "
              f"Train Loss = {train_loss/len(train_loader):.4f}, "
              f"Val Loss = {val_loss/len(val_loader):.4f}")
```

### Evaluation Metrics

We evaluate models using the following metrics:

1. **Accuracy**: Overall correct classification rate
2. **Precision**: True positives / (True positives + False positives)
3. **Recall**: True positives / (True positives + False negatives)
4. **F1 Score**: Harmonic mean of precision and recall
5. **AUC-ROC**: Area under the Receiver Operating Characteristic curve
6. **Confusion Matrix**: Visualization of prediction errors

### Visualization Types

The project includes various visualizations:

1. **Architecture Diagrams**: Visual representations of model architectures
2. **Training Curves**: Loss and accuracy during training
3. **ROC Curves**: True positive rate vs. false positive rate
4. **Confusion Matrices**: Visualization of model predictions
5. **Feature Importance**: Attention weights and feature contributions
6. **Model Comparisons**: Performance metrics across different models

## Project Structure

```
mental_health_ai/
├── data/                      # Data handling modules
│   ├── dataset_loader.py      # Dataset loading utilities
│   ├── eeg/                   # EEG data processing
│   │   └── preprocess_eeg.py  # EEG preprocessing
│   ├── audio/                 # Audio data processing
│   │   └── preprocess_audio.py # Audio preprocessing
│   └── text/                  # Text data processing
│       └── preprocess_text.py # Text preprocessing
├── models/                    # Model implementations
│   ├── eeg_models.py          # EEG model architectures
│   ├── audio_models.py        # Audio model architectures
│   ├── text_models.py         # Text model architectures
│   └── fusion_models.py       # Multimodal fusion models
├── train/                     # Training scripts
│   ├── train_eeg_model.py     # EEG model training
│   ├── train_audio_model.py   # Audio model training
│   ├── train_text_model.py    # Text model training
│   └── train_fusion_model.py  # Fusion model training
├── evaluate/                  # Evaluation scripts
│   └── evaluate.py            # Model evaluation
├── visualizations/            # Visualization outputs
│   ├── architecture/          # Model architecture diagrams
│   ├── training/              # Training curves
│   ├── roc/                   # ROC curve analysis
│   └── comparison/            # Model comparison charts
├── clinical_insights/         # Clinical interpretation modules
│   ├── risk_assessment.py     # Depression risk assessment
│   └── report_generator.py    # Clinical report generation
├── app.py                     # Main application
├── trained_chatbot.py         # Chatbot interface
├── preprocess_all_data.py     # Data preprocessing script
├── train_all_models.py        # Model training script
└── run_pipeline.py            # Complete pipeline execution
```

## Running the Pipeline

You can run the entire pipeline with a single command:

```bash
python run_pipeline.py
```

This will:
1. Preprocess all data (EEG, audio, text)
2. Train all models
3. Run the chatbot interface

You can also run specific steps of the pipeline:

```bash
python run_pipeline.py --step preprocess  # Only preprocess data
python run_pipeline.py --step train       # Only train models
python run_pipeline.py --step chatbot     # Only run chatbot
```

Or focus on specific modalities:

```bash
python run_pipeline.py --modality eeg    # Only process EEG data
python run_pipeline.py --modality audio  # Only process audio data
python run_pipeline.py --modality text   # Only process text data
```

## Model Comparison Results

Our preliminary experiments show the following performance comparison:

| Model    | Accuracy | Precision | Recall  | F1 Score | AUC     | Training Time (s) |
|----------|----------|-----------|---------|----------|---------|-------------------|
| SimpleNN | 0.855    | 0.798     | 0.882   | 0.838    | 0.941   | 0.29              |
| LSTM     | 0.500    | 0.450     | 0.788   | 0.573    | 0.532   | 2.13              |
| CNN      | 0.900    | 0.849     | 0.929   | 0.888    | 0.970   | 0.86              |

The CNN model achieves the best performance across all metrics, with an accuracy of 90% and an AUC of 0.97.

## Future Work

1. **Advanced Architectures**: Implement transformer-based models for each modality
2. **Cross-modal Attention**: Explore attention mechanisms between modalities
3. **Explainable AI**: Enhance model interpretability for clinical applications
4. **Real-time Processing**: Optimize for real-time inference
5. **Clinical Validation**: Validate system with clinical datasets and expert feedback

## Dependencies

- Python 3.8+
- PyTorch 1.9+
- NumPy
- Pandas
- Scikit-learn
- MNE (for EEG processing)
- Librosa (for audio processing)
- NLTK and Transformers (for text processing)
- Matplotlib and Seaborn (for visualization)
- Streamlit (for web interface)

## References

1. Lawhern, V. J., et al. (2018). EEGNet: A compact convolutional neural network for EEG-based brain-computer interfaces. Journal of Neural Engineering.
2. Schirrmeister, R. T., et al. (2017). Deep learning with convolutional neural networks for EEG decoding and visualization. Human Brain Mapping.
3. Livingstone, S. R., & Russo, F. A. (2018). The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS). PLoS ONE.
4. Cao, H., et al. (2014). CREMA-D: Crowd-sourced Emotional Multimodal Actors Dataset. IEEE Transactions on Affective Computing.
5. Kim, Y. (2014). Convolutional neural networks for sentence classification. EMNLP.
6. Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. Neural Computation.
