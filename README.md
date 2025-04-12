# Translational AI and Cognitive Neuroscience

A comprehensive AI-enabled mental healthcare project using multimodal deep learning with EEG, audio, and text data.

## Project Overview

This project implements a suite of deep learning models for mental health classification using three different modalities:
- EEG (electroencephalogram) data
- Audio data
- Text data

The models are designed to detect signs of depression and provide insights for mental healthcare professionals.

## Features

- **Multimodal Analysis**: Process and analyze EEG signals, audio recordings, and text data
- **Multiple Model Architectures**: Implementation of various neural network architectures for each modality
- **Comprehensive Evaluation**: Detailed performance metrics and visualizations
- **Cross-Modality Comparison**: Compare the effectiveness of different data modalities
- **Real Dataset Support**: Trained and evaluated on real datasets

## Project Structure

```
mental_health_ai/
├── data/                  # Data processing modules
│   ├── audio/             # Audio data processing
│   ├── eeg/               # EEG data processing
│   └── text/              # Text data processing
├── models/                # Model architectures
│   ├── audio_models.py    # Audio model architectures
│   ├── eeg_models.py      # EEG model architectures
│   ├── text_models.py     # Text model architectures
│   └── fusion_models.py   # Multimodal fusion models
├── train/                 # Training scripts
├── utils/                 # Utility functions
├── clinical_insights/     # Clinical analysis tools
└── results/               # Results and visualizations
    ├── audio/             # Audio model results
    ├── eeg/               # EEG model results
    ├── text/              # Text model results
    └── comparison/        # Cross-modality comparisons
```

## Model Architectures

### EEG Models
- SimpleModel: A basic neural network model with configurable hidden layers
- EEGNet: A compact convolutional neural network designed specifically for EEG signal processing
- DeepConvNet: A deeper CNN architecture for EEG classification
- ShallowConvNet: A shallow CNN architecture optimized for motor imagery classification
- EEGCNN: A 1D CNN model for temporal EEG signal processing
- EEGLSTM: An LSTM-based model for sequential EEG data processing
- EEGTransformer: A transformer-based model for capturing long-range dependencies in EEG signals

### Audio Models
- AudioCNN: A convolutional neural network for audio feature classification
- AudioLSTM: An LSTM-based model for audio feature classification
- AudioCRNN: A convolutional recurrent neural network for audio feature classification
- AudioResNet: A ResNet-inspired model for audio feature classification

### Text Models
- TextCNN: A convolutional neural network for text feature classification
- TextLSTM: An LSTM-based model for text feature classification
- TextBiLSTM: A bidirectional LSTM-based model for text feature classification
- TextTransformer: A transformer-based model for text feature classification

## Results

### Cross-Modality Comparison (Best Models)

| Modality | Accuracy | Precision | Recall | F1 Score | AUC |
|----------|----------|-----------|--------|----------|-----|
| EEG      | 0.550    | 0.538     | 0.700  | 0.609    | 0.570 |
| Audio    | 0.930    | 0.925     | 0.943  | 0.934    | 0.973 |
| Text     | 0.925    | 0.933     | 0.925  | 0.929    | 0.970 |

## Getting Started

### Prerequisites

- Python 3.8+
- PyTorch 1.9+
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/translational-ai-cognitive-neuroscience.git
cd translational-ai-cognitive-neuroscience
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Usage

#### Training Models

To train all models on all modalities:
```bash
python train_all_models.py
```

To train models for a specific modality:
```bash
python train_all_eeg_models.py  # For EEG models
python train_all_audio_models.py  # For audio models
python train_all_text_models.py  # For text models
```

#### Comparing Results

To compare results across modalities:
```bash
python compare_all_modalities.py
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Thanks to all the researchers who have contributed to the field of AI in mental healthcare
- Special thanks to the open-source community for providing tools and libraries that made this project possible