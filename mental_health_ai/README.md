# AI-Enabled Mental Healthcare

A comprehensive Python project for mental health analysis using multimodal deep learning with EEG, audio, and text data.

## Overview

This project aims to develop advanced AI models for mental health assessment, focusing on depression detection and risk level estimation. By leveraging multimodal data (EEG signals, audio recordings, and text transcripts), the system provides a holistic approach to mental health analysis.

## Features

- **Multimodal Data Processing**: Specialized preprocessing pipelines for EEG, audio, and text data
- **Advanced Model Architectures**: Implementation of CNNs, LSTMs, Transformers, and custom fusion models
- **Comprehensive Evaluation**: Multiple metrics including accuracy, F1-score, precision, recall, and confusion matrices
- **Clinical Insights**: Risk level assessment, modality contribution analysis, and suggested observations
- **Visualization Tools**: Interactive visualizations for model interpretability
- **Web Interface**: Optional Streamlit/Gradio demo for interactive inference

## Project Structure

```
mental_health_ai/
├── data/                  # Data loading and preprocessing
│   ├── eeg/               # EEG data processing
│   ├── audio/             # Audio data processing
│   └── text/              # Text data processing
├── models/                # Model architectures
│   ├── eeg_models.py      # EEG-specific models
│   ├── audio_models.py    # Audio-specific models
│   ├── text_models.py     # Text-specific models
│   └── fusion_models.py   # Multimodal fusion models
├── train/                 # Training and evaluation
│   ├── train.py           # Main training script
│   ├── evaluate.py        # Evaluation metrics
│   └── config.py          # Training configurations
├── utils/                 # Helper functions
│   ├── data_utils.py      # Data utilities
│   ├── model_utils.py     # Model utilities
│   └── visualization.py   # Basic visualization tools
├── notebooks/             # Exploratory analysis
├── visualization/         # Advanced visualization tools
├── clinical_insights/     # Clinical interpretation modules
└── app.py                 # Streamlit/Gradio web interface
```

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/mental_health_ai.git
cd mental_health_ai

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Data Preparation

1. Download the DEAP dataset for EEG data
2. Download the DAIC-WOZ dataset for audio and text data
3. Run the preprocessing scripts:
   ```
   python -m data.preprocess_eeg
   python -m data.preprocess_audio
   python -m data.preprocess_text
   ```

### Training Models

```bash
# Train a single modality model
python -m train.train --modality eeg --model cnn

# Train a fusion model
python -m train.train --modality fusion --fusion_type late
```

### Evaluation

```bash
python -m train.evaluate --model_path models/saved/fusion_model.pt
```

### Web Interface

We provide two options for running the web interface:

1. **Full Web Interface**: This includes all features of the Mental Health AI system.

```bash
streamlit run app.py
```

2. **Simple Demo Interface**: This is a lightweight version that uses synthetic data for demonstration.

```bash
streamlit run streamlit_app.py
```

## Results

The project evaluates multiple model architectures:
- CNN (baseline for EEG and audio)
- LSTM (for time-sequenced data)
- Transformer (for text)
- BiLSTM + Attention
- 1D CNN + GRU
- Multimodal fusion model (custom)
- Ensemble model
- Pretrained BERT (text) + CNN (EEG/audio) fusion

Performance metrics and visualizations are available in the `results` directory.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- DEAP dataset for EEG data
- DAIC-WOZ dataset for audio and text data
- All contributors and researchers in the field of AI for mental health
