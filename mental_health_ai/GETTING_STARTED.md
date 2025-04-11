# Getting Started with Mental Health AI

This guide will help you set up and run the Mental Health AI project from scratch.

## Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/mental-health-ai.git
cd mental-health-ai
```

## Step 2: Set Up the Environment

### Create a Virtual Environment

#### On Windows:
```bash
python -m venv venv
venv\Scripts\activate
```

#### On macOS/Linux:
```bash
python -m venv venv
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

If the requirements.txt file is missing, install the following packages:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn torch streamlit nltk
```

## Step 3: Run a Demo

### Option 1: Run the Minimal Demo (Recommended for First-Time Setup)

The minimal demo requires only basic dependencies and runs quickly:

#### On Windows:
```bash
run_minimal_demo.bat
```

#### On macOS/Linux:
```bash
chmod +x run_minimal_demo.sh
./run_minimal_demo.sh
```

This will:
1. Set up a virtual environment
2. Install minimal dependencies (numpy, matplotlib, scikit-learn)
3. Generate synthetic data
4. Train three models (LogisticRegression, RandomForest, NaiveBayes)
5. Evaluate the models
6. Generate comparison visualizations in the `minimal_results` directory

### Option 2: Run the Full Quick Demo

The full quick demo trains and evaluates three different neural network models on synthetic data:

#### On Windows:
```bash
run_demo.bat
```

#### On macOS/Linux:
```bash
chmod +x run_demo.sh
./run_demo.sh
```

Or manually:
```bash
python quick_demo.py
```

This will:
1. Generate synthetic data
2. Train three models (SimpleNN, LSTM, CNN)
3. Evaluate the models
4. Generate comparison visualizations in the `demo_results` directory

## Step 4: Generate Visualizations

To generate comprehensive visualizations for the project:

```bash
python generate_visualizations.py
```

This will create visualizations in the `visualizations` directory, organized into:
- `architecture/`: Model architecture diagrams
- `roc/`: ROC curve analysis
- `loss/`: Loss comparison visualizations
- `comparison/`: Model comparison visualizations

## Step 5: Run the Chatbot Interface

To run the chatbot interface:

```bash
python run_pipeline.py --step chatbot
```

This will start a Streamlit web application that you can access in your browser.

## Step 6: Process Real Data (Optional)

To preprocess real EEG, audio, and text data:

```bash
python preprocess_all_data.py
```

You can also preprocess specific modalities:

```bash
python preprocess_eeg_data.py    # Only preprocess EEG data
python preprocess_audio_data.py  # Only preprocess audio data
python preprocess_text_data.py   # Only preprocess text data
```

## Step 7: Train Models on Real Data (Optional)

To train models on real data:

```bash
python train_all_models.py
```

You can also train specific modalities:

```bash
python train_all_models.py --modality eeg    # Only train EEG models
python train_all_models.py --modality audio  # Only train audio models
python train_all_models.py --modality text   # Only train text models
```

## Step 8: Run the Full Pipeline (Optional)

To run the entire pipeline (preprocessing, training, and chatbot):

```bash
python run_pipeline.py
```

## Troubleshooting

### Missing Directories

If you encounter errors about missing directories, create them manually:

```bash
mkdir -p data/eeg/raw data/eeg/processed
mkdir -p data/audio/raw data/audio/processed
mkdir -p data/text/raw data/text/processed
mkdir -p models
mkdir -p results/eeg results/audio results/text results/fusion
mkdir -p visualizations
```

### CUDA Issues

If you encounter CUDA-related errors, the models will fall back to CPU. This is slower but will still work.

### Dependency Issues

If you encounter dependency issues, try installing the specific versions:

```bash
pip install torch==1.9.0
pip install numpy==1.21.0
pip install pandas==1.3.0
pip install matplotlib==3.4.2
pip install seaborn==0.11.1
pip install scikit-learn==0.24.2
pip install streamlit==1.0.0
pip install nltk==3.6.2
```

## Project Structure Overview

```
mental_health_ai/
├── data/                      # Data handling modules
│   ├── eeg/                   # EEG data processing
│   ├── audio/                 # Audio data processing
│   └── text/                  # Text data processing
├── models/                    # Model implementations
├── visualizations/            # Visualization outputs
├── demo_results/              # Quick demo results
├── quick_demo.py              # Quick demo script
├── generate_visualizations.py # Visualization script
├── preprocess_all_data.py     # Data preprocessing script
├── train_all_models.py        # Model training script
└── run_pipeline.py            # Complete pipeline execution
```

## For More Information

See the detailed README files:
- `README.md`: Basic project information
- `README_DETAILED.md`: Detailed project documentation

## Contact

If you encounter any issues, please contact the project maintainer at your.email@example.com.
