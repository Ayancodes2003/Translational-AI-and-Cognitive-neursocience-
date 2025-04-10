"""
Web Interface for Mental Health AI

This module provides a Streamlit web interface for the Mental Health AI system.
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import json
import tempfile
import librosa
import librosa.display
import mne
import io
from PIL import Image
import base64
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from clinical_insights.risk_assessment import RiskAssessor
from clinical_insights.modality_contribution import ModalityContributionAnalyzer
from clinical_insights.report_generator import ClinicalReportGenerator
from utils.visualization import plot_eeg_signals, plot_spectrogram, plot_modality_contribution


# Set page configuration
st.set_page_config(
    page_title="Mental Health AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Define functions for loading models
@st.cache_resource
def load_model(model_path):
    """
    Load a trained model.
    
    Args:
        model_path (str): Path to the model checkpoint
    
    Returns:
        nn.Module: Loaded model
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(model_path, map_location=device)
    
    # TODO: Create model based on checkpoint and load state dict
    
    model.eval()
    return model, device


@st.cache_data
def load_sample_data():
    """
    Load sample data for demonstration.
    
    Returns:
        tuple: (eeg_data, audio_data, text_data)
    """
    # TODO: Load sample data
    
    return eeg_data, audio_data, text_data


# Define functions for processing uploads
def process_eeg_upload(uploaded_file):
    """
    Process uploaded EEG file.
    
    Args:
        uploaded_file (UploadedFile): Uploaded EEG file
    
    Returns:
        numpy.ndarray: Processed EEG data
    """
    # Save uploaded file to temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.edf') as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name
    
    try:
        # Load EEG data
        raw = mne.io.read_raw_edf(tmp_path, preload=True)
        
        # Extract data
        data = raw.get_data()
        
        # Preprocess data
        # TODO: Implement preprocessing
        
        return data
    finally:
        # Remove temporary file
        os.unlink(tmp_path)


def process_audio_upload(uploaded_file):
    """
    Process uploaded audio file.
    
    Args:
        uploaded_file (UploadedFile): Uploaded audio file
    
    Returns:
        numpy.ndarray: Processed audio data
    """
    # Save uploaded file to temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name
    
    try:
        # Load audio data
        audio, sr = librosa.load(tmp_path, sr=16000)
        
        # Preprocess audio
        # TODO: Implement preprocessing
        
        return audio
    finally:
        # Remove temporary file
        os.unlink(tmp_path)


def process_text_input(text):
    """
    Process text input.
    
    Args:
        text (str): Input text
    
    Returns:
        str: Processed text
    """
    # Preprocess text
    # TODO: Implement preprocessing
    
    return text


# Define functions for visualization
def plot_eeg(eeg_data, channel_names=None):
    """
    Plot EEG data.
    
    Args:
        eeg_data (numpy.ndarray): EEG data
        channel_names (list, optional): Channel names
    
    Returns:
        matplotlib.figure.Figure: Figure object
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot EEG data
    if channel_names is None:
        channel_names = [f'Channel {i+1}' for i in range(eeg_data.shape[0])]
    
    for i in range(min(5, eeg_data.shape[0])):  # Plot up to 5 channels
        plt.plot(eeg_data[i], label=channel_names[i])
    
    plt.xlabel('Time (samples)')
    plt.ylabel('Amplitude')
    plt.title('EEG Signal')
    plt.legend()
    
    return fig


def plot_audio(audio_data, sr=16000):
    """
    Plot audio waveform and spectrogram.
    
    Args:
        audio_data (numpy.ndarray): Audio data
        sr (int): Sampling rate
    
    Returns:
        matplotlib.figure.Figure: Figure object
    """
    fig, ax = plt.subplots(2, 1, figsize=(10, 8))
    
    # Plot waveform
    librosa.display.waveshow(audio_data, sr=sr, ax=ax[0])
    ax[0].set_title('Audio Waveform')
    
    # Plot spectrogram
    D = librosa.amplitude_to_db(np.abs(librosa.stft(audio_data)), ref=np.max)
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log', ax=ax[1])
    ax[1].set_title('Audio Spectrogram')
    fig.colorbar(ax[1].collections[0], ax=ax[1], format='%+2.0f dB')
    
    plt.tight_layout()
    
    return fig


def get_report_html(report):
    """
    Generate HTML for clinical report.
    
    Args:
        report (dict): Clinical report
    
    Returns:
        str: HTML string
    """
    html = f"""
    <div style="padding: 20px; border: 1px solid #ddd; border-radius: 5px; background-color: #f9f9f9;">
        <h2 style="text-align: center; color: #333;">Mental Health Assessment Report</h2>
        <p style="text-align: center; color: #666;">Generated on: {report['timestamp']}</p>
        
        <div style="margin-top: 20px;">
            <h3>Depression Assessment</h3>
            <p><strong>Depression Probability:</strong> {report['depression_probability']:.1%}</p>
    """
    
    if 'phq8_score' in report:
        html += f"""
            <p><strong>PHQ-8 Score:</strong> {report['phq8_score']}</p>
        """
        if 'phq8_interpretation' in report:
            html += f"""
            <p><strong>Interpretation:</strong> {report['phq8_interpretation']}</p>
            """
    
    if 'risk_level' in report:
        html += f"""
        <div style="margin-top: 20px;">
            <h3>Risk Assessment</h3>
            <p><strong>Risk Level:</strong> {report['risk_level']}</p>
        </div>
        """
    
    if 'observations' in report:
        html += f"""
        <div style="margin-top: 20px;">
            <h3>Clinical Observations</h3>
            <ul>
        """
        for observation in report['observations']:
            html += f"""
                <li>{observation}</li>
            """
        html += f"""
            </ul>
        </div>
        """
    
    if 'suggestions' in report:
        html += f"""
        <div style="margin-top: 20px;">
            <h3>Recommendations</h3>
            <ul>
        """
        for suggestion in report['suggestions']:
            html += f"""
                <li>{suggestion}</li>
            """
        html += f"""
            </ul>
        </div>
        """
    
    html += """
    </div>
    """
    
    return html


# Main application
def main():
    # Sidebar
    st.sidebar.title("Mental Health AI")
    st.sidebar.image("https://img.icons8.com/color/96/000000/brain--v2.png", width=100)
    
    # Navigation
    page = st.sidebar.selectbox("Navigation", ["Home", "Assessment", "About"])
    
    if page == "Home":
        show_home_page()
    elif page == "Assessment":
        show_assessment_page()
    elif page == "About":
        show_about_page()


def show_home_page():
    st.title("AI-Enabled Mental Healthcare")
    st.subheader("Multimodal Deep Learning for Mental Health Assessment")
    
    st.markdown("""
    Welcome to the Mental Health AI platform, a comprehensive system for mental health assessment using multimodal deep learning.
    
    ### Features
    
    - **Multimodal Analysis**: Combines EEG signals, audio recordings, and text data for a holistic assessment
    - **Advanced AI Models**: Utilizes state-of-the-art deep learning models for accurate predictions
    - **Clinical Insights**: Provides risk assessment, modality contributions, and personalized recommendations
    - **Interactive Interface**: Upload your own data or use sample data for demonstration
    
    ### How It Works
    
    1. **Data Collection**: The system analyzes three types of data:
       - EEG signals capturing brain activity
       - Audio recordings capturing speech patterns
       - Text data capturing linguistic patterns
    
    2. **AI Analysis**: Advanced deep learning models process each modality and combine them for comprehensive assessment
    
    3. **Clinical Insights**: The system generates a detailed report with risk assessment, observations, and recommendations
    
    ### Get Started
    
    Navigate to the **Assessment** page to try out the system with your own data or sample data.
    """)
    
    st.image("https://img.icons8.com/color/96/000000/mental-health.png", width=100)


def show_assessment_page():
    st.title("Mental Health Assessment")
    
    # Initialize session state
    if 'eeg_data' not in st.session_state:
        st.session_state.eeg_data = None
    if 'audio_data' not in st.session_state:
        st.session_state.audio_data = None
    if 'text_data' not in st.session_state:
        st.session_state.text_data = None
    if 'report' not in st.session_state:
        st.session_state.report = None
    
    # Data input section
    st.header("Data Input")
    
    # Create tabs for different input methods
    tab1, tab2 = st.tabs(["Upload Your Data", "Use Sample Data"])
    
    with tab1:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("EEG Data")
            eeg_file = st.file_uploader("Upload EEG file (EDF format)", type=["edf"])
            if eeg_file is not None:
                st.session_state.eeg_data = process_eeg_upload(eeg_file)
                st.success("EEG data uploaded successfully!")
        
        with col2:
            st.subheader("Audio Data")
            audio_file = st.file_uploader("Upload audio file (WAV format)", type=["wav"])
            if audio_file is not None:
                st.session_state.audio_data = process_audio_upload(audio_file)
                st.success("Audio data uploaded successfully!")
        
        with col3:
            st.subheader("Text Data")
            text_input = st.text_area("Enter text (e.g., speech transcript, journal entry)")
            if text_input:
                st.session_state.text_data = process_text_input(text_input)
                st.success("Text data processed successfully!")
    
    with tab2:
        if st.button("Load Sample Data"):
            eeg_data, audio_data, text_data = load_sample_data()
            st.session_state.eeg_data = eeg_data
            st.session_state.audio_data = audio_data
            st.session_state.text_data = text_data
            st.success("Sample data loaded successfully!")
    
    # Data visualization section
    if st.session_state.eeg_data is not None or st.session_state.audio_data is not None or st.session_state.text_data is not None:
        st.header("Data Visualization")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.session_state.eeg_data is not None:
                st.subheader("EEG Signal")
                fig = plot_eeg(st.session_state.eeg_data)
                st.pyplot(fig)
        
        with col2:
            if st.session_state.audio_data is not None:
                st.subheader("Audio Signal")
                fig = plot_audio(st.session_state.audio_data)
                st.pyplot(fig)
        
        if st.session_state.text_data is not None:
            st.subheader("Text Data")
            st.text_area("Processed Text", st.session_state.text_data, height=100, disabled=True)
    
    # Analysis section
    if st.session_state.eeg_data is not None or st.session_state.audio_data is not None or st.session_state.text_data is not None:
        st.header("Analysis")
        
        if st.button("Generate Assessment"):
            with st.spinner("Analyzing data..."):
                # TODO: Implement actual analysis
                
                # For demonstration, create a mock report
                st.session_state.report = {
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'depression_probability': 0.65,
                    'risk_level': 'Moderate',
                    'modality_contributions': {
                        'eeg': 0.4,
                        'audio': 0.35,
                        'text': 0.25
                    },
                    'observations': [
                        "Moderate probability of depression detected.",
                        "EEG patterns show significant indicators of altered brain activity.",
                        "Speech patterns show notable changes in vocal characteristics.",
                        "Overall risk assessment indicates moderate risk for depression. Regular monitoring is recommended."
                    ],
                    'suggestions': [
                        "Consider consulting a mental health professional",
                        "Increase self-care activities",
                        "Monitor mood changes",
                        "Practice mindfulness and relaxation techniques"
                    ]
                }
                
                st.success("Assessment generated successfully!")
    
    # Results section
    if st.session_state.report is not None:
        st.header("Assessment Results")
        
        # Display report
        st.markdown(get_report_html(st.session_state.report), unsafe_allow_html=True)
        
        # Display modality contributions
        if 'modality_contributions' in st.session_state.report:
            st.subheader("Modality Contributions")
            
            contributions = st.session_state.report['modality_contributions']
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Bar chart
                fig, ax = plt.subplots(figsize=(8, 5))
                ax.bar(contributions.keys(), contributions.values())
                ax.set_xlabel('Modality')
                ax.set_ylabel('Contribution')
                ax.set_title('Modality Contribution - Bar Chart')
                st.pyplot(fig)
            
            with col2:
                # Pie chart
                fig, ax = plt.subplots(figsize=(8, 5))
                ax.pie(contributions.values(), labels=contributions.keys(), autopct='%1.1f%%')
                ax.set_title('Modality Contribution - Pie Chart')
                st.pyplot(fig)
        
        # Download report
        st.download_button(
            label="Download Report (JSON)",
            data=json.dumps(st.session_state.report, indent=4),
            file_name="mental_health_report.json",
            mime="application/json"
        )


def show_about_page():
    st.title("About Mental Health AI")
    
    st.markdown("""
    ### Project Overview
    
    Mental Health AI is a comprehensive system for mental health assessment using multimodal deep learning. The system analyzes EEG signals, audio recordings, and text data to provide a holistic assessment of mental health, with a focus on depression detection.
    
    ### Technical Details
    
    The system uses a variety of deep learning models:
    
    - **EEG Models**: CNN, LSTM, BiLSTM with Attention, 1D CNN, 1D CNN + GRU, Transformer
    - **Audio Models**: CNN, LSTM, BiLSTM with Attention, 2D CNN, 1D CNN + GRU, Transformer
    - **Text Models**: CNN, LSTM, BiLSTM with Attention, 1D CNN, BERT, Transformer
    - **Fusion Models**: Early Fusion, Late Fusion, Intermediate Fusion, Cross-Modal Attention, Hierarchical Fusion, Ensemble
    
    ### Datasets
    
    The system is trained on publicly available datasets:
    
    - **DEAP Dataset**: EEG and physiological signals for emotion analysis
    - **DAIC-WOZ Dataset**: Audio, video, and text data for depression analysis
    
    ### References
    
    1. Koelstra, S., et al. (2012). DEAP: A Database for Emotion Analysis using Physiological Signals. IEEE Transactions on Affective Computing, 3(1), 18-31.
    
    2. Gratch, J., et al. (2014). The Distress Analysis Interview Corpus of Human and Computer Interviews. Proceedings of the Ninth International Conference on Language Resources and Evaluation (LREC'14).
    
    ### Contact
    
    For more information, please contact [email@example.com](mailto:email@example.com).
    """)


if __name__ == "__main__":
    main()
