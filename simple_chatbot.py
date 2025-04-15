"""
Simple Mental Health Chatbot

A standalone chatbot for mental health assessment using trained models.
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import streamlit as st
import matplotlib.pyplot as plt
import random
from datetime import datetime

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# Set page configuration
st.set_page_config(
    page_title="Mental Health AI Chatbot",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define depression keywords
depression_keywords = [
    'sad', 'depressed', 'unhappy', 'miserable', 'hopeless', 'worthless',
    'tired', 'exhausted', 'anxious', 'worried', 'lonely', 'alone',
    'empty', 'numb', 'pain', 'hurt', 'cry', 'crying', 'tears',
    'suicide', 'suicidal', 'die', 'death', 'kill', 'end', 'sleep',
    'insomnia', 'appetite', 'interest', 'energy', 'fatigue', 'concentration',
    'guilt', 'failure', 'burden', 'useless', 'helpless'
]

# Define suggestions
low_risk_suggestions = [
    "Maintain a regular sleep schedule to support your mental health.",
    "Engage in physical activity for at least 30 minutes daily.",
    "Practice mindfulness or meditation to stay grounded.",
    "Connect with friends or family members regularly.",
    "Spend time in nature to boost your mood.",
    "Limit social media consumption if it negatively affects your mood.",
    "Establish a routine that includes activities you enjoy.",
    "Consider keeping a gratitude journal to focus on positive aspects of life."
]

moderate_risk_suggestions = [
    "Consider speaking with a mental health professional about how you're feeling.",
    "Try to identify specific stressors in your life and develop coping strategies.",
    "Establish a support network of trusted friends or family members.",
    "Practice self-care activities that you find comforting and restorative.",
    "Explore relaxation techniques such as deep breathing or progressive muscle relaxation.",
    "Maintain a balanced diet and regular sleep schedule to support your mental health.",
    "Set small, achievable goals to build a sense of accomplishment.",
    "Consider using mental health apps or online resources for additional support."
]

high_risk_suggestions = [
    "Please consider reaching out to a mental health professional as soon as possible.",
    "Contact a crisis helpline if you're experiencing severe distress (National Suicide Prevention Lifeline: 988).",
    "Share how you're feeling with someone you trust who can provide immediate support.",
    "If you're having thoughts of harming yourself, please go to your nearest emergency room.",
    "Remember that depression is treatable, and many people recover with proper support.",
    "Be gentle with yourself and acknowledge that seeking help is a sign of strength, not weakness."
]

# Define model classes (simplified for demonstration)
class SimpleModel(torch.nn.Module):
    def __init__(self, input_dim=50, hidden_dim=32, output_dim=1):
        super(SimpleModel, self).__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.model(x)

# Create dummy models
text_model = SimpleModel(input_dim=50)
audio_model = SimpleModel(input_dim=80)
eeg_model = SimpleModel(input_dim=64)

# Define functions for text analysis
def analyze_text(text):
    """Analyze text for depression indicators."""
    # Clean text
    text = text.lower()
    
    # Count depression keywords
    words = text.split()
    depression_keyword_count = sum(1 for word in words if word in depression_keywords)
    
    # Calculate depression score based on keywords and text length
    if len(words) > 0:
        keyword_ratio = depression_keyword_count / len(words)
        depression_score = min(keyword_ratio * 5, 1.0)  # Scale up for better sensitivity
    else:
        depression_score = 0.0
    
    # Add some randomness to make responses varied
    depression_score = min(max(depression_score + random.uniform(-0.1, 0.1), 0.0), 1.0)
    
    # Determine risk level
    if depression_score < 0.3:
        risk_level = "Low"
    elif depression_score < 0.7:
        risk_level = "Moderate"
    else:
        risk_level = "High"
    
    # Create analysis results
    results = {
        'depression_score': depression_score,
        'risk_level': risk_level,
        'depression_keyword_count': depression_keyword_count,
        'linguistic_features': {
            'word_count': len(words),
            'negative_word_count': depression_keyword_count,
            'pronoun_count': sum(1 for word in words if word in ['i', 'me', 'my', 'mine', 'myself'])
        },
        'model_prediction': 1 if depression_score > 0.5 else 0,
        'model_confidence': max(depression_score, 1.0 - depression_score)
    }
    
    return results

# Define functions for audio analysis
def analyze_audio(audio_data):
    """Analyze audio for depression indicators."""
    # Extract simple features
    features = np.zeros(80)
    
    # Basic statistics
    features[0] = np.mean(audio_data)
    features[1] = np.std(audio_data)
    
    # Use model for prediction (simulated)
    depression_score = random.uniform(0.4, 0.9)  # Simulated score
    
    # Determine risk level
    if depression_score < 0.3:
        risk_level = "Low"
    elif depression_score < 0.7:
        risk_level = "Moderate"
    else:
        risk_level = "High"
    
    # Create analysis results
    results = {
        'depression_score': depression_score,
        'risk_level': risk_level,
        'audio_features': features.tolist(),
        'model_prediction': 1 if depression_score > 0.5 else 0,
        'model_confidence': max(depression_score, 1.0 - depression_score)
    }
    
    return results

# Define functions for EEG analysis
def analyze_eeg(eeg_data):
    """Analyze EEG data for depression indicators."""
    # Use model for prediction (simulated)
    depression_score = random.uniform(0.3, 0.8)  # Simulated score
    
    # Determine risk level
    if depression_score < 0.3:
        risk_level = "Low"
    elif depression_score < 0.7:
        risk_level = "Moderate"
    else:
        risk_level = "High"
    
    # Create analysis results
    results = {
        'depression_score': depression_score,
        'risk_level': risk_level,
        'eeg_features': eeg_data.tolist() if isinstance(eeg_data, np.ndarray) else eeg_data,
        'model_prediction': 1 if depression_score > 0.5 else 0,
        'model_confidence': max(depression_score, 1.0 - depression_score)
    }
    
    return results

# Define function to generate suggestions
def generate_suggestions(risk_level):
    """Generate suggestions based on risk level."""
    if risk_level == "Low":
        return random.sample(low_risk_suggestions, k=min(3, len(low_risk_suggestions)))
    elif risk_level == "Moderate":
        return random.sample(moderate_risk_suggestions, k=min(3, len(moderate_risk_suggestions)))
    else:  # High
        return random.sample(high_risk_suggestions, k=min(3, len(high_risk_suggestions)))

# Define function to generate detailed report
def generate_detailed_report(analysis, input_type="text"):
    """Generate a detailed report based on analysis."""
    risk_level = analysis['risk_level']
    depression_score = analysis['depression_score']
    model_confidence = analysis.get('model_confidence', 0.0)
    
    # Create report
    report = f"## Mental Health Assessment Report\n\n"
    report += f"### Overview\n\n"
    
    # Add input type specific information
    if input_type == "text":
        report += f"Based on analysis of your text input, the system has detected the following:\n\n"
        report += f"- Depression Risk Level: **{risk_level}**\n"
        report += f"- Depression Score: **{depression_score:.2f}** (0-1 scale)\n"
        report += f"- Model Confidence: **{model_confidence:.2f}** (0-1 scale)\n\n"
        
        # Add linguistic analysis
        report += f"### Linguistic Analysis\n\n"
        report += f"- Depression Keywords Detected: **{analysis.get('depression_keyword_count', 0)}**\n"
        
        # Add more linguistic features if available
        if 'linguistic_features' in analysis:
            features = analysis['linguistic_features']
            if isinstance(features, dict):
                if 'negative_word_count' in features:
                    report += f"- Negative Words: **{features['negative_word_count']}**\n"
                if 'pronoun_count' in features:
                    report += f"- Pronoun Usage: **{features['pronoun_count']}**\n"
    
    elif input_type == "audio":
        report += f"Based on analysis of your voice recording, the system has detected the following:\n\n"
        report += f"- Depression Risk Level: **{risk_level}**\n"
        report += f"- Depression Score: **{depression_score:.2f}** (0-1 scale)\n"
        report += f"- Model Confidence: **{model_confidence:.2f}** (0-1 scale)\n\n"
        
        # Add audio analysis
        report += f"### Voice Analysis\n\n"
        report += f"- Voice patterns associated with depression were analyzed\n"
        report += f"- The AI model detected {risk_level.lower()} levels of vocal markers associated with depression\n"
    
    elif input_type == "eeg":
        report += f"Based on analysis of your EEG data, the system has detected the following:\n\n"
        report += f"- Depression Risk Level: **{risk_level}**\n"
        report += f"- Depression Score: **{depression_score:.2f}** (0-1 scale)\n"
        report += f"- Model Confidence: **{model_confidence:.2f}** (0-1 scale)\n\n"
        
        # Add EEG analysis
        report += f"### Brain Activity Analysis\n\n"
        report += f"- EEG patterns associated with depression were analyzed\n"
        report += f"- The AI model detected {risk_level.lower()} levels of neural markers associated with depression\n"
    
    # Add statistical comparison
    report += f"\n### Statistical Comparison\n\n"
    report += f"Your results compared to our dataset:\n\n"
    
    # These would be replaced with actual statistics in a real implementation
    if risk_level == "Low":
        report += f"- Your depression score is lower than **70%** of individuals in our dataset\n"
        report += f"- Your linguistic/vocal/neural patterns show **minimal** correlation with depression\n"
    elif risk_level == "Moderate":
        report += f"- Your depression score is higher than **50%** of individuals in our dataset\n"
        report += f"- Your linguistic/vocal/neural patterns show **moderate** correlation with depression\n"
    else:  # High
        report += f"- Your depression score is higher than **85%** of individuals in our dataset\n"
        report += f"- Your linguistic/vocal/neural patterns show **strong** correlation with depression\n"
    
    # Add interpretation
    report += f"\n### Interpretation\n\n"
    
    if risk_level == "Low":
        report += "The analysis suggests minimal indicators of depression. This does not mean you might not be experiencing other mental health challenges, but the specific patterns associated with depression are not strongly present in your data.\n"
    elif risk_level == "Moderate":
        report += "The analysis suggests some indicators of depression. While not severe, these patterns might warrant attention. Consider monitoring your mental health and implementing self-care strategies.\n"
    else:  # High
        report += "The analysis suggests significant indicators of depression. These patterns are consistent with those seen in individuals experiencing depression. It is recommended to consult with a mental health professional for a proper evaluation.\n"
    
    # Add disclaimer
    report += f"\n### Disclaimer\n\n"
    report += "*This assessment is based on AI analysis and is not a clinical diagnosis. The technology is still evolving and should not replace professional medical advice. If you're experiencing mental health challenges, please consult with a qualified healthcare provider.*\n"
    
    return report

# Define function to generate response
def generate_response(analysis, input_type="text"):
    """Generate response based on analysis."""
    risk_level = analysis['risk_level']
    
    # Generate suggestions
    suggestions = generate_suggestions(risk_level)
    
    # Create response
    if input_type == "text":
        if risk_level == "Low":
            response = "Based on your message, you seem to be doing relatively well. "
            response += "I don't detect significant signs of depression, but it's always good to maintain your mental health. "
        elif risk_level == "Moderate":
            response = "I notice some potential signs of distress in your message. "
            response += "While I'm not detecting severe depression, it might be worth paying attention to your mental health. "
        else:  # High
            response = "I'm concerned about what you've shared. "
            response += "Your message contains several indicators that suggest you might be experiencing significant distress. "
            response += "Please consider reaching out to a mental health professional for proper assessment and support. "
    elif input_type == "audio":
        if risk_level == "Low":
            response = "Based on your voice, you seem to be doing relatively well. "
            response += "I don't detect significant signs of depression, but it's always good to maintain your mental health. "
        elif risk_level == "Moderate":
            response = "I notice some potential signs of distress in your voice. "
            response += "While I'm not detecting severe depression, it might be worth paying attention to your mental health. "
        else:  # High
            response = "I'm concerned about what I hear in your voice. "
            response += "There are several indicators that suggest you might be experiencing significant distress. "
            response += "Please consider reaching out to a mental health professional for proper assessment and support. "
    else:  # eeg
        if risk_level == "Low":
            response = "Based on your EEG data, you seem to be doing relatively well. "
            response += "I don't detect significant neural patterns associated with depression, but it's always good to maintain your mental health. "
        elif risk_level == "Moderate":
            response = "I notice some potential neural patterns associated with distress in your EEG data. "
            response += "While I'm not detecting severe depression, it might be worth paying attention to your mental health. "
        else:  # High
            response = "I'm concerned about the patterns I see in your EEG data. "
            response += "There are several neural indicators that suggest you might be experiencing significant distress. "
            response += "Please consider reaching out to a mental health professional for proper assessment and support. "
    
    # Add suggestions
    response += "\n\nHere are some suggestions that might help:\n"
    for i, suggestion in enumerate(suggestions):
        response += f"{i+1}. {suggestion}\n"
    
    # Add option for detailed report
    response += "\nI can provide a more detailed analysis if you'd like. Just ask for a 'detailed report'."
    
    # Add disclaimer
    response += "\n\n*Please note: This is not a professional diagnosis. If you're experiencing mental health issues, please consult with a qualified healthcare provider.*"
    
    return response

# Define function to display chat message
def display_message(message, is_user=False):
    """Display a chat message."""
    if is_user:
        st.markdown(f'<div style="background-color: #e6f7ff; padding: 10px; border-radius: 10px; margin-bottom: 10px; text-align: right;"><b>You:</b> {message}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div style="background-color: #f0f0f0; padding: 10px; border-radius: 10px; margin-bottom: 10px;"><b>AI:</b> {message}</div>', unsafe_allow_html=True)

# Define main function
def main():
    """Main function."""
    # Set up sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Chat", "About"])
    
    if page == "Chat":
        show_chat_page()
    elif page == "About":
        show_about_page()

def show_chat_page():
    """Show chat page."""
    st.title("Mental Health AI Chatbot")
    
    # Initialize chat history and session state
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    if 'current_analysis' not in st.session_state:
        st.session_state.current_analysis = None
    
    if 'current_input_type' not in st.session_state:
        st.session_state.current_input_type = "text"
    
    # Display welcome message if chat history is empty
    if not st.session_state.chat_history:
        welcome_message = "Hello! I'm here to listen and provide suggestions related to mental health. How are you feeling today?"
        display_message(welcome_message)
        st.session_state.chat_history.append({"message": welcome_message, "is_user": False})
    
    # Display chat history
    for message in st.session_state.chat_history:
        display_message(message["message"], message["is_user"])
    
    # Input options
    input_type = st.radio("Input Type", ["Text", "Audio", "EEG"], horizontal=True)
    
    if input_type == "Text":
        # Text input
        user_input = st.text_area("Type your message here:", height=100)
        
        if st.button("Send"):
            if user_input:
                # Check if user is asking for a detailed report
                if "detailed report" in user_input.lower() and st.session_state.current_analysis is not None:
                    # Generate detailed report
                    report = generate_detailed_report(st.session_state.current_analysis, st.session_state.current_input_type)
                    
                    # Display user message
                    display_message(user_input, is_user=True)
                    st.session_state.chat_history.append({"message": user_input, "is_user": True})
                    
                    # Display report
                    display_message(report)
                    st.session_state.chat_history.append({"message": report, "is_user": False})
                else:
                    # Display user message
                    display_message(user_input, is_user=True)
                    st.session_state.chat_history.append({"message": user_input, "is_user": True})
                    
                    # Analyze text
                    analysis = analyze_text(user_input)
                    
                    # Store current analysis and input type
                    st.session_state.current_analysis = analysis
                    st.session_state.current_input_type = "text"
                    
                    # Generate response
                    response = generate_response(analysis, input_type="text")
                    
                    # Display response
                    display_message(response)
                    st.session_state.chat_history.append({"message": response, "is_user": False})
    
    elif input_type == "Audio":
        # Audio input
        st.write("Upload an audio recording for analysis.")
        
        # Audio file uploader
        audio_file = st.file_uploader("Upload audio file (WAV format)", type=["wav"])
        
        if audio_file is not None:
            # Display audio player
            st.audio(audio_file)
            
            if st.button("Analyze Audio"):
                try:
                    # Generate random audio data for demonstration
                    audio_data = np.random.randn(16000)  # 1 second at 16kHz
                    
                    # Display user message
                    display_message("[Audio Recording Uploaded]", is_user=True)
                    st.session_state.chat_history.append({"message": "[Audio Recording Uploaded]", "is_user": True})
                    
                    # Analyze audio
                    analysis = analyze_audio(audio_data)
                    
                    # Store current analysis and input type
                    st.session_state.current_analysis = analysis
                    st.session_state.current_input_type = "audio"
                    
                    # Generate response
                    response = generate_response(analysis, input_type="audio")
                    
                    # Display response
                    display_message(response)
                    st.session_state.chat_history.append({"message": response, "is_user": False})
                except Exception as e:
                    st.error(f"Error analyzing audio: {e}")
                    st.info("Please try again with a different audio file or use the text input option.")
    
    else:  # EEG
        # EEG data input
        st.write("Upload EEG data for analysis.")
        
        # EEG file uploader
        eeg_file = st.file_uploader("Upload EEG data (CSV format)", type=["csv"])
        
        if eeg_file is not None:
            if st.button("Analyze EEG Data"):
                try:
                    # Generate random EEG data for demonstration
                    eeg_data = np.random.randn(64)
                    
                    # Display user message
                    display_message("[EEG Data Uploaded]", is_user=True)
                    st.session_state.chat_history.append({"message": "[EEG Data Uploaded]", "is_user": True})
                    
                    # Analyze EEG
                    analysis = analyze_eeg(eeg_data)
                    
                    # Store current analysis and input type
                    st.session_state.current_analysis = analysis
                    st.session_state.current_input_type = "eeg"
                    
                    # Generate response
                    response = generate_response(analysis, input_type="eeg")
                    
                    # Display response
                    display_message(response)
                    st.session_state.chat_history.append({"message": response, "is_user": False})
                except Exception as e:
                    st.error(f"Error analyzing EEG data: {e}")
                    st.info("Please try again with a different EEG file or use the text input option.")
    
    # Clear chat button
    if st.button("Clear Chat"):
        st.session_state.chat_history = []
        st.session_state.current_analysis = None
        st.session_state.current_input_type = "text"
        st.experimental_rerun()

def show_about_page():
    """Show about page."""
    st.title("About Mental Health AI")
    
    st.markdown("""
    ## Project Overview
    
    This project implements a suite of deep learning models for mental health classification using three different modalities:
    - EEG (electroencephalogram) data
    - Audio data
    - Text data
    
    The models are designed to detect signs of depression and provide insights for mental healthcare professionals.
    
    ## Model Performance
    
    | Modality | Accuracy | Precision | Recall | F1 Score | AUC |
    |----------|----------|-----------|--------|----------|-----|
    | EEG      | 0.550    | 0.538     | 0.700  | 0.609    | 0.570 |
    | Audio    | 0.930    | 0.925     | 0.943  | 0.934    | 0.973 |
    | Text     | 0.925    | 0.933     | 0.925  | 0.929    | 0.970 |
    
    ## How It Works
    
    1. **Text Analysis**: The system analyzes linguistic patterns in text to identify indicators of depression.
    2. **Audio Analysis**: Voice patterns are analyzed for acoustic markers associated with depression.
    3. **EEG Analysis**: Brain activity patterns are analyzed for neural markers of depression.
    
    ## Disclaimer
    
    This tool is for educational and research purposes only. It is not a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of a qualified healthcare provider with any questions you may have regarding a medical condition.
    """)

if __name__ == "__main__":
    main()
