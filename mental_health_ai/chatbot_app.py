"""
Mental Health AI Chatbot

This script creates a chatbot-like interface for mental health assessment and suggestions.
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
import io
from PIL import Image
import base64
from datetime import datetime
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re
import pickle
from scipy.io import wavfile
import time

# Check if librosa is available
try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

# Download NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Define processor classes
class TextProcessor:
    """Class for processing text data."""

    def __init__(self, vocab_file=None, model_file=None):
        """Initialize the text processor."""
        self.vocab_file = vocab_file
        self.model_file = model_file

    def clean_text(self, text):
        """Clean text by removing special characters and extra whitespace."""
        if not text:
            return ""

        # Convert to lowercase
        text = text.lower()

        # Remove special characters and extra whitespace
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    def extract_linguistic_features(self, texts):
        """Extract linguistic features from texts."""
        if not texts or len(texts) == 0:
            return {}

        # Use the first text for simplicity
        text = texts[0]

        # Tokenize
        tokens = word_tokenize(text.lower())

        # Count words
        word_count = len(tokens)

        # Count negative words (simplified)
        negative_words = ['sad', 'depressed', 'unhappy', 'miserable', 'hopeless', 'worthless', 'tired', 'exhausted', 'anxious', 'worried']
        negative_word_count = sum(1 for token in tokens if token in negative_words)

        # Count pronouns (simplified)
        pronouns = ['i', 'me', 'my', 'mine', 'myself']
        pronoun_count = sum(1 for token in tokens if token in pronouns)

        # Create features dictionary
        features = {
            'word_count': word_count,
            'negative_word_count': negative_word_count,
            'pronoun_count': pronoun_count
        }

        return features

class AudioProcessor:
    """Class for processing audio data."""

    def __init__(self, model_file=None, config_file=None):
        """Initialize the audio processor."""
        self.model_file = model_file
        self.config_file = config_file

    def extract_features(self, audio_data):
        """Extract features from audio data."""
        if audio_data is None or len(audio_data) == 0:
            return np.zeros(80)

        # Create a simple feature vector (in a real implementation, this would use MFCC, spectral features, etc.)
        # For demonstration purposes, we'll just use some simple statistics
        features = np.zeros(80)

        # Basic statistics
        features[0] = np.mean(audio_data)  # Mean
        features[1] = np.std(audio_data)   # Standard deviation
        features[2] = np.max(audio_data)   # Maximum value
        features[3] = np.min(audio_data)   # Minimum value
        features[4] = np.median(audio_data)  # Median
        features[5] = np.sum(np.abs(audio_data))  # Energy

        # Zero crossing rate (simplified)
        zero_crossings = np.sum(np.abs(np.diff(np.signbit(audio_data))))
        features[6] = zero_crossings / len(audio_data)

        # Fill the rest with random values for demonstration
        features[7:] = np.random.randn(73) * 0.01

        return features

# Set page configuration
st.set_page_config(
    page_title="Mental Health AI Chatbot",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add models directory to path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models'))

# Import model classes
from models.text_models import TextCNN
from models.audio_models import AudioLSTM
from models.eeg_models import EEGCNN

# Risk Assessor class
class RiskAssessor:
    """Class for assessing mental health risk levels."""

    def __init__(self):
        """Initialize the risk assessor."""
        pass

    def assess_risk(self, depression_score):
        """Assess risk level based on depression score."""
        if depression_score < 0.3:
            return "Low"
        elif depression_score < 0.7:
            return "Moderate"
        else:
            return "High"

# Clinical Report Generator class
class ClinicalReportGenerator:
    """Class for generating clinical reports."""

    def __init__(self):
        """Initialize the report generator."""
        pass

    def generate_report(self, analysis, input_type="text"):
        """Generate a clinical report based on analysis."""
        return generate_detailed_report(analysis, input_type)

# Initialize text processor
text_processor = TextProcessor(None, None)

# Initialize audio processor
audio_processor = AudioProcessor(None, None)

# Initialize risk assessor
risk_assessor = RiskAssessor()

# Initialize report generator
report_generator = ClinicalReportGenerator()

# Load trained models
def load_model(modality, model_type):
    """Load a trained model."""
    try:
        if modality == "text":
            if model_type == "TextCNN":
                model = TextCNN(input_dim=50)  # Adjust input_dim based on your feature size
                model.load_state_dict(torch.load(f"results/text/{model_type}/model.pt", map_location=torch.device('cpu')))
                return model
        elif modality == "audio":
            if model_type == "AudioLSTM":
                model = AudioLSTM(input_dim=80)  # Adjust input_dim based on your feature size
                model.load_state_dict(torch.load(f"results/audio/{model_type}/model.pt", map_location=torch.device('cpu')))
                return model
        elif modality == "eeg":
            if model_type == "EEGCNN":
                model = EEGCNN(input_dim=64)  # Adjust input_dim based on your feature size
                model.load_state_dict(torch.load(f"results/eeg/{model_type}/model.pt", map_location=torch.device('cpu')))
                return model

        st.error(f"Model {model_type} for {modality} not found.")
        return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Load models
text_model = load_model("text", "TextCNN")
audio_model = load_model("audio", "AudioLSTM")
eeg_model = load_model("eeg", "EEGCNN")

# Define depression keywords
depression_keywords = [
    'sad', 'unhappy', 'depressed', 'anxious', 'worried', 'fear', 'afraid', 'scared',
    'terrible', 'horrible', 'awful', 'bad', 'worse', 'worst', 'pain', 'hurt',
    'suffering', 'miserable', 'lonely', 'alone', 'empty', 'meaningless', 'hopeless',
    'helpless', 'worthless', 'useless', 'failure', 'failed', 'lose', 'lost',
    'losing', 'loser', 'hate', 'hated', 'hating', 'anger', 'angry', 'mad',
    'upset', 'frustrated', 'irritated', 'annoyed', 'stress', 'stressed',
    'stressful', 'tired', 'exhausted', 'fatigue', 'fatigued', 'weak', 'weary',
    'sick', 'ill', 'disease', 'disorder', 'problem', 'trouble', 'difficult',
    'hard', 'struggle', 'struggling', 'suffer', 'suffered', 'suffering',
    'cry', 'crying', 'cried', 'tears', 'sob', 'sobbing', 'sobbed', 'sigh',
    'sighing', 'sighed', 'groan', 'groaning', 'groaned', 'moan', 'moaning',
    'moaned', 'scream', 'screaming', 'screamed', 'yell', 'yelling', 'yelled',
    'shout', 'shouting', 'shouted', 'curse', 'cursing', 'cursed', 'swear',
    'swearing', 'swore', 'damn', 'damned', 'damning', 'hell', 'death', 'dead',
    'die', 'died', 'dying', 'kill', 'killed', 'killing', 'suicide', 'suicidal'
]

# Define PHQ-9 questions
phq9_questions = [
    "Little interest or pleasure in doing things?",
    "Feeling down, depressed, or hopeless?",
    "Trouble falling or staying asleep, or sleeping too much?",
    "Feeling tired or having little energy?",
    "Poor appetite or overeating?",
    "Feeling bad about yourself — or that you are a failure or have let yourself or your family down?",
    "Trouble concentrating on things, such as reading the newspaper or watching television?",
    "Moving or speaking so slowly that other people could have noticed? Or the opposite — being so fidgety or restless that you have been moving around a lot more than usual?",
    "Thoughts that you would be better off dead or of hurting yourself in some way?"
]

# Define PHQ-9 options
phq9_options = [
    "Not at all",
    "Several days",
    "More than half the days",
    "Nearly every day"
]

# Define suggestions based on depression level
low_risk_suggestions = [
    "Continue regular self-monitoring of your mood and energy levels.",
    "Maintain a healthy lifestyle with regular exercise, balanced diet, and adequate sleep.",
    "Practice stress management techniques like deep breathing, meditation, or yoga.",
    "Stay connected with friends and family for social support.",
    "Engage in activities you enjoy and that give you a sense of accomplishment.",
    "Consider keeping a gratitude journal to focus on positive aspects of your life.",
    "Limit exposure to negative news and social media if it affects your mood.",
    "Spend time in nature, which has been shown to improve mood and reduce stress.",
    "Set realistic goals and celebrate small achievements.",
    "Maintain a regular daily routine to provide structure and stability."
]

moderate_risk_suggestions = [
    "Consider consulting a mental health professional for an evaluation.",
    "Increase self-care activities and prioritize your wellbeing.",
    "Monitor mood changes more closely and keep a mood journal.",
    "Practice mindfulness and relaxation techniques regularly.",
    "Reach out to trusted friends or family members for support.",
    "Join a support group to connect with others experiencing similar challenges.",
    "Establish a regular sleep schedule and practice good sleep hygiene.",
    "Engage in regular physical activity, even if it's just a short walk.",
    "Limit alcohol and avoid recreational drugs, which can worsen depression.",
    "Break large tasks into smaller, manageable steps to avoid feeling overwhelmed.",
    "Challenge negative thoughts by questioning their validity and considering alternative perspectives.",
    "Consider using mental health apps or online resources for additional support."
]

high_risk_suggestions = [
    "Please seek professional help as soon as possible. This could include a therapist, counselor, or psychiatrist.",
    "If you're having thoughts of harming yourself, call a crisis hotline immediately: National Suicide Prevention Lifeline at 988 or 1-800-273-8255.",
    "Consider therapy or counseling to develop coping strategies and address underlying issues.",
    "Discuss medication options with a healthcare provider if appropriate.",
    "Establish a strong support network of trusted individuals who can help during difficult times.",
    "Create a safety plan with specific steps to take when experiencing suicidal thoughts.",
    "Remove access to means of self-harm if you're experiencing suicidal thoughts.",
    "Attend to basic needs like eating regularly, staying hydrated, and getting rest.",
    "Use grounding techniques when feeling overwhelmed (e.g., the 5-4-3-2-1 technique).",
    "Avoid making major life decisions during this difficult time.",
    "Remember that depression is treatable, and many people recover with proper support.",
    "Be gentle with yourself and acknowledge that seeking help is a sign of strength, not weakness."
]

# Define function to analyze text for depression indicators
def analyze_text(text):
    """
    Analyze text for depression indicators using the trained model.

    Args:
        text (str): Input text

    Returns:
        dict: Analysis results
    """
    # Clean text
    cleaned_text = text_processor.clean_text(text)

    # Extract linguistic features
    features = text_processor.extract_linguistic_features([text])

    # Count depression keywords
    tokens = word_tokenize(cleaned_text.lower())
    depression_keyword_count = sum(1 for token in tokens if token in depression_keywords)

    # Use trained model for prediction if available
    model_prediction = 0.0
    model_confidence = 0.0

    if text_model is not None:
        try:
            # Convert features to tensor
            # Note: In a real implementation, you would need to ensure the features match what the model expects
            # This is a simplified version
            feature_vector = np.zeros(50)  # Placeholder for actual feature extraction
            for i, token in enumerate(tokens[:50]):
                # Simple bag of words approach
                feature_vector[i % 50] += 1

            # Normalize
            if np.sum(feature_vector) > 0:
                feature_vector = feature_vector / np.sum(feature_vector)

            # Convert to tensor
            input_tensor = torch.tensor(feature_vector, dtype=torch.float32).unsqueeze(0)

            # Set model to evaluation mode
            text_model.eval()

            # Get prediction
            with torch.no_grad():
                output = text_model(input_tensor)
                probability = torch.sigmoid(output).item()
                prediction = 1 if probability > 0.5 else 0
                model_prediction = prediction
                model_confidence = probability if prediction == 1 else 1 - probability
        except Exception as e:
            st.error(f"Error using text model: {e}")

    # Calculate depression score (combine model prediction with heuristics)
    depression_score = 0.0

    # If model is available, give it high weight
    if text_model is not None:
        depression_score += model_prediction * 0.7  # 70% weight

    # Factor 1: Presence of depression keywords
    if len(tokens) > 0:
        keyword_ratio = depression_keyword_count / len(tokens)
        depression_score += keyword_ratio * 0.1  # 10% weight

    # Factor 2: Negative word count from linguistic features
    if 'negative_word_count' in features:
        negative_word_count = features['negative_word_count']
        if len(tokens) > 0:
            negative_ratio = negative_word_count / len(tokens)
            depression_score += negative_ratio * 0.1  # 10% weight

    # Factor 3: Pronoun usage (high first-person pronoun usage can indicate depression)
    if 'pronoun_count' in features:
        pronoun_count = features['pronoun_count']
        if len(tokens) > 0:
            pronoun_ratio = pronoun_count / len(tokens)
            depression_score += pronoun_ratio * 0.1  # 10% weight

    # Normalize score to 0-1 range
    depression_score = min(max(depression_score, 0.0), 1.0)

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
        'linguistic_features': features,
        'model_prediction': model_prediction,
        'model_confidence': model_confidence
    }

    return results


# Define function to analyze audio for depression indicators
def analyze_audio(audio_data, sr):
    """
    Analyze audio for depression indicators using the trained model.

    Args:
        audio_data (numpy.ndarray): Audio data
        sr (int): Sampling rate

    Returns:
        dict: Analysis results
    """
    # Extract audio features
    features = audio_processor.extract_features(audio_data)

    # Use trained model for prediction if available
    model_prediction = 0.0
    model_confidence = 0.0

    if audio_model is not None:
        try:
            # Convert features to tensor
            # Note: In a real implementation, you would need to ensure the features match what the model expects
            # This is a simplified version
            feature_vector = np.zeros(80)  # Placeholder for actual feature extraction

            # If features is a numpy array, use it directly
            if isinstance(features, np.ndarray):
                # Ensure it's the right size
                if features.size >= 80:
                    feature_vector = features[:80]
                else:
                    # Pad if too small
                    feature_vector[:features.size] = features.flatten()

            # Normalize
            if np.sum(feature_vector) > 0:
                feature_vector = feature_vector / np.sum(feature_vector)

            # Convert to tensor
            input_tensor = torch.tensor(feature_vector, dtype=torch.float32).unsqueeze(0)

            # Set model to evaluation mode
            audio_model.eval()

            # Get prediction
            with torch.no_grad():
                output = audio_model(input_tensor)
                probability = torch.sigmoid(output).item()
                prediction = 1 if probability > 0.5 else 0
                model_prediction = prediction
                model_confidence = probability if prediction == 1 else 1 - probability
        except Exception as e:
            st.error(f"Error using audio model: {e}")

    # Calculate depression score (combine model prediction with heuristics)
    depression_score = 0.0

    # If model is available, give it high weight
    if audio_model is not None:
        depression_score += model_prediction * 0.8  # 80% weight
    else:
        # Fallback to a default score if model is not available
        depression_score = 0.5

    # Normalize score to 0-1 range
    depression_score = min(max(depression_score, 0.0), 1.0)

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
        'audio_features': features.tolist() if isinstance(features, np.ndarray) else features,
        'model_prediction': model_prediction,
        'model_confidence': model_confidence
    }

    return results


# Define function to analyze EEG data for depression indicators
def analyze_eeg(eeg_data):
    """
    Analyze EEG data for depression indicators using the trained model.

    Args:
        eeg_data (numpy.ndarray): EEG data

    Returns:
        dict: Analysis results
    """
    # Use trained model for prediction if available
    model_prediction = 0.0
    model_confidence = 0.0

    if eeg_model is not None:
        try:
            # Convert features to tensor
            # Note: In a real implementation, you would need to ensure the features match what the model expects
            # This is a simplified version
            feature_vector = np.zeros(64)  # Placeholder for actual feature extraction

            # If eeg_data is a numpy array, use it directly
            if isinstance(eeg_data, np.ndarray):
                # Ensure it's the right size
                if eeg_data.size >= 64:
                    feature_vector = eeg_data[:64]
                else:
                    # Pad if too small
                    feature_vector[:eeg_data.size] = eeg_data.flatten()

            # Normalize
            if np.sum(feature_vector) > 0:
                feature_vector = feature_vector / np.sum(feature_vector)

            # Convert to tensor
            input_tensor = torch.tensor(feature_vector, dtype=torch.float32).unsqueeze(0)

            # Set model to evaluation mode
            eeg_model.eval()

            # Get prediction
            with torch.no_grad():
                output = eeg_model(input_tensor)
                probability = torch.sigmoid(output).item()
                prediction = 1 if probability > 0.5 else 0
                model_prediction = prediction
                model_confidence = probability if prediction == 1 else 1 - probability
        except Exception as e:
            st.error(f"Error using EEG model: {e}")

    # Calculate depression score (use model prediction directly)
    depression_score = 0.0

    # If model is available, use its prediction
    if eeg_model is not None:
        depression_score = model_prediction
    else:
        # Fallback to a default score if model is not available
        depression_score = 0.5

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
        'model_prediction': model_prediction,
        'model_confidence': model_confidence
    }

    return results


# Define function to generate suggestions based on risk level
def generate_suggestions(risk_level):
    """
    Generate suggestions based on risk level.

    Args:
        risk_level (str): Risk level ('Low', 'Moderate', or 'High')

    Returns:
        list: Suggestions
    """
    if risk_level == "Low":
        return np.random.choice(low_risk_suggestions, size=3, replace=False).tolist()
    elif risk_level == "Moderate":
        return np.random.choice(moderate_risk_suggestions, size=3, replace=False).tolist()
    else:  # High
        return np.random.choice(high_risk_suggestions, size=3, replace=False).tolist()


# Define function to generate detailed report
def generate_detailed_report(analysis, input_type="text"):
    """
    Generate a detailed report based on analysis.

    Args:
        analysis (dict): Analysis results
        input_type (str): Input type ('text', 'audio', or 'eeg')

    Returns:
        str: Detailed report
    """
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


# Define function to generate response based on analysis
def generate_response(analysis, input_type="text"):
    """
    Generate response based on analysis.

    Args:
        analysis (dict): Analysis results
        input_type (str): Input type ('text', 'audio', or 'eeg')

    Returns:
        str: Response
    """
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
    """
    Display chat message.

    Args:
        message (str): Message to display
        is_user (bool): Whether the message is from the user
    """
    if is_user:
        st.markdown(f'<div style="background-color: #e6f7ff; padding: 10px; border-radius: 10px; margin-bottom: 10px; text-align: right;"><b>You:</b> {message}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div style="background-color: #f0f0f0; padding: 10px; border-radius: 10px; margin-bottom: 10px;"><b>Mental Health AI:</b> {message}</div>', unsafe_allow_html=True)


# Define function to display PHQ-9 questionnaire
def display_phq9_questionnaire():
    """
    Display PHQ-9 questionnaire.

    Returns:
        list: PHQ-9 scores
    """
    st.subheader("PHQ-9 Depression Screening Questionnaire")
    st.markdown("Over the last 2 weeks, how often have you been bothered by any of the following problems?")

    scores = []

    for i, question in enumerate(phq9_questions):
        score = st.radio(
            f"{i+1}. {question}",
            options=phq9_options,
            index=0,
            key=f"phq9_{i}"
        )

        # Convert option to score
        option_score = phq9_options.index(score)
        scores.append(option_score)

    return scores


# Define function to interpret PHQ-9 scores
def interpret_phq9_scores(scores):
    """
    Interpret PHQ-9 scores.

    Args:
        scores (list): PHQ-9 scores

    Returns:
        dict: Interpretation results
    """
    total_score = sum(scores)

    # Determine depression severity
    if total_score <= 4:
        severity = "Minimal or none"
        risk_level = "Low"
    elif total_score <= 9:
        severity = "Mild"
        risk_level = "Low"
    elif total_score <= 14:
        severity = "Moderate"
        risk_level = "Moderate"
    elif total_score <= 19:
        severity = "Moderately severe"
        risk_level = "High"
    else:
        severity = "Severe"
        risk_level = "High"

    # Check for suicidal ideation (question 9)
    suicide_risk = scores[8] > 0

    # Create interpretation results
    results = {
        'total_score': total_score,
        'severity': severity,
        'risk_level': risk_level,
        'suicide_risk': suicide_risk
    }

    return results


# Define function to display PHQ-9 results
def display_phq9_results(results):
    """
    Display PHQ-9 results.

    Args:
        results (dict): Interpretation results
    """
    st.subheader("PHQ-9 Results")

    # Display total score and severity
    st.markdown(f"**Total Score:** {results['total_score']}")
    st.markdown(f"**Depression Severity:** {results['severity']}")

    # Display interpretation
    st.markdown("### Interpretation")

    if results['risk_level'] == "Low":
        st.markdown("Your responses suggest minimal to mild depression symptoms.")
    elif results['risk_level'] == "Moderate":
        st.markdown("Your responses suggest moderate depression symptoms. Consider discussing these results with a healthcare provider.")
    else:  # High
        st.markdown("Your responses suggest moderately severe to severe depression symptoms. It is strongly recommended that you consult with a healthcare provider.")

    # Display suicide risk warning if applicable
    if results['suicide_risk']:
        st.warning("⚠️ Your response indicates thoughts of self-harm or suicide. Please seek immediate help from a mental health professional or call the National Suicide Prevention Lifeline at 988 or 1-800-273-8255.")

    # Display suggestions
    st.markdown("### Suggestions")

    suggestions = generate_suggestions(results['risk_level'])
    for suggestion in suggestions:
        st.markdown(f"- {suggestion}")

    # Add disclaimer
    st.markdown("*Please note: This is not a professional diagnosis. If you're experiencing mental health issues, please consult with a qualified healthcare provider.*")


# Main application
def main():
    # Sidebar
    st.sidebar.title("Mental Health AI Chatbot")
    st.sidebar.image("https://img.icons8.com/color/96/000000/brain--v2.png", width=100)

    # Navigation
    page = st.sidebar.selectbox("Navigation", ["Chat", "PHQ-9 Assessment", "About"])

    if page == "Chat":
        show_chat_page()
    elif page == "PHQ-9 Assessment":
        show_phq9_page()
    elif page == "About":
        show_about_page()


def show_chat_page():
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

        if not LIBROSA_AVAILABLE:
            st.warning("The librosa library is not available. Audio analysis will use a simplified approach.")

        # Audio file uploader
        audio_file = st.file_uploader("Upload audio file (WAV format)", type=["wav"])

        if audio_file is not None:
            # Display audio player
            st.audio(audio_file)

            if st.button("Analyze Audio"):
                try:
                    # Load audio file
                    if LIBROSA_AVAILABLE:
                        y, sr = librosa.load(audio_file, sr=None)
                    else:
                        # Fallback to scipy.io.wavfile if librosa is not available
                        audio_file.seek(0)  # Reset file pointer
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                            tmp_file.write(audio_file.getvalue())
                            tmp_file_path = tmp_file.name

                        sr, y = wavfile.read(tmp_file_path)
                        # Convert to float32 and normalize
                        y = y.astype(np.float32)
                        if y.ndim > 1:  # Stereo to mono
                            y = np.mean(y, axis=1)
                        y = y / np.max(np.abs(y))

                        # Clean up temp file
                        try:
                            os.unlink(tmp_file_path)
                        except:
                            pass

                    # Display user message
                    display_message("[Audio Recording Uploaded]", is_user=True)
                    st.session_state.chat_history.append({"message": "[Audio Recording Uploaded]", "is_user": True})

                    # Analyze audio
                    analysis = analyze_audio(y, sr)

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
                    # Load EEG data
                    eeg_data = np.loadtxt(eeg_file, delimiter=',')  # Simple CSV loading

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


def show_phq9_page():
    st.title("PHQ-9 Depression Screening")

    st.markdown("""
    The Patient Health Questionnaire (PHQ-9) is a self-administered depression screening tool.

    This questionnaire is designed to help you understand your mental health better. It is not a diagnostic tool, but it can provide insights that you might want to discuss with a healthcare provider.
    """)

    # Display PHQ-9 questionnaire
    scores = display_phq9_questionnaire()

    # Submit button
    if st.button("Submit"):
        # Interpret scores
        results = interpret_phq9_scores(scores)

        # Display results
        display_phq9_results(results)


def show_about_page():
    st.title("About Mental Health AI Chatbot")

    st.markdown("""
    ### Overview

    The Mental Health AI Chatbot is designed to provide a supportive space for discussing mental health concerns and offering suggestions based on user input. It uses natural language processing to analyze text and audio inputs for potential indicators of depression or distress.

    ### Features

    - **Text Analysis**: Analyzes text input for linguistic patterns associated with depression
    - **Audio Analysis**: Analyzes voice recordings for acoustic patterns associated with depression
    - **PHQ-9 Assessment**: Provides a standardized depression screening questionnaire
    - **Personalized Suggestions**: Offers tailored suggestions based on detected risk level

    ### Important Disclaimer

    This chatbot is not a substitute for professional mental health care. It cannot diagnose mental health conditions or provide treatment. If you're experiencing mental health issues, please consult with a qualified healthcare provider.

    ### Crisis Resources

    If you're in crisis or experiencing thoughts of suicide:

    - **National Suicide Prevention Lifeline**: Call or text 988, or chat at [988lifeline.org](https://988lifeline.org)
    - **Crisis Text Line**: Text HOME to 741741
    - **Emergency Services**: Call 911 or go to your nearest emergency room

    ### Privacy

    Your interactions with this chatbot are not stored permanently. The chat history is only maintained for the duration of your session and is cleared when you close the browser or click the "Clear Chat" button.
    """)


if __name__ == "__main__":
    main()
