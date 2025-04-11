"""
Trained Chatbot

This script runs the Mental Health AI chatbot with trained models.
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
from datetime import datetime
import nltk
from nltk.tokenize import word_tokenize
import re

# Download NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# from data.text.preprocess_text import TextProcessor
from clinical_insights.risk_assessment import RiskAssessor
from clinical_insights.report_generator import ClinicalReportGenerator

# Set page configuration
st.set_page_config(
    page_title="Mental Health AI Chatbot",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define a simple text processor class
class SimpleTextProcessor:
    def __init__(self):
        pass

    def clean_text(self, text):
        # Simple text cleaning
        return text.lower()

    def extract_linguistic_features(self, texts):
        # Simple feature extraction
        features = {}
        text = texts[0]

        # Count tokens
        tokens = text.split()
        features['token_count'] = len(tokens)

        # Count unique tokens
        features['unique_token_count'] = len(set(tokens))

        # Calculate lexical diversity
        features['lexical_diversity'] = features['unique_token_count'] / features['token_count'] if features['token_count'] > 0 else 0

        # Count depression keywords
        depression_keywords = ['sad', 'unhappy', 'depressed', 'anxious', 'worried', 'fear', 'afraid', 'scared']
        features['depression_keyword_count'] = sum(1 for token in tokens if token in depression_keywords)

        # Count first-person pronouns
        first_person_pronouns = ['i', 'me', 'my', 'mine', 'myself', 'we', 'us', 'our', 'ours', 'ourselves']
        features['first_person_pronoun_count'] = sum(1 for token in tokens if token in first_person_pronouns)

        # Count negative words
        negative_words = ['no', 'not', 'never', 'none', 'nothing', 'nowhere', 'neither', 'nor', 'nobody']
        features['negative_word_count'] = sum(1 for token in tokens if token in negative_words)

        return features

# Initialize text processor
text_processor = SimpleTextProcessor()

# Initialize risk assessor
risk_assessor = RiskAssessor()

# Initialize clinical report generator
report_generator = ClinicalReportGenerator()

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

# Define function to analyze text for depression indicators
def analyze_text(text):
    """
    Analyze text for depression indicators.

    Args:
        text (str): Input text

    Returns:
        dict: Analysis results
    """
    # Clean text
    cleaned_text = text_processor.clean_text(text)

    # Extract linguistic features
    features = text_processor.extract_linguistic_features([text])

    # Assess risk
    risk_assessment = risk_assessor.assess_risk_from_text(features)

    return risk_assessment


# Define function to generate response based on analysis
def generate_response(analysis):
    """
    Generate response based on analysis.

    Args:
        analysis (dict): Analysis results

    Returns:
        str: Response
    """
    # Generate clinical report
    report = report_generator.generate_report(analysis)

    # Create response
    risk_level = analysis['risk_level']

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

    # Add observations
    response += "\n\nObservations:\n"
    for observation in report['observations']:
        response += f"- {observation}\n"

    # Add suggestions
    response += "\n\nHere are some suggestions that might help:\n"
    for suggestion in report['suggestions']:
        response += f"- {suggestion}\n"

    # Add disclaimer
    response += "\n*Please note: This is not a professional diagnosis. If you're experiencing mental health issues, please consult with a qualified healthcare provider.*"

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

    # Generate clinical report
    report = report_generator.generate_report(results)

    # Display suggestions
    st.markdown("### Suggestions")

    for suggestion in report['suggestions']:
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

    # Initialize chat history
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    # Display welcome message if chat history is empty
    if not st.session_state.chat_history:
        welcome_message = "Hello! I'm here to listen and provide suggestions related to mental health. How are you feeling today?"
        display_message(welcome_message)
        st.session_state.chat_history.append({"message": welcome_message, "is_user": False})

    # Display chat history
    for message in st.session_state.chat_history:
        display_message(message["message"], message["is_user"])

    # Text input
    user_input = st.text_area("Type your message here:", height=100)

    if st.button("Send"):
        if user_input:
            # Display user message
            display_message(user_input, is_user=True)
            st.session_state.chat_history.append({"message": user_input, "is_user": True})

            # Analyze text
            analysis = analyze_text(user_input)

            # Generate response
            response = generate_response(analysis)

            # Display response
            display_message(response)
            st.session_state.chat_history.append({"message": response, "is_user": False})

    # Clear chat button
    if st.button("Clear Chat"):
        st.session_state.chat_history = []
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
        results = risk_assessor.assess_risk_from_phq9(scores)

        # Display results
        display_phq9_results(results)


def show_about_page():
    st.title("About Mental Health AI Chatbot")

    st.markdown("""
    ### Overview

    The Mental Health AI Chatbot is designed to provide a supportive space for discussing mental health concerns and offering suggestions based on user input. It uses natural language processing to analyze text inputs for potential indicators of depression or distress.

    ### Features

    - **Text Analysis**: Analyzes text input for linguistic patterns associated with depression
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
