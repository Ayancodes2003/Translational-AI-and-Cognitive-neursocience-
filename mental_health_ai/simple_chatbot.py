"""
Simple Mental Health AI Chatbot

This script creates a simple chatbot-like interface for mental health assessment and suggestions.
"""

import os
import sys
import numpy as np
import streamlit as st
import re
from datetime import datetime
import nltk
from nltk.tokenize import word_tokenize

# Download NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Set page configuration
st.set_page_config(
    page_title="Mental Health AI Chatbot",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

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

# Define function to clean text
def clean_text(text):
    """
    Clean text by removing special characters, numbers, and extra whitespace.
    
    Args:
        text (str): Input text
    
    Returns:
        str: Cleaned text
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

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
    cleaned_text = clean_text(text)
    
    # Tokenize text
    tokens = word_tokenize(cleaned_text.lower())
    
    # Count depression keywords
    depression_keyword_count = sum(1 for token in tokens if token in depression_keywords)
    
    # Count total words
    word_count = len(tokens)
    
    # Count unique words
    unique_word_count = len(set(tokens))
    
    # Calculate lexical diversity
    lexical_diversity = unique_word_count / word_count if word_count > 0 else 0
    
    # Count first-person pronouns
    first_person_pronouns = ['i', 'me', 'my', 'mine', 'myself', 'we', 'us', 'our', 'ours', 'ourselves']
    first_person_pronoun_count = sum(1 for token in tokens if token in first_person_pronouns)
    
    # Count negative words
    negative_words = ['no', 'not', 'never', 'none', 'nothing', 'nowhere', 'neither', 'nor', 'nobody']
    negative_word_count = sum(1 for token in tokens if token in negative_words)
    
    # Calculate depression score (simple heuristic)
    # Higher score indicates higher likelihood of depression
    depression_score = 0.0
    
    # Factor 1: Presence of depression keywords
    if word_count > 0:
        keyword_ratio = depression_keyword_count / word_count
        depression_score += keyword_ratio * 0.4  # 40% weight
    
    # Factor 2: Negative word count
    if word_count > 0:
        negative_ratio = negative_word_count / word_count
        depression_score += negative_ratio * 0.3  # 30% weight
    
    # Factor 3: First-person pronoun usage
    if word_count > 0:
        pronoun_ratio = first_person_pronoun_count / word_count
        depression_score += pronoun_ratio * 0.2  # 20% weight
    
    # Factor 4: Lexical diversity (lower diversity can indicate depression)
    if lexical_diversity < 0.3:  # Low lexical diversity
        depression_score += 0.1  # 10% weight
    
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
        'word_count': word_count,
        'unique_word_count': unique_word_count,
        'lexical_diversity': lexical_diversity,
        'first_person_pronoun_count': first_person_pronoun_count,
        'negative_word_count': negative_word_count
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

# Define function to generate response based on analysis
def generate_response(analysis):
    """
    Generate response based on analysis.
    
    Args:
        analysis (dict): Analysis results
    
    Returns:
        str: Response
    """
    risk_level = analysis['risk_level']
    depression_score = analysis['depression_score']
    
    # Generate suggestions
    suggestions = generate_suggestions(risk_level)
    
    # Create response
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
    
    # Add suggestions
    response += "\n\nHere are some suggestions that might help:\n"
    for i, suggestion in enumerate(suggestions):
        response += f"{i+1}. {suggestion}\n"
    
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
        results = interpret_phq9_scores(scores)
        
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
