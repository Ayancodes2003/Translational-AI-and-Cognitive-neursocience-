"""
Trained Chatbot

This script runs the Mental Health AI chatbot with trained models.
"""

import os
import sys
import random
from datetime import datetime
import streamlit as st
import nltk

# Download NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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

# Define a simple risk assessor class
class SimpleRiskAssessor:
    def __init__(self):
        pass

    def assess_risk_from_text(self, text_features):
        """
        Assess risk level from text features.

        Args:
            text_features (dict): Text features

        Returns:
            dict: Risk assessment results
        """
        # Initialize risk score
        risk_score = 0.0

        # Factor 1: Depression keyword count
        if 'depression_keyword_count' in text_features and 'token_count' in text_features:
            if text_features['token_count'] > 0:
                keyword_ratio = text_features['depression_keyword_count'] / text_features['token_count']
                risk_score += keyword_ratio * 0.4  # 40% weight

        # Factor 2: Negative word count
        if 'negative_word_count' in text_features and 'token_count' in text_features:
            if text_features['token_count'] > 0:
                negative_ratio = text_features['negative_word_count'] / text_features['token_count']
                risk_score += negative_ratio * 0.3  # 30% weight

        # Factor 3: First-person pronoun usage
        if 'first_person_pronoun_count' in text_features and 'token_count' in text_features:
            if text_features['token_count'] > 0:
                pronoun_ratio = text_features['first_person_pronoun_count'] / text_features['token_count']
                risk_score += pronoun_ratio * 0.2  # 20% weight

        # Factor 4: Lexical diversity (lower diversity can indicate depression)
        if 'lexical_diversity' in text_features:
            lexical_diversity = text_features['lexical_diversity']
            if lexical_diversity < 0.3:  # Low lexical diversity
                risk_score += 0.1  # 10% weight

        # Normalize risk score to 0-1 range
        risk_score = min(max(risk_score, 0.0), 1.0)

        # Determine risk level
        if risk_score < 0.3:
            risk_level = "Low"
        elif risk_score < 0.7:
            risk_level = "Moderate"
        else:
            risk_level = "High"

        # Create risk assessment results
        results = {
            'risk_score': risk_score,
            'risk_level': risk_level,
            'modality': 'text'
        }

        return results

    def assess_risk_from_phq9(self, phq9_scores):
        """
        Assess risk level from PHQ-9 scores.

        Args:
            phq9_scores (list): PHQ-9 scores

        Returns:
            dict: Risk assessment results
        """
        # Calculate total score
        total_score = sum(phq9_scores)

        # Determine depression severity
        if total_score <= 4:
            severity = "Minimal or none"
            risk_level = "Low"
            risk_score = total_score / 27  # Normalize to 0-1 range
        elif total_score <= 9:
            severity = "Mild"
            risk_level = "Low"
            risk_score = total_score / 27
        elif total_score <= 14:
            severity = "Moderate"
            risk_level = "Moderate"
            risk_score = total_score / 27
        elif total_score <= 19:
            severity = "Moderately severe"
            risk_level = "High"
            risk_score = total_score / 27
        else:
            severity = "Severe"
            risk_level = "High"
            risk_score = total_score / 27

        # Check for suicidal ideation (question 9)
        suicide_risk = phq9_scores[8] > 0

        # Create risk assessment results
        results = {
            'risk_score': risk_score,
            'risk_level': risk_level,
            'severity': severity,
            'total_score': total_score,
            'suicide_risk': suicide_risk,
            'modality': 'phq9'
        }

        return results

# Initialize risk assessor
risk_assessor = SimpleRiskAssessor()

# Define a simple clinical report generator class
class SimpleClinicalReportGenerator:
    def __init__(self):
        # Initialize suggestions
        self.low_risk_suggestions = [
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

        self.moderate_risk_suggestions = [
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

        self.high_risk_suggestions = [
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

        # Initialize observations
        self.low_risk_observations = [
            "Your responses suggest minimal to mild depression symptoms.",
            "You appear to be managing your mental health effectively.",
            "Your linguistic patterns do not show significant indicators of depression.",
            "Your overall risk assessment indicates low risk for depression. Continued monitoring is recommended."
        ]

        self.moderate_risk_observations = [
            "Your responses suggest moderate depression symptoms.",
            "There are some indicators of potential distress in your communication patterns.",
            "Your linguistic patterns show some indicators associated with depression.",
            "Your overall risk assessment indicates moderate risk for depression. Regular monitoring is recommended."
        ]

        self.high_risk_observations = [
            "Your responses suggest moderately severe to severe depression symptoms.",
            "There are significant indicators of distress in your communication patterns.",
            "Your linguistic patterns show strong indicators associated with depression.",
            "Your overall risk assessment indicates high risk for depression. Professional intervention is recommended."
        ]

    def generate_observations(self, risk_level, modality_contributions=None, suicide_risk=False):
        """
        Generate observations based on risk level and modality contributions.

        Args:
            risk_level (str): Risk level ('Low', 'Moderate', or 'High')
            modality_contributions (dict): Modality contributions
            suicide_risk (bool): Whether suicide risk is detected

        Returns:
            list: Observations
        """
        # Select observations based on risk level
        if risk_level == "Low":
            observations = self.low_risk_observations.copy()
        elif risk_level == "Moderate":
            observations = self.moderate_risk_observations.copy()
        else:  # High
            observations = self.high_risk_observations.copy()

        # Add suicide risk observation if applicable
        if suicide_risk:
            observations.append("⚠️ Your responses indicate thoughts of self-harm or suicide. Please seek immediate help from a mental health professional or call the National Suicide Prevention Lifeline at 988 or 1-800-273-8255.")

        return observations

    def generate_suggestions(self, risk_level, num_suggestions=3):
        """
        Generate suggestions based on risk level.

        Args:
            risk_level (str): Risk level ('Low', 'Moderate', or 'High')
            num_suggestions (int): Number of suggestions to generate

        Returns:
            list: Suggestions
        """
        import random

        # Select suggestions based on risk level
        if risk_level == "Low":
            suggestions = self.low_risk_suggestions
        elif risk_level == "Moderate":
            suggestions = self.moderate_risk_suggestions
        else:  # High
            suggestions = self.high_risk_suggestions

        # Randomly select suggestions
        selected_suggestions = random.sample(suggestions, min(num_suggestions, len(suggestions)))

        # Always include suicide hotline for high risk
        if risk_level == "High" and "If you're having thoughts of harming yourself, call a crisis hotline immediately: National Suicide Prevention Lifeline at 988 or 1-800-273-8255." not in selected_suggestions:
            selected_suggestions.insert(0, "If you're having thoughts of harming yourself, call a crisis hotline immediately: National Suicide Prevention Lifeline at 988 or 1-800-273-8255.")

        return selected_suggestions

    def generate_report(self, risk_assessment, text_features=None, audio_features=None, phq9_scores=None, sample_id=None):
        """
        Generate clinical report.

        Args:
            risk_assessment (dict): Risk assessment results
            text_features (dict): Text features
            audio_features (dict): Audio features
            phq9_scores (list): PHQ-9 scores
            sample_id (int): Sample ID

        Returns:
            dict: Clinical report
        """
        # Extract risk level and score
        risk_level = risk_assessment['risk_level']
        risk_score = risk_assessment['risk_score']

        # Extract modality contributions
        modality_contributions = risk_assessment.get('modality_contributions', {})

        # Extract suicide risk
        suicide_risk = False
        if 'suicide_risk' in risk_assessment:
            suicide_risk = risk_assessment['suicide_risk']
        elif phq9_scores is not None:
            suicide_risk = phq9_scores[8] > 0

        # Generate observations
        observations = self.generate_observations(risk_level, modality_contributions, suicide_risk)

        # Generate suggestions
        suggestions = self.generate_suggestions(risk_level)

        # Create report
        from datetime import datetime

        report = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'sample_id': sample_id,
            'depression_probability': risk_score,
            'risk_level': risk_level,
            'observations': observations,
            'suggestions': suggestions
        }

        # Add modality contributions if available
        if modality_contributions:
            report['modality_contributions'] = modality_contributions

        # Add PHQ-9 results if available
        if phq9_scores is not None:
            report['phq9_total_score'] = sum(phq9_scores)
            report['phq9_scores'] = phq9_scores
            report['suicide_risk'] = suicide_risk

        return report

# Initialize clinical report generator
report_generator = SimpleClinicalReportGenerator()

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
    # Clean text (not used but kept for clarity)
    text_processor.clean_text(text)

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
