"""
Text Data Preprocessing Module

This module handles the preprocessing of text data from the DAIC-WOZ dataset.
It includes functions for loading, tokenization, and feature extraction.
"""

import os
import numpy as np
import pandas as pd
import json
import glob
import pickle
import logging
import re
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import BertTokenizer, BertModel
import torch
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Download NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')


class TextProcessor:
    """Class for processing text data from the DAIC-WOZ dataset."""
    
    def __init__(self, data_path, output_path=None, use_bert=True):
        """
        Initialize the Text processor.
        
        Args:
            data_path (str): Path to the DAIC-WOZ dataset
            output_path (str, optional): Path to save processed data
            use_bert (bool): Whether to use BERT for feature extraction
        """
        self.data_path = data_path
        self.output_path = output_path or os.path.join(data_path, 'processed')
        os.makedirs(self.output_path, exist_ok=True)
        self.use_bert = use_bert
        
        # Load depression labels
        self.labels_path = os.path.join(data_path, 'labels', 'train_split_Depression_AVEC2017.csv')
        if os.path.exists(self.labels_path):
            self.labels_df = pd.read_csv(self.labels_path)
        else:
            logger.warning(f"Labels file not found: {self.labels_path}")
            self.labels_df = None
        
        # Initialize BERT tokenizer and model if needed
        if use_bert:
            logger.info("Initializing BERT tokenizer and model")
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            self.model = BertModel.from_pretrained('bert-base-uncased')
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model.to(self.device)
            self.model.eval()
    
    def load_transcript(self, participant_id):
        """
        Load transcript data for a specific participant.
        
        Args:
            participant_id (int): Participant ID
            
        Returns:
            list: List of participant's utterances
        """
        # Format participant ID with leading zeros
        participant_str = f"{participant_id:03d}"
        
        # Path to the transcript file
        transcript_file = os.path.join(self.data_path, participant_str, f"{participant_str}_TRANSCRIPT.csv")
        
        # Check if file exists
        if not os.path.exists(transcript_file):
            raise FileNotFoundError(f"Transcript file not found: {transcript_file}")
        
        # Load transcript
        transcript_df = pd.read_csv(transcript_file)
        
        # Extract participant's utterances
        participant_utterances = transcript_df[transcript_df['speaker'] == 'Participant']['value'].tolist()
        
        return participant_utterances
    
    def preprocess_text(self, text_list):
        """
        Preprocess a list of text utterances.
        
        Args:
            text_list (list): List of text utterances
            
        Returns:
            str: Preprocessed text
        """
        # Combine all utterances
        combined_text = ' '.join(text_list)
        
        # Convert to lowercase
        text = combined_text.lower()
        
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def extract_features_tfidf(self, preprocessed_texts, max_features=1000):
        """
        Extract TF-IDF features from preprocessed texts.
        
        Args:
            preprocessed_texts (list): List of preprocessed texts
            max_features (int): Maximum number of features
            
        Returns:
            numpy.ndarray: TF-IDF feature matrix
        """
        logger.info("Extracting TF-IDF features")
        
        # Initialize TF-IDF vectorizer
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        # Fit and transform texts
        X = vectorizer.fit_transform(preprocessed_texts).toarray()
        
        # Save vectorizer
        with open(os.path.join(self.output_path, 'tfidf_vectorizer.pkl'), 'wb') as f:
            pickle.dump(vectorizer, f)
        
        return X
    
    def extract_features_bert(self, preprocessed_texts, max_length=512):
        """
        Extract BERT features from preprocessed texts.
        
        Args:
            preprocessed_texts (list): List of preprocessed texts
            max_length (int): Maximum sequence length for BERT
            
        Returns:
            numpy.ndarray: BERT feature matrix
        """
        logger.info("Extracting BERT features")
        
        features = []
        
        with torch.no_grad():
            for text in tqdm(preprocessed_texts, desc="Extracting BERT features"):
                # Tokenize text
                inputs = self.tokenizer(
                    text,
                    return_tensors='pt',
                    padding='max_length',
                    truncation=True,
                    max_length=max_length
                )
                
                # Move inputs to device
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Get BERT embeddings
                outputs = self.model(**inputs)
                
                # Use CLS token embedding as feature vector
                cls_embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                
                features.append(cls_embedding.flatten())
        
        return np.vstack(features)
    
    def extract_linguistic_features(self, text_list):
        """
        Extract linguistic features from a list of text utterances.
        
        Args:
            text_list (list): List of text utterances
            
        Returns:
            dict: Dictionary of linguistic features
        """
        # Combine all utterances
        combined_text = ' '.join(text_list)
        
        # Tokenize
        tokens = word_tokenize(combined_text.lower())
        
        # Get stopwords
        stop_words = set(stopwords.words('english'))
        
        # Calculate features
        features = {}
        
        # Total word count
        features['word_count'] = len(tokens)
        
        # Average word length
        word_lengths = [len(word) for word in tokens if word.isalpha()]
        features['avg_word_length'] = np.mean(word_lengths) if word_lengths else 0
        
        # Unique word count
        features['unique_word_count'] = len(set(tokens))
        
        # Lexical diversity (unique words / total words)
        features['lexical_diversity'] = features['unique_word_count'] / features['word_count'] if features['word_count'] > 0 else 0
        
        # Stopword ratio
        stop_word_count = sum(1 for word in tokens if word in stop_words)
        features['stopword_ratio'] = stop_word_count / features['word_count'] if features['word_count'] > 0 else 0
        
        # Sentence count
        sentences = [s for s in combined_text.split('.') if s.strip()]
        features['sentence_count'] = len(sentences)
        
        # Average sentence length (in words)
        sentence_lengths = [len(s.split()) for s in sentences]
        features['avg_sentence_length'] = np.mean(sentence_lengths) if sentence_lengths else 0
        
        # Sentiment-related words (very basic approach)
        positive_words = ['good', 'happy', 'positive', 'nice', 'great', 'excellent', 'wonderful', 'love', 'enjoy', 'like']
        negative_words = ['bad', 'sad', 'negative', 'terrible', 'awful', 'horrible', 'hate', 'dislike', 'angry', 'depressed', 'depression', 'anxiety', 'worried', 'stress', 'tired', 'exhausted']
        
        features['positive_word_count'] = sum(1 for word in tokens if word in positive_words)
        features['negative_word_count'] = sum(1 for word in tokens if word in negative_words)
        features['sentiment_ratio'] = features['positive_word_count'] / (features['positive_word_count'] + features['negative_word_count']) if (features['positive_word_count'] + features['negative_word_count']) > 0 else 0.5
        
        # First-person pronoun usage
        first_person_pronouns = ['i', 'me', 'my', 'mine', 'myself']
        features['first_person_count'] = sum(1 for word in tokens if word in first_person_pronouns)
        features['first_person_ratio'] = features['first_person_count'] / features['word_count'] if features['word_count'] > 0 else 0
        
        return features
    
    def process_all_participants(self):
        """
        Process all participants in the dataset.
        
        Returns:
            tuple: (X, y) where X is the feature matrix and y is the label vector
        """
        logger.info("Processing all participants")
        
        # Get list of participant directories
        participant_dirs = glob.glob(os.path.join(self.data_path, '[0-9][0-9][0-9]'))
        participant_ids = [int(os.path.basename(d)) for d in participant_dirs]
        
        all_preprocessed_texts = []
        all_linguistic_features = []
        all_labels = []
        
        for participant_id in tqdm(participant_ids, desc="Processing participants"):
            try:
                # Get depression label
                if self.labels_df is not None:
                    label_row = self.labels_df[self.labels_df['Participant_ID'] == participant_id]
                    if len(label_row) == 0:
                        logger.warning(f"No label found for participant {participant_id}")
                        continue
                    
                    phq8_binary = label_row['PHQ8_Binary'].values[0]
                    phq8_score = label_row['PHQ8_Score'].values[0]
                else:
                    # If no labels file, try to find labels in the participant directory
                    label_file = os.path.join(self.data_path, f"{participant_id:03d}", f"{participant_id:03d}_P.json")
                    if os.path.exists(label_file):
                        with open(label_file, 'r') as f:
                            label_data = json.load(f)
                        phq8_binary = 1 if label_data.get('PHQ8_Score', 0) >= 10 else 0
                        phq8_score = label_data.get('PHQ8_Score', 0)
                    else:
                        logger.warning(f"No label found for participant {participant_id}")
                        continue
                
                # Load transcript
                utterances = self.load_transcript(participant_id)
                
                # Preprocess text
                preprocessed_text = self.preprocess_text(utterances)
                
                # Extract linguistic features
                linguistic_features = self.extract_linguistic_features(utterances)
                
                # Store data
                all_preprocessed_texts.append(preprocessed_text)
                all_linguistic_features.append(list(linguistic_features.values()))
                all_labels.append([phq8_binary, phq8_score])
                
                logger.info(f"Processed participant {participant_id}")
            except Exception as e:
                logger.error(f"Error processing participant {participant_id}: {e}")
        
        # Extract features
        if self.use_bert:
            X_text = self.extract_features_bert(all_preprocessed_texts)
        else:
            X_text = self.extract_features_tfidf(all_preprocessed_texts)
        
        # Convert linguistic features to numpy array
        X_ling = np.array(all_linguistic_features)
        
        # Combine features
        X = np.hstack([X_text, X_ling])
        
        # Convert labels to numpy array
        y = np.array(all_labels)
        
        # Save processed data
        np.save(os.path.join(self.output_path, 'text_features.npy'), X)
        np.save(os.path.join(self.output_path, 'text_labels.npy'), y)
        
        # Save feature information
        if len(all_linguistic_features) > 0:
            feature_info = {
                'text_feature_dim': X_text.shape[1],
                'linguistic_feature_names': list(self.extract_linguistic_features(['dummy text']).keys())
            }
            with open(os.path.join(self.output_path, 'text_feature_info.pkl'), 'wb') as f:
                pickle.dump(feature_info, f)
        
        logger.info("Finished processing all participants")
        
        return X, y
    
    def create_dataset_splits(self, test_size=0.2, val_size=0.1, random_state=42):
        """
        Create train/val/test splits from the processed data.
        
        Args:
            test_size (float): Proportion of data to use for testing
            val_size (float): Proportion of data to use for validation
            random_state (int): Random seed for reproducibility
            
        Returns:
            dict: Dictionary containing the data splits
        """
        from sklearn.model_selection import train_test_split
        
        logger.info("Creating dataset splits")
        
        # Load processed data
        X = np.load(os.path.join(self.output_path, 'text_features.npy'))
        y = np.load(os.path.join(self.output_path, 'text_labels.npy'))
        
        # Split into train+val and test
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y[:, 0]
        )
        
        # Split train+val into train and val
        val_ratio = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=val_ratio, random_state=random_state, stratify=y_train_val[:, 0]
        )
        
        # Create dataset dictionary
        dataset = {
            'X_train': X_train,
            'y_train': y_train,
            'X_val': X_val,
            'y_val': y_val,
            'X_test': X_test,
            'y_test': y_test
        }
        
        # Save dataset splits
        with open(os.path.join(self.output_path, 'text_dataset.pkl'), 'wb') as f:
            pickle.dump(dataset, f)
        
        logger.info("Finished creating dataset splits")
        
        return dataset


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Process text data from the DAIC-WOZ dataset')
    parser.add_argument('--data_path', type=str, default='data/text/raw',
                        help='Path to the DAIC-WOZ dataset')
    parser.add_argument('--output_path', type=str, default='data/text/processed',
                        help='Path to save processed data')
    parser.add_argument('--use_bert', action='store_true',
                        help='Use BERT for feature extraction')
    
    args = parser.parse_args()
    
    # Check if data exists
    if not os.path.exists(args.data_path):
        print(f"Data path {args.data_path} does not exist. Please download the DAIC-WOZ dataset.")
    else:
        # Process data
        processor = TextProcessor(args.data_path, args.output_path, args.use_bert)
        processor.process_all_participants()
        processor.create_dataset_splits()
