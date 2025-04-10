"""
Audio Data Preprocessing Module

This module handles the preprocessing of audio data from the DAIC-WOZ dataset.
It includes functions for loading, feature extraction, and dataset creation.
"""

import os
import numpy as np
import pandas as pd
import librosa
import pickle
import glob
import json
import logging
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define constants
SAMPLE_RATE = 16000  # Hz
FRAME_LENGTH = 0.025  # 25ms
FRAME_STRIDE = 0.010  # 10ms
N_MFCC = 40  # Number of MFCC features
N_MELS = 128  # Number of Mel bands
N_FFT = 2048  # FFT window size


class AudioProcessor:
    """Class for processing audio data from the DAIC-WOZ dataset."""
    
    def __init__(self, data_path, output_path=None):
        """
        Initialize the Audio processor.
        
        Args:
            data_path (str): Path to the DAIC-WOZ dataset
            output_path (str, optional): Path to save processed data
        """
        self.data_path = data_path
        self.output_path = output_path or os.path.join(data_path, 'processed')
        os.makedirs(self.output_path, exist_ok=True)
        
        # Load depression labels
        self.labels_path = os.path.join(data_path, 'labels', 'train_split_Depression_AVEC2017.csv')
        if os.path.exists(self.labels_path):
            self.labels_df = pd.read_csv(self.labels_path)
        else:
            logger.warning(f"Labels file not found: {self.labels_path}")
            self.labels_df = None
    
    def load_audio(self, participant_id):
        """
        Load audio data for a specific participant.
        
        Args:
            participant_id (int): Participant ID
            
        Returns:
            numpy.ndarray: Audio waveform
        """
        # Format participant ID with leading zeros
        participant_str = f"{participant_id:03d}"
        
        # Path to the audio file
        audio_file = os.path.join(self.data_path, participant_str, f"{participant_str}_AUDIO.wav")
        
        # Check if file exists
        if not os.path.exists(audio_file):
            raise FileNotFoundError(f"Audio file not found: {audio_file}")
        
        # Load audio
        audio, _ = librosa.load(audio_file, sr=SAMPLE_RATE, mono=True)
        
        return audio
    
    def extract_features(self, audio):
        """
        Extract features from audio data.
        
        Args:
            audio (numpy.ndarray): Audio waveform
            
        Returns:
            dict: Dictionary of audio features
        """
        # Calculate frame parameters
        frame_length = int(SAMPLE_RATE * FRAME_LENGTH)
        frame_stride = int(SAMPLE_RATE * FRAME_STRIDE)
        
        # Extract MFCCs
        mfccs = librosa.feature.mfcc(
            y=audio, 
            sr=SAMPLE_RATE, 
            n_mfcc=N_MFCC,
            n_fft=N_FFT,
            hop_length=frame_stride,
            win_length=frame_length
        )
        
        # Extract Mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=SAMPLE_RATE,
            n_mels=N_MELS,
            n_fft=N_FFT,
            hop_length=frame_stride,
            win_length=frame_length
        )
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Extract chroma features
        chroma = librosa.feature.chroma_stft(
            y=audio,
            sr=SAMPLE_RATE,
            n_fft=N_FFT,
            hop_length=frame_stride,
            win_length=frame_length
        )
        
        # Extract spectral contrast
        contrast = librosa.feature.spectral_contrast(
            y=audio,
            sr=SAMPLE_RATE,
            n_fft=N_FFT,
            hop_length=frame_stride
        )
        
        # Extract zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(
            y=audio,
            frame_length=frame_length,
            hop_length=frame_stride
        )
        
        # Extract RMS energy
        rms = librosa.feature.rms(
            y=audio,
            frame_length=frame_length,
            hop_length=frame_stride
        )
        
        # Extract pitch (fundamental frequency)
        f0, voiced_flag, voiced_probs = librosa.pyin(
            y=audio,
            fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C7'),
            sr=SAMPLE_RATE,
            hop_length=frame_stride
        )
        
        # Calculate statistics for each feature
        feature_stats = {}
        
        # Function to calculate statistics
        def calc_stats(feature, name):
            feature_stats[f"{name}_mean"] = np.mean(feature)
            feature_stats[f"{name}_std"] = np.std(feature)
            feature_stats[f"{name}_max"] = np.max(feature)
            feature_stats[f"{name}_min"] = np.min(feature)
            feature_stats[f"{name}_range"] = np.ptp(feature)
            feature_stats[f"{name}_median"] = np.median(feature)
            feature_stats[f"{name}_q25"] = np.percentile(feature, 25)
            feature_stats[f"{name}_q75"] = np.percentile(feature, 75)
        
        # Calculate statistics for each feature
        for i in range(N_MFCC):
            calc_stats(mfccs[i], f"mfcc_{i+1}")
        
        calc_stats(np.mean(mel_spec_db, axis=0), "mel_spec")
        calc_stats(np.mean(chroma, axis=0), "chroma")
        calc_stats(np.mean(contrast, axis=0), "contrast")
        calc_stats(zcr[0], "zcr")
        calc_stats(rms[0], "rms")
        
        # Handle NaN values in pitch
        f0_clean = f0[~np.isnan(f0)]
        if len(f0_clean) > 0:
            calc_stats(f0_clean, "pitch")
        else:
            # If no pitch detected, set default values
            feature_stats["pitch_mean"] = 0
            feature_stats["pitch_std"] = 0
            feature_stats["pitch_max"] = 0
            feature_stats["pitch_min"] = 0
            feature_stats["pitch_range"] = 0
            feature_stats["pitch_median"] = 0
            feature_stats["pitch_q25"] = 0
            feature_stats["pitch_q75"] = 0
        
        # Calculate speaking rate (syllables per second)
        # This is a rough estimate based on energy peaks
        energy = librosa.feature.rms(y=audio, hop_length=frame_stride)[0]
        peaks = librosa.util.peak_pick(energy, pre_max=3, post_max=3, pre_avg=3, post_avg=5, delta=0.1, wait=10)
        duration = len(audio) / SAMPLE_RATE
        speaking_rate = len(peaks) / duration
        feature_stats["speaking_rate"] = speaking_rate
        
        # Calculate speech duration and pauses
        silence_threshold = 0.01
        is_silence = energy < silence_threshold
        silence_runs = librosa.util.find_runs(is_silence)
        
        # Calculate total speech duration (non-silence)
        speech_frames = np.sum(~is_silence)
        speech_duration = speech_frames * frame_stride / SAMPLE_RATE
        feature_stats["speech_duration"] = speech_duration
        
        # Calculate pause statistics
        if len(silence_runs) > 0:
            pause_durations = [(run[1] - run[0]) * frame_stride / SAMPLE_RATE 
                              for run in silence_runs if run[1] - run[0] > 3]  # Pauses longer than 30ms
            
            if len(pause_durations) > 0:
                feature_stats["pause_count"] = len(pause_durations)
                feature_stats["pause_mean_duration"] = np.mean(pause_durations)
                feature_stats["pause_max_duration"] = np.max(pause_durations)
                feature_stats["pause_total_duration"] = np.sum(pause_durations)
                feature_stats["pause_rate"] = len(pause_durations) / duration
            else:
                feature_stats["pause_count"] = 0
                feature_stats["pause_mean_duration"] = 0
                feature_stats["pause_max_duration"] = 0
                feature_stats["pause_total_duration"] = 0
                feature_stats["pause_rate"] = 0
        else:
            feature_stats["pause_count"] = 0
            feature_stats["pause_mean_duration"] = 0
            feature_stats["pause_max_duration"] = 0
            feature_stats["pause_total_duration"] = 0
            feature_stats["pause_rate"] = 0
        
        return feature_stats
    
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
        
        all_features = []
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
                
                # Load audio
                audio = self.load_audio(participant_id)
                
                # Extract features
                features = self.extract_features(audio)
                
                # Convert features to vector
                feature_vector = np.array(list(features.values()))
                
                # Store features and labels
                all_features.append(feature_vector)
                all_labels.append([phq8_binary, phq8_score])
                
                logger.info(f"Processed participant {participant_id}")
            except Exception as e:
                logger.error(f"Error processing participant {participant_id}: {e}")
        
        # Convert to numpy arrays
        X = np.vstack(all_features)
        y = np.array(all_labels)
        
        # Normalize features
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        
        # Save processed data
        np.save(os.path.join(self.output_path, 'audio_features.npy'), X)
        np.save(os.path.join(self.output_path, 'audio_labels.npy'), y)
        
        # Save feature names
        if len(all_features) > 0:
            feature_names = list(self.extract_features(np.zeros(1000)).keys())
            with open(os.path.join(self.output_path, 'audio_feature_names.pkl'), 'wb') as f:
                pickle.dump(feature_names, f)
        
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
        X = np.load(os.path.join(self.output_path, 'audio_features.npy'))
        y = np.load(os.path.join(self.output_path, 'audio_labels.npy'))
        
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
        with open(os.path.join(self.output_path, 'audio_dataset.pkl'), 'wb') as f:
            pickle.dump(dataset, f)
        
        logger.info("Finished creating dataset splits")
        
        return dataset


def download_daic_woz_dataset(output_dir):
    """
    Function to guide users through downloading the DAIC-WOZ dataset.
    
    Args:
        output_dir (str): Directory to save instructions
    """
    instructions = """
    # DAIC-WOZ Dataset Download Instructions
    
    The DAIC-WOZ dataset is not directly downloadable without registration. Follow these steps to obtain it:
    
    1. Visit the DAIC-WOZ dataset website: https://dcapswoz.ict.usc.edu/
    
    2. Click on the "Apply Now DAIC-WOZ" button and fill out the form.
    
    3. You will receive credentials to download the dataset via email.
    
    4. Download the dataset and extract it to the 'data/audio/raw' directory in this project.
    
    5. Run the preprocessing script:
       ```
       python -m data.audio.preprocess_audio
       ```
    """
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Write instructions to file
    with open(os.path.join(output_dir, 'daic_woz_download_instructions.md'), 'w') as f:
        f.write(instructions)
    
    print(f"Instructions for downloading the DAIC-WOZ dataset have been saved to {output_dir}/daic_woz_download_instructions.md")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Process audio data from the DAIC-WOZ dataset')
    parser.add_argument('--data_path', type=str, default='data/audio/raw',
                        help='Path to the DAIC-WOZ dataset')
    parser.add_argument('--output_path', type=str, default='data/audio/processed',
                        help='Path to save processed data')
    parser.add_argument('--download_instructions', action='store_true',
                        help='Generate download instructions for the DAIC-WOZ dataset')
    
    args = parser.parse_args()
    
    if args.download_instructions:
        download_daic_woz_dataset(os.path.dirname(args.data_path))
    else:
        # Check if data exists
        if not os.path.exists(args.data_path):
            print(f"Data path {args.data_path} does not exist. Generating download instructions...")
            download_daic_woz_dataset(os.path.dirname(args.data_path))
        else:
            # Process data
            processor = AudioProcessor(args.data_path, args.output_path)
            processor.process_all_participants()
            processor.create_dataset_splits()
