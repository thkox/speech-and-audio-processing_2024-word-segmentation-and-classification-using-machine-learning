import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

# Define the directories for the datasets
background_sound_dir = 'files/datasets/background_sound'
foreground_sound_dir = 'files/datasets/foreground_sound'


# Function to load a .wav file and extract MFCC and Mel spectrogram features
def load_and_extract_features(file_path, n_mfcc=96, n_mels=80, n_fft=1024, hop_length=512):
    """
    Load an audio file and extract MFCC and Mel spectrogram features.

    Args:
        file_path (str): Path to the audio file.
        n_mfcc (int): Number of MFCC features to extract.
        n_mels (int): Number of Mel bands to generate.
        n_fft (int): FFT window size.
        hop_length (int): Number of samples between successive frames.

    Returns:
        np.ndarray: Mean MFCC features.
        np.ndarray: Log Mel spectrogram.
    """
    print(f"Loading and extracting features from {file_path}")
    audio, sample_rate = librosa.load(file_path, sr=None)

    # Extract MFCC features
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
    mfccs_mean = np.mean(mfccs.T, axis=0)

    # Compute the Mel spectrogram
    mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_fft=n_fft, hop_length=hop_length,
                                                     n_mels=n_mels)

    # Convert to log scale (dB). We'll use the peak power (max) as reference.
    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)

    # Plotting the Mel spectrogram
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(log_mel_spectrogram, sr=sample_rate, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel spectrogram')
    plt.tight_layout()
    plt.show()

    return mfccs_mean, log_mel_spectrogram


# Function to load all .wav files in a directory and extract features
def load_audio_files_and_extract_features(directory, label, n_mfcc=96, n_mels=80, n_fft=1024, hop_length=512):
    """
    Load all audio files in a directory and extract MFCC and Mel spectrogram features.

    Args:
        directory (str): Path to the directory containing audio files.
        label (int): Label for the audio files (0 for background, 1 for foreground).
        n_mfcc (int): Number of MFCC features to extract.
        n_mels (int): Number of Mel bands to generate.
        n_fft (int): FFT window size.
        hop_length (int): Number of samples between successive frames.

    Returns:
        np.ndarray: Extracted features.
        np.ndarray: Labels for the features.
    """
    print(f"Loading audio files from {directory} and extracting features")
    features = []
    labels = []
    for file_name in os.listdir(directory):
        if file_name.endswith('.wav'):
            file_path = os.path.join(directory, file_name)
            mfccs_mean, _ = load_and_extract_features(file_path, n_mfcc, n_mels, n_fft, hop_length)
            features.append(mfccs_mean)
            labels.append(label)
    return np.array(features), np.array(labels)


# Main execution for feature extraction
def extract_features(n_mfcc=96, n_mels=80, n_fft=1024, hop_length=512):
    """
    Extract features from both background and foreground sound datasets.

    Args:
        n_mfcc (int): Number of MFCC features to extract.
        n_mels (int): Number of Mel bands to generate.
        n_fft (int): FFT window size.
        hop_length (int): Number of samples between successive frames.

    Returns:
        np.ndarray: All extracted features.
        np.ndarray: Corresponding labels.
    """
    print("Starting feature extraction")

    # Load audio files and extract features
    background_features, background_labels = load_audio_files_and_extract_features(background_sound_dir, 0, n_mfcc,
                                                                                   n_mels, n_fft, hop_length)
    foreground_features, foreground_labels = load_audio_files_and_extract_features(foreground_sound_dir, 1, n_mfcc,
                                                                                   n_mels, n_fft, hop_length)

    # Concatenate the features and labels of both background and foreground sounds
    all_features = np.concatenate((background_features, foreground_features), axis=0)
    all_labels = np.concatenate((background_labels, foreground_labels), axis=0)

    # Shuffle the data
    all_features, all_labels = shuffle(all_features, all_labels, random_state=0)

    print("Feature extraction completed and saved")
    return all_features, all_labels
