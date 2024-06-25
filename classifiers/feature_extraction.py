import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

# Define the directories for the datasets
BACKGROUND_SOUND_DIR = 'files/datasets/background_sound'
FOREGROUND_SOUND_DIR = 'files/datasets/foreground_sound'

# Define constants for feature extraction parameters
N_MFCC = 20  # Number of Mel-frequency cepstral coefficients (MFCCs) features to extract
N_MELS = 96  # Number of Mel bands to generate
N_FFT = 400  # FFT window size
HOP_LENGTH = 200  # Number of samples between successive frames


def load_and_extract_features(file_path, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH, show_plots=False):
    """
    Load an audio file and extract MFCC features and mel spectrogram for each frame.

    Args:
        file_path (str): Path to the audio file.
        n_mfcc (int): Number of MFCC features to extract.
        n_fft (int): FFT window size.
        hop_length (int): Number of samples between successive frames.
        show_plots (bool): Whether to display plots of the mel spectrogram.

    Returns:
        np.ndarray: MFCC features and mel spectrogram for each frame.
    """
    print(f"Loading and extracting features from {file_path}")
    audio, sample_rate = librosa.load(file_path, sr=None)

    # Apply Hamming window to the audio signal
    audio_windowed = np.hamming(len(audio)) * audio

    # Extract MFCC features for each frame
    mfccs = librosa.feature.mfcc(y=audio_windowed, sr=sample_rate, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)

    # Extract mel spectrogram
    mel_spec = librosa.feature.melspectrogram(y=audio_windowed, sr=sample_rate, n_mels=N_MELS, n_fft=n_fft, hop_length=hop_length)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)  # Convert to decibel scale

    if show_plots:
        # Plot the mel spectrogram
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(mel_spec_db, x_axis='time', y_axis='mel', sr=sample_rate, hop_length=hop_length)
        plt.colorbar(format='%+2.0f dB')
        plt.title(f'Mel spectrogram for {os.path.basename(file_path)}')
        plt.tight_layout()
        plt.show()

    return mfccs.T, mel_spec_db.T  # Return MFCCs and mel spectrogram for each frame (transposed for convenience)


def load_audio_files_and_extract_features(directory, label, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH, show_plots=False):
    """
    Load all audio files in a directory and extract MFCC features and mel spectrogram for each frame.

    Args:
        directory (str): Path to the directory containing audio files.
        label (int): Label for the audio files (0 for background, 1 for foreground).
        n_mfcc (int): Number of MFCC features to extract.
        n_fft (int): FFT window size.
        hop_length (int): Number of samples between successive frames.
        show_plots (bool): Whether to display plots of the mel spectrogram.

    Returns:
        np.ndarray: Extracted MFCC features for all frames.
        np.ndarray: Extracted mel spectrogram for all frames.
        np.ndarray: Labels for the features.
    """
    print(f"Loading audio files from {directory} and extracting features")
    all_mfccs = []
    all_mel_specs = []
    all_labels = []
    for file_name in os.listdir(directory):
        if file_name.endswith('.wav'):
            file_path = os.path.join(directory, file_name)
            mfccs, mel_spec = load_and_extract_features(file_path, n_mfcc, n_fft, hop_length, show_plots)
            all_mfccs.append(mfccs)
            all_mel_specs.append(mel_spec)
            labels = np.full(mfccs.shape[0], label)
            all_labels.append(labels)
    return np.vstack(all_mfccs), np.vstack(all_mel_specs), np.hstack(all_labels)


def extract_features(n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH, shuffle_data=False, show_plots=False):
    """
    Extract features from both background and foreground sound datasets.

    Args:
        n_mfcc (int): Number of MFCC features to extract.
        n_fft (int): FFT window size.
        hop_length (int): Number of samples between successive frames.
        shuffle_data (bool): Whether to shuffle the data.
        show_plots (bool): Whether to display plots of the mel spectrogram.

    Returns:
        np.ndarray: All extracted MFCC features.
        np.ndarray: All extracted mel spectrogram features.
        np.ndarray: Corresponding labels.
    """
    print("Starting feature extraction")

    # Load audio files and extract features for each frame
    background_mfccs, background_mel_specs, background_labels = load_audio_files_and_extract_features(BACKGROUND_SOUND_DIR, 0, n_mfcc,
                                                                                                      n_fft, hop_length, show_plots)
    foreground_mfccs, foreground_mel_specs, foreground_labels = load_audio_files_and_extract_features(FOREGROUND_SOUND_DIR, 1, n_mfcc,
                                                                                                      n_fft, hop_length, show_plots)

    # Concatenate the features and labels of both background and foreground sounds
    all_mfccs = np.concatenate((background_mfccs, foreground_mfccs), axis=0)
    all_mel_specs = np.concatenate((background_mel_specs, foreground_mel_specs), axis=0)
    all_labels = np.concatenate((background_labels, foreground_labels), axis=0)

    # Shuffle the data if specified
    if shuffle_data:
        all_mfccs, all_mel_specs, all_labels = shuffle(all_mfccs, all_mel_specs, all_labels, random_state=0)

    print("Feature extraction completed and saved")
    return all_mfccs, all_mel_specs, all_labels
