import os
import numpy as np
import librosa
import matplotlib.pyplot as plt

# Define the directories for the datasets
background_sound_dir = 'files/datasets/background_sound'
foreground_sound_dir = 'files/datasets/foreground_sound'


# Function to load a .wav file and extract MFCC features
def load_and_extract_features(file_path):
    print(f"Loading and extracting features from {file_path}")
    audio, sample_rate = librosa.load(file_path, sr=None)
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=80)
    mfccs_mean = np.mean(mfccs.T, axis=0)

    # Compute the Mel spectrogram
    mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sample_rate)

    # Convert to log scale (dB). We'll use the peak power (max) as reference.
    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)

    # Plotting the Mel spectrogram
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(log_mel_spectrogram, sr=sample_rate, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel spectrogram')
    plt.tight_layout()
    plt.show()
    return mfccs_mean


# Function to load all .wav files in a directory and extract features
def load_audio_files_and_extract_features(directory, label):
    print(f"Loading audio files from {directory} and extracting features")
    features = []
    labels = []
    for file_name in os.listdir(directory):
        if file_name.endswith('.wav'):
            file_path = os.path.join(directory, file_name)
            mfccs_mean = load_and_extract_features(file_path)
            features.append(mfccs_mean)
            labels.append(label)
    return np.array(features), np.array(labels)


# Main execution for feature extraction
def extract_features():
    print("Starting feature extraction")
    # Load audio files and extract features
    background_data = load_audio_files_and_extract_features(background_sound_dir, 0)
    foreground_data = load_audio_files_and_extract_features(foreground_sound_dir, 1)

    # Create DataFrame for each dataset
    background_df = pd.DataFrame(background_data)
    foreground_df = pd.DataFrame(foreground_data)

    # Print DataFrame to console
    print("Background dataset:")
    print(background_df)
    print("Foreground dataset:")
    print(foreground_df)

    # Save DataFrame to text file
    background_df.to_csv('files/output/feature_extraction_from_background.txt', sep='\t', index=False)
    foreground_df.to_csv('files/output/feature_extraction_from_foreground.txt', sep='\t', index=False)

    print("Feature extraction completed and saved")
