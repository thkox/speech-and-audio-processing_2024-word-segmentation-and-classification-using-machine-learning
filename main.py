import load_database as cda
from classifiers import feature_extraction as fe
from classifiers import mlp
from classifiers import rnn
from classifiers import svm
from classifiers import least_squares as ls
from scipy.ndimage import median_filter
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display


# # # Create the datasets that are necessary for the app
# cda.create_datasets()
# #
# # # Feature extraction of both background and foreground datasets
# fe.extract_features(shuffle_data=False, show_plots=False) # to create and save the features
#
# # Train the classifiers
# svm.train()
# mlp.train()
# rnn.train()
# ls.train()
#
# # Load the trained models
# svm_model = svm.load_model()
# mlp_model = mlp.load_model()
# rnn_model = rnn.load_model()
# ls_model = ls.load_model()
# print("SVM model:", svm_model)
# print("MLP model:", mlp_model)
# print("RNN model:", rnn_model)
# print("Least Squares model:", ls_model)


def detect_voice_intervals(predictions, frame_rate, min_length=1):
    intervals = []
    in_foreground = False
    start_time = 0
    length = 0

    for i, label in enumerate(predictions):
        if label == 1:
            if not in_foreground:
                in_foreground = True
                start_time = i / frame_rate
            length += 1
        else:
            if in_foreground and length >= min_length:
                end_time = i / frame_rate
                intervals.append((start_time, end_time))
            in_foreground = False
            length = 0

    if in_foreground and length >= min_length:
        end_time = len(predictions) / frame_rate
        intervals.append((start_time, end_time))

    for interval in intervals:
        print(f"Voice from {interval[0]:.4f} sec to {interval[1]:.4f} sec")

    return intervals


def plot_audio_with_intervals(audio, sample_rate, intervals):
    times = np.arange(len(audio)) / sample_rate
    plt.figure(figsize=(15, 6))
    plt.plot(times, audio, label="Audio waveform")
    for interval in intervals:
        plt.axvspan(interval[0], interval[1], color='red', alpha=0.3,
                    label="Detected voice interval" if interval == intervals[0] else "")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title("Audio waveform with detected voice intervals")
    plt.legend()
    plt.show()


# check a test file to predict the labels
# test_file = '/home/theo/Downloads/LibriSpeech/test-clean/2094/142345/2094-142345-0008.flac'
test_file = '/home/theo/PycharmProjects/speech-and-audio-processing/files/datasets/background_sound/Lab41-SRI-VOiCES-rm1-none-mc02-lav-clo.wav'


# Extract features from the test file
_, features, sample_rate, audio = fe.load_and_extract_features(test_file, show_plots=False)
hop_length = fe.HOP_LENGTH
frame_rate = sample_rate / hop_length

# Median Filter
L = 3

# Predict the labels using the trained models
svm_predictions = svm.predict(features)
print("SVM predictions:", svm_predictions)
svm_predictions_median = median_filter(svm_predictions, size=L)
print("SVM predictions after median filter:", svm_predictions_median)

mlp_predictions = mlp.predict(features)
print("MLP predictions:", mlp_predictions)
mlp_predictions_median = median_filter(mlp_predictions, size=L)
print("MLP predictions after median filter:", mlp_predictions_median)

rnn_predictions = rnn.predict(features)
print("RNN predictions:", rnn_predictions)
rnn_predictions_median = median_filter(rnn_predictions, size=L)
print("RNN predictions after median filter:", rnn_predictions_median)

ls_predictions = ls.predict(features)
print("Least Squares predictions:", ls_predictions)
ls_predictions_median = median_filter(ls_predictions, size=L)
print("Least Squares predictions after median filter:", ls_predictions_median)

# Combine the predictions
combined_predictions = np.vstack(
    (svm_predictions_median, mlp_predictions_median, rnn_predictions_median, ls_predictions_median))

# Majority voting
majority_voting = np.sign(np.sum(combined_predictions, axis=0))
majority_voting_median = median_filter(majority_voting, size=L)
print("Majority voting predictions:", majority_voting)

intervals = detect_voice_intervals(svm_predictions, frame_rate)

plot_audio_with_intervals(audio, sample_rate, intervals)


print("=====================================")
