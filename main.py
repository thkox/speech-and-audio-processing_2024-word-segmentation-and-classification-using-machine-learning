import load_database as cda
from classifiers import feature_extraction as fe
import numpy as np
import os
from scipy.ndimage import median_filter
from classifiers import mlp, rnn, svm, least_squares as ls

# Create the datasets that are necessary for the app
cda.create_datasets()

# Feature extraction of both background and foreground datasets
fe.extract_features(shuffle_data=True, show_plots=False)  # to create and save the features

# Train the classifiers
svm.train()
mlp.train()
rnn.train()
ls.train()

# Load the trained models
svm_model = svm.load_model()
mlp_model = mlp.load_model()
rnn_model = rnn.load_model()
ls_model = ls.load_model()

# check a test file to predict the labels
# test_file = '/home/theo/Downloads/LibriSpeech/test-clean/2094/142345/2094-142345-0008.flac'
test_file = 'files/datasets/test/test_2.mp3'

# get the labels from the test file
# labels = fe.webrtc_vad_speech_detection(test_file)


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

rnn_predictions = rnn.predict(features, n_of_files=1)
print("RNN predictions:", rnn_predictions)
rnn_predictions_median = np.squeeze(median_filter(rnn_predictions, size=L))
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


def show_predictions(predictions, frame_rate, title):
    intervals = fe.detect_voice_intervals(predictions, frame_rate)
    fe.plot_audio_with_intervals(audio, sample_rate, intervals, title)


# Show the predictions
show_predictions(svm_predictions, frame_rate, "SVM predictions")
show_predictions(mlp_predictions, frame_rate, "MLP predictions")
show_predictions(rnn_predictions, frame_rate, "RNN predictions")
show_predictions(ls_predictions, frame_rate, "Least Squares predictions")
show_predictions(majority_voting, frame_rate, "Majority voting predictions")

print("=====================================")
