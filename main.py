import load_database as cda
from classifiers import feature_extraction as fe
import numpy as np
import os
from scipy.ndimage import median_filter
from classifiers import mlp, rnn, svm, least_squares as ls

# # # Create the datasets that are necessary for the app
# cda.create_datasets()
# #
# # Feature extraction of both background and foreground datasets
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


# check a test file to predict the labels
# test_file = '/home/theo/Downloads/LibriSpeech/test-clean/2094/142345/2094-142345-0008.flac'
test_file = 'files/datasets/test/Lab41-SRI-VOiCES-rm1-none-sp2785-ch163322-sg0030-mc01-stu-clo-dg080.wav'

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

intervals = fe.detect_voice_intervals(svm_predictions_median, frame_rate)

fe.plot_audio_with_intervals(audio, sample_rate, intervals)

print("=====================================")
