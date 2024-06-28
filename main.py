from classifiers import feature_extraction as fe
import numpy as np
import load_database as cda
from scipy.ndimage import median_filter
from classifiers import mlp, rnn, svm, least_squares as ls
from classifiers.speech_to_text import transcribe_audio, show_predictions

# Create the datasets that are necessary for the app
cda.create_datasets()

# Feature extraction of both background and foreground datasets
fe.extract_features(shuffle_data=False, show_plots=False)  # to create and save the features

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
test_file = 'files/datasets/test/Lab41-SRI-VOiCES-rm1-babb-sp0176-ch123271-sg0019-mc01-stu-clo-dg030.wav'

# get the labels from the test file
# labels = fe.webrtc_vad_speech_detection(test_file)


# Extract features from the test file
_, features, sample_rate, audio = fe.load_and_extract_features(test_file, show_plots=False)
hop_length = fe.HOP_LENGTH
frame_rate = sample_rate / hop_length

L = 5  # Window size for median filter

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

# get the intervals of the voice
intervals_original, texts = transcribe_audio(test_file)


# Show the predictions
show_predictions(audio, sample_rate, intervals_original, svm_predictions_median, frame_rate, "SVM predictions")
show_predictions(audio, sample_rate, intervals_original, mlp_predictions_median, frame_rate, "MLP predictions")
show_predictions(audio, sample_rate, intervals_original, rnn_predictions_median, frame_rate, "RNN predictions")
show_predictions(audio, sample_rate, intervals_original, ls_predictions_median, frame_rate, "Least Squares predictions")

print("=====================================")
