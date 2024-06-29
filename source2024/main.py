from source2024 import feature_extraction as fe
import numpy as np
import load_database as ld
from scipy.ndimage import median_filter
from source2024.classifiers import rnn, svm, mlp, least_squares as ls
from source2024.speech_to_text import transcribe_audio, show_predictions

# # Create the datasets that are necessary for the app
ld.create_datasets()
#
# # Feature extraction of both background and foreground datasets
# fe.extract_features(shuffle_data=True, show_plots=False)  # to create and save the features
#
# # Train the source2024
# svm.train()
# mlp.train()
# rnn.train()
# ls.train()

# Load the trained models
svm_model = svm.load_model()
mlp_model = mlp.load_model()
rnn_model = rnn.load_model()
ls_model = ls.load_model()

# # check a test file to predict the labels
# test_file = 'auxiliary2024/datasets/test/Lab41-SRI-VOiCES-rm1-babb-sp0175-ch129587-sg0019-mc01-stu-clo-dg000.wav'
#
#
# # Extract features from the test file
# _, features, sample_rate, audio = fe.load_and_extract_features(test_file, show_plots=False)
# hop_length = fe.HOP_LENGTH
# frame_rate = sample_rate / hop_length
#
# L = 5  # Window size for median filter
#
# # Predict the labels using the trained models
# svm_predictions = svm.predict(features)
# svm_predictions_median = median_filter(svm_predictions, size=L)
#
# mlp_predictions = mlp.predict(features)
# mlp_predictions_median = median_filter(mlp_predictions, size=L)
#
# rnn_predictions = rnn.predict(features, n_of_files=1)
# rnn_predictions_median = np.squeeze(median_filter(rnn_predictions, size=L))
#
# ls_predictions = ls.predict(features)
# ls_predictions_median = median_filter(ls_predictions, size=L)
#
# # get the intervals of the voice
# intervals_original, texts = transcribe_audio(test_file)
#
#
# # Show the predictions
# show_predictions(audio, sample_rate, intervals_original, svm_predictions_median, frame_rate, "SVM")
# show_predictions(audio, sample_rate, intervals_original, mlp_predictions_median, frame_rate, "MLP")
# show_predictions(audio, sample_rate, intervals_original, rnn_predictions_median, frame_rate, "RNN")
# show_predictions(audio, sample_rate, intervals_original, ls_predictions_median, frame_rate, "Least_Squares")

print("=====================================")
