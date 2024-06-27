import load_database as cda
from classifiers import feature_extraction as fe
from classifiers import mlp
from classifiers import rnn
from classifiers import svm
from classifiers import least_squares as ls

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

# check a test file to predict the labels
test_file = 'files/datasets/test/Lab41-SRI-VOiCES-rm3-musi-mc16-bar-tbc.wav'

# Extract features from the test file
_, features = fe.load_and_extract_features(test_file)


# Predict the labels using the trained models
svm_predictions = svm.predict(features)
print("SVM predictions:", svm_predictions)
mlp_predictions = mlp.predict(features)
print("MLP predictions:", mlp_predictions)
rnn_predictions = rnn.predict(features)
print("RNN predictions:", rnn_predictions)
ls_predictions = ls.predict(features)
print("Least Squares predictions:", ls_predictions)
