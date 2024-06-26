import create_datasets_for_app as cda
import classifiers.feature_extraction as fe
import classifiers.mlp as mlp
import classifiers.rnn as rnn
import classifiers.svm as svm

# Create the datasets that are necessary for the app
cda.create_datasets()

# Feature extraction of both background and foreground datasets
# mfccs, features, labels = fe.extract_features(shuffle_data=False, show_plots=False) # to create and save the features
mfccs, features, labels = fe.load_features('files/output/features.npz') # to load the features

print("Extracted features shape:", features.shape)
print("Extracted labels shape:", labels.shape)

# Train the classifiers
# svm_model = svm.train(features, labels)
# mlp_model = mlp.train(features, labels)
# rnn_model = rnn.train(features, labels)


# Load the trained models
svm_model = svm.load_model()
mlp_model = mlp.load_model()
rnn_model = rnn.load_model()
print("SVM model:", svm_model)
print("MLP model:", mlp_model)
print("RNN model:", rnn_model)

# Predict the labels using the trained models
svm_predictions = svm.predict(svm_model, features)
mlp_predictions = mlp.predict(mlp_model, features)
rnn_predictions = rnn.predict(rnn_model, features)
