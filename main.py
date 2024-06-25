# import create_datasets_for_app as cda
import classifiers.feature_extraction as fe
import classifiers.model_trainning as mt

# Create the datasets that are necessary for the app
# cda.create_datasets()

# Feature extraction of both background and foreground datasets
mfccs, features, labels = fe.extract_features(shuffle_data=False, show_plots=False)
print("Extracted features shape:", features.shape)
print("Extracted labels shape:", labels.shape)

# print("Features:", features)
# print("Labels:", labels)

# Train the classifiers
mt.train_classifiers(features, labels)
