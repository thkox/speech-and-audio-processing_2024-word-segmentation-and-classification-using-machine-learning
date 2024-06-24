# import create_datasets_for_app as cda
import classifiers.feature_extraction as fe
# import classifiers.model_trainning as mt

# Create the datasets that are necessary for the app
# cda.create_datasets()

# Feature extraction of both background and foreground datasets
features, labels = fe.extract_features()
print("Extracted features shape:", features.shape)
print("Extracted labels shape:", labels.shape)

# Loop through each sample and print its features and label
for i in range(len(features)):
    print(f"Sample {i + 1}:")
    print("Features:", features[i])
    print("Label:", labels[i])
    print()
# Train the classifiers
# mt.train_classifiers()
