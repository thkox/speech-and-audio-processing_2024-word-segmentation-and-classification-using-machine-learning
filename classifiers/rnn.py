import numpy as np
import tensorflow as tf
from classifiers import feature_extraction as fe
import os

OUTPUT_DIR = 'files/output/classifiers'


def preprocess_features(features, labels, n_of_files=548):
    """
    Preprocess features and labels to match the required input shape for the RNN model.

    Args:
        features (np.ndarray): The extracted features.
        labels (np.ndarray): The labels corresponding to the features.
        n_of_files (int): The number of files to split the features into.

    Returns:
        np.ndarray, np.ndarray: Preprocessed features and labels.
    """
    timesteps, n_mels = features.shape
    print("n_mels:", n_mels)
    print("timesteps:", timesteps)
    assert timesteps % n_of_files == 0, "Total timesteps must be divisible by n_of_files"

    # Reshape features and labels to fit the model input shape
    # First reshape to split into files
    features = features.reshape((n_of_files, timesteps // n_of_files, n_mels))

    # Adjust labels to match the number of files
    labels = labels.reshape((n_of_files, timesteps // n_of_files))  # Ensuring labels match feature shapes

    return features, labels



def train(output_dir=OUTPUT_DIR):
    """
    Train RNN classifier on extracted features and save the model.

    Args:
        output_dir (str): Directory to save the trained model.

    Returns:
        Sequential: Trained RNN classifier.
    """

    if fe.load_features() is None:
        return
    else:
        _, features, labels = fe.load_features()

    print("=====================================")
    print("Training RNN classifier")

    # Preprocess features and labels
    features, labels = preprocess_features(features, labels)

    # Initialize RNN model
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Input(shape=(features.shape[1], features.shape[2])))  # Input layer based on features shape
    model.add(tf.keras.layers.SimpleRNN(32, activation='sigmoid', return_sequences=True))  # Change here
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(features, labels, epochs=10, batch_size=32, validation_split=0.2)

    # Save the trained model
    model_filename = os.path.join(output_dir, 'rnn_model.keras')
    os.makedirs(os.path.dirname(model_filename), exist_ok=True)
    model.save(model_filename)

    print("RNN training completed")
    print("=====================================")
    return model


def load_model(output_dir=OUTPUT_DIR):
    """
    Load the trained RNN model from the given path.

    Returns:
        Sequential: Loaded RNN classifier.
    """
    model_path = os.path.join(output_dir, 'rnn_model.keras')
    if not os.path.isfile(model_path):
        print("The RNN model does not exist. Please train the model first.")
        return
    print("Loading RNN model from", model_path)
    rnn_clf = tf.keras.models.load_model(model_path)
    print("RNN model loaded successfully")
    return rnn_clf


def predict(features):
    """
    Predict labels for new data using the trained RNN model.

    Args:
        features (np.ndarray): New data for prediction.

    Returns:
        np.ndarray: Predicted labels.
    """
    features, _ = preprocess_features(features, np.zeros((features.shape[0],)), n_of_files=1)
    rnn_model = load_model()
    predictions = rnn_model.predict(features)
    return (predictions > 0.5).astype(int)  # Assuming binary classification

