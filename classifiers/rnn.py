import numpy as np
import tensorflow as tf
from classifiers import feature_extraction as fe
import os

OUTPUT_DIR = 'files/output/classifiers'


def get_divisor(n, k):
    divisors = []

    # Find all divisors of n
    for i in range(1, int(n ** 0.5) + 1):
        if n % i == 0:
            divisors.append(i)
            if i != n // i:
                divisors.append(n // i)

    # Sort divisors in ascending order
    divisors.sort()

    # Check if there are at least 4 divisors
    if len(divisors) >= 5:
        return divisors[k]  # 4th divisor (index 3 in 0-based indexing)
    else:
        return None  # Not enough divisors (should handle this case as needed)


def preprocess_features(features, labels, n_of_files=1):
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
    # find the 4th divisor of the timesteps

    timesteps, n_mels = features.shape
    n_of_files = get_divisor(timesteps, 11)
    features, labels = preprocess_features(features, labels, n_of_files)

    # Initialize RNN model
    model = tf.keras.models.Sequential()
    model.add(
        tf.keras.layers.Input(shape=(features.shape[1], features.shape[2])))  # Input layer based on features shape
    model.add(tf.keras.layers.SimpleRNN(32, activation='sigmoid', return_sequences=True))  # Change here
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(features, labels, epochs=40, batch_size=32, validation_split=0.1)

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


def predict(features, n_of_files=1):
    """
    Predict labels for new data using the trained RNN model.

    Args:
        features (np.ndarray): New data for prediction.

    Returns:
        np.ndarray: Predicted labels.
    """
    features, _ = preprocess_features(features, np.zeros((features.shape[0],)), n_of_files)
    rnn_model = load_model()
    predictions = rnn_model.predict(features)
    return (predictions > 0.5).astype(int)  # Assuming binary classification
