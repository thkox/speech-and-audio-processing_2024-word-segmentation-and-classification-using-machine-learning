import os
import joblib
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from source2024 import feature_extraction as fe

OUTPUT_DIR = 'auxiliary2024/output/classifiers'


def preprocess_data(x, y=None):
    """
    Preprocess the input data and add bias term.

    Args:
        x (np.ndarray): Input features.
        y (np.ndarray): Labels.

    Returns:
        tf.Tensor, tf.Tensor: Preprocessed features and labels.
    """
    x = tf.convert_to_tensor(x, dtype=tf.float32)
    if x.shape.rank == 1:
        x = tf.expand_dims(x, axis=-1)
    if y is not None:
        y = tf.convert_to_tensor(y, dtype=tf.float32)
        if y.shape.rank == 1:
            y = tf.expand_dims(y, axis=-1)
        return x, y
    return x


def add_bias_term(features):
    """
    Add a bias term to the features.

    Args:
        features (tf.Tensor): Input features.

    Returns:
        tf.Tensor: Features with bias term added.
    """
    return tf.concat([tf.ones((features.shape[0], 1), dtype=tf.float32), features], axis=1)


def train(output_dir=OUTPUT_DIR):
    """
    Train Least Squares classifier on extracted features and save the model.
    """

    features_labels = fe.load_features()
    if features_labels is None:
        print("No features found. Aborting training.")
        return

    _, features, labels = features_labels

    if features is None or labels is None:
        print("Error loading features. Aborting training.")
        return

    # Convert labels to -1 and 1
    labels = np.where(labels == 0, -1, 1)

    print("=====================================")
    print("Training Least Squares classifier")

    # Split data into training and validation sets
    x_train, x_val, y_train, y_val = train_test_split(features, labels, test_size=0.1, random_state=0)

    # Preprocess data and add bias term
    x_train, y_train = preprocess_data(x_train, y_train)
    x_train = add_bias_term(x_train)

    # Use tf.linalg.lstsq to solve for weights
    weights = tf.linalg.lstsq(x_train, y_train, fast=False)

    # Validate the model
    x_val, y_val = preprocess_data(x_val, y_val)
    x_val = add_bias_term(x_val)
    val_predictions = tf.sign(tf.matmul(x_val, weights))

    # Calculate accuracy
    accuracy = accuracy_score(y_val.numpy().flatten(), val_predictions.numpy().flatten())
    print("Least Squares Accuracy: {:.2f}%".format(accuracy * 100))

    # Save the trained weights
    os.makedirs(output_dir, exist_ok=True)
    model_filename = os.path.join(output_dir, 'ls_model.pkl')
    joblib.dump(weights.numpy(), model_filename)

    print("Least Squares training completed")
    print("=====================================")

    return weights.numpy()


def load_model(output_dir=OUTPUT_DIR):
    """
    Load the trained Least Squares model from the given path.

    Returns:
        tf.Tensor: Loaded Least Squares weights.
    """
    model_path = os.path.join(output_dir, 'ls_model.pkl')
    if not os.path.isfile(model_path):
        print("The Least Squares model does not exist. Please train the model first.")
        return
    print("Loading Least Squares model from", model_path)
    weights = joblib.load(model_path)
    print("Least Squares model loaded successfully")
    return weights


def predict(features):
    """
    Predict the labels of given features using the trained Least Squares model.

    Args:
        features (np.ndarray): Features to predict the labels.

    Returns:
        np.ndarray: Predicted labels.
    """

    # Add bias term
    features = np.column_stack([np.ones(len(features)), features])

    # Compute predictions
    predictions = np.sign(features @ load_model())  # Matrix multiplication and sign function

    # Convert predictions to 0 and 1
    binary_predictions = np.where(predictions == -1, 0, 1)

    # Reshape the predictions to (number of labels,)
    binary_predictions = binary_predictions.ravel()

    return binary_predictions
