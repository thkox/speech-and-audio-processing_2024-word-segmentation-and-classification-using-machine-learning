from sklearn.model_selection import train_test_split
from source2024 import feature_extraction as fe
import joblib
import os
import numpy as np
import tensorflow as tf


OUTPUT_DIR = 'auxiliary2024/output/classifiers'


def preprocess_data(x, y=None):
    x = tf.constant(x, dtype=tf.float32)
    if x.shape.rank == 1:
        x = tf.expand_dims(x, axis=-1)  # Make it a column vector
    if y is not None:
        y = tf.constant(y, dtype=tf.float32)
        if y.shape.rank == 1:
            y = tf.expand_dims(y, axis=-1)  # Make it a column vector
        return x, y
    else:
        return x


def train(output_dir=OUTPUT_DIR):
    """
    Train Least Squares classifier on extracted features and save the model.
    """

    if fe.load_features() is None:
        return
    else:
        _, features, labels = fe.load_features()

    if features is None or labels is None:
        print("Error loading features. Aborting training.")
        return

    # Convert labels to -1 and 1
    labels = np.where(labels == 0, -1, 1)

    print("=====================================")
    print("Training Least Squares classifier")

    # Split data into training and validation sets
    x_train, x_val, y_train, y_val = train_test_split(features, labels, test_size=0.1, random_state=0)

    # Ensure features and labels are 2D tensors
    x_train, y_train = preprocess_data(x_train, y_train)

    # Add bias term to features
    x_train = tf.concat([tf.ones((x_train.shape[0], 1), dtype=tf.float32), x_train], axis=1)

    # Use tf.linalg.lstsq to solve for weights
    weights = tf.linalg.lstsq(x_train, y_train, fast=False)

    # Validate the model
    x_val, y_val = preprocess_data(x_val, y_val)

    # Add bias term to validation features
    x_val = tf.concat([tf.ones((x_val.shape[0], 1), dtype=tf.float32), x_val], axis=1)

    # Compute validation predictions
    val_predictions = tf.sign(tf.matmul(x_val, weights))

    # Convert predictions to 0 and 1
    val_binary_predictions = np.where(val_predictions.numpy() == -1, 0, 1)

    # Calculate validation accuracy
    val_accuracy = np.mean(val_binary_predictions == y_val.numpy().ravel())
    print(f"Validation Accuracy: {val_accuracy}")

    # Save the trained weights
    model_filename = os.path.join(output_dir, 'ls_model.pkl')
    os.makedirs(os.path.dirname(model_filename), exist_ok=True)
    joblib.dump(weights.numpy(), model_filename)

    print("Least Squares training completed")
    print("=====================================")

    return weights.numpy()


def load_model(output_dir=OUTPUT_DIR):
    """
    Load the trained Least Squares model from the given path.

    Returns:
        np.ndarray: Loaded Least Squares weights.
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
