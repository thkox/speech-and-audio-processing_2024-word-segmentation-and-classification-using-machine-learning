import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import os

OUTPUT_DIR = 'files/output'


# TODO: you can remove tensorflow and use keras only
def train_rnn(features, labels, output_dir=OUTPUT_DIR):
    """
    Train RNN classifier on extracted features and save the model.

    Args:
        features (np.ndarray): Extracted features (time series data).
        labels (np.ndarray): Corresponding labels.
        output_dir (str): Directory to save the trained model.

    Returns:
        Sequential: Trained RNN classifier.
    """
    print("=====================================")
    print("Training RNN classifier")

    # Ensure the labels are in categorical format
    labels = tf.keras.utils.to_categorical(labels)

    # Check if the features array is 2D (samples, features)
    if len(features.shape) == 2:
        # Reshape to (samples, timesteps, features)
        features = features.reshape((features.shape[0], features.shape[1], 1))
    elif len(features.shape) != 3:
        raise ValueError("Features array must be 2D or 3D. Current shape: {}".format(features.shape))

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=0)

    # Initialize RNN model with Input layer
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Input(shape=(X_train.shape[1], X_train.shape[2])))
    model.add(tf.keras.layers.SimpleRNN(50, activation='relu'))
    model.add(tf.keras.layers.Dense(labels.shape[1], activation='softmax'))

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(X_train, y_train, epochs=3, batch_size=32, validation_split=0.2)

    # Evaluate the model
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"RNN Accuracy: {accuracy}")

    # Save the trained model
    model_filename = os.path.join(output_dir, 'rnn_model.keras')

    # Check if the directory exists, if not, create it
    os.makedirs(os.path.dirname(model_filename), exist_ok=True)

    model.save(model_filename)

    # Predict on test set for classification report
    y_pred = model.predict(X_test)
    y_pred_classes = y_pred.argmax(axis=-1)
    y_test_classes = y_test.argmax(axis=-1)

    # Print classification report
    print("RNN Classification Report:")
    print(classification_report(y_test_classes, y_pred_classes))

    print("RNN training completed")
    print("=====================================")
    return model


def load_rnn_model(output_dir=OUTPUT_DIR):
    """
    Load the trained RNN model from the given path.

    Returns:
        Sequential: Loaded RNN classifier.
    """
    model_filename = os.path.join(output_dir, 'rnn_model.h5')
    print("Loading RNN model from", model_filename)
    model = tf.keras.models.load_model(model_filename)
    print("RNN model loaded successfully")
    return model


def predict_with_rnn(model, features):
    """
    Predict labels for new data using the trained RNN model.

    Args:
        model (Sequential): Trained RNN classifier.
        features (np.ndarray): New data for prediction.

    Returns:
        np.ndarray: Predicted labels.
    """
    print("Making predictions with RNN model")
    if len(features.shape) == 2:
        features = features.reshape((features.shape[0], features.shape[1], 1))
    elif len(features.shape) != 3:
        raise ValueError("Features array must be 2D or 3D. Current shape: {}".format(features.shape))
    predictions = model.predict(features)
    return predictions.argmax(axis=-1)
