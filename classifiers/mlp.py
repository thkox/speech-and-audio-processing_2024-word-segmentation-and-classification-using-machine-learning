from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

OUTPUT_DIR = 'files/output'


def train_mlp(features, labels, output_dir=OUTPUT_DIR):
    """
    Train MLP classifier on extracted features and save the model.

    Args:
        features (np.ndarray): Extracted features (MFCCs and mel spectrogram).
        labels (np.ndarray): Corresponding labels.
        output_dir (str): Directory to save the trained model.

    Returns:
        MLPClassifier: Trained MLP classifier.
    """
    print("=====================================")
    print("Training MLP classifier")

    # Split data into training and test sets
    x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=0)

    # Initialize MLP classifier with three layers
    mlp_clf = MLPClassifier(hidden_layer_sizes=(100, 50, 25), max_iter=300, random_state=0, early_stopping=True) # TODO Change the values

    # Train MLP classifier
    mlp_clf.fit(x_train, y_train)

    # Predict on test set
    y_pred = mlp_clf.predict(x_test)

    # Evaluate performance
    accuracy = accuracy_score(y_test, y_pred)
    print(f"MLP Accuracy: {accuracy}")

    # Save the trained model
    model_filename = os.path.join(output_dir, 'mlp_model.pkl')

    # Check if the directory exists, if not, create it
    os.makedirs(os.path.dirname(model_filename), exist_ok=True)

    joblib.dump(mlp_clf, model_filename)

    # Print classification report
    print("MLP Classification Report:")
    print(classification_report(y_test, y_pred))

    print("MLP training completed")
    print("=====================================")
    return mlp_clf


def load_mlp_model(output_dir=OUTPUT_DIR):
    """
    Load the trained MLP model from the given path.

    Returns:
        MLPClassifier: Loaded MLP classifier.
    """
    print("Loading MLP model from", os.path.join(output_dir, 'mlp_model.pkl'))
    mlp_clf = joblib.load(os.path.join(output_dir, 'mlp_model.pkl'))
    print("MLP model loaded successfully")
    return mlp_clf
