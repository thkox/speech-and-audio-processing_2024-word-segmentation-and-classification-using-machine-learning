from sklearn.neural_network import MLPClassifier
from source2024 import feature_extraction as fe
import joblib
import os

OUTPUT_DIR = 'auxiliary2024/output/classifiers'


def train(output_dir=OUTPUT_DIR):
    """
    Train MLP classifier on extracted features and save the model.

    Args:
        output_dir (str): Directory to save the trained model.

    Returns:
        MLPClassifier: Trained MLP classifier.
    """

    if fe.load_features() is None:
        return
    else:
        _, features, labels = fe.load_features()

    print("=====================================")
    print("Training MLP classifier")

    # Initialize MLP classifier with three layers
    mlp_clf = MLPClassifier(hidden_layer_sizes=(512, 256, 128), max_iter=300, random_state=0,
                            early_stopping=True)  # TODO Change the values

    # Train MLP classifier
    mlp_clf.fit(features, labels)

    # Save the trained model
    model_filename = os.path.join(output_dir, 'mlp_model.pkl')

    # Check if the directory exists, if not, create it
    os.makedirs(os.path.dirname(model_filename), exist_ok=True)

    joblib.dump(mlp_clf, model_filename)

    print("MLP training completed")
    print("=====================================")
    return mlp_clf


def load_model(output_dir=OUTPUT_DIR):
    """
    Load the trained MLP model from the given path.

    Returns:
        MLPClassifier: Loaded MLP classifier.
    """
    model_path = os.path.join(output_dir, 'mlp_model.pkl')
    if not os.path.isfile(model_path):
        print("The MLP model does not exist. Please train the model first.")
        return
    print("Loading MLP model from", model_path)
    mlp_clf = joblib.load(model_path)
    print("MLP model loaded successfully")
    return mlp_clf


def predict(features):
    """
    Predict the labels of given features using the trained MLP model.

    Args:
        model (MLPClassifier): Trained MLP classifier.
        features (np.ndarray): Features to predict the labels.

    Returns:
        np.ndarray: Predicted labels.
    """
    return load_model().predict(features)
