from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from classifiers import feature_extraction as fe
import joblib
import os

OUTPUT_DIR = 'files/output/classifiers'


def train(output_dir=OUTPUT_DIR):
    """
    Train SVM classifier on extracted features and save the model.

    Args:
        output_dir (str): Directory to save the trained model.

    Returns:
        SVC: Trained SVM classifier.
    """

    if fe.load_features() is None:
        return
    else:
        _, features, labels = fe.load_features()

    print("=====================================")
    print("Training SVM classifier")

    # Split data into training and test sets
    x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=0)

    # Initialize SVM classifier
    # svm_clf = SVC(kernel='linear', random_state=0) -> old code
    svm_clf = LinearSVC(random_state=0)

    # Train SVM classifier
    svm_clf.fit(x_train, y_train)

    # Predict on test set
    y_pred = svm_clf.predict(x_test)

    # Evaluate performance
    accuracy = accuracy_score(y_test, y_pred)
    print(f"SVM Accuracy: {accuracy}")

    # Save the trained model
    model_filename = os.path.join(output_dir, 'svm_model.pkl')

    # Check if the directory exists, if not, create it
    os.makedirs(os.path.dirname(model_filename), exist_ok=True)

    joblib.dump(svm_clf, model_filename)

    # Print classification report
    print("SVM Classification Report:")
    print(classification_report(y_test, y_pred))

    print("SVM training completed")
    print("=====================================")
    return svm_clf


def load_model(output_dir=OUTPUT_DIR):
    """
    Load the trained SVM model from the given path.

    Returns:
        SVC: Loaded SVM classifier.
    """
    model_path = os.path.join(output_dir, 'svm_model.pkl')
    if not os.path.isfile(model_path):
        print("The SVM model does not exist. Please train the model first.")
        return
    print("Loading SVM model from", model_path)
    svm_clf = joblib.load(model_path)
    print("SVM model loaded successfully")
    return svm_clf


def predict(features):
    """
    Predict the labels of given features using the trained SVM model.
    """
    return load_model().predict(features)
