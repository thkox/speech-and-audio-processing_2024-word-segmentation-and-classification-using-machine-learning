from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

OUTPUT_DIR = 'files/output'


def train_svm(features, labels, output_dir = OUTPUT_DIR):
    """
    Train SVM classifier on extracted features and save the model.

    Args:
        features (np.ndarray): Extracted features (MFCCs and mel spectrogram).
        labels (np.ndarray): Corresponding labels.
        output_dir (str): Directory to save the trained model.

    Returns:
        SVC: Trained SVM classifier.
    """
    print("Training SVM classifier")

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=0)

    # Initialize SVM classifier
    # svm_clf = SVC(kernel='linear', random_state=0) -> old code
    svm_clf = LinearSVC(random_state=0)

    # Train SVM classifier
    svm_clf.fit(X_train, y_train)

    # Predict on test set
    y_pred = svm_clf.predict(X_test)

    # Evaluate performance
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")

    # Save the trained model
    model_filename = os.path.join(output_dir, 'svm_model.pkl')

    # Check if the directory exists, if not, create it
    os.makedirs(os.path.dirname(model_filename), exist_ok=True)

    joblib.dump(svm_clf, model_filename)

    # Print classification report
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    print("SVM training completed")
    return svm_clf
