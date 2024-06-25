import classifiers.svm as svm


def train_classifiers(features, labels):
    # Train SVM classifier
    svm_classifier = svm.train_svm(features, labels)

    return svm_classifier
