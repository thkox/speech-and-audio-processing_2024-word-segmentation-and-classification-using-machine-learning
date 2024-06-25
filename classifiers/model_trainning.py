import classifiers.svm as svm


def train_classifiers(features, labels, output_dir):
    # Train SVM classifier
    svm_classifier = svm.train_svm(features, labels, output_dir)

    return svm_classifier
