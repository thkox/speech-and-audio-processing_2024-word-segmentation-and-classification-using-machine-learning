import classifiers.svm as svm
import classifiers.mlp as mlp


def train_classifiers(features, labels):
    # Train SVM classifier
    svm_classifier = svm.train_svm(features, labels)
    mlp_classifier = mlp.train_mlp(features, labels)

    return svm_classifier, mlp_classifier
