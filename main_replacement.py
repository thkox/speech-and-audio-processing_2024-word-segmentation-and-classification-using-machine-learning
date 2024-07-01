import inquirer
import os
import source2024.load_database as ldb
from source2024.classifiers import rnn, svm, mlp, least_squares as ls
import sys
from source2024 import feature_extraction as fe
from scipy.ndimage import median_filter
import numpy as np

# Constants
BACKGROUND_SOUND_DIR = "auxiliary2024/datasets/background_sound"
FOREGROUND_SOUND_DIR = "auxiliary2024/datasets/foreground_sound"
TEST_SOUND_DIR = "auxiliary2024/datasets/test"


# Helper Functions
def check_dataset_exists():
    return (
            os.path.exists(BACKGROUND_SOUND_DIR)
            and os.path.exists(FOREGROUND_SOUND_DIR)
            and os.path.exists(TEST_SOUND_DIR)
    )


def check_features_exist():
    return os.path.exists("auxiliary2024/output/features.npz")


def check_model_exists(model_name):
    return os.path.exists(f"auxiliary2024/output/{model_name}_model.pkl")


def get_audio_files():
    return os.listdir(TEST_SOUND_DIR)


def is_interactive():
    return sys.stdin.isatty()


def load_dataset():
    print("Loading the dataset...")
    ldb.create_datasets()


def extract_features():
    print("Extracting features from the datasets...")
    fe.extract_features(shuffle_data=True, show_plots=False)


def train_model(model):
    if not check_model_exists(model):
        print(f"Training the {model} model...")
        if model == "SVM":
            svm.train()
        elif model == "MLP Three Layers":
            mlp.train()
        elif model == "RNN":
            rnn.train()
        elif model == "Least Squares":
            ls.train()
        elif model == "All 4 Classifiers":
            svm.train()
            mlp.train()
            rnn.train()
            ls.train()


def transcribe_audio(transcribe_option):
    if transcribe_option == "From the test in the database?":
        audio_files = get_audio_files()
        questions = [
            inquirer.List(
                "audio_file",
                message="Please select the audio file to transcribe:",
                choices=audio_files,
            )
        ]
        answers = inquirer.prompt(questions)
        audio_file = answers["audio_file"]
        print(f"Transcribing audio file: {audio_file}")

        # Load the trained models
        svm.load_model()
        mlp.load_model()
        rnn.load_model()
        ls.load_model()

        # Extract features from the selected audio file
        test_file = os.path.join(TEST_SOUND_DIR, audio_file)
        _, features, sample_rate, audio = fe.load_and_extract_features(test_file, show_plots=False)
        hop_length = fe.HOP_LENGTH
        frame_rate = sample_rate / hop_length

        L = 5  # Window size for median filter

        # Predict using the trained models
        svm_predictions = svm.predict(features)
        svm_predictions_median = median_filter(svm_predictions, size=L)
        print("SVM predictions after median filter:", svm_predictions_median)

        mlp_predictions = mlp.predict(features)
        mlp_predictions_median = median_filter(mlp_predictions, size=L)
        print("MLP predictions after median filter:", mlp_predictions_median)

        rnn_predictions = rnn.predict(features, n_of_files=1)
        rnn_predictions_median = np.squeeze(median_filter(rnn_predictions, size=L))
        print("RNN predictions after median filter:", rnn_predictions_median)

        ls_predictions = ls.predict(features)
        ls_predictions_median = median_filter(ls_predictions, size=L)
        print("Least Squares predictions after median filter:", ls_predictions_median)

        # Show the predictions
        show_predictions(audio, sample_rate, svm_predictions_median, frame_rate, "SVM predictions")
        show_predictions(audio, sample_rate, mlp_predictions_median, frame_rate, "MLP predictions")
        show_predictions(audio, sample_rate, rnn_predictions_median, frame_rate, "RNN predictions")
        show_predictions(audio, sample_rate, ls_predictions_median, frame_rate, "Least Squares predictions")

    elif transcribe_option == "From a file of your choice?":
        print("Please provide the path to the audio file to transcribe.")
        # Add logic to handle file input and transcription here


def show_predictions(audio, sample_rate, predictions, frame_rate, title):
    intervals = fe.detect_voice_intervals(predictions, frame_rate)
    fe.plot_audio_with_intervals(audio, sample_rate, intervals, title)


# Main Function
def main():
    if not is_interactive():
        print("This script needs to be run in an interactive terminal.")
        return

    while True:
        options = [
            "Load the necessary dataset",
            "Extract features from the dataset",
            "Train the models",
            "Transcribe an audio file",
            "Quit",
        ]
        answers = inquirer.prompt(
            [
                inquirer.List(
                    "option", message="What do you want to do?", choices=options
                )
            ]
        )

        if answers["option"] == options[0]:  # Load dataset
            if not check_dataset_exists():
                load_dataset()
            else:
                print("Dataset already exists. Skipping dataset loading.\n")
        elif answers["option"] == options[1]:  # Extract features
            if not check_features_exist():
                extract_features()
            else:
                print("Features already exist. Skipping feature extraction.\n")
        elif answers["option"] == options[2]:  # Train models
            models = [
                "SVM",
                "MLP Three Layers",
                "RNN",
                "Least Squares",
                "All 4 Classifiers",
                "Back",
            ]
            answers = inquirer.prompt(
                [
                    inquirer.List(
                        "model",
                        message="Which model do you want to train?",
                        choices=models,
                    )
                ]
            )
            if answers["model"] != "Back":
                train_model(answers["model"])
        elif answers["option"] == options[3]:  # Transcribe audio
            transcribe_options = [
                "From the test in the database?",
                "Back",
            ]
            answers = inquirer.prompt(
                [
                    inquirer.List(
                        "transcribe_option",
                        message="Do you want to transcribe an audio file?",
                        choices=transcribe_options,
                    )
                ]
            )
            if answers["transcribe_option"] != "Back":
                transcribe_audio(answers["transcribe_option"])
        elif answers["option"] == options[4]:  # Quit
            print("Quitting the program.")
            break


if __name__ == "__main__":
    main()
