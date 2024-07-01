import inquirer
import os
import sys
import numpy as np
from scipy.ndimage import median_filter
from source2024 import load_database as ldb
from source2024.classifiers import rnn, svm, mlp, least_squares as ls
from source2024.feature_extraction import extract_features, load_and_extract_features, HOP_LENGTH, detect_voice_intervals, plot_audio_with_intervals
from source2024.speech_to_text import transcribe_audio, show_predictions

# Constants
BACKGROUND_SOUND_DIR = "auxiliary2024/datasets/background_sound"
FOREGROUND_SOUND_DIR = "auxiliary2024/datasets/foreground_sound"
TEST_SOUND_DIR = "auxiliary2024/datasets/test"


# Helper Functions
def check_dataset_exists():
    return (
            os.path.exists(os.path.join(os.getcwd(), BACKGROUND_SOUND_DIR))
            and os.path.exists(os.path.join(os.getcwd(), FOREGROUND_SOUND_DIR))
            and os.path.exists(os.path.join(os.getcwd(), TEST_SOUND_DIR))
    )


def check_features_exist():
    return os.path.exists(os.path.join(os.getcwd(), "auxiliary2024/output/features.npz"))


def check_model_exists(model_name):
    return os.path.exists(os.path.join(os.getcwd(), f"auxiliary2024/output/{model_name}_model.pkl"))


def get_audio_files():
    return os.listdir(os.path.join(os.getcwd(), TEST_SOUND_DIR))


def is_interactive():
    return sys.stdin.isatty()


def load_dataset():
    print("Loading the dataset...")
    ldb.create_datasets()


def extract_features_wrapper():
    print("Extracting features from the datasets...")
    extract_features(shuffle_data=False, show_plots=False)


def train_models():
    models = ["SVM", "MLP Three Layers", "RNN", "Least Squares"]

    for model in models:
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
                extract_features_wrapper()
            else:
                print("Features already exist. Skipping feature extraction.\n")
        elif answers["option"] == options[2]:  # Train models
            train_models()
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
            if answers["transcribe_option"] == "From the test in the database?":
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
                _, features, sample_rate, audio = load_and_extract_features(test_file, show_plots=False)
                hop_length = HOP_LENGTH
                frame_rate = sample_rate / hop_length

                # get the intervals of the voice
                intervals_original, texts = transcribe_audio(test_file)

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
                show_predictions(audio, sample_rate, intervals_original, svm_predictions_median, frame_rate, "SVM", audio_file)
                show_predictions(audio, sample_rate, intervals_original, mlp_predictions_median, frame_rate, "MLP", audio_file)
                show_predictions(audio, sample_rate, intervals_original, rnn_predictions_median, frame_rate, "RNN", audio_file)
                show_predictions(audio, sample_rate, intervals_original, ls_predictions_median, frame_rate,"Least_Squares", audio_file)

            elif answers["transcribe_option"] == "Back":
                continue

        elif answers["option"] == options[4]:  # Quit
            print("Quitting the program.")
            break


if __name__ == "__main__":
    main()
