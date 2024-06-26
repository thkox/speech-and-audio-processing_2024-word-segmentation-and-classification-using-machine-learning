import inquirer
import os
import load_database as ldb
from classifiers import svm, mlp, rnn
import sys

# Constants
BACKGROUND_SOUND_DIR = "files/datasets/background_sound"
FOREGROUND_SOUND_DIR = "files/datasets/foreground_sound"
TEST_SOUND_DIR = "files/datasets/test"


# Helper Functions
def check_dataset_exists():
    return (
            os.path.exists(BACKGROUND_SOUND_DIR)
            and os.path.exists(FOREGROUND_SOUND_DIR)
            and os.path.exists(TEST_SOUND_DIR)
    )


def check_model_exists(model_name):
    return os.path.exists(f"files/output/{model_name}_model.pkl")


def get_audio_files():
    return os.listdir(TEST_SOUND_DIR)


def is_interactive():
    return sys.stdin.isatty()


def load_dataset():
    print("Loading the dataset...")
    ldb.create_datasets()


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
            print("Training Least Squares...")  # Replace with actual training
        elif model == "All 4 Classifiers":
            svm.train()
            mlp.train()
            rnn.train()
            print("Training Least Squares...")  # Replace with actual training


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
        # Add transcription function call here
    elif transcribe_option == "From a file of your choice?":
        print("Please provide the path to the audio file to transcribe.")
        # Add logic to handle file input and transcription here


# Main Function
def main():
    if not is_interactive():
        print("This script needs to be run in an interactive terminal.")
        return

    while True:
        options = [
            "Load the necessary dataset",
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
        elif answers["option"] == options[1]:  # Train models
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
        elif answers["option"] == options[2]:  # Transcribe audio
            transcribe_options = [
                "From the test in the database?",
                "From a file of your choice?",
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
        elif answers["option"] == options[3]:  # Quit
            print("Quitting the program.")
            break


if __name__ == "__main__":
    main()
