import os
import platform
import subprocess


def create_conda_env():
    # Define the command to create the conda environment
    command = "conda env create -f environment.yml"

    try:
        # Run the command
        subprocess.check_call(command, shell=True)
        print("Conda environment created successfully.")
    except subprocess.CalledProcessError as e:
        print("Failed to create conda environment.")
        print(e)


def activate_conda_env():
    env_name = "speech-and-audio-processing"
    activation_command = ""

    if platform.system() == "Windows":
        activation_command = f"activate {env_name}"
    else:
        activation_command = f"source activate {env_name}"

    print(f"To activate the conda environment, run: {activation_command}")


if __name__ == "__main__":
    create_conda_env()
    activate_conda_env()
