import os
import platform
import subprocess


def create_conda_env():
    env_name = "speech-and-audio-processing"

    # Define the command to remove the conda environment if it exists
    remove_command = f"conda env remove --name {env_name}"

    try:
        # Run the remove command
        subprocess.check_call(remove_command, shell=True)
        print("Existing conda environment removed successfully.")
    except subprocess.CalledProcessError:
        print("No existing conda environment to remove.")

    # Define the command to create the conda environment
    create_command = "conda env create -f environment.yml"

    try:
        # Run the create command
        subprocess.check_call(create_command, shell=True)
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
