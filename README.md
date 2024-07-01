# Project Setup Instructions

## Prerequisites

Ensure you have Anaconda or Miniconda installed on your system. You can download it from [here](https://www.anaconda.com/products/distribution).

## Setup Instructions

1. Clone the repository and navigate to the project directory:

   ```sh
   git clone https://github.com/thkox/speech-and-audio-processing
   cd https://github.com/thkox/speech-and-audio-processing

2. Download the dataset from [here](https://www.anaconda.com/products/distribution). **Hint**: You don't need to have an AWS account to download the dataset. You can download it directly from the link provided using the aws cli.
3. Extract the dataset to the `auxiliary2024/input/` directory. The dataset should be in the following format:


    auxiliary2024/input
    ├── VOiCES_devkit
    │   ├── distant-16k
    │   ├── references
    │   ├── source-16k


4. Run the setup script to create the conda environment:

    ```sh
    python setup.py
    ```
## Activating the Conda Environment

After the setup script completes, you can activate the conda environment using the following command:

### On Windows , MacOS, and Linux

    conda activate speech-and-audio-processing

The environment.yml file specifies the required dependencies:
- Python 3.11
- TensorFlow 2.16.1
- NumPy 1.25.*
- librosa 0.10.*
- pandas
- inquirer
- speechrecognition
- pydub

### Running the program

After activating the conda environment, you can run the program using either `main.py` or `main_replacement.py`.  
- `main.py` is the original script that runs the entire process from loading the dataset, extracting features, training the models, and making predictions.  
- `main_replacement.py` is an interactive script that allows you to choose which part of the process you want to run. It provides options to load the dataset, extract features, train the models, and transcribe an audio file.