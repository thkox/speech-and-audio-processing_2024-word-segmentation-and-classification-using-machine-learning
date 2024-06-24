# Project Setup Instructions

## Prerequisites

Ensure you have Anaconda or Miniconda installed on your system. You can download it from [here](https://www.anaconda.com/products/distribution).

## Setup Instructions

1. Clone the repository and navigate to the project directory:

   ```sh
   git clone https://github.com/thkox/speech-and-audio-processing
   cd https://github.com/thkox/speech-and-audio-processing

2. Run the setup script to create the conda environment:

    ```sh
    python setup.py
    ```
## Activating the Conda Environment

After the setup script completes, you can activate the conda environment using the following command:

### On Windows

    python setup.py

### On macOS and Linux

    source activate speech-and-audio-processing

## Additional Information

The environment.yml file specifies the required dependencies:
- Python 3.11
- NumPy 1.25.x
- librosa 0.10.0
- TensorFlow 2.13.0


### Summary

- The `environment.yml` file contains the environment configuration.
- The `setup.py` script automates the creation of the conda environment and provides instructions for activating it.
- The `README.md` file contains detailed instructions for users to set up and activate the environment.

Place these files in the root directory of your project repository. Users can then follow the instructions in the `README.md` to set up their environment.
