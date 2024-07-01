import os
import shutil

# Define the root directory of the VOiCES dataset
VOiCES_SOUNDS_DIR = 'auxiliary2024/input/VOiCES_devkit'

# Define the directories for the new datasets
BACKGROUND_SOUND_DIR = 'auxiliary2024/datasets/background_sound'
FOREGROUND_SOUND_DIR = 'auxiliary2024/datasets/foreground_sound'
TEST_SOUND_DIR = 'auxiliary2024/datasets/test'


# Function to copy auxiliary2024 to the new directory
def copy_files(src_dir, dest_dir, condition_func=None):
    for root, dirs, files in os.walk(src_dir):
        for file in files:
            if file.endswith('.wav') and (condition_func is None or condition_func(file)):
                src_file_path = os.path.join(root, file)
                dest_file_path = os.path.join(dest_dir, file)

                # Skip if the file already exists in the destination directory
                if os.path.exists(dest_file_path):
                    print(f'Skipped: {src_file_path} already exists in {dest_file_path}')
                    continue

                # Copy files without checking duration
                shutil.copy2(src_file_path, dest_file_path)
                print(f'Copied: {src_file_path} to {dest_file_path}')


# Condition function for background sounds (rm1 and mc02)
def background_condition(file):
    return 'rm1' in file and 'mc02' in file


# Condition function for foreground sounds (rm1, mc01, none, clo)
def foreground_condition(file):
    return 'rm1' in file and 'mc01' in file and 'clo' in file


def test_condition(file):
    return 'rm1' in file and 'mc01' in file and 'clo' in file


def create_datasets():
    # Create the new directories if they don't exist
    os.makedirs(BACKGROUND_SOUND_DIR, exist_ok=True)
    os.makedirs(FOREGROUND_SOUND_DIR, exist_ok=True)
    os.makedirs(TEST_SOUND_DIR, exist_ok=True)

    # Copy foreground sound auxiliary2024
    foreground_sound_src_dir = os.path.join(VOiCES_SOUNDS_DIR, 'distant-16k', 'speech', 'train', 'rm1')
    copy_files(foreground_sound_src_dir, FOREGROUND_SOUND_DIR, foreground_condition)

    # Copy background sound auxiliary2024
    background_sound_src_dir = os.path.join(VOiCES_SOUNDS_DIR, 'distant-16k', 'distractors', 'rm1')
    copy_files(background_sound_src_dir, BACKGROUND_SOUND_DIR, background_condition)

    # Copy test dataset auxiliary2024 (both background and foreground)
    test_dataset_src_dir = os.path.join(VOiCES_SOUNDS_DIR, 'distant-16k', 'speech', 'test', 'rm1')
    copy_files(test_dataset_src_dir, TEST_SOUND_DIR, test_condition)

    print('File copying completed.')
