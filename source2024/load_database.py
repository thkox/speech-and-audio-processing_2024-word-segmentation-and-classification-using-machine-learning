import os
import shutil
import wave

# Define the root directory of the VOiCES dataset
VOiCES_SOUNDS_DIR = 'auxiliary2024/input/VOiCES_devkit'

# Define the directories for the new datasets
BACKGROUND_SOUND_DIR = 'auxiliary2024/datasets/background_sound'
FOREGROUND_SOUND_DIR = 'auxiliary2024/datasets/foreground_sound'
TEST_SOUND_DIR = 'auxiliary2024/datasets/test'


# Function to copy auxiliary2024 to the new directory
def copy_files(src_dir, dest_dir, condition_func=None, max_duration=None):
    total_duration = 0
    files_duration = []  # List to hold tuples of (file_path, duration)

    for root, dirs, files in os.walk(src_dir):
        for file in files:
            if file.endswith('.wav') and (condition_func is None or condition_func(file)):
                src_file_path = os.path.join(root, file)
                dest_file_path = os.path.join(dest_dir, file)

                # Calculate file duration
                with wave.open(src_file_path, 'r') as wav_file:
                    duration = wav_file.getnframes() / wav_file.getframerate()

                files_duration.append((src_file_path, duration))

    # Calculate total duration of all auxiliary2024
    total_duration = sum(duration for _, duration in files_duration)

    # Check if total duration exceeds max_duration
    if max_duration is not None and total_duration > max_duration:
        # Calculate scaling factor
        scaling_factor = max_duration / total_duration

        # Copy and scale durations
        for src_file_path, duration in files_duration:
            scaled_duration = duration * scaling_factor
            dest_file_path = os.path.join(dest_dir, os.path.basename(src_file_path))

            # Use ffmpeg or similar to trim/cut the audio file to the scaled duration
            # Here we assume you have ffmpeg installed and available in PATH
            os.system(f'ffmpeg -i {src_file_path} -ss 0 -t {scaled_duration} -y {dest_file_path}')

            print(f'Copied and scaled: {src_file_path} to {dest_file_path} ({scaled_duration:.2f} sec)')

        return max_duration  # Return max_duration as total duration

    else:
        # Copy auxiliary2024 without scaling
        for src_file_path, duration in files_duration:
            dest_file_path = os.path.join(dest_dir, os.path.basename(src_file_path))
            shutil.copy2(src_file_path, dest_file_path)
            print(f'Copied: {src_file_path} to {dest_file_path} ({duration:.2f} sec)')

        return total_duration


# Condition function for background sounds (rm1 and mc02)
def background_condition(file):
    return 'rm1' in file and 'mc02' in file


# Condition function for foreground sounds (rm1, mc01, none, clo)
def foreground_condition(file):
    return 'rm1' in file and 'mc01' in file and 'none' in file and 'clo' in file


def test_condition(file):
    return 'rm1' in file and 'mc01' in file and 'clo' in file


def create_datasets():
    # Create the new directories if they don't exist
    os.makedirs(BACKGROUND_SOUND_DIR, exist_ok=True)
    os.makedirs(FOREGROUND_SOUND_DIR, exist_ok=True)
    os.makedirs(TEST_SOUND_DIR, exist_ok=True)

    # Copy foreground sound auxiliary2024
    foreground_sound_src_dir = os.path.join(VOiCES_SOUNDS_DIR, 'distant-16k', 'speech', 'train', 'rm1', 'none')
    foreground_duration = copy_files(foreground_sound_src_dir, FOREGROUND_SOUND_DIR, foreground_condition)

    # Copy background sound auxiliary2024
    background_sound_src_dir = os.path.join(VOiCES_SOUNDS_DIR, 'distant-16k', 'distractors', 'rm1')
    copy_files(background_sound_src_dir, BACKGROUND_SOUND_DIR, background_condition, max_duration=foreground_duration)

    # Copy test dataset auxiliary2024 (both background and foreground)
    test_dataset_src_dir = os.path.join(VOiCES_SOUNDS_DIR, 'distant-16k', 'speech', 'test', 'rm1')
    copy_files(test_dataset_src_dir, TEST_SOUND_DIR, test_condition)

    print('File copying completed.')
