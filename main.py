import os
import shutil

# Define the root directory of the VOiCES dataset
voices_root_dir = 'files/input/'

# Define the directories for the new datasets
background_sound_dir = 'files/datasets/background_sound'
foreground_sound_dir = 'files/datasets/foreground_sound'
test_dataset_dir = 'files/datasets/test'

# Create the new directories if they don't exist
os.makedirs(background_sound_dir, exist_ok=True)
os.makedirs(foreground_sound_dir, exist_ok=True)
os.makedirs(test_dataset_dir, exist_ok=True)


# Function to copy files to the new directory
def copy_files(src_dir, dest_dir, condition_func=None):
    for root, dirs, files in os.walk(src_dir):
        for file in files:
            if file.endswith('.wav') and (condition_func is None or condition_func(file)):
                src_file_path = os.path.join(root, file)
                dest_file_path = os.path.join(dest_dir, file)
                shutil.copy2(src_file_path, dest_file_path)
                print(f'Copied: {src_file_path} to {dest_file_path}')


# Copy background sound files
background_sound_src_dir = os.path.join(voices_root_dir, 'distant-16k', 'distractors', 'rm1')
copy_files(background_sound_src_dir, background_sound_dir)

# Copy foreground sound files
foreground_sound_src_dir = os.path.join(voices_root_dir, 'distant-16k', 'speech', 'train', 'rm1', 'none')
copy_files(foreground_sound_src_dir, foreground_sound_dir)

# Copy test dataset files (both background and foreground)
test_dataset_src_dir = os.path.join(voices_root_dir, 'distant-16k', 'speech', 'test', 'rm1')
copy_files(test_dataset_src_dir, test_dataset_dir)

print('File copying completed.')


