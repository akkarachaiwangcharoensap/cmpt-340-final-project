"""
import os
import numpy as np


def get_npy_file_shape(directory):
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.npy'):
                file_path = os.path.join(root, file)
                npy_array = np.load(file_path)
                return file_path, npy_array.shape


project17_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd())))
train_image_dir = os.path.join(project17_dir, 'TrainingData', 'Full')

file_path, file_shape = get_npy_file_shape(train_image_dir)

print(f"File Path: {file_path}")
print(f"File Shape: {file_shape}")
"""

