"""
import os
import random
import shutil

def create_output_dirs(base_dir: str, dir_names: list) -> None:
    for dir_name in dir_names:
        full_path = os.path.join(base_dir, dir_name)
        os.makedirs(full_path, exist_ok=True)
        print(f"Directory created: {full_path}")

def get_all_npy_files(root_dir: str):
    npy_files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith('.npy'):
                npy_files.append(os.path.join(dirpath, filename))
    return npy_files

def split_dataset(full_image_dir, mask_image_dir, train_full_dir, train_mask_dir, val_full_dir, val_mask_dir, train_ratio=0.8):
    full_images = sorted(get_all_npy_files(full_image_dir))
    mask_images = sorted(get_all_npy_files(mask_image_dir))

    if len(full_images) != len(mask_images):
        print(f"Mismatch in number of files: {len(full_images)} FullImages vs {len(mask_images)} MaskedImages.")
        return

    paired_images = list(zip(full_images, mask_images))
    random.shuffle(paired_images)

    train_size = int(len(paired_images) * train_ratio)
    train_images = paired_images[:train_size]
    val_images = paired_images[train_size:]

    for full_image, mask_image in train_images:
        shutil.copy(full_image, os.path.join(train_full_dir, os.path.basename(full_image)))
        shutil.copy(mask_image, os.path.join(train_mask_dir, os.path.basename(mask_image)))

    for full_image, mask_image in val_images:
        shutil.copy(full_image, os.path.join(val_full_dir, os.path.basename(full_image)))
        shutil.copy(mask_image, os.path.join(val_mask_dir, os.path.basename(mask_image)))

    print(f"Training set size: {len(train_images)}")
    print(f"Validation set size: {len(val_images)}")

if __name__ == '__main__':
    project17_dir = os.path.dirname(os.path.dirname(os.getcwd()))

    full_image_dir = project17_dir + "/FullImages"
    mask_image_dir = project17_dir + "/MaskedImages"

    train_data_dir = os.path.join(project17_dir, 'TrainingData')
    val_data_dir = os.path.join(project17_dir, 'ValidatingData')

    create_output_dirs(train_data_dir, ['Full', 'Mask'])
    create_output_dirs(val_data_dir, ['Full', 'Mask'])

    train_full_dir = os.path.join(train_data_dir, 'Full')
    train_mask_dir = os.path.join(train_data_dir, 'Mask')
    val_full_dir = os.path.join(val_data_dir, 'Full')
    val_mask_dir = os.path.join(val_data_dir, 'Mask')

    # To improve the model's training, I randomly selected 80% of the patients for training and 20% for validation.
    split_dataset(full_image_dir, mask_image_dir, train_full_dir, train_mask_dir, val_full_dir, val_mask_dir, train_ratio=0.8)
"""