import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
from segmentation_models import Xnet

def clip_hu_values(hu_values, min_value=-1000, max_value=2000):
    return np.clip(hu_values, min_value, max_value)

def normalize_hu_values(hu_values, min_value=-1000, max_value=2000):
    return (hu_values - min_value) / (max_value - min_value)

def process_hu_values(hu_values, min_value=-1000, max_value=2000):
    clipped_values = clip_hu_values(hu_values, min_value, max_value)
    normalized_values = normalize_hu_values(clipped_values, min_value, max_value)
    return normalized_values

class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.image_files = sorted(os.listdir(image_dir))
        self.mask_files = sorted(os.listdir(mask_dir))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_files[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_files[idx])

        image = np.load(image_path)
        mask = np.load(mask_path)

        image = process_hu_values(image)

        image = np.expand_dims(image, axis=0)
        mask = np.expand_dims(mask, axis=0)

        return image, mask

project17_dir = "/content/drive/MyDrive"
train_image_dir = project17_dir + '/TrainingData/Full'
train_mask_dir = project17_dir + '/TrainingData/Mask'
val_image_dir = project17_dir + '/ValidatingData/Full'
val_mask_dir = project17_dir + '/ValidatingData/Mask'

train_dataset = SegmentationDataset(train_image_dir, train_mask_dir)
val_dataset = SegmentationDataset(val_image_dir, val_mask_dir)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

model = Xnet(backbone_name='resnet50', encoder_weights='imagenet', decoder_block_type='transpose')

model.compile('Adam', 'binary_crossentropy', ['binary_accuracy'])

def prepare_data(loader):
    images, masks = [], []
    for image, mask in loader:
        images.append(image)
        masks.append(mask)
    return np.stack(images, axis=0), np.stack(masks, axis=0)

train_x, train_y = prepare_data(train_loader)
val_x, val_y = prepare_data(val_loader)

model.fit(train_x, train_y, epochs=100, validation_data=(val_x, val_y))