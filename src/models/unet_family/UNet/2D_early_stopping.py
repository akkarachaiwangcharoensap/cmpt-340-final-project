import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from tqdm.auto import tqdm
import torch.nn.functional as F


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

        image = torch.from_numpy(image).float()
        mask = torch.from_numpy(mask).float()

        return image, mask


# set path
project17_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd())))
train_image_dir = project17_dir + '/TrainingData/Full'
train_mask_dir = project17_dir + '/TrainingData/Mask'
val_image_dir = project17_dir + '/ValidatingData/Full'
val_mask_dir = project17_dir + '/ValidatingData/Mask'

# load data
train_dataset = SegmentationDataset(train_image_dir, train_mask_dir)
val_dataset = SegmentationDataset(val_image_dir, val_mask_dir)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)


# DIY encoder
class UNetWithResNet50(nn.Module):
    def __init__(self, num_classes=1):
        super(UNetWithResNet50, self).__init__()

        self.adjust_conv = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=3, stride=1, padding=1)
        self.encoder = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.encoder_layers = [
            self.encoder.layer1,
            self.encoder.layer2,
            self.encoder.layer3,
            self.encoder.layer4
        ]

        self.upconv4 = self._upconv(2048, 512)
        self.conv4 = self._conv_block(512 + 2048, 512)
        self.upconv3 = self._upconv(512, 256)
        self.conv3 = self._conv_block(256 + 1024, 256)
        self.upconv2 = self._upconv(256, 128)
        self.conv2 = self._conv_block(128 + 512, 128)
        self.upconv1 = self._upconv(128, 64)
        self.conv1 = self._conv_block(64 + 256, 64)
        self.upconv0 = self._upconv(64, 32)
        self.final_conv = nn.Conv2d(32, num_classes, kernel_size=1)

    def _upconv(self, in_channels, out_channels):
        return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def _conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.adjust_conv(x)

        skip_connections = []
        x = self.encoder.relu(self.encoder.bn1(self.encoder.conv1(x)))
        x = self.encoder.maxpool(x)

        for layer in self.encoder_layers:
            x = layer(x)
            skip_connections.append(x)

        skip_connections = skip_connections[::-1]

        x = self.upconv4(x)
        skip_connections[0] = F.interpolate(skip_connections[0], scale_factor=2, mode='bilinear', align_corners=False)
        x = torch.cat((x, skip_connections[0]), dim=1)
        x = self.conv4(x)

        x = self.upconv3(x)
        skip_connections[1] = F.interpolate(skip_connections[1], scale_factor=2, mode='bilinear', align_corners=False)
        x = torch.cat((x, skip_connections[1]), dim=1)
        x = self.conv3(x)

        x = self.upconv2(x)
        skip_connections[2] = F.interpolate(skip_connections[2], scale_factor=2, mode='bilinear', align_corners=False)
        x = torch.cat((x, skip_connections[2]), dim=1)
        x = self.conv2(x)

        x = self.upconv1(x)
        skip_connections[3] = F.interpolate(skip_connections[3], scale_factor=2, mode='bilinear', align_corners=False)
        x = torch.cat((x, skip_connections[3]), dim=1)
        x = self.conv1(x)
        x = self.upconv0(x)

        x = self.final_conv(x)

        return torch.sigmoid(x)


# loss function
model = UNetWithResNet50(num_classes=1)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)


class EarlyStopping:
    def __init__(self, patience=5, verbose=False):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ...')
        torch.save(model.state_dict(), 'checkpoint.pth')
        self.val_loss_min = val_loss


# early stopping method
early_stopping = EarlyStopping(patience=5, verbose=True)

def train_model(model, train_loader, val_loader, epochs=100):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        print(f"Epoch {epoch + 1}/{epochs}")

        with tqdm(total=len(train_loader), desc="Training", unit="batch") as pbar:
            for images, masks in train_loader:
                images, masks = images.to(device), masks.to(device)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, masks)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

                pbar.update(1)
                pbar.set_postfix({'loss': loss.item()})

        val_loss = validate_model(model, val_loader)
        print(f"Epoch {epoch + 1}/{epochs}, Training Loss: {total_loss / len(train_loader)}, Val Loss: {val_loss}")

        early_stopping(val_loss, model)

        if early_stopping.early_stop:
            print("Early stopping")
            break


# validate model
def validate_model(model, val_loader):
    model.eval()
    total_loss = 0
    with tqdm(total=len(val_loader), desc="Validating", unit="batch") as pbar:
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                loss = criterion(outputs, masks)
                total_loss += loss.item()
                pbar.update(1)
    return total_loss / len(val_loader)


# train model
train_model(model, train_loader, val_loader)