import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from typing import Iterable, Callable
from scipy import ndimage as nd

# !!! Change to your own paths !!!
image_folder = r'H:\FullImages\FullImages'
masked_folder = r'H:\MaskedImages\MaskedImages'

'''
    Resize image with optional anti-aliasing
'''


def downsample(img: np.ndarray, *, frac=4, anti_alias=False, sigma=1) -> np.ndarray:
    if not anti_alias:
        return img[::frac, ::frac]
    return nd.gaussian_filter(img[::frac, ::frac], sigma=[sigma, sigma])


'''
    segment image using the hounsfield window for lung tissue [-600, 1500]
    adjusted to max of 2500 from tumor brightness investigation
'''


def segment(img: np.ndarray,*,low = -600,high = 2500) -> np.ndarray:
    return np.clip(
        img,
        low, high
    )


'''
    Will see if there's object in the image from the path name
'''


def is_tumour(img_path: str) -> bool:
    return np.load(img_path, allow_pickle=True).sum() > 0


'''
    Traverse image file directory and generate masked path from that
'''


def get_training_files(image_folder: str, masked_folder: str) -> Iterable:
    for (root, folders, files) in tqdm(os.walk(image_folder), desc='finding images'):

        for file in files:

            if os.path.splitext(file)[1] == '.npy':
                parts = file.split('_')
                masked_file = parts[0] + '_' + parts[1] + '_' + 'mask.npy'

                masked_path = os.path.join(
                    masked_folder,
                    os.path.split(root)[1],
                    masked_file
                )
                yield os.path.join(root, file), masked_path


'''
    Calculate number of true tumors and randomly sample same amount from 
    healthy images
'''


def balance_data(image_folder: str, masked_folder: str) -> pd.DataFrame:
    files = pd.DataFrame(
        get_training_files(image_folder, masked_folder),
        columns=['Image Path', 'Masked Path']
    )
    print('Labelling mask')
    labels = []
    for masked in tqdm(files['Masked Path'], desc='labelling Images'):
        labels.append(is_tumour(masked))
    files['Tumor'] = labels

    tumors = files[files['Tumor']]
    healthy = files[~files['Tumor']]

    n_tumors = tumors.shape[0]
    balanced_healthy = healthy.sample(n_tumors)
    print(f'Number of tumor images: {tumors.shape[0]}, Number of Healty images: {balanced_healthy.shape[0]}')
    return pd.concat([tumors, balanced_healthy]).reset_index()





if __name__ == '__main__':
    # balanced_data = balance_data(image_folder,masked_folder)
    # balanced_data.to_csv('balanced_data.csv',index=False)
    img_file = r'full_dir/LIDC-IDRI-0001_N000_S086_Full.npy'
    import matplotlib.pyplot as plt

    plt.subplot(1, 2, 1)
    plt.imshow(segment(downsample(np.load(img_file, allow_pickle=True))))
    plt.subplot(1, 2, 2)
    plt.imshow(segment(downsample(np.load(img_file, allow_pickle=True), anti_alias=True)))

    plt.show()
