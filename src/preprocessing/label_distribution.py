import os
import numpy as np
from multiprocessing import Pool
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.stats import mode, skew
import pandas as pd
from scipy.sparse import coo_matrix
def numpy_path(path:str):
    for (root, folder, files) in os.walk(path):
        for file in files:
            if '.npy' in file:
                yield os.path.join(root,file)

            
DATA_DIR = r'H:\data_splits\data_splits\train_data'

# masked_paths = numpy_path(masked_folder)
# image_paths = numpy_path(image_folder)


def is_air(img_path:str,*,air_thresh:float = 0.9):
    img = np.load(img_path,allow_pickle=True).flatten()
    # if 90% or more of the image is air
    return (( img < -1000).sum() / img.size) > 0.9
        
def is_tumour(img_path:str):
    return np.load(img_path,allow_pickle=True).sum() > 0

# def patient_dist():
#     patients = [os.path.join(masked_folder,img) for img in os.listdir(masked_folder)]

#     def label_istrue(patient_folder):
#         masks = [os.path.join(patient_folder,img) for img in os.listdir(patient_folder)]
#         return np.array([is_tumour(mask) for mask in masks])
 
#     labels = []
#     for patient in tqdm(patients):
#         labels.append(label_istrue(patient))

#     labels = labels
#     tumour_sizes = []
#     no_tumours = 0
#     for patient in tqdm(labels, desc = 'calculating sizes'):
#         size = patient.sum() 
#         if size == 0: no_tumours += 1
#         tumour_sizes.append(size)

 
#     print(no_tumours)

#     plt.hist(tumour_sizes,bins = 50)
#     plt.title('Z-size of tumor per patient')
#     plt.xlabel('Number of Slices with tumors')
#     plt.ylabel('Count')
#     plt.text(20,120,f'Most Common Value = {mode(tumour_sizes).mode}\n# of patients w no tumors = {no_tumours}')
#     plt.show()

    
# def percentage_air():
#     labels = []
#     for file in tqdm(image_paths):
#         labels.append(is_air(file))
#     labels = np.array(labels)
#     air = labels.sum()
#     chest = labels.size - air
#     plt.bar(['Chest Images','Outliers'],[chest,air])
#     plt.show()
def image_brightness(img_path,mask_path):
    img = np.load(img_path, allow_pickle=True)
    mask = np.load(mask_path, allow_pickle=True)
    return img * (mask != 0)
def get_paths(row):
    prefix = f"{row['patient_id']}_{row['nodule_id']}_{row['slice_id']}"
    return (os.path.join(
        DATA_DIR,
        prefix+"_Full.npy"
    ),
    os.path.join(
        DATA_DIR,
        prefix+"_Mask.npy"
    ))
def tumor_brightness(metadata,split_labels):
    training_ids = split_labels['patient_id'].where(split_labels['split_to'] == 'train_data')
    local_ids = metadata.where(metadata['patient_id'].isin(training_ids)).dropna()
    tqdm.pandas(desc='Generating Image paths')
    args = local_ids.progress_apply(get_paths, axis = 1)
    with Pool(os.cpu_count()) as p:
        vals = p.starmap(image_brightness,args)
    vals = np.array(vals)
    indexes = vals.shape[0]*np.random.sample(4)
    fig, axes = plt.subplots(2,2)
    for i in range(indexes.shape[0]):
        ax = axes[i // 2, i % 2]
        ax.imshow(vals[np.uint32(indexes[i]),:,:])
        ax.set_title(f'Scan {i}')
    fig.tight_layout()
    plt.figure()
    
    
    vals = vals[vals > 0].flatten()

    plt.hist(vals,bins = 100)
    plt.xlabel('Hounsfield Level')
    plt.ylabel('Count')
    plt.title('Tumor Hounsfield levels')
    plt.text(1000,25000,f'99% Value {np.percentile(vals,99)}')
    plt.text(1000,23000,f'Max Value {vals.max()}')

    plt.show()

# Number of images with tumours
# def label_dist():
#     labels = []
#     n_images = 0
#     for file in tqdm(masked_paths):
#         labels.append(is_tumour(file))
#     for file in tqdm(image_paths):
#         n_images += 1
#     labels = np.array(labels)

#     tumour_count = labels.sum()
#     healthy = labels.shape[0] - tumour_count

#     plt.bar(['Number of \nTumour Images', 'Number of \nHealthy Images'],[tumour_count / n_images,healthy / n_images])
#     plt.title('Label Distribution')
#     plt.xlabel(f'classes of {n_images} images')
#     plt.show()

if __name__ == '__main__':
    meta = pd.read_csv('new_metadata.csv')
    splits = pd.read_csv('labelled_patient_sorted_metadata.csv')

    tumor_brightness(meta,splits)
