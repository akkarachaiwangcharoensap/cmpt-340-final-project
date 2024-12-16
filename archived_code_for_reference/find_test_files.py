"""
import multiprocessing as mp
import numpy as np
import nibabel as nib
import pydicom as dcm
import matplotlib.pyplot as plt
import os
from src.preprocessing.preprocessing_pipeline import down_sample
from src.app.conversion import convert_to_numpy
from tqdm import tqdm
dataset_path = r"D:\school stuff\cmpt340\manifest-1600709154662\LIDC-IDRI"
def mse(scan1,scan2):
    return ((scan1 - scan2) ** 2).mean()


def traverse_dataset(path:str):
    for (root, folder, files) in os.walk(path):
        for file in files:
            if '.dcm' in file:
                yield os.path.join(root,file)

def min_mse(scan_path:str):
    dcms = traverse_dataset(dataset_path)
    scan = convert_to_numpy(scan_path)
    min_paths = []
    min_error = 100_000_000
    for file in tqdm(dcms,desc="Calculating errorubuntu"):
        try:
            dcm_scan = dcm.dcmread(file,force=True).pixel_array
        except Exception as e:
            continue
        if dcm_scan.shape == (512,512):
            error = mse(scan,down_sample(dcm_scan))
            if error < min_error:
                min_error = error
                min_path = [file]
            elif error == min_error:
                min_paths.append(file)
    return min_path


if __name__ == "__main__":
    scan_dir = r"H:\ten_samples_from_test_dataset\ten_samples_from_test_dataset"
    scans = [os.path.join(scan_dir, file) for file in os.listdir(scan_dir)]
    with mp.Pool(processes=os.cpu_count()) as p:
        results = p.map(min_mse,scans)

    with open('Scans.txt','w') as f:
        f.write("\n".join([f"{scan},{result}" for scan,result in zip(scans,results)] ))
    for i,(scan,result) in enumerate(zip(scans,results)):
        plt.subplot(1,2,1)
        plt.imshow(convert_to_numpy(scan))
        plt.subplot(1,2,2)
        plt.imshow(convert_to_numpy(result[0]))
        plt.savefig(f"Scan_{i}.png")
"""