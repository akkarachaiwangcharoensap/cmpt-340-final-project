import os
import csv
import shutil
from pathlib import Path

import pandas as pd
import pylidc as pl
import numpy as np
from tqdm import tqdm
from pylidc.utils import consensus
from scipy import ndimage as nd

# fix the conflict between numpy and pylidc, do not remove those lines
np.bool = np.bool_
np.int = int


def down_sample(img: np.ndarray, frac=2, anti_alias=False, sigma=1) -> np.ndarray:
    if not anti_alias:
        return img[::frac, ::frac]
    return nd.gaussian_filter(img[::frac, ::frac], sigma=[sigma, sigma])


def preprocess_pipeline(npy_path: str) -> np.ndarray:
    return down_sample(np.load(npy_path))


def categorize_and_save_csv(sorted_metadata_path: str, labelled_sorted_metadata_path: str):
    # Load the data
    df_metadata = pd.read_csv(sorted_metadata_path)

    # The rule is: if index % 5 in [0, 1, 2] -> train; if index % 5 == 3 -> validation; if index % 5 == 4 -> test.
    # So, we can achieve 6:2:2 without any patient leaks
    df_metadata['split_to'] = df_metadata.index % 5
    df_metadata['split_to'] = df_metadata['split_to'].apply(
        lambda x: 'train_data' if x in [0, 1, 2] else 'validation_data' if x == 3 else 'test_data')

    df_metadata.to_csv(labelled_sorted_metadata_path, index=False)


def split_balanced_train_valid_test_dataset(full_dir: str, mask_dir: str, labelled_sorted_metadata_path: str):
    def move_files_to_folder(row, _full_dir, _mask_dir, _base_destination_dir):
        patient_id = row['patient_id']
        split_to = row['split_to']

        destination_folder = Path(_base_destination_dir) / split_to
        destination_folder.mkdir(parents=True, exist_ok=True)

        for source_dir in [_full_dir, _mask_dir]:
            matching_file_names = [f for f in os.listdir(source_dir) if f.startswith(patient_id) and f.endswith('.npy')]
            for matched_file_name in matching_file_names:
                from_file_path = Path(source_dir) / matched_file_name
                to_file_path = destination_folder / matched_file_name

                processed_img = preprocess_pipeline(str(from_file_path))
                np.save(to_file_path, processed_img)

    # read csv
    df_metadata = pd.read_csv(labelled_sorted_metadata_path)
    base_destination_dir = "data_splits"

    tqdm.pandas(desc="Processing all patients")
    df_metadata.progress_apply(move_files_to_folder, axis=1, args=(full_dir, mask_dir, base_destination_dir))


def generate_full_and_mask_npy_512(unhealthy_full_image_dir_name: str, unhealthy_masked_image_dir_name: str,
                                   metadata_filename: str, error_log_filename: str):
    # mkdir
    os.makedirs(unhealthy_full_image_dir_name, exist_ok=True)
    os.makedirs(unhealthy_masked_image_dir_name, exist_ok=True)

    with open(metadata_filename, mode='w', newline='') as csvfile, open(error_log_filename, mode='w',
                                                                        newline='') as errorfile:
        writer = csv.writer(csvfile)
        error_writer = csv.writer(errorfile)

        writer.writerow(["patient_id", "nodule_id", "slice_id", "malignancy", "num_nodules_pixels"])
        error_writer.writerow(["patient_id", "error_message"])

        # get all patients
        all_patient_3d_scans = pl.query(pl.Scan).all()

        # iterate all patients' 3D scans
        for one_patient_3d_scan in tqdm(all_patient_3d_scans):
            try:
                nodules_annotation = one_patient_3d_scan.cluster_annotations()
                lung_3d_vol = one_patient_3d_scan.to_volume()

                # skip the patients without nodules
                if len(nodules_annotation) == 0:
                    continue

                for nodule_idx, nodule in enumerate(nodules_annotation):
                    confidence_level = 0.5
                    padding = 0
                    mask_3d, consensus_bounding_box, _ = consensus(nodule, confidence_level, padding)

                    # We calculate the malignancy information
                    malignancy_scores = [annotation.malignancy for annotation in nodule]
                    malignancy_median = int(np.median(malignancy_scores))

                    for nodule_slice_idx in range(mask_3d.shape[2]):

                        actual_slice_idx = consensus_bounding_box[2].start + nodule_slice_idx

                        full_npy = lung_3d_vol[:, :, actual_slice_idx]  # todo
                        full_npy[full_npy == -0] = 0

                        mask_npy = np.zeros((512, 512), dtype=np.uint8)
                        mask_npy[consensus_bounding_box[0].start:consensus_bounding_box[0].stop,
                        consensus_bounding_box[1].start:consensus_bounding_box[1].stop] = \
                            mask_3d[:, :, nodule_slice_idx] * malignancy_median  # todo

                        # calculate the actual number of pixels in the nodule region
                        num_nodules_pixels = np.sum(mask_3d[:, :, nodule_slice_idx] > 0)

                        # skip the slices without nodules
                        if num_nodules_pixels == 0:
                            continue

                        # Assign file names
                        patient_id_part = one_patient_3d_scan.patient_id
                        nodule_part = f"N{str(nodule_idx).zfill(3)}"
                        actual_slice_part = f"S{str(actual_slice_idx).zfill(3)}"

                        full_filename = f"{patient_id_part}_{nodule_part}_{actual_slice_part}_Full.npy"
                        mask_filename = f"{patient_id_part}_{nodule_part}_{actual_slice_part}_Mask.npy"

                        # save npy files
                        np.save(os.path.join(unhealthy_full_image_dir_name, full_filename), full_npy)
                        np.save(os.path.join(unhealthy_masked_image_dir_name, mask_filename), mask_npy)

                        # write metadata
                        writer.writerow(
                            [patient_id_part, nodule_part, actual_slice_part, malignancy_median, num_nodules_pixels])

            except Exception as e:
                error_writer.writerow([one_patient_3d_scan.patient_id, str(e)])


def refactor_dataset_into_full_and_mask_dirs(source_dir_path):
    # mkdir
    full_dir_path = os.path.join(source_dir_path, "full_dir")
    mask_dir_path = os.path.join(source_dir_path, "mask_dir")
    os.makedirs(full_dir_path, exist_ok=True)
    os.makedirs(mask_dir_path, exist_ok=True)

    # read
    for filename in os.listdir(source_dir_path):
        source_file_path = os.path.join(source_dir_path, filename)

        # check and move
        if "_Full" in filename:
            new_filename = filename.replace("_Full", "")
            destination_path = os.path.join(full_dir_path, new_filename)
            shutil.move(source_file_path, destination_path)
        elif "_Mask" in filename:
            new_filename = filename.replace("_Mask", "")
            destination_path = os.path.join(mask_dir_path, new_filename)
            shutil.move(source_file_path, destination_path)
        else:
            print(f"Warning: '{filename}' does not contain '_Full' or '_Mask' and will be skipped.")


def compare_folders(folder1: str, folder2: str):
    # Get the list of file names in each folder
    files_in_folder1 = set(os.listdir(folder1))
    files_in_folder2 = set(os.listdir(folder2))

    # Check if the file counts are the same
    if len(files_in_folder1) != len(files_in_folder2):
        print(
            f"File counts differ: {folder1} has {len(files_in_folder1)} files, {folder2} has {len(files_in_folder2)} files.")
        return False

    # Check if file names match exactly between both folders
    if files_in_folder1 == files_in_folder2:
        print("Both folders have the same number of files, and all file names match.")
        return True
    else:
        # Identify files missing in folder2 and files missing in folder1
        missing_in_folder2 = files_in_folder1 - files_in_folder2
        missing_in_folder1 = files_in_folder2 - files_in_folder1

        if missing_in_folder2:
            print(f"Files missing in {folder2}: {missing_in_folder2}")
        if missing_in_folder1:
            print(f"Files missing in {folder1}: {missing_in_folder1}")

        return False


if __name__ == '__main__':
    # generate_full_and_mask_npy_512("full_dir", "mask_dir", "new_metadata.csv", "err.csv")
    # preprocess_pipeline("full_dir", "mask_dir")
    # categorize_and_save_csv("patient_sorted_metadata.csv","labelled_patient_sorted_metadata.csv")
    # split_balanced_train_valid_test_dataset("full_dir", "mask_dir", "labelled_patient_sorted_metadata.csv")
    # refactor_dataset_into_full_and_mask_dirs("data_splits/validation_data")
    compare_folders("data_splits/validation_data/full_dir", "data_splits/validation_data/mask_dir")
