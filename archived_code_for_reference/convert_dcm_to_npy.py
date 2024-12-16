"""
import csv
import os
from typing import List
import numpy as np
import pydicom
import pylidc as pl
from pylidc.utils import consensus
from tqdm import tqdm

np.bool = np.bool_
np.int = int


def create_output_dirs(dir_names: List[str]) -> None:
    for dir_name in dir_names:
        os.makedirs(dir_name, exist_ok=True)
        print(f"Directory created or already exists: {dir_name}")


def normalizeSlice(sliceInNP, min_hu=-1000, max_hu=3000) -> np.ndarray:
    return sliceInNP


class CTImageConverter:
    def __init__(self, project_dir: str, data_path: str, confidence_level: float = 0.5, padding: int = 0):
        self.project_dir = project_dir
        self.data_path = data_path
        self.confidence_level = confidence_level
        self.padding = padding
        self.image_dir = os.path.join(self.project_dir, "FullImages")
        self.mask_dir = os.path.join(self.project_dir, "MaskedImages")
        create_output_dirs([self.image_dir, self.mask_dir])

    def get_pids(self) -> List[str]:
        return [name for name in os.listdir(self.data_path) if os.path.isdir(os.path.join(self.data_path, name))]

    def convert_image_to_npy_with_filter(self, ls_pids: List[str]):
        for pid in tqdm(ls_pids):
            try:
                scan = pl.query(pl.Scan).filter(pl.Scan.patient_id == pid).first()
                nodules_annotations = scan.cluster_annotations()
                vol = scan.to_volume()
                print(f"Patient ID: {pid}, CT Shape: {vol.shape}, Nodules: {len(nodules_annotations)}")

                patient_full_image_dir = os.path.join(self.image_dir, pid)
                patient_masked_image_dir = os.path.join(self.mask_dir, pid)
                os.makedirs(patient_full_image_dir, exist_ok=True)
                os.makedirs(patient_masked_image_dir, exist_ok=True)

                for nodule_idx, nodule in enumerate(nodules_annotations):
                    malignancy_levels = [anno.malignancy for anno in nodule]
                    average_malignancy = int(np.round(np.mean(malignancy_levels)))

                    consensus_mask, consensus_bbox, _ = consensus(nodule, self.confidence_level, self.padding)

                    for nodule_slice in range(consensus_mask.shape[2]):
                        num_pixel_of_mask = np.sum(consensus_mask[:, :, nodule_slice])
                        if num_pixel_of_mask == 0:
                            continue

                        patient_id_part = f"p{pid[-4:]}"
                        slice_part = f"s{str(nodule_slice).zfill(3)}"
                        mask_name = f"{patient_id_part}_{slice_part}_mask.npy"
                        nodule_name = f"{patient_id_part}_{slice_part}_full.npy"

                        full_mask = np.zeros((512, 512), dtype=np.uint8)
                        full_mask[consensus_bbox[0].start:consensus_bbox[0].stop,
                                  consensus_bbox[1].start:consensus_bbox[1].stop] = (
                                      consensus_mask[:, :, nodule_slice] * average_malignancy
                                  )

                        mask_path = os.path.join(patient_masked_image_dir, mask_name)
                        np.save(mask_path, full_mask)

                        sliceInNP = vol[:, :, nodule_slice]
                        normalizedSlice = normalizeSlice(sliceInNP)

                        full_image_path = os.path.join(patient_full_image_dir, nodule_name)
                        np.save(full_image_path, normalizedSlice)


            except Exception as e:
                print(f"Error processing patient {pid}: {e}")

    def convert_image_to_npy_without_filter(self, ls_pids: List[str]):
        csv_file = os.path.join(self.project_dir, "ctmetadata.csv")
        with open(csv_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["PatientID", "numPixelsOfMask", "TotalTumors", "Manufacturer", "ModelName"])

        for pid in tqdm(ls_pids):
            try:
                scan = pl.query(pl.Scan).filter(pl.Scan.patient_id == pid).first()
                nodules_annotations = scan.cluster_annotations()
                vol = scan.to_volume()
                print(f"Patient ID: {pid}, CT Shape: {vol.shape}, Nodules: {len(nodules_annotations)}")

                patient_full_image_dir = os.path.join(self.image_dir, pid)
                patient_masked_image_dir = os.path.join(self.mask_dir, pid)
                os.makedirs(patient_full_image_dir, exist_ok=True)
                os.makedirs(patient_masked_image_dir, exist_ok=True)

                dicom_data = scan.load_all_dicom_images()[0]
                manufacturer = dicom_data.Manufacturer if 'Manufacturer' in dicom_data else 'Unknown'
                model_name = dicom_data.ManufacturerModelName if 'ManufacturerModelName' in dicom_data else 'Unknown'

                for nodule_idx, nodule in enumerate(nodules_annotations):
                    malignancy_levels = [anno.malignancy for anno in nodule]
                    average_malignancy = int(np.round(np.mean(malignancy_levels)))

                    consensus_mask, consensus_bbox, _ = consensus(nodule, self.confidence_level, self.padding)

                    for nodule_slice in range(consensus_mask.shape[2]):
                        num_pixel_of_mask = np.sum(consensus_mask[:, :, nodule_slice])

                        patient_id_part = f"p{pid[-4:]}"
                        slice_part = f"s{str(nodule_slice).zfill(3)}"
                        mask_name = f"{patient_id_part}_{slice_part}_mask.npy"
                        nodule_name = f"{patient_id_part}_{slice_part}_full.npy"

                        full_mask = np.zeros((512, 512), dtype=np.uint8)
                        full_mask[consensus_bbox[0].start:consensus_bbox[0].stop,
                                  consensus_bbox[1].start:consensus_bbox[1].stop] = (
                                      consensus_mask[:, :, nodule_slice] * average_malignancy
                                  )

                        mask_path = os.path.join(patient_masked_image_dir, mask_name)
                        np.save(mask_path, full_mask)


                        sliceInNP = vol[:, :, nodule_slice]
                        normalizedSlice = normalizeSlice(sliceInNP)

                        full_image_path = os.path.join(patient_full_image_dir, nodule_name)
                        np.save(full_image_path, normalizedSlice)

                        with open(csv_file, mode='a', newline='') as file:
                            writer = csv.writer(file)
                            writer.writerow([pid, num_pixel_of_mask, len(nodules_annotations),
                                             manufacturer, model_name])

            except Exception as e:
                print(f"Error processing patient {pid}: {e}")


if __name__ == '__main__':
    project17_dir = os.path.dirname(os.path.dirname(os.getcwd()))
    original_data_path = "/Users/josh/Downloads/2024_3_PROJECT_17/36_PID_DCM_Dataset"

    converter = CTImageConverter(project17_dir, original_data_path)

    ls_pid = converter.get_pids()
    converter.convert_image_to_npy_without_filter(ls_pid)
"""
