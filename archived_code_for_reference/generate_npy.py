"""
import csv
import pylidc as pl
import numpy as np
import os
from tqdm import tqdm
from pylidc.utils import consensus

# fix the conflict between numpy and pylidc, do not remove those lines
np.bool = np.bool_
np.int = int


def generate_full_and_mask_npy(unhealthy_full_image_dir_name: str, unhealthy_masked_image_dir_name: str,
                               metadata_filename: str):
    # mkdir
    os.makedirs(unhealthy_full_image_dir_name, exist_ok=True)
    os.makedirs(unhealthy_masked_image_dir_name, exist_ok=True)

    # init metadata csv
    with open(metadata_filename, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["patient_id", "nodule_id", "slice_id", "malignancy", "mask_area"])

        # all_patient_three_dim_scans = pl.query(pl.Scan).filter(pl.Scan.patient_id == "LIDC-IDRI-0024")
        all_patient_three_dim_scans = pl.query(pl.Scan).all()

        for one_patient_three_dim_scan in tqdm(all_patient_three_dim_scans, desc="Processing Patients", unit="patient"):

            try:
                vol = one_patient_three_dim_scan.to_volume()

                nodules_annotations = one_patient_three_dim_scan.cluster_annotations()

                # current pid has more than zero nodules
                if len(nodules_annotations) > 0:
                    for nodule_idx, nodule in enumerate(nodules_annotations):
                        # get the median of malignancy scores from all doctors
                        malignancy_scores = [annotation.malignancy for annotation in nodule]
                        malignancy_median = int(np.median(malignancy_scores))

                        # set confidence_level and padding to include additional surrounding areas as part of the nodule
                        confidence_level = 0.5
                        padding = 0
                        consensus_3D_mask, consensus_3D_bbox, _ = consensus(nodule, confidence_level, padding)

                        # traverse each nodule save its slices
                        for mask_z_idx in range(consensus_3D_mask.shape[2]):
                            full_slice = vol[:, :, mask_z_idx]
                            full_mask = np.zeros((512, 512), dtype=np.uint8)

                            sum_nodules_pixels = np.sum(consensus_3D_mask[:, :, mask_z_idx])
                            if sum_nodules_pixels > 0:
                                full_mask[consensus_3D_bbox[0].start:consensus_3D_bbox[0].stop,
                                consensus_3D_bbox[1].start:consensus_3D_bbox[1].stop] = (
                                        consensus_3D_mask[:, :, mask_z_idx] * malignancy_median
                                )

                                # give it a name
                                patient_id_part = one_patient_three_dim_scan.patient_id
                                slice_part = f"S{str(mask_z_idx).zfill(3)}"
                                nodule_part = f"N{str(nodule_idx).zfill(3)}"
                                full_filename = f"{patient_id_part}_{nodule_part}_{slice_part}_Full.npy"
                                mask_filename = f"{patient_id_part}_{nodule_part}_{slice_part}_Mask.npy"

                                # save npy files
                                np.save(os.path.join(unhealthy_full_image_dir_name, full_filename), full_slice)
                                np.save(os.path.join(unhealthy_masked_image_dir_name, mask_filename), full_mask)

                                # save the metadata of the two files to the csv
                                # pid, #nodule, #slice, malignancy
                                print("non-healthy")
                                writer.writerow(
                                    [patient_id_part, nodule_part, slice_part, malignancy_median, sum_nodules_pixels])

            except Exception as e:
                print(f"Error: {e}")
"""