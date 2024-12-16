import pylidc as pl
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

# fix the conflict between numpy and pylidc, do not remove those lines
np.bool = np.bool_
np.int = int


def sort_meta_data(metadata_file: str):
    metadata = pd.read_csv(metadata_file)

    patient_stats = (
        metadata
        .groupby("patient_id")
        .agg(
            total_slices=("slice_id", "count"),
            avg_mask_area=("num_nodules_pixels", "mean")
        )
        .reset_index()
    )

    patient_stats = patient_stats.sort_values(by="avg_mask_area", ascending=False).reset_index(drop=True)

    output_file = "patient_sorted_metadata.csv"
    patient_stats.to_csv(output_file, index=False)


def analyze_patient_data_distribution_train_validate_test(file_path):
    # Step 1: Load the data
    data = pd.read_csv(file_path)

    # Step 2: Visualize average pixels by patient ID
    plt.figure(figsize=(12, 6))
    plt.bar(data['patient_id'], data['avg_mask_area'])
    plt.xlabel('Patient ID')
    plt.ylabel('Average Pixels')
    plt.title('Average Pixels by Patient ID')
    plt.xticks(rotation=90)  # Rotate x-axis labels for readability
    plt.tight_layout()
    plt.show()

    # Step 3: Categorize each patient into training, validation, or test set based on the modulo operation.
    # The rule is as follows: if index % 5 in [0, 1, 2] -> train; if index % 5 == 3 -> validation; if index % 5 == 4 -> test.
    data['set_type'] = data.index % 5
    data['set_type'] = data['set_type'].apply(
        lambda x: 'train' if x in [0, 1, 2] else 'validation' if x == 3 else 'test')

    # Step 4: Calculate the total number of slices for each group
    slice_summary = data.groupby('set_type')['total_slices'].sum().reset_index()

    # Step 5: Calculate the percentage of slices in each group
    total_slices = slice_summary['total_slices'].sum()
    slice_summary['percentage'] = (slice_summary['total_slices'] / total_slices) * 100

    # Display the final summary table
    slice_summary.to_csv("slice_summary.csv", index=False)

    return slice_summary


def count_total_patients():
    all_patient_scans = pl.query(pl.Scan).all()
    total_patient_count = len(all_patient_scans)
    print("Total number of patients in LIDC-IDRI dataset:", total_patient_count)


def count_pid():
    file_path = 'patient_sorted_metadata_2.csv'
    data = pd.read_csv(file_path)

    unique_patient_ids_count = data['patient_id'].nunique()
    print("Unique Patient IDs count:", unique_patient_ids_count)


def extract_lung(img: np.ndarray) -> np.ndarray:
    return np.clip(img, -600, 1500)


def compare_extraction(npy_path: str):
    plt.subplot(1, 2, 1)
    plt.imshow((np.load(npy_path, allow_pickle=True)))
    plt.subplot(1, 2, 2)
    plt.imshow(extract_lung((np.load(npy_path, allow_pickle=True))))
    plt.show()


def research_on_split(labelled_sorted_metadata_path: str):
    df = pd.read_csv(labelled_sorted_metadata_path)
    slice_summary = df.groupby('split_to')['total_slices'].sum().reset_index()
    slice_summary['percentage'] = slice_summary['total_slices'] * 100 / slice_summary['total_slices'].sum()
    slice_summary['percentage'] = slice_summary['percentage'].apply(lambda x: str(x) + "%")
    slice_summary.to_csv("stat_analysis_on_split", index=False)


if __name__ == '__main__':
    # sort_meta_data("new_metadata.csv")
    # count_total_patients()
    # count_pid()
    # compare_extraction("full_dir/LIDC-IDRI-0001_N000_S086_Full.npy")
    # research_on_split("labelled_patient_sorted_metadata.csv")
    pass
