import os
import nibabel
import numpy as np
import pydicom
import SimpleITK as sitk


def nii_to_dicom(nii_path, output_path):
    dcm_image = sitk.GetImageFromArray(nibabel.load(nii_path).get_fdata().astype(np.int16))
    sitk.WriteImage(dcm_image, output_path)


def dicom_to_numpy(dicom_path):
    ds = pydicom.dcmread(dicom_path).pixel_array
    # ds_array_float = ds.pixel_array
    # ds_array_int = ds.pixel_array.astype('uint16')
    # np.savetxt("ds_array_float.txt", ds_array_float, fmt="%f", delimiter=",")
    # np.savetxt("ds_array_int.txt", ds_array_int, fmt="%d", delimiter=",")
    return ds


def convert_ten_nii_to_dicom(nii_dir_path: str, dcm_dir_path: str):
    for nii_file_name in os.listdir(nii_dir_path):
        if nii_file_name.endswith('.nii.gz'):
            nii_file_path = os.path.join(nii_dir_path, nii_file_name)
            dcm_file_name = nii_file_name.replace('.nii.gz', '.dicom')
            dcm_file_path = os.path.join(dcm_dir_path, dcm_file_name)
            nii_to_dicom(nii_file_path, dcm_file_path)


def compare_content(nii_dir_path, dcm_dir_path):
    for nii_file_name in os.listdir(nii_dir_path):
        if nii_file_name.endswith('.nii.gz'):
            nii_file_path = os.path.join(nii_dir_path, nii_file_name)
            dcm_file_name = nii_file_name.replace('.nii.gz', '.dicom')
            dcm_file_path = os.path.join(dcm_dir_path, dcm_file_name)
            print(nii_file_path, dcm_file_path)
            np_data_from_nii = nibabel.load(nii_file_path).get_fdata()
            np_data_from_dcm = dicom_to_numpy(dcm_file_path)

            if np.array_equal(np_data_from_nii, np_data_from_dcm):
                print("YES! Original nii.gz data and DCM data are identical.", np_data_from_nii.shape, np_data_from_dcm.shape)
            else:
                print("NO! Data mismatch between original NumPy array and DICOM data.")


if __name__ == '__main__':
    # convert_ten_nii_to_dicom("/Users/davishan/Desktop/ten_samples_from_test_dataset",
    #                          "/Users/davishan/Desktop/ten_samples_from_test_dataset_dicom_new_int")

    compare_content("/Users/davishan/Desktop/ten_samples_from_test_dataset",
                    "/Users/davishan/Desktop/CMPT_340/2024_3_project_17"
                    "/ten_ct_slice_samples_from_various_patients_in_test_dataset")

    pass



