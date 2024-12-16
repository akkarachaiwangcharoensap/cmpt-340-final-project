"""
import pydicom
from src.models.unet_family.nnUNet.nnUNet_2D import nnunetv2_inference_api
import matplotlib.pyplot as plt
import os
import numpy as np

if __name__ == '__main__':
    nii_path = r"H:\ten_samples_from_test_dataset\ten_samples_from_test_dataset\lungCT_LIDC-IDRI-0001-N000-S090_0000.nii.gz"
    path = r"D:\school stuff\cmpt340\manifest-1600709154662\LIDC-IDRI\LIDC-IDRI-0004\01-01-2000-NA-NA-91780\3000534.000000-NA-58228\1-021.dcm"


    nii_scan = conversion.convert_to_numpy(nii_path)
    scan = conversion.convert_to_numpy(path)
    print(nii_scan.shape,scan.shape,down_sample(scan).shape)


    #Test inference
    file_name = os.path.split(nii_path)[1].split('.')[0]
    print(file_name)
    input_data = {file_name:down_sample(scan)}
    print(input_data)
    print(down_sample(scan).shape)
    # prediction = nnunetv2_inference_api(input_ct_npy_dict=input_data)
    # print(prediction)

from scipy import ndimage as nd


def down_sample(img: np.ndarray, frac=2, anti_alias=False, sigma=1) -> np.ndarray:
    if not anti_alias:
        return img[::frac, ::frac]
    return nd.gaussian_filter(img[::frac, ::frac], sigma=[sigma, sigma])


def readDcm(dcm_path: str):
    dcm_file = pydicom.dcmread(dcm_path)
    npy_data = dcm_file.pixel_array
    print(npy_data.shape)
    assert npy_data.shape == (256, 256)

    file_name = os.path.basename(dcm_path).replace('.dicom', '')
    print(file_name)
    return nnunetv2_inference_api({file_name: npy_data})


# "TEST 6"
pred = readDcm('/Users/davishan/Desktop/lungCT_LIDC-IDRI-0334-N008-S103_0000.dicom')
plt.imshow(pred.get("lungCT_LIDC-IDRI-0334-N008-S103_0000"), cmap='gray')  # Use 'gray' for grayscale images
plt.colorbar()
plt.title("testing API")
plt.show()

"TEST 8"
# pred = readDcm('/Users/davishan/Desktop/lungCT_LIDC-IDRI-0750-N002-S089_0000.dcm')
# plt.imshow(pred.get("lungCT_LIDC-IDRI-0750-N002-S089_0000"), cmap='gray')  # Use 'gray' for grayscale images
# plt.colorbar()
# plt.title("testing API")
# plt.show()

# "TEST"
# npy1 = pydicom.dcmread('/Users/davishan/Desktop/int8_image_testing.dicom').pixel_array
# npy2 = nibabel.load('/Users/davishan/Desktop/ten_samples_from_test_dataset/lungCT_LIDC-IDRI-0022-N000-S105_0000.nii.gz').get_fdata()
# # np.savetxt("text1.txt", npy1, "%.1f")
# # np.savetxt("text2.txt", npy2, "%.1f")
#
# print(npy1[0][0])
# print(npy2[0][0])
#
# print(npy1.shape, npy2.shape)
# print(np.array_equal(npy1, npy2))
"""















