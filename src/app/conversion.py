import numpy as np
import nibabel as nib
import pydicom as pdcm
import os
import base64
'''
    TODO: implement standardization logic from Han
'''
def convert_to_numpy(dicom_path:str) -> np.ndarray:
    return read_scan(dicom_path)

'''
    Uses Min-Max normalization and maps that back to [0,2**bit_depth] range 
    it uses the range of the distribution from [0,1] to not increase contrast
'''
def convert_to_png(scan:np.ndarray,*,houn_offset = 3000,bit_depth = 8) -> np.ndarray:
    shifted = scan + houn_offset
    normalized = (shifted - shifted.min()) / (shifted.max() - shifted.min())
    return (normalized * 2**bit_depth * (normalized.max() - normalized.min())).astype(np.uint8)

'''
    Genral function to automatically read our filetypes
'''
def read_scan(path:str) -> np.ndarray:
    if '.gz' in os.path.splitext(path)[1]:
        return nib.load(path).get_fdata()
    elif '.dcm' in os.path.splitext(path)[1] or '.dicom' in os.path.splitext(path)[1]:
        return pdcm.dcmread(path).pixel_array
    else:
        return np.load(path,allow_pickle=True)



# Function to convert an image file to a Base64 string
def image_to_base64(image_path):
    try:
        with open(image_path, "rb") as image_file:
            # Read the image file as binary data
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        return encoded_string
    except FileNotFoundError:
        return "File not found. Please check the file path."

