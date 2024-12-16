import numpy as np

# validate passed, I think

if __name__ == '__main__':
    # LIDC-IDRI-0078,N005,S003,4,2520
    npy_data_path_full = ("/Users/davishan/Desktop/CMPT_340/2024_3_project_17/src/preprocessing/full_dir_512/LIDC-IDRI-0066_N001_S568_Full.npy")
    npy_data_path_mask = ("/Users/davishan/Desktop/CMPT_340/2024_3_project_17/src/preprocessing/mask_dir_512/LIDC-IDRI-0066_N001_S568_Mask.npy")

    mask = np.load(npy_data_path_mask)
    np.savetxt('sample_mask.txt', mask, fmt='%s')
    print("mask size: " + str(mask.shape))

    full = np.load(npy_data_path_full)
    np.savetxt('sample_full.txt', full, fmt='%s')
    print("full size: " + str(full.shape))


