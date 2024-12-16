import os
import re
import subprocess
import shutil
from pathlib import Path
import nibabel as nib
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm
import numpy as np
import torch.cuda
from typing import Dict


def convert_2d_npy_to_nii(source_folder, destination_folder):
    # Traverse all .npy files in the source folder
    for filename in tqdm(os.listdir(source_folder)):
        if filename.endswith('.npy'):
            # Full path to the .npy file
            npy_path = os.path.join(source_folder, filename)

            # Load the 2D .npy file
            data = np.load(npy_path)

            # Ensure that data is 2D
            if data.ndim != 2:
                print(f"Skipping {filename}: Expected 2D data, got {data.ndim}D.")
                continue

            # Create a NIfTI image with an identity affine matrix (for 2D images)
            nii_image = nib.Nifti1Image(data, affine=np.eye(4))

            # Define the output path for the .nii.gz file
            nii_filename = filename.replace('.npy', '.nii.gz')
            nii_path = os.path.join(destination_folder, nii_filename)

            # Save the .nii.gz file
            nib.save(nii_image, nii_path)


def refactor_data(source_folder: str, destination_folder: str):
    for root, dirs, files in os.walk(source_folder):
        for file in files:
            source_file_path = os.path.join(root, file)
            # Copy the file to the destination folder
            shutil.move(source_file_path, destination_folder)


def fix_naming_issue_on_imagesTr(old_dir_path: str, new_dir_path: str):
    for filename in tqdm(os.listdir(old_dir_path)):
        if filename.startswith('LIDC'):
            post_fix = '.nii.gz'
            new_filename = filename.replace(post_fix, '')
            new_filename = 'lungCT_' + new_filename.replace('_', '-') + '_0000' + post_fix

            old_file_path = os.path.join(old_dir_path, filename)
            new_file_path = os.path.join(new_dir_path, new_filename)

            shutil.copyfile(old_file_path, new_file_path)


def fix_naming_issue_on_labelsTr(old_dir_path: str, new_dir_path: str):
    for filename in tqdm(os.listdir(old_dir_path)):
        if filename.startswith('LIDC'):
            post_fix = '.nii.gz'
            new_filename = filename.replace(post_fix, '')
            new_filename = 'lungCT_' + new_filename.replace('_', '-') + post_fix

            old_file_path = os.path.join(old_dir_path, filename)
            new_file_path = os.path.join(new_dir_path, new_filename)

            shutil.copyfile(old_file_path, new_file_path)


def set_env_variables():
    # Set environment variables for nnU-Net paths
    nnunet_root = Path(__file__).parent
    os.environ['nnUNet_raw'] = str(nnunet_root / 'nnUNet_raw')
    os.environ['nnUNet_preprocessed'] = str(nnunet_root / 'nnUNet_preprocessed')
    os.environ['nnUNet_results'] = str(nnunet_root / 'nnUNet_results')

    """
        export nnUNet_raw="/Users/davishan/Desktop/CMPT_340/2024_3_project_17/src/models/unet_family/nnUNet/nnUNet_raw"
        export nnUNet_preprocessed="/Users/davishan/Desktop/CMPT_340/2024_3_project_17/src/models/unet_family/nnUNet/nnUNet_raw"
        export nnUNet_results="/Users/davishan/Desktop/CMPT_340/2024_3_project_17/src/models/unet_family/nnUNet/nnUNet_results"
    """


def run_preprocess_and_train_single_v2(res_enc_plan: str, fold: str,
                                       dataset_id: str = "777", configuration: str = "2d",
                                       nnUNetTrainer_Xepochs: str = "nnUNetTrainer_500epochs",
                                       device: str = "cuda"):
    # preprocess and verify
    preprocess_command = [
        "nnUNetv2_plan_and_preprocess",
        "-d", dataset_id,
        "-pl", res_enc_plan,
        "--verify_dataset_integrity"
    ]
    "!nnUNetv2_plan_and_preprocess -d 777 -pl nnUNetPlannerResEncM --verify_dataset_integrity"
    print("Please manually run the following command to  preprocess your dataset\n",
          " ".join(preprocess_command))

    # Training command, internally determine if continue to train from the checkpoint
    train_command = [
        "nnUNetv2_train",
        "-tr", nnUNetTrainer_Xepochs,
        "-p", res_enc_plan,
        "--c",
        "-device", device,
        dataset_id, configuration, fold
    ]
    "!nnUNetv2_train -tr nnUNetTrainer_500epochs -p nnUNetResEncUNetLPlans --c -device cuda cuda 777 2d 0"
    "!nnUNetv2_train -tr nnUNetTrainer_500epochs -p nnUNetResEncUNetMPlans --c -device cuda 777 2d 0"
    print("Please manually run the following command to train your model with your preprocessed dataset\n",
          " ".join(train_command))


def let_us_pip():
    # pip_command = [sys.executable, "-m", "pip", "install", "triton", "nnunetv2", "acvl_utils==0.2"]
    # subprocess.run(pip_command, check=True)
    pass


def run_preprocess_and_train_pipeline(res_enc_plan: str, fold: str):
    let_us_pip()
    set_env_variables()

    run_preprocess_and_train_single_v2(res_enc_plan, fold)


def run_nnunet_prediction(input_folder: str, output_folder: str,
                          res_enc_plan: str, device: str,
                          dataset_id: str = "777", configuration: str = "2d",
                          nnUNetTrainer_Xepochs: str = "nnUNetTrainer_500epochs",
                          fold: str = "0") -> None:
    # let_us_pip()
    set_env_variables()

    # start prediction
    predict_command = [
        "nnUNetv2_predict",
        "-tr", nnUNetTrainer_Xepochs,
        "-p", res_enc_plan,
        "-d", dataset_id,
        "-f", fold,
        "-c", configuration,
        "-i", input_folder,
        "-o", output_folder,
        "-chk", "checkpoint_best.pth",
        "--continue_prediction",
        "-device", device
    ]
    _ = ("!nnUNetv2_predict -tr nnUNetTrainer_500epochs -p nnUNetResEncUNetMPlans -d 777 -f 0 -c 2d "
         "-i /content/gdrive/MyDrive/nnUNet/nnUNet_raw/Dataset777_LungCT/imagesTs/ts_full "
         "-o /content/gdrive/MyDrive/nnUNet/nnUNet_raw/Dataset777_LungCT/imagesTs/ts_prediction "
         "--continue_prediction")
    # print("Please manually run the following command to predict\n",
    #       " ".join(predict_command))

    result = subprocess.run(predict_command, check=True, stdout=subprocess.PIPE, text=True)
    print(result.stdout)


def dice_coefficient_only_for_nodule(pred, gt):
    # union of a non-zero section
    intersection = np.sum((pred > 0) * (gt > 0))
    # compute dice
    return (2. * intersection) / (np.sum(pred > 0) + np.sum(gt > 0))


def get_test_dataset_dice_coefficient(pred_folder: str, ground_true_folder: str) -> None:
    dice_scores = []

    for file_name in os.listdir(ground_true_folder):
        if file_name.endswith('.nii.gz'):
            pred_path = os.path.join(pred_folder, file_name)
            gt_path = os.path.join(ground_true_folder, file_name)
            pred_npy = nib.load(pred_path).get_fdata()
            gt_npy = nib.load(gt_path).get_fdata()

            # there are some white slices in the test folder, we do skip them when computing dice coff
            if np.all(gt_npy == 0):
                continue

            dice_score = dice_coefficient_only_for_nodule(pred_npy, gt_npy)
            dice_scores.append(dice_score)

    print(np.mean(dice_scores))


def make_label_binary(input_dir_path: str, output_dir_path: str) -> None:
    for file_name in tqdm(os.listdir(input_dir_path)):
        if file_name.endswith('.nii.gz'):
            file_path = os.path.join(input_dir_path, file_name)
            nifti_img = nib.load(file_path)
            npy_data = nifti_img.get_fdata()
            npy_data = npy_data.astype(np.uint8)

            npy_data[npy_data > 1] = 1
            modified_nifti_img = nib.Nifti1Image(npy_data, nifti_img.affine, nifti_img.header)

            nib.save(modified_nifti_img, os.path.join(output_dir_path, file_name))


def new_filter_and_produce_metadata(corrected_base_path):
    list_metadata = []

    original_dir = os.path.join(corrected_base_path, 'original')
    mask_dir = os.path.join(corrected_base_path, 'mask')

    for mask_file_name in tqdm(os.listdir(mask_dir)):

        if not mask_file_name.endswith('.nii.gz'):
            continue

        mask_file_path = os.path.join(mask_dir, mask_file_name)
        mask_nifti_img = nib.load(mask_file_path)
        mask_npy_data = mask_nifti_img.get_fdata().astype(np.uint8)

        # too small, make prediction too hard
        sum_mask = np.sum(mask_npy_data)
        if sum_mask <= 8:
            continue

        og_file_name = mask_file_name.replace('.nii.gz', '') + '_0000' + '.nii.gz'

        # find matched file
        og_file_path = os.path.join(original_dir, og_file_name)
        assert os.path.isfile(og_file_path)

        list_metadata.append((og_file_path, mask_file_path, sum_mask))

    df_metadata = pd.DataFrame(list_metadata, columns=['og_file_path', 'mask_file_path', 'mask_pixel_number'])
    df_metadata.to_csv('nnunet_metadata.csv', index=False)


def produce_sorted_metadata(metadata_path):
    df = pd.read_csv(metadata_path)
    df['PatientID'] = df['og_file_path'].str.extract(r'LIDC-IDRI-(\d{4})')[0]

    df_avg = df.groupby('PatientID')['mask_pixel_number'].mean().reset_index()

    df_avg_sorted = df_avg.sort_values(by='mask_pixel_number', ascending=False).reset_index(drop=True)

    df_avg_sorted.columns = ['pid', 'avg_pixel']

    df_avg_sorted['split_to'] = df_avg_sorted.index.map(lambda x: 'train' if x % 5 < 4 else 'test')

    df_avg_sorted.to_csv('sorted_nnunet_metadata.csv', index=False)


def add_split_to_col(nnunet_metadata_path, sorted_metadata_path):
    # Load data
    nnunet_metadata_df = pd.read_csv(nnunet_metadata_path, dtype=str)
    sorted_metadata_df = pd.read_csv(sorted_metadata_path, dtype=str)

    output_list = []
    for index, row in tqdm(nnunet_metadata_df.iterrows()):

        og_file_path = row['og_file_path']
        mask_file_path = row['mask_file_path']
        mask_pixel_number = row['mask_pixel_number']

        patient_id = re.search(r'LIDC-IDRI-(\d{4})', row['og_file_path'])
        if patient_id:
            patient_id = patient_id.group(1)
        else:
            assert False

        split_to = sorted_metadata_df[sorted_metadata_df['pid'] == patient_id]['split_to'].iloc[0]
        print(split_to)

        output_list.append((og_file_path, mask_file_path, mask_pixel_number, split_to))

    df_metadata = pd.DataFrame(output_list, columns=['og_file_path', 'mask_file_path', 'mask_pixel_number', 'split_to'])
    df_metadata.to_csv('nnunet_metadata_with_split_to.csv', index=False)


def split_dataset(nnunet_metadata_with_split_to: str):
    df = pd.read_csv(nnunet_metadata_with_split_to, dtype=str)

    imagesTr = 'nnUNet_raw/Dataset777_LungCT/corrected_dataset/imagesTr'
    labelsTr = 'nnUNet_raw/Dataset777_LungCT/corrected_dataset/labelsTr'
    ts_full = 'nnUNet_raw/Dataset777_LungCT/corrected_dataset/ts_full'
    ts_label = 'nnUNet_raw/Dataset777_LungCT/corrected_dataset/ts_label'

    for index, row in tqdm(df.iterrows(), total=len(df)):
        og_file_path = row['og_file_path']
        mask_file_path = row['mask_file_path']
        # mask_pixel_number = row['mask_pixel_number']
        split_to = row['split_to']

        if split_to == 'train':
            shutil.move(og_file_path, imagesTr)
            shutil.move(mask_file_path, labelsTr)
        elif split_to == 'test':
            shutil.move(og_file_path, ts_full)
            shutil.move(mask_file_path, ts_label)


def display_gt_and_pred_images():
    nii_dicts = {
        "sample1_orig": "nnUNet_raw/Dataset777_LungCT/imagesTs/ts_full/lungCT_LIDC-IDRI-0597-N000-S102_0000.nii.gz",
        "sample1_true_mask": "nnUNet_raw/Dataset777_LungCT/imagesTs/ts_label/lungCT_LIDC-IDRI-0597-N000-S102.nii.gz",
        "sample1_pred_mask": "nnUNet_raw/Dataset777_LungCT/imagesTs/ts_label_prediction/lungCT_LIDC-IDRI-0597-N000-S102.nii.gz",
        "sample2_orig": "nnUNet_raw/Dataset777_LungCT/imagesTs/ts_full/lungCT_LIDC-IDRI-0785-N004-S141_0000.nii.gz",
        "sample2_true_mask": "nnUNet_raw/Dataset777_LungCT/imagesTs/ts_label/lungCT_LIDC-IDRI-0785-N004-S141.nii.gz",
        "sample2_pred_mask": "nnUNet_raw/Dataset777_LungCT/imagesTs/ts_label_prediction/lungCT_LIDC-IDRI-0785-N004-S141.nii.gz",
        "sample3_orig": "nnUNet_raw/Dataset777_LungCT/imagesTs/ts_full/lungCT_LIDC-IDRI-0469-N000-S128_0000.nii.gz",
        "sample3_true_mask": "nnUNet_raw/Dataset777_LungCT/imagesTs/ts_label/lungCT_LIDC-IDRI-0469-N000-S128.nii.gz",
        "sample3_pred_mask": "nnUNet_raw/Dataset777_LungCT/imagesTs/ts_label_prediction/lungCT_LIDC-IDRI-0469-N000-S128.nii.gz",
    }

    # Load all nii.gz data
    nii_data = {k: nib.load(v).get_fdata() for k, v in nii_dicts.items()}

    # Create a figure and 3x3 grid
    fig, axs = plt.subplots(3, 3, figsize=(12, 12))
    fig.suptitle("Comparison Among Original, True Mask, and Predicted Mask", fontsize=16)

    # Display an order list for each row
    order = [
        ["sample1_orig", "sample1_true_mask", "sample1_pred_mask"],
        ["sample2_orig", "sample2_true_mask", "sample2_pred_mask"],
        ["sample3_orig", "sample3_true_mask", "sample3_pred_mask"]
    ]

    # Titles for each subplot
    labels = [
        ["Original Image (Sample 1)", "True Mask (Sample 1)", "Predicted Mask (Sample 1)"],
        ["Original Image (Sample 2)", "True Mask (Sample 2)", "Predicted Mask (Sample 2)"],
        ["Original Image (Sample 3)", "True Mask (Sample 3)", "Predicted Mask (Sample 3)"]
    ]

    # Plotting each image in the defined order
    for i in range(3):
        for j in range(3):
            key = order[i][j]
            axs[i, j].imshow(nii_data[key], cmap="gray")  # Display data in grayscale
            axs[i, j].set_title(labels[i][j], fontsize=15)
            axs[i, j].axis("off")

    plt.tight_layout()
    plt.subplots_adjust(top=0.92, hspace=0.1, wspace=0.1)
    plt.show()


def nnunetv2_inference_api(input_ct_npy_dict: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """
    :param
        {"lungCT_LIDC-IDRI-0001-N000-S086_0000": original_npy_256x256_data,
         "lungCT_LIDC-IDRI-0001-N000-S087_0000": original_npy_256x256_data})
    :return
        {"lungCT_LIDC-IDRI-0001-N000-S086_0000": predicted_npy_256x256_data,
         "lungCT_LIDC-IDRI-0001-N000-S087_0000": predicted_npy_256x256_data})

    notes:
        ! Please use original file name from the test dataset, otherwise model will not read them !
    """

    # preprocessing
    nnunet_root = Path(__file__).parent
    ts_prod_og_dir = str(nnunet_root / 'nnUNet_raw' / 'Dataset777_LungCT' / 'imagesTs' / 'ts_prod_og')
    ts_prod_predict_dir = str(nnunet_root / 'nnUNet_raw' / 'Dataset777_LungCT' / 'imagesTs' / 'ts_prod_predict')
    # os.makedirs(ts_prod_og_dir, exist_ok=True)
    # os.makedirs(ts_prod_predict_dir, exist_ok=True)

    for npy_input_name, npy_input_data in input_ct_npy_dict.items():
        affine = np.eye(4)
        nii_image = nib.Nifti1Image(npy_input_data, affine)
        file_save_path = os.path.join(ts_prod_og_dir, npy_input_name + '.nii.gz')
        nib.save(nii_image, file_save_path)

    # processing
    run_nnunet_prediction(input_folder=ts_prod_og_dir,
                          output_folder=ts_prod_predict_dir,
                          res_enc_plan="nnUNetResEncUNetLPlans",
                          device="cuda" if torch.cuda.is_available() else "cpu")

    # postprocessing
    output_ct_npy_dict = {}
    for input_file_name in input_ct_npy_dict.keys():
        output_file_name = input_file_name.replace('_0000', '') + '.nii.gz'
        output_file_path = os.path.join(ts_prod_predict_dir, output_file_name)

        if os.path.exists(output_file_path):
            mask_nii_prediction = nib.load(output_file_path)
            mask_npy_prediction = mask_nii_prediction.get_fdata()
            output_ct_npy_dict[input_file_name] = mask_npy_prediction
        else:
            print(f"Warning: Prediction output not found for {output_file_name}")
            assert False

    return output_ct_npy_dict


if __name__ == '__main__':
    "split done"
    # refactor_data("nnUNet_raw/Dataset777_LungCT/labelsTr",
    #               "nnUNet_raw/Dataset777_LungCT/labelsTr")

    "comparison done"
    # compare_folders("nnUNet_raw/Dataset777_LungCT/imagesTr",
    #                 "nnUNet_raw/Dataset777_LungCT/labelsTr")

    "convertion done"
    # convert_2d_npy_to_nii("nnUNet_raw/Dataset777_LungCT/imagesTs_/ts_mask_dir",
    #                       "nnUNet_raw/Dataset777_LungCT/imagesTs/ts_label_dir")

    "fix naming issues"
    # fix_naming_issue_on_imagesTr('nnUNet_raw/Dataset777_LungCT/imagesTs/ts_full_',
    #                             'nnUNet_raw/Dataset777_LungCT/imagesTs/ts_full')
    # fix_naming_issue_on_labelsTr('nnUNet_raw/Dataset777_LungCT/imagesTs/ts_label_',
    #                             'nnUNet_raw/Dataset777_LungCT/imagesTs/ts_label')

    "make label binary"
    # make_label_binary('nnUNet_raw/Dataset777_LungCT/imagesTs/ts_label_',
    #                   'nnUNet_raw/Dataset777_LungCT/imagesTs/ts_label')
    # make_label_binary('nnUNet_raw/Dataset777_LungCT/labelsTr_',
    #                   'nnUNet_raw/Dataset777_LungCT/labelsTr')

    "produce new metadata"
    # new_filter_and_produce_metadata('nnUNet_raw/Dataset777_LungCT/corrected_dataset')
    # produce_sorted_metadata('nnunet_metadata.csv')
    # add_split_to_col('nnunet_metadata.csv', 'sorted_nnunet_metadata.csv')
    # split_dataset('nnunet_metadata_with_split_to.csv')

    "! start model training !"
    # run_preprocess_and_train_pipeline(res_enc_plan="nnUNetResEncUNetLPlans", fold="0")

    "! start model predication !"
    # run_nnunet_prediction(input_folder="nnUNet_raw/Dataset777_LungCT/imagesTs/ts_prod_og",
    #                       output_folder="nnUNet_raw/Dataset777_LungCT/imagesTs/ts_prod_predict",
    #                       res_enc_plan="nnUNetResEncUNetLPlans",
    #                       device="cpu")

    "! test model accuracy !"
    # get_test_dataset_dice_coefficient(pred_folder='nnUNet_raw/Dataset777_LungCT/imagesTs/ts_prediction',
    #                                   ground_true_folder='nnUNet_raw/Dataset777_LungCT/imagesTs/ts_label')

    "display nice images"
    # display_gt_and_pred_images()

    "test api"
    # nii_data = nib.load("/Users/davishan/Desktop/ten_samples_from_test_dataset"
    #                     "/lungCT_LIDC-IDRI-0022-N000-S105_0000.nii.gz")
    # npy_data = nii_data.get_fdata()
    # res = nnunetv2_inference_api({"lungCT_LIDC-IDRI-0022-N000-S105_0000": npy_data})
    # plt.imshow(res.get("lungCT_LIDC-IDRI-0022-N000-S105_0000"), cmap='gray')  # Use 'gray' for grayscale images
    # plt.colorbar()
    # plt.title("testing API")
    # plt.show()
    pass
