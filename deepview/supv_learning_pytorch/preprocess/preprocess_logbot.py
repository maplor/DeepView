# This Python file uses the following encoding: utf-8
# preprocess the logbot (2 bird species) and save window of data as .npz
# run this file directly
# todo: modify the paths to the GUI setting

import os
import glob
import pandas as pd
from process_utils import *
from tqdm import tqdm

#--------------------prepare paths-------------------------
species = "omizunagidori"
# species = "umineko"

root_path = r"D:\logbot-data\OstukaPaperData\data"

output_dir_path = os.path.join(root_path, "preprocessed_data")

path_target = os.path.join(root_path, "raw-data", species, "**.csv")
raw_data_path_list = sorted(glob.glob(path_target))

animal_id_path = os.path.join(root_path, "id-files", "animal_id.csv")

if species == "omizunagidori":
    label_id_path = os.path.join(root_path, "id-files", "label_id_omizunagidori.csv")
elif species == "umineko":
    label_id_path = os.path.join(root_path, "id-files", "label_id_umineko.csv")


# new folder to save labeled segment data
labelled_data_base_dir = os.path.join(root_path, "labelled", species)
# new folder to save unlabeled segment data
unlabelled_data_base_dir = os.path.join(root_path, "npz_format", "shuffled_20_v2", species)

#--------------------read raw data-----------------------------
df_animal_id = pd.read_csv(animal_id_path)

# # todo: test code
# raw_data_path_list = raw_data_path_list[:10]

for raw_data_path in tqdm(raw_data_path_list, total=len(raw_data_path_list)):
    (
        species,
        year,
        animal_tag,
        animal_id,
        acc_sampling_rate,
        correct_timestamp,
        back_mount
    ) = get_raw_date_information(raw_data_path,
                                 animal_id_path)
    df = read_raw_data_and_refine_timestamp(raw_data_path,
                                            correct_timestamp)
    # df.columns = ['timestamp', 'acc_x', 'acc_y', 'acc_z', 'label']
    df_list = divide_df_if_timestamp_gap_detected_2(df,
                                                    acc_sampling_rate*5*60)
    # Note: this will be 25*5*60 = 125*60 seconds gap in divide_df_if_timestamp_gap_detected

    df = run_resampling_and_concat_df(df_list,
                                      acc_sampling_rate,  # 25hz
                                      remove_sec=3,
                                      check_df=False)

    df = preprocess_sensor_data(df,
                                clipping=True,
                                clipping_threshold=8,
                                method="none",
                                check_df=False)

    # if debug_test_mode == True:
    #     print(f"| debug mode -> do not save data |")
    # else:
    save_preprocessed_data(df,
                           output_dir_path,
                           species,
                           animal_id,
                           label_id_path)

print(f"-----------------------------------")
print(f"raw data preprocessing completed !")
print(f"-----------------------------------")

#--------------------generate segment data-----------------------------
# load files
target_path = os.path.join(output_dir_path, species, "**.csv")
preprocessed_data_path_list = sorted(glob.glob(target_path))
##-------------------1 for labeled data--------------------------------
print("Extract sliding windows from preprocessed data (.csv) and save them as .npz files")
for preprocessed_data_path in tqdm(preprocessed_data_path_list, total=len(preprocessed_data_path_list)):
    print("-----------------------------------------------------------------------")
    animal_id = os.path.basename(preprocessed_data_path).replace(".csv", "")
    print(animal_id, end=": ")

    # extract windows
    (
        X_list,
        label_id_list,
        timestamp_list,
        labelled_flag_list,
        labelled_X_list,
        labelled_label_id_list,
        labelled_timestamp_list,
        timestamp_gap_idx_list
    ) = extract_sliding_windows(preprocessed_data_path,
                                sliding_window_size=50,
                                sliding_window_step_size=25)
    print(f"N of extracted windows: {len(X_list)}")
    print(f"N of labelled windows:  {len(labelled_X_list)}")
    print(f"N of timestamp gaps:    {len(timestamp_gap_idx_list)}")

    if len(labelled_X_list) > 0:
        # save labelled data
        npz_file_dir = os.path.join(labelled_data_base_dir, animal_id)
        print("Saving labelled windows as npz ...")

        # if debug_test_mode == True:
        #     print(f"| debug mode -> do not save data |")
        # else:
        save_labelled_windows_as_npz(animal_id,
                                     npz_file_dir,
                                     labelled_X_list,
                                     labelled_label_id_list,
                                     labelled_timestamp_list)

print(f"----------------------------------------")
print(f"Labelled window extraction completed !")
print(f"----------------------------------------")

##-------------------2 for unlabeled data------------------------------
for preprocessed_data_path in tqdm(preprocessed_data_path_list, total=len(preprocessed_data_path_list)):
    print("-----------------------------------------------------------------------")
    animal_id = os.path.basename(preprocessed_data_path).replace(".csv", "")
    print(animal_id, end=": ")

    # extract windows
    (
        unX_list,
        unlabel_id_list,
        untimestamp_list,
        unlabelled_flag_list,
        unlabelled_X_list,
        unlabelled_label_id_list,
        unlabelled_timestamp_list,
        untimestamp_gap_idx_list
    ) = extract_sliding_windows(
        preprocessed_data_path,
        sliding_window_size=50,
        sliding_window_step_size=25
    )

    # shuffle extracted windows
    (
        index_list_random,
        X_list_random,
        label_id_list_random,
        timestamp_list_random,
        labelled_flag_list_random
    ) = get_shuffled_list(unX_list,
                          unlabel_id_list,
                          untimestamp_list,
                          unlabelled_flag_list,
                          random_seed=558)

    # save all data as npz (1 file 20 windows)
    num_windows_per_npz_file = 20
    npz_file_dir = os.path.join(unlabelled_data_base_dir, animal_id)
    print(f"npz_file_dir: {npz_file_dir}")
    print("Saving all windows as npz ...")
    # if debug_test_mode == True:
    #     print(f"| debug mode -> do not save data |")
    # else:
    save_blocks_of_windows_as_npz(num_windows_per_npz_file,
                                  animal_id,
                                  npz_file_dir,
                                  index_list_random,
                                  X_list_random,
                                  label_id_list_random,
                                  timestamp_list_random,
                                  labelled_flag_list_random)

print(f"----------------------------------------")
print(f"Unlabelled window extraction completed !")
print(f"----------------------------------------")