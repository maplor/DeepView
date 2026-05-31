import os
import random
import logging

import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)


def extract_sliding_windows(preprocessed_data_path,
                            sliding_window_size=50,
                            sliding_window_step_size=25):
    '''
    Args:
        preprocessed_data_path (str): path for preprocessed csv file
        sliding_window_size (int): number of data points in a time-window = window size
        sliding_window_step_size (int): number of data points that the sliding window moves every step
    Returns:
        Data of extracted windows as lists
    '''

    # species = os.path.basename(os.path.dirname(preprocessed_data_path))
    # animal_id = os.path.basename(preprocessed_data_path).replace('.csv', '')
    # read preprocessed data
    df = pd.read_csv(preprocessed_data_path, low_memory=False)
    logger.info("Length of df: %s", len(df))
    logger.info("Extracting sliding windows")

    window_size = sliding_window_size
    window_step_size = sliding_window_step_size

    X_list = []
    label_id_list = []
    timestamp_list = []
    labelled_flag_list = []

    labelled_X_list = []
    labelled_label_id_list = []
    labelled_timestamp_list = []

    feature_columns = ["acc_x", "acc_y", "acc_z"]

    timestamp_gap_idx_list = []
    X_zeros_idx_list = []

    df_values = df.values
    for i in range(0, len(df_values) - window_size, window_step_size):
        # for i in tqdm(range(0, len(df_values)-window_size, window_step_size)):
        timestamp_tmp = df_values[i:i + window_size, 1]
        label_id_tmp = df_values[i:i + window_size, 6]
        X_tmp = df_values[i:i + window_size, 2:5]

        # timestamp_tmp = list(df["unixtime"][i:i+window_size])
        # label_id_tmp = list(df["label_id"][i:i+window_size])
        X_zeros, labelled, timestamp_gap = check_before_saving(
            X=X_tmp,
            timestamp=list(timestamp_tmp),
            label_id=list(label_id_tmp)
        )

        if timestamp_gap == True:
            timestamp_gap_idx_list.append(i)
        elif X_zeros == True:
            X_zeros_idx_list.append(i)
        else:
            # X_tmp = df_values[i:i+window_size, 2:5] # acc_x, acc_y, acc_z

            # Note: consider adding only data that are not labelled
            # -> if you want to do so, comment out the below 3 lines and
            # use the if else statement below
            X_list.append(X_tmp)
            timestamp_list.append(timestamp_tmp)
            label_id_list.append(label_id_tmp)
            # X_tmp = (df[feature_columns][i:i+window_size])
            # X_list.append(np.array(X_tmp))
            # label_id_list.append(np.array(label_id_tmp))
            # timestamp_list.append(np.array(timestamp_tmp))

            if labelled == True:
                labelled_flag_list.append(True)
                labelled_X_list.append(X_tmp)
                labelled_label_id_list.append(label_id_tmp)
                labelled_timestamp_list.append(timestamp_tmp)
            else:
                labelled_flag_list.append(False)
                # X_list.append(X_tmp)
                # timestamp_list.append(timestamp_tmp)
                # label_id_list.append(label_id_tmp)

    return (X_list,
            label_id_list,
            timestamp_list,
            labelled_flag_list,
            labelled_X_list,
            labelled_label_id_list,
            labelled_timestamp_list,
            timestamp_gap_idx_list)


def check_before_saving(X, timestamp, label_id):
    # todo: 感觉没必要啊。。。
    '''Check extracted data before saving as npz file
    Args:
        X (numpy.ndarray): array of signal data (float) of a window
        timestamp (numpy.ndarray): array of timestamp of a window
        label_id (numpy.ndarray): array of label (int)
    Returns:
        X_zeros (bool): the number of data points with zero values >= 5?
        labelled (bool): all data points are labelled or not
        timestamp_gap (bool): is there a time gap equal to or larger than 2.00?
    '''

    # check X
    # Return True if X has more zeros
    num_zeros_in_X = np.sum(X == 0)
    if num_zeros_in_X >= 5:
        X_zeros = True
    else:
        X_zeros = False

    # check timestamp
    # Return True if there is a gap of more than 2 seconds
    start_timestamp = timestamp[0]
    end_timestamp = timestamp[-1]
    time_diff = end_timestamp - start_timestamp
    if time_diff < 2.00:  # a window of 2 sec -> time gap should be less than 2.00
        timestamp_gap = False
    else:
        timestamp_gap = True

    # check label
    # True if all data points are labeled and all labels are the same
    num_na_in_label_id = np.sum(np.isnan(label_id))
    num_unique_label_id = len(np.unique(label_id))
    if num_na_in_label_id == 0 and num_unique_label_id == 1:
        labelled = True
    else:
        labelled = False

    return X_zeros, labelled, timestamp_gap


def save_labelled_windows_as_npz(animal_id,
                                 npz_file_dir,
                                 labelled_X_list,
                                 labelled_label_id_list,
                                 labelled_timestamp_list):
    '''
    Args:
        animal_id (str):
        npz_file_dir (str):
        labelled_X_list (list):
        labelled_label_id_list (list):
        labelled_timestamp_list (list):
    Returns:
        None
    '''
    labelled_window_counter = 0
    for i in range(0, len(labelled_X_list)):
        # for i in tqdm(range(0, len(labelled_X_list))):
        X = np.array([labelled_X_list[i]]).astype("float64")
        label_id = np.array([labelled_label_id_list[i]]).astype("float64")
        timestamp = np.array([labelled_timestamp_list[i]]).astype("float64")

        if os.path.exists(npz_file_dir) == False:
            os.makedirs(npz_file_dir)

        npz_file_name = animal_id + "_labelled_" + str(labelled_window_counter).zfill(5)
        npz_file_path = os.path.join(npz_file_dir, npz_file_name)

        # animal_id_list = [animal_id]
        # Do not assign the same variable, or you will get error here

        np.savez_compressed(file=npz_file_path,
                            X=X,
                            label_id=label_id,
                            timestamp=timestamp,
                            animal_id=animal_id)

        labelled_window_counter += 1
    return


# Shuffle the index of the original list with random.sample(),
# append according to that randomized index -> return the reordered list
def get_shuffled_list(X_list,
                      label_id_list,
                      timestamp_list,
                      labelled_flag_list,
                      random_seed=558):
    index_list = list(range(0, len(X_list)))
    random.seed(random_seed)
    index_list_random = random.sample(index_list, len(index_list))

    X_list_random = []
    label_id_list_random = []
    timestamp_list_random = []
    labelled_flag_list_random = []
    # animal_id_list_random = []

    # shuffle
    for i in index_list_random:
        X_list_random.append(X_list[i])
        label_id_list_random.append(label_id_list[i])
        timestamp_list_random.append(timestamp_list[i])
        labelled_flag_list_random.append(labelled_flag_list)

    return (index_list_random,
            X_list_random,
            label_id_list_random,
            timestamp_list_random,
            labelled_flag_list_random)


def save_blocks_of_windows_as_npz(num_windows_per_npz_file,
                                  animal_id,
                                  npz_file_dir,
                                  index_list_random,
                                  X_list_random,
                                  label_id_list_random,
                                  timestamp_list_random,
                                  labelled_flag_list_random):
    # todo 好像和get_shuffled_list差不多
    '''
    Args:
        num_windows_per_npz_file: int
        animal_id: str
        npz_file_dir: str
        index_list_random: list
    Returns:
        None
    '''
    index_block = []
    X_block = []
    label_id_block = []
    timestamp_block = []
    animal_id_block = []
    block_counter = 0

    for i in range(0, len(index_list_random)):
        # for i in tqdm(range(0, len(index_list_random))):
        index_block.append(index_list_random[i])
        X_block.append(X_list_random[i])
        label_id_block.append(label_id_list_random[i])
        timestamp_block.append(timestamp_list_random[i])
        # animal_id is constant because we shuffled the list of data from the same individual
        animal_id_block.append([animal_id])

        if (i + 1) % num_windows_per_npz_file == 0:

            X_block_array = np.array(X_block).astype("float64")
            # Keep data as float to maintain missing values
            label_id_block_array = np.array(label_id_block).astype("float64")
            timestamp_block_array = np.array(timestamp_block).astype("float64")
            # animal_id_block_array = np.array(animal_id_block)

            if os.path.exists(npz_file_dir) == False:
                os.makedirs(npz_file_dir)
            npz_file_name = animal_id + "_" + str(block_counter).zfill(5)
            npz_file_path = npz_file_dir + npz_file_name

            np.savez_compressed(file=npz_file_path,
                                X=X_block_array,
                                label_id=label_id_block_array,
                                timestamp=timestamp_block_array,
                                animal_id=animal_id_block)

            # Initialize block after saving
            index_block = []
            X_block = []
            label_id_block = []
            timestamp_block = []
            animal_id_block = []

            block_counter += 1
