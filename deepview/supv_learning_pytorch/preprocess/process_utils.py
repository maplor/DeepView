# This Python file uses the following encoding: utf-8
# code from Otsuka's project: data_preprocess_logbot.py


# import torch
import random
import numpy as np
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def get_raw_date_information(raw_data_path, animal_id_path):
    '''Extract meta data (information) of (every single) raw data
    Args:
        raw_data_path (str): path for raw data (csv file)
        animal_id_path (str): path for meta information (csv file)
    Returns:
        meta data about the raw data (extracted information from the csv files)
    '''
    # todo: 如果这部分没用，可以简化
    s = raw_data_path
    s = os.path.basename(s)

    target = '20'  # All data were collected since 2017
    index = s.find(target)
    species = s[:index]
    species = species.lower()
    year = s[index:index + 4]
    year = int(year)

    # Extract information from raw data file name
    s = raw_data_path
    s = os.path.basename(s)
    target = "_raw_data_"
    index = s.find(target)
    animal_tag = s[index + 10:]
    # print(animal_tag)
    target = "_lb"
    index = animal_tag.find(target)
    animal_tag = animal_tag[:index]
    # print(animal_tag)

    df_animal_id = pd.read_csv(animal_id_path, low_memory=False)
    df_target_row = df_animal_id[(df_animal_id["animal_tag"] == animal_tag) \
                                 & (df_animal_id["species"] == species) \
                                 & (df_animal_id["year"] == year)]
    # display(df_target_row)
    animal_id = df_target_row["animal_id"].values[0]
    acc_sampling_rate = df_target_row["acc_sampling_rate"].values.astype(int)[0]
    correct_timestamp = df_target_row["correct_timestamp"].values.astype(int)[0]

    # because the sensor placement has two settings, mark flag at here
    back = df_target_row["back"].values.astype(int)[0]
    if back == 1:
        back_mount = True
    elif back == 0:
        back_mount = False
    return (species, year, animal_tag, animal_id,
            acc_sampling_rate, correct_timestamp, back_mount)


def read_raw_data_and_refine_timestamp(raw_data_path, correct_timestamp):
    '''Refine timestamp if needed
    Args:
        raw_data_path (str): path for raw data (csv file)
        correct_timestamp (int): 1: correct timestamp 0: do not correct timestamp
    Returns:
        df (DataFrame): a data frame with refined timestamps
    '''
    # Message
    print("Reading raw data -> ", end="")

    # load csv
    column_list = ['timestamp', 'acc_x', 'acc_y', 'acc_z', 'label']
    df = pd.read_csv(raw_data_path, low_memory=False, usecols=column_list)
    # datetime
    s = df['timestamp'].str.replace('T', ' ').str.replace('Z', '')
    df = df.drop('timestamp', axis=1)
    s_datetime = pd.to_datetime(s)  # to datetime64[ns]
    df.insert(loc=0, column='datetime', value=s_datetime)
    # round at 1 millisecond
    df['datetime'] = df['datetime'].dt.round('1L')
    # unixtime
    unixtime = df['datetime'].apply(lambda t: t.timestamp())
    df.insert(loc=1, column='unixtime', value=unixtime)

    # Refine timestamp data if needed. (because some csv files have different sampling rate)
    # Strange timestamp (that of OM1901) should be corrected here.
    if correct_timestamp == 1:
        print("Strange timestamp -> Correcting timestamp ...")
        SAMPLING_RATE_25Hz = 25
        SAMPLING_RATE_31Hz = 31
        l_datetime_ = list(df['datetime'])
        l_datetime = []
        l_unixtime = []
        # Keep the value only at .000 ms, convert all the rest np.nan
        for i in range(0, len(l_datetime_)):
            if l_datetime_[i].value % 1000000000 == 0:
                l_datetime.append(l_datetime_[i])
                l_unixtime.append(l_datetime_[i].timestamp())  # unixtimeに変換
            else:
                l_datetime.append(np.nan)
                l_unixtime.append(np.nan)
        df['datetime'] = l_datetime
        df['unixtime'] = l_unixtime
        # interpolate unixtime data
        # Note that the last second is not linearly interpolated properly
        # and the unixtime (datetime) is duplicated.
        # Fill NaN values using an interpolation method.
        df['unixtime'] = df['unixtime'].interpolate(method='linear',
                                                    limit=SAMPLING_RATE_31Hz - 1)
        df['datetime'] = pd.to_datetime(df['unixtime'], unit='s')
        df['datetime'] = df['datetime'].dt.round('1L')  # round at 1 millisecond

    print("Length of df: ", len(df))

    return df


def divide_df_if_timestamp_gap_detected_2(
        df,
        gap_min_limit=125  # 2 hours + alpha
):
    ''' Divide data frame before resampling
        ( Otherwise, it takes a lot of time for resampling. )
    Args:
        df (DataFrame): a raw data
        gap_min_limit (int):
            default -> 125 min (2 hours + 5 min)
            For logbot data, there may be a gap of 10-12 hours.
            gap_min_limit of 2 hours is enough to detect the large time gap.
    Returns:
        df_list (list): a list of divided data frames
    '''
    # message
    print("Checking timestamp gap -> ", end="")

    # settings
    # SAMPLING_RATE = acc_sampling_rate
    GAP_SEC_LIMIT = 1 * gap_min_limit
    large_gap_detector = []  # bool list
    large_gap_detect_index_list = [0]  # index list: the first index should be 0

    # Check timestamps of all data
    for i in range(1, len(df)):
        diff = df['unixtime'][i] - df['unixtime'][i - 1]  # gap time in seconds (e.g. 0.040 sec)
        # if there is a gap of more than GAP_SEC_LIMIT, divide data frame
        # if diff > SAMPLING_RATE * GAP_SEC_LIMIT:  # gap_min_limit = 5: 25 * 60 * 5 = 7500 sec (125 min)
        if diff > GAP_SEC_LIMIT:  # gap_min_limit = 125
            large_gap_detector.append(True)
            large_gap_detect_index_list.append(i)
            # print(i)
        else:
            large_gap_detector.append(False)
    # print(large_gap_detect_index_list)

    # If there is more than one timestamp gap,
    # split the data frame and save them as a list
    df_list = []
    if len(large_gap_detect_index_list) > 1:
        print(str(len(large_gap_detect_index_list) - 1), "timestamp gap(s) detected -> ", end="")
        for i in range(0, len(large_gap_detect_index_list)):
            # print(i)
            if i == 0:
                df_tmp = df[0:large_gap_detect_index_list[i + 1]]
                df_list.append(df_tmp)
            elif i != 0 and i < len(large_gap_detect_index_list) - 1:
                df_tmp = df[large_gap_detect_index_list[i]:large_gap_detect_index_list[i + 1]]
                df_list.append(df_tmp)
            else:
                df_tmp = df[large_gap_detect_index_list[i]:]
                df_list.append(df_tmp)
            # print("df:", i)
            # display(df_tmp.head(3))
            # display(df_tmp.tail(3))
    else:
        df_list.append(df)
        print("No timestamp gap detected -> ", end="")

    print("N of dataframe:", len(df_list))

    return df_list

def run_resampling_and_concat_df(df_list, acc_sampling_rate, remove_sec=3, check_df=False):
    '''
    Args:
        df_list (list): a list of divided data frames
        acc_sampling_rate (int): sampling rate of acceleration data
        remove_sec (int): the first few seconds to be removed
                            due to measurement errors  (many zeros)
        check_df (bool): whether to show datafram or not
    Return:
        df (DataFrame): a data frame (combined resampled data frames)
    '''
    OUTPUT_SAMPLIGN_RATE = 25
    INTERMEDIATE_SAMPLING_RATE = 100
    start_index = acc_sampling_rate * remove_sec
    df_concat = pd.DataFrame()

    # 31Hz, df divided      -> resampling each data frames and concat
    # 31Hz, df not divided  -> resampling
    # 25Hz, df divided      -> concat
    # 25Hz, df not divided  -> none
    if acc_sampling_rate == 31:
        if len(df_list) > 1:
            for i in range(0, len(df_list)):
                # If the data frame contains more than one minute of recordings,
                # use the data frame, otherwise, discard it.
                if len(df_list[i]) > (acc_sampling_rate * 60):
                    # delete the first several seconds with noisy data
                    df_list[i] = df_list[i][start_index:]
                    df_list[i].reset_index(inplace=True)
                    # if check_df == True:
                    #     display(df_list[i].head(5))
                    df_resampled = resampling(
                        df=df_list[i],
                        intermediate_sampling_rate=INTERMEDIATE_SAMPLING_RATE,
                        output_sampling_rate=OUTPUT_SAMPLIGN_RATE
                    )
                    df_concat = pd.concat([df_concat, df_resampled])
                    if check_df == True:
                        print("Length of current df: ", len(df_resampled))
                else:
                    print("Recording time is too short -> discard the current df")
        else:
            # If the original dataframe was not divided into multiple dfs
            # resample the first dataframe in df_list
            # remove the first several seconds (remove_sec)
            df_list[0] = df_list[0][start_index:]
            df_list[0].reset_index(inplace=True)
            # if check_df == True:
            #     display(df_list[0].head(5))
            df_resampled = resampling(
                df=df_list[0],
                intermediate_sampling_rate=INTERMEDIATE_SAMPLING_RATE,
                output_sampling_rate=OUTPUT_SAMPLIGN_RATE)
            df_concat = pd.concat([df_concat, df_resampled])

    elif acc_sampling_rate == 25:
        if len(df_list) > 1:
            for i in range(0, len(df_list)):
                # If the data frame contains more than one minute of recordings,
                # use the data frame, otherwise, discard it.
                if len(df_list[i]) > (acc_sampling_rate * 60):
                    # remove the first several seconds (remove_sec)
                    df_list[i] = df_list[i][start_index:]
                    df_list[i].reset_index(inplace=True)
                    # if check_df == True:
                    #     display(df_list[i].head(5))
                    df_concat = pd.concat([df_concat, df_list[i]])
                    if check_df == True:
                        print("Length of current df: ", len(df_list[i]))
                else:
                    print("Recording time is too short -> discard the current df")
        else:
            df_list[0] = df_list[0][start_index:]
            df_list[0].reset_index(inplace=True)
            # if check_df == True:
            #     display(df_list[0].head(5))
            df_concat = pd.concat([df_concat, df_list[0]])
    else:
        print("Unknonw acc_sampling_rate")

    df = df_concat
    # Reset index because we removed the first several seconds
    df.reset_index(inplace=True, drop=True)
    df = df.drop("index", axis=1)
    # if check_df == True:
    #     display(df.head(5))
    #     print("Length of concatenated df: ", len(df))

    return df


def resampling(df, intermediate_sampling_rate=100, output_sampling_rate=25):
    '''
    Resampling data: 31Hz -> 100 Hz -> 25 Hz

    Args:
        df (DataFrame): a data frame of sensor data with original sampling rate
        intermediate_sampling_rate (int): default value = 100 (100 Hz)
        output_sampling_rate (int): default value = 25 (25 Hz)
    Return:
        df (DataFrame): a resampled data frame with sampling rate = output_sampling_rate
    '''
    # Message
    print("Resampling -> ", end="")

    # Settings
    if intermediate_sampling_rate == 100:
        asfreq_param_intermediate = "10L"
    elif intermediate_sampling_rate == 1000:
        asfreq_param_intermediate = "1L"
    else:
        print("invalid intermediate_sampling_rate")

    if output_sampling_rate == 25:
        asfreq_param_output = "40L"
    elif output_sampling_rate == 50:
        asfreq_param_output = "20L"
    else:
        print("invalid output_sampling_rate")

    # If there are any duplicate rows, delete them all (basically delete the last second)
    # If there is no milliseconds after the timstamp, the index will be duplicated.
    if np.sum(df['unixtime'].duplicated()) > 1:
        # print(len(df[df['unixtime'].duplicated()]))
        df.drop_duplicates(subset='datetime', keep=False, inplace=True)
        # print(len(df[df['unixtime'].duplicated()]))
        print("duplicated index detected -> duplicates removed")
    else:
        print("No duplicates")

    # up-sampling to 1000Hz and interpolate data
    df.set_index("datetime", inplace=True, drop=False)
    # todo, why upsampling to such a high frenquency???
    df = df.asfreq(asfreq_param_intermediate)
    # display(df[:32])
    # 1000 Hz: data points (samples) every 1 msec
    # 31 Hz: data points (samples) every about 32 msec
    # ( 25 Hz: data points (samples) every 40 msec )
    # -> The above process will result in a loss of 31 ~ 32 data points,
    # which will be filled by the linear interpolation below (limit = 60 is sufficient)
    df["acc_x"] = df["acc_x"].astype(np.float64).interpolate(method='linear', limit=60)
    df["acc_y"] = df["acc_y"].astype(np.float64).interpolate(method='linear', limit=60)
    df["acc_z"] = df["acc_z"].astype(np.float64).interpolate(method='linear', limit=60)
    df["label"] = df["label"].interpolate(method='ffill', limit=60)  # fill with previous label
    # df["animal_tag"] = df["animal_tag"].interpolate(method='ffill', limit=60) # fill with previous animal_tag
    # display(df[:32])

    #  down-sampling to 25Hz
    df = df.asfreq(asfreq_param_output)
    df = df.drop('datetime', axis=1)
    df = df.drop('unixtime', axis=1)
    df.reset_index(inplace=True)
    unixtime = df['datetime'].apply(lambda t: t.timestamp())
    df.insert(loc=1, column='unixtime', value=unixtime)
    # print("resampling completed")
    # print(len(df) % SAMPLING_RATE_25Hz)
    # display(df[:32])

    return df


def preprocess_sensor_data(df,
                           clipping=True,
                           clipping_threshold=8,
                           method="none",
                           check_df=False):
    '''
    clipping using clip method (pandas)
    standardization using sklearn.preprocessing.StandardScaler()
    interpolation using scipy.interpolate.interp1d()
    '''

    if len(df) == 0:
        print("No Sensor Data")
    else:
        # clipping (just to make sure that measurement errors are removed)
        # Note: ±8G
        # # todo: 真的需要吗？阈值怎么确定？
        # if clipping == True:
        #     df["acc_x"] = df["acc_x"].clip(lower=-clipping_threshold,
        #                                    upper=clipping_threshold)
        #     df["acc_y"] = df["acc_y"].clip(lower=-clipping_threshold,
        #                                    upper=clipping_threshold)
        #     df["acc_z"] = df["acc_z"].clip(lower=-clipping_threshold,
        #                                    upper=clipping_threshold)
        # else:
        #     print("No Clipping")

        # if check_df == True:
        #     display(df[:5])
        #     print(df.describe())

        # Note: we implemented the below pre-processing methods,
        # but did not use the standardization nor normalizetion in dl-wabc study
        if method == "standardization":
            # print("Applying standardizing to all sensor data")
            scaling_columns = ["acc_x", "acc_y", "acc_z"]
            sensor_data = df[scaling_columns]
            scaler = StandardScaler().fit(sensor_data.values)
            scaled_sensor_data = scaler.transform(sensor_data.values)
            df[scaling_columns] = scaled_sensor_data
            # if check_df == True:
            #     display(df[:5])
            #     print(df.describe())
        elif method == "normalization":
            # print("Applying min-max normalization to all sensor data")
            scaling_columns = ["acc_x", "acc_y", "acc_z"]
            sensor_data = df[scaling_columns]
            scaler = MinMaxScaler().fit(sensor_data.values)
            scaled_sensor_data = scaler.transform(sensor_data.values)
            df[scaling_columns] = scaled_sensor_data
            # if check_df == True:
            #     display(df[:5])
            #     print(df.describe())
        # elif method == "none":
        # print("No standardization nor normalization method applied")

    return df


def save_preprocessed_data(df, output_dir_path, species, animal_id, label_id_path):
    '''
    Args:
        df (DataFrame): a preprocess data frame
        output_dir_path (str): a path to save the preprocessed data frame
        species (str): species name, such as omizunagidori, umineko
                        (Japanese names of streaked shearwaters and black-tailled gulls)
        animal_id (str): animal id, such as OM2101, UM1901
        label_id_path: label file path

    Return:
        None
    '''

    if len(df) == 0:
        print("No Sensor Data -> No Data Saved")
    else:
        df = df.loc[:, ['datetime', 'unixtime',
                        'acc_x', 'acc_y', 'acc_z', 'label']]
        # if species == "omizunagidori":
        #     label_id_path = "../data/id_files/label_id_omizunagidori.csv"
        # elif species == "umineko":
        #     label_id_path = "../data/id_files/label_id_umineko.csv"

        df["label"] = df["label"].astype("object")
        df_label_id = pd.read_csv(label_id_path, low_memory=False)
        df_label_id["label"] = df_label_id["label"].astype("object")
        df_merge = pd.merge(df, df_label_id[["label", "label_id"]],
                            how='left', on='label')
        df = df_merge

        save_dir = os.path.join(output_dir_path, species)
        if os.path.exists(save_dir) == False:
            os.makedirs(save_dir)

        df_save_path = os.path.join(save_dir, str(animal_id) + ".csv")

        df.to_csv(df_save_path, index=False)
        print("Preprocessed Data Saved")
        print("-----------------------------------------------------")


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

    # # print("Preprocessed data: ", os.path.basename(preprocessed_data_path))
    # species = os.path.basename(os.path.dirname(preprocessed_data_path))
    # animal_id = os.path.basename(preprocessed_data_path).replace('.csv', '')
    # read preprocessed data
    df = pd.read_csv(preprocessed_data_path, low_memory=False)
    print("length of df: ", len(df))
    print("Extracting sliding windows ...")

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
            # print(f"npz saved X: {X_block_array.shape} label_id: {label_id_block_array.shape})

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

        # if (i+1) % 10000 == 0:
        # print(f"{i+1} -> npz saved X: {X_block_array.shape} label_id: {label_id_block_array.shape}")
