import os
import logging

import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)


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
    target = "_lb"
    index = animal_tag.find(target)
    animal_tag = animal_tag[:index]

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
    logger.info("Reading raw data")

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
        logger.info("Strange timestamp detected; correcting timestamp")
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

    logger.info("Length of df: %s", len(df))

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
    logger.info("Checking timestamp gap")

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
        else:
            large_gap_detector.append(False)

    # If there is more than one timestamp gap,
    # split the data frame and save them as a list
    df_list = []
    if len(large_gap_detect_index_list) > 1:
        logger.info("%s timestamp gap(s) detected", len(large_gap_detect_index_list) - 1)
        for i in range(0, len(large_gap_detect_index_list)):
            if i == 0:
                df_tmp = df[0:large_gap_detect_index_list[i + 1]]
                df_list.append(df_tmp)
            elif i != 0 and i < len(large_gap_detect_index_list) - 1:
                df_tmp = df[large_gap_detect_index_list[i]:large_gap_detect_index_list[i + 1]]
                df_list.append(df_tmp)
            else:
                df_tmp = df[large_gap_detect_index_list[i]:]
                df_list.append(df_tmp)
            # display(df_tmp.head(3))
            # display(df_tmp.tail(3))
    else:
        df_list.append(df)
        logger.info("No timestamp gap detected")

    logger.info("N of dataframe: %s", len(df_list))

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
                        logger.info("Length of current df: %s", len(df_resampled))
                else:
                    logger.warning("Recording time is too short; discarding the current df")
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
                        logger.info("Length of current df: %s", len(df_list[i]))
                else:
                    logger.warning("Recording time is too short; discarding the current df")
        else:
            df_list[0] = df_list[0][start_index:]
            df_list[0].reset_index(inplace=True)
            # if check_df == True:
            #     display(df_list[0].head(5))
            df_concat = pd.concat([df_concat, df_list[0]])
    else:
        logger.warning("Unknown acc_sampling_rate: %s", acc_sampling_rate)

    df = df_concat
    # Reset index because we removed the first several seconds
    df.reset_index(inplace=True, drop=True)
    df = df.drop("index", axis=1)
    # if check_df == True:
    #     display(df.head(5))

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
    logger.info("Resampling")

    # Settings
    if intermediate_sampling_rate == 100:
        asfreq_param_intermediate = "10L"
    elif intermediate_sampling_rate == 1000:
        asfreq_param_intermediate = "1L"
    else:
        logger.warning("Invalid intermediate_sampling_rate: %s", intermediate_sampling_rate)

    if output_sampling_rate == 25:
        asfreq_param_output = "40L"
    elif output_sampling_rate == 50:
        asfreq_param_output = "20L"
    else:
        logger.warning("Invalid output_sampling_rate: %s", output_sampling_rate)

    # If there are any duplicate rows, delete them all (basically delete the last second)
    # If there is no milliseconds after the timstamp, the index will be duplicated.
    if np.sum(df['unixtime'].duplicated()) > 1:
        df.drop_duplicates(subset='datetime', keep=False, inplace=True)
        logger.info("Duplicated index detected; duplicates removed")
    else:
        logger.info("No duplicates")

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
    # display(df[:32])

    return df