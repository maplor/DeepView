# This Python file uses the following encoding: utf-8
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

label_str2num = {}
# label_str2num['stationary'] = 100
# label_str2num['ground_stationary'] = 100
# label_str2num['preening'] = 100
# label_str2num['ground_active'] = 200
# label_str2num['bathing'] = 300
# label_str2num['bathing_poss'] = 300
# label_str2num['flying_active'] = 400
# label_str2num['flying_passive'] = 400
# label_str2num['foraging_poss'] = 500
# label_str2num['poss_foraging'] = 500
# label_str2num['foraging_fish_poss'] = 600
# label_str2num['foraging_fish_poss'] = 600
# label_str2num['foraging_insect_poss'] = 700
# label_str2num['forgaing_insect'] = 700
# label_str2num['foraging_non'] = 700
# label_str2num['foraging_steal'] = 700
# label_str2num['body_shaking'] = 700
# label_str2num['unknown'] = 700
# label_str2num['nan'] = 0
# label_str_list = list(label_str2num.keys())

# omizunagidori
#D:\logbot-data\OstukaPaperData\data\id-files\label_id_omizunagidori.csv
label_str2num['stationary'] = 200
label_str2num['preening'] = 201
label_str2num['bathing'] = 300

label_str2num['flight_take_off'] = 400
label_str2num['flight_cruising'] = 401
label_str2num['foraging_dive'] = 501
label_str2num['surface_seizing'] = 502
label_str_list = list(label_str2num.keys())


GRAVITATIONAL_ACCELERATION = 9.80665
date_format = '%Y-%m-%dT%H:%M:%S.%fZ'
GYROSCOPE_SCALE = 10


#---------------------------logbot--------------------------
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

def run_resampling_and_concat_df(df_list,
                                 acc_sampling_rate,
                                 INTERMEDIATE_SAMPLING_RATE=100,
                                 remove_sec=3,
                                 check_df=False):
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
    OUTPUT_SAMPLIGN_RATE = acc_sampling_rate
    # OUTPUT_SAMPLIGN_RATE = 25
    # INTERMEDIATE_SAMPLING_RATE = 100
    start_index = acc_sampling_rate * remove_sec
    df_concat = pd.DataFrame()

    # 31Hz, df divided      -> resampling each data frames and concat
    # 31Hz, df not divided  -> resampling
    # 25Hz, df divided      -> concat
    # 25Hz, df not divided  -> none
    # if acc_sampling_rate == 31:
    if acc_sampling_rate != INTERMEDIATE_SAMPLING_RATE:
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

    elif acc_sampling_rate == INTERMEDIATE_SAMPLING_RATE:
    # elif acc_sampling_rate == 25:
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

    # df = df_concat
    # Reset index because we removed the first several seconds
    df_concat.reset_index(inplace=True, drop=True)
    if 'index' in df_concat.columns:
        df_concat = df_concat.drop("index", axis=1)
    # if check_df == True:
    #     display(df.head(5))
    #     print("Length of concatenated df: ", len(df))

    return df_concat


def check_if_has_str(listdata):
    # check if a list contains string values
    for l in listdata:
        if type(l) == str:
            return True
    return False

def resampling(df, intermediate_sampling_rate=100, output_sampling_rate=25):
    if np.sum(df['unixtime'].duplicated()) > 1:
        # print(len(df[df['unixtime'].duplicated()]))
        df.drop_duplicates(subset='datetime', keep=False, inplace=True)
        # print(len(df[df['unixtime'].duplicated()]))
        print("duplicated index detected -> duplicates removed")
    else:
        print("No duplicates")

    # Generate original time indices
    original_time = np.arange(len(df)) / intermediate_sampling_rate
    # Generate new time indices
    new_length = int(len(df) * output_sampling_rate / intermediate_sampling_rate)
    new_time = np.arange(new_length) / output_sampling_rate
    # Create a new DataFrame to store the resampled values
    resampled_df = pd.DataFrame(index=new_time)
    for column in df.columns:
        # Step 1: Extract unique characters from the column
        unique_chars = df[column].unique()

        if check_if_has_str(unique_chars):

            # Step 2: Create a dictionary that maps each character to a unique integer
            char_to_int = {char: i for i, char in enumerate(unique_chars, start=1)}
            # Step 3: Use the dictionary to replace the characters with their corresponding integers
            df[column] = df[column].map(char_to_int)

            # Create interpolation function for each column
            interp_func = interp1d(original_time, df[column].values, kind='linear', fill_value='extrapolate')
            # Generate new values at the new sampling rate
            resampled_df[column] = interp_func(new_time)

            # Step 4: Create an inverse mapping dictionary
            int_to_char = {v: k for k, v in char_to_int.items()}
            # Step 5: Use the inverse mapping dictionary to convert the integers back to the original characters
            resampled_df[column] = resampled_df[column].map(int_to_char)
        else:
            # Create interpolation function for each column
            interp_func = interp1d(original_time, df[column].values, kind='linear', fill_value='extrapolate')
            # Generate new values at the new sampling rate
            resampled_df[column] = interp_func(new_time)

    resampled_df.drop(columns=['index'], inplace=True)
    return resampled_df

def resampling_otsuka(df, intermediate_sampling_rate=100, output_sampling_rate=25):
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
