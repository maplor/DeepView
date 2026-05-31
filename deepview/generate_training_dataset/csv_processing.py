import numpy as np
import pandas as pd

from deepview.generate_training_dataset.utils import (
    divide_df_if_timestamp_gap_detected_2,
    run_resampling_and_concat_df,
)
from deepview.utils.auxiliaryfunctions import read_config


# Function to calculate velocity
def calculate_velocity(row):
    if not np.isnan(row['latitude']) and not np.isnan(row['longitude']):
        if 'prev_lat' in calculate_velocity.__dict__:
            distance = ((row['latitude'] - calculate_velocity.prev_lat)**2 +
                        (row['longitude'] - calculate_velocity.prev_lon)**2)**0.5
            velocity = distance  # You may want to divide by time if you have it
        else:
            velocity = np.nan
        calculate_velocity.prev_lat = row['latitude']
        calculate_velocity.prev_lon = row['longitude']
    else:
        velocity = np.nan
    return velocity


def z_score_normalization(df):
    # Calculate mean and standard deviation for each column
    mean_values = df.mean()
    std_deviations = df.std()

    # Apply Z-score normalization for each column
    normalized_df = (df - mean_values) / std_deviations

    return normalized_df


def format_timestamp(df):
    # if 'datetime' not in df.columns:
    s = df['timestamp'].str.replace('T', ' ').str.replace('Z', '')
    # df = df.drop('timestamp', axis=1)
    s_datetime = pd.to_datetime(s)  # to datetime64[ns]
    df.insert(loc=0, column='datetime', value=s_datetime)
    # round at 1 millisecond
    df['datetime'] = df['datetime'].dt.round('1L')
    # unixtime
    unixtime = df['datetime'].apply(lambda t: t.timestamp())
    df.insert(loc=1, column='unixtime', value=unixtime)
    return df


def process_gps(df):
    # identify if gps exists
    # if exists, calculate velocity and angle
    gps_len = len(df)
    df_columns = df.columns
    if ('latitude' in df_columns) or\
        ('longitude' in df_columns):
        # Extract rows where both latitude and longitude are not NaN
        df_non_nan = df.dropna(subset=['latitude', 'longitude'])

        # get sampling rate of GPS signal,newlen*oldHz/oldlen得到GPShz，所以这里传newlen
        gps_len = len(df_non_nan)

        # Calculate differences, handling NaN by filling with zeros
        df_non_nan['lat_diff'] = np.radians(df_non_nan['latitude'].diff())
        df_non_nan['lon_diff'] = np.radians(df_non_nan['longitude'].diff())

        # Convert latitude to radians, handling NaN by filling with zeros
        df_non_nan['lat1'] = np.radians(df_non_nan['latitude'].shift())
        df_non_nan['lat2'] = np.radians(df_non_nan['latitude'])

        # Calculate time difference in seconds
        df_non_nan['timestamp'] = pd.to_datetime(df_non_nan['timestamp'])
        df_non_nan['time_diff'] = df_non_nan['timestamp'].diff().dt.total_seconds()

        # Haversine formula
        a = (np.sin(df_non_nan['lat_diff'] / 2) ** 2 +
             np.cos(df_non_nan['lat1']) * np.cos(df_non_nan['lat2']) * np.sin(df_non_nan['lon_diff'] / 2) ** 2)
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        R = 6371000  # Earth radius in meters
        df_non_nan['distance'] = R * c
        # Calculate velocity (m/s)
        df_non_nan['GPS_velocity'] = df_non_nan['distance'] / df_non_nan['time_diff']

        # Calculate bearing
        x = np.sin(df_non_nan['lon_diff']) * np.cos(df_non_nan['lat2'])
        y = (np.cos(df_non_nan['lat1']) * np.sin(df_non_nan['lat2']) -
             np.sin(df_non_nan['lat1']) * np.cos(df_non_nan['lat2']) * np.cos(df_non_nan['lon_diff']))
        initial_bearing = np.arctan2(x, y)
        initial_bearing = np.degrees(initial_bearing)
        df_non_nan['GPS_bearing'] = (initial_bearing + 360) % 360

        # Merge velocity and bearing back to the original dataframe
        df = df.merge(df_non_nan[['GPS_velocity', 'GPS_bearing']], left_index=True, right_index=True, how='left')

    return df, gps_len


# ---------------------------read raw sensor data--------------------------------

def read_process_csv(root, file, sample_rate=25):
    """
    data most contains rows: timestamp and label
    timestamp: transfer string to unixtime
    """
    df = pd.read_csv(file)
    # add velocity and angles if GPS sensor exists
    df, gps_len = process_gps(df)

    # Create a new column 'label_flag' where NaN rows in 'label' are 0 and others are 1
    df['label_flag'] = df['label'].notna().astype(int)
    # 如果整个文件都没有标签，那直接给label赋值为unknown
    if 1 not in df['label_flag'].unique():
        df['label'] = 'unknown'

    # fulfill nan values
    df = df.bfill().ffill()

    df = format_timestamp(df)  # 在这里timestamp字符串生成datetime和unixtime


    # calculate sampling rate, the input is timestamp
    INTERMEDIATE_SAMPLING_RATE = int(1/np.mean(np.diff(df['unixtime'].values)))
    if INTERMEDIATE_SAMPLING_RATE == 0:  # if the sampling rate is the same, should be 1
        INTERMEDIATE_SAMPLING_RATE = 1

    # divide data if time_gap exists
    df_list = divide_df_if_timestamp_gap_detected_2(df, int(sample_rate) * 5 * 60)

    # change sampling rate
    newdf = run_resampling_and_concat_df(df_list,
                                      int(sample_rate),
                                      INTERMEDIATE_SAMPLING_RATE,
                                      remove_sec=3,
                                      check_df=False)

    root_cfg = read_config(root.config)
    label_dict = root_cfg['label_dict']
    # create label_id (int) for label (str)
    newdf['label_id'] = newdf['label'].map(label_dict)  # todo, very long time
    # df['label_id'] = df['label'].map(label_str2num)
    # 因为角度是一段距离内的角度累计，需要除以时间
    gps_sampling_rate = (gps_len * float(sample_rate)) / len(newdf)
    newdf['GPS_bearing'] = newdf['GPS_bearing'] * (gps_sampling_rate)

    # # process timestamp
    # newdf['timestamp'] = pd.to_datetime(newdf['unixtime'],
    #                                     unit='s'.dt.strftime('%Y-%m-%dT%H:%M:%S.%fZ'))
    newdf['index'] = newdf.index
    return newdf