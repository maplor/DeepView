
import math
import logging
import os
import os.path
import warnings

from functools import lru_cache
from pathlib import Path
# from PIL import Image

import numpy as np
import pandas as pd
import yaml
import glob
import pickle
from datetime import datetime
import sqlite3
from deepview.clustering_pytorch import training
# from deeplabcut.utils import (
#     auxiliaryfunctions,
#     conversioncode,
#     auxfun_models,
#     auxfun_multianimal,
# )
# from deeplabcut.utils.auxfun_videos import VideoReader
# from deeplabcut.pose_estimation_tensorflow.config import load_config
# from deeplabcut.modelzoo.utils import parse_available_supermodels


from deepview.utils import (
    auxiliaryfunctions,
    conversioncode,
    get_deepview_path,
    read_plainconfig,
    get_training_set_folder,
    get_unsupervised_set_folder,
    attempt_to_make_folder,
)

from deepview.generate_training_dataset.utils import (
    label_str_list,
    label_str2num,
    GRAVITATIONAL_ACCELERATION,
    divide_df_if_timestamp_gap_detected_2,
    run_resampling_and_concat_df,
    date_format,
    GYROSCOPE_SCALE,
)

from deepview.utils.auxiliaryfunctions import (
    read_config,
)


def _robust_path_split(path):
    sep = "\\" if "\\" in path else "/"
    splits = path.rsplit(sep, 1)
    if len(splits) == 1:
        parent = "."
        file = splits[0]
    elif len(splits) == 2:
        parent, file = splits
    else:
        raise ("Unknown filepath split for path {}".format(path))
    filename, ext = os.path.splitext(file)
    return parent, filename, ext


def merge_annotateddatasets(cfg, trainingsetfolder_full):
    """
    Merges all the h5 files for all labeled-datasets (from individual videos).

    This is a bit of a mess because of cross platform compatibility.

    Within platform comp. is straightforward. But if someone labels on windows and wants to train on a unix cluster or colab...

    #------by xia---------
    the csv file contains every sample filename and the label positions (by human)
    """

    AnnotationData = []
    data_path = Path(os.path.join(cfg["project_path"], "labeled-data"))
    files = cfg["file_sets"].keys()

    ## removed for loop, assume only one file
    _, filename, _ = _robust_path_split(files)
    file_path = os.path.join(
        data_path / filename, f'CollectedData_{cfg["scorer"]}.pkl'
    )
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        conversioncode.guarantee_multiindex_rows(data)
        if data.columns.levels[0][0] != cfg["scorer"]:
            print(
                f"{file_path} labeled by a different scorer. This data will not be utilized in training dataset creation. If you need to merge datasets across scorers, see https://github.com/DeepLabCut/DeepLabCut/wiki/Using-labeled-data-in-DeepLabCut-that-was-annotated-elsewhere-(or-merge-across-labelers)"
            )
        AnnotationData.append(data)
    except FileNotFoundError:
        print(file_path, " not found (perhaps not annotated).")

    if not len(AnnotationData):
        print(
            "Annotation data was not found by splitting video paths (from config['video_sets']). An alternative route is taken..."
        )
        AnnotationData = conversioncode.merge_windowsannotationdataONlinuxsystem(cfg)
        if not len(AnnotationData):
            print("No data was found!")
            return

    AnnotationData = pd.concat(AnnotationData).sort_index()

    # When concatenating DataFrames with misaligned column labels,
    # all sorts of reordering may happen (mainly depending on 'sort' and 'join')
    # Ensure the 'bodyparts' level agrees with the order in the config file.
    bodyparts = cfg["bodyparts"]
    AnnotationData = AnnotationData.reindex(
        bodyparts, axis=1, level=AnnotationData.columns.names.index("bodyparts")
    )

    # save data, data is dataframe, include raw data and annotations (by human)
    # (deeplabcut) row: sample name, column: label types
    # (deepview) todo: change the dataframe, I want to label the start and end time of activities for each channel
    filename = os.path.join(trainingsetfolder_full, f'CollectedData_{cfg["scorer"]}')
    # AnnotationData.to_hdf(filename + ".h5", key="df_with_missing", mode="w")
    with open(filename + ".pkl", 'wb') as f:
        pickle.dump(AnnotationData, f)
    # human readable. human labels of every samples
    AnnotationData.to_csv(filename + ".csv")

    return AnnotationData
    # return splits

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

def preprocess_datasets(root, progress_update, cfg, allsetfolder, sample_rate):
    """
    for each sensor data file, preprocess it and save as pkl file into
    rootpath/unsupervised-datasets/allDataSet folder
    raw sensor data path saved at: rootpath/config.yaml, file_sets parameter
    """

    # get raw sensor data full paths
    filenames = cfg["file_sets"]
    sorted_filenames = sorted(filenames)
    filenum = len(filenames)
    # read data
    # AnnotationData = []
    for idx, file in enumerate(sorted_filenames):
        progress_update.emit(int((idx+1) / filenum * 100))

        parent, filename, _ = _robust_path_split(file)
        file_path = os.path.join(
            allsetfolder, filename + f'_%sHz.pkl ' % sample_rate
            # allsetfolder, filename+f'_{cfg["scorer"]}.pkl'
        )
        # reading raw data here...
        # TODO 这里有个bug，当sampling rate变化时，不会触发重新处理数据的bug
        try:
            if os.path.isfile(file_path):
                print('Raw sensor data already exists at %s ' % file_path)
                # with open(file_path, 'rb') as f:
                #     data = pickle.load(f)
            else:
                data = read_process_csv(root, file, sample_rate)  # return dataframe
                with open(file_path, 'wb') as f:
                    pickle.dump(data, f)
                db_path = os.path.join(cfg["project_path"], "db", "database.db")
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()
                conn.execute('BEGIN TRANSACTION')
                try:
                    # 选择需要的列
                    columns = [
                        'logger_id', 'animal_tag', 'datetime', 'timestamp', 'unixtime', 'latitude', 'longitude',
                        'acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z', 
                        'mag_x', 'mag_y', 'mag_z', 'illumination', 'pressure', 'GPS_velocity', 
                        'GPS_bearing', 'temperature', 'label_id', 'label', 'label_flag'
                    ]
                    data = data[columns]
                    # 填充缺失值
                    data.fillna({
                        'logger_id': 'default_logger_id',
                        'animal_tag': 'default_animal_tag',
                        'datetime': '1970-01-01 00:00:00.000',
                        'timestamp': 'default_timestamp',
                        'unixtime': 0,
                        'latitude': 0.0,
                        'longitude': 0.0,
                        'acc_x': 0.0,
                        'acc_y': 0.0,
                        'acc_z': 0.0,
                        'gyro_x': 0.0,
                        'gyro_y': 0.0,
                        'gyro_z': 0.0,
                        'mag_x': 0.0,
                        'mag_y': 0.0,
                        'mag_z': 0.0,
                        'illumination': 0.0,
                        'pressure': 0.0,
                        'GPS_velocity': 0.0,
                        'GPS_bearing': 0.0,
                        'temperature': 0.0,
                        'label_id': 0,
                        'label': 'default_label',
                        'label_flag': 0
                    }, inplace=True)

                    # # 填充timestamp列
                    # data.loc[data['timestamp'] == 'default_timestamp', 'timestamp'] = data['datetime'].apply(
                    #     lambda x: x.replace(' ', 'T') + 'Z')

                    # 格式化 datetime 列
                    data['datetime'] = pd.to_datetime(data['datetime']).dt.strftime('%Y-%m-%d %H:%M:%S.%f').str[:-3]

                    # 转换为数组
                    data = data.to_records(index=False)
                    cursor.executemany('''
                    INSERT INTO raw_data (
                        logger_id, animal_tag, datetime, timestamp, unixtime, latitude, longitude,
                        acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, mag_x, mag_y, mag_z,
                        illumination, pressure, GPS_velocity, GPS_bearing, temperature,
                        label_id, label, label_flag
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', data)
                    # 提交事务
                    conn.commit()
                except Exception as e:
                    conn.rollback()
                    print(f"错误: {e}")
                conn.close()

                # with open(file_path, 'wb') as f:
                #     pickle.dump(data, f)
            # conversioncode.guarantee_multiindex_rows(data)
            # AnnotationData.append(data)
        except FileNotFoundError:
            print(file_path, " not found raw sensor data, please create data first.")

    # if not len(AnnotationData):
    #     print("No data was found!")
    #     return
    return


def create_training_dataset(
        root,
    progress_update,
    config,
    sample_rate=None,
):
    """Creates a training dataset.
    Returns
    -------
    list(tuple) or None
        If training dataset was successfully created, a list of tuples is returned.
        The first two elements in each tuple represent the training fraction and the
        shuffle value. The last two elements in each tuple are arrays of integers
        representing the training and test indices.

        Returns None if training dataset could not be created.

    Notes
    -----
    Use the function ``add_new_videos`` at any stage of the project to add more videos
    to the project.

    Examples
    --------

    Linux/MacOS

    >>> deeplabcut.create_training_dataset(
            '/analysis/project/reaching-task/config.yaml', num_shuffles=1,
        )

    Windows

    >>> deeplabcut.create_training_dataset(
            'C:\\Users\\Ulf\\looming-task\\config.yaml', Shuffles=[3,17,5],
        )
    """

    # Loading metadata from config.yaml file:
    cfg = auxiliaryfunctions.read_config(config)  # project_path/config.yaml

   # remove if multianimal
   #  scorer = cfg["scorer"]  # part of project name, string
    project_path = cfg["project_path"]
    # Create path for training sets & store data there. Path: training_datasets/iteration_0/..
    trainingsetfolder = auxiliaryfunctions.get_unsupervised_set_folder()
    # Create folder for above path. Path concatenation OS platform independent
    auxiliaryfunctions.attempt_to_make_folder(
        Path(os.path.join(project_path, str(trainingsetfolder))), recursive=True
    )  # WindowsPath('C:/Users/dell/Desktop/xia-logbot-2024-04-19/unsupervised-datasets/allDataSet')

    # preprocess data
    # print('preprocessing data using min-max norm...')

    preprocess_datasets(
        root,
        progress_update,
        cfg,
        Path(os.path.join(project_path, trainingsetfolder)),
        sample_rate,
    )


    ################################################################################
    # Creating file structure for unsupervised/supervised training &
    # Test files as well as pose_yaml files (containing training and testing information)
    #################################################################################
    unsup_modelfoldername = auxiliaryfunctions.get_unsup_model_folder(cfg)

    auxiliaryfunctions.attempt_to_make_folder(
        str(Path(config).parents[0] / unsup_modelfoldername)
    )  # for all data

    path_unsup_train_config = str(
        os.path.join(
            cfg["project_path"],
            Path(unsup_modelfoldername),
            "model_cfg.yaml",
        )
    )
    # Make training file! 读文件路径，是存到training datasets里的两个文件
    trainingsetfolder = auxiliaryfunctions.get_training_set_folder(cfg)
    (
        datafilename,
        metadatafilename,
    ) = auxiliaryfunctions.get_data_and_metadata_filenames(
        trainingsetfolder, cfg
    )
    items2change = {
        "project_path": str(cfg["project_path"]),  # 最外层路径
        "dataset": Path(os.path.join(project_path, trainingsetfolder)),
        "sample_rate": int(sample_rate),
        "net_type": "AE_CNN",
        "lr_init": 0.0001,
        'batch_size': 32,
        'max_epochs': 100,
        'data_length': 180,
        # 'data_colunms': ['acc_x', 'acc_y', 'acc_z']
    }
    dvparent_path = auxiliaryfunctions.get_deepview_path()
    defaultconfigfile = os.path.join(dvparent_path, "model_cfg.yaml")
    _ = MakeTrain_yaml(
        items2change, path_unsup_train_config, defaultconfigfile)


    sup_modelfoldername = auxiliaryfunctions.get_sup_model_folder(cfg)

    auxiliaryfunctions.attempt_to_make_folder(
        str(Path(config).parents[0] / sup_modelfoldername) + "/train"
    )
    auxiliaryfunctions.attempt_to_make_folder(
        str(Path(config).parents[0] / sup_modelfoldername) + "/test"
    )


    path_train_config = str(
        os.path.join(
            cfg["project_path"],
            Path(sup_modelfoldername),
            "train",
            "model_cfg.yaml",
        )
    )
    path_test_config = str(
        os.path.join(
            cfg["project_path"],
            Path(sup_modelfoldername),
            "test",
            "model_cfg.yaml",
        )
    )

    # Make training file! 读文件路径，是存到training datasets里的两个文件
    trainingsetfolder = auxiliaryfunctions.get_training_set_folder(cfg)
    (
        datafilename,
        metadatafilename,
    ) = auxiliaryfunctions.get_data_and_metadata_filenames(
        trainingsetfolder, cfg
    )


    dvparent_path = auxiliaryfunctions.get_deepview_path()
    defaultconfigfile = os.path.join(dvparent_path, "model_cfg.yaml")
    trainingdata = MakeTrain_yaml(
        items2change, path_train_config, defaultconfigfile)

    keys2save = [
        "dataset",
        "net_type",
        "init_weights",
    ]
    MakeTest_pose_yaml(trainingdata, keys2save, path_test_config)
    print(
        "The training dataset is successfully created. Use the function 'train_network' to start training. Happy training!"
    )
    return

# --------------make yaml files-------------------
def ParseYaml(configfile):
    raw = open(configfile).read()
    docs = []
    for raw_doc in raw.split("\n---"):
        try:
            docs.append(yaml.load(raw_doc, Loader=yaml.SafeLoader))
        except SyntaxError:
            docs.append(raw_doc)
    return docs


def MakeTrain_yaml(
    itemstochange, saveasconfigfile, defaultconfigfile, items2drop={}
):
    docs = ParseYaml(defaultconfigfile)
    for key in items2drop.keys():
        # print(key, "dropping?")
        if key in docs[0].keys():
            docs[0].pop(key)

    for key in itemstochange.keys():
        docs[0][key] = itemstochange[key]

    with open(saveasconfigfile, "w") as f:
        yaml.dump(docs[0], f)

    return docs[0]

def MakeTest_pose_yaml(
    dictionary,
    keys2save,
    saveasfile,
    nmsradius=None,
    minconfidence=None,
    sigma=None,
    locref_smooth=None,
):
    dict_test = {}
    for key in keys2save:
        dict_test[key] = dictionary[key]

    # adding important values for multianiaml project:
    if nmsradius is not None:
        dict_test["nmsradius"] = nmsradius
    if minconfidence is not None:
        dict_test["minconfidence"] = minconfidence
    if sigma is not None:
        dict_test["sigma"] = sigma
    if locref_smooth is not None:
        dict_test["locref_smooth"] = locref_smooth

    dict_test["scoremap_dir"] = "test"
    with open(saveasfile, "w") as f:
        yaml.dump(dict_test, f)
