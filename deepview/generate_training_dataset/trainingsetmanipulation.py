
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
from deepview.generate_training_dataset.annotations import (
    _robust_path_split,
    merge_annotateddatasets,
)
from deepview.generate_training_dataset.csv_processing import (
    calculate_velocity,
    format_timestamp,
    process_gps,
    read_process_csv,
    z_score_normalization,
)
from deepview.generate_training_dataset.yaml_config import (
    MakeTest_pose_yaml,
    MakeTrain_yaml,
    ParseYaml,
)


logger = logging.getLogger(__name__)


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
                logger.info("Raw sensor data already exists at %s", file_path)
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
                except Exception:
                    conn.rollback()
                    logger.exception("Failed to insert raw sensor data into database")
                conn.close()

                # with open(file_path, 'wb') as f:
                #     pickle.dump(data, f)
            # conversioncode.guarantee_multiindex_rows(data)
            # AnnotationData.append(data)
        except FileNotFoundError:
            logger.warning("%s not found raw sensor data, please create data first.", file_path)

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
    logger.info(
        "The training dataset is successfully created. Use the function 'train_network' to start training. Happy training!"
    )
    return
