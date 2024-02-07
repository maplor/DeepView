

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
import pickle
from datetime import datetime

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
    attempt_to_make_folder,
)

from deepview.generate_training_dataset.utils import (
    label_str_list,
    label_str2num,
    GRAVITATIONAL_ACCELERATION,
    date_format,
    GYROSCOPE_SCALE,
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


def read_process_csv(file, filename, string_to_value):
    # load data
    df = pd.read_csv(file)
    needed_column = ['logger_id', 'animal_tag', 'timestamp', 'acc_x', 'acc_y', 'acc_z',
                     'latitude', 'longitude', 'gyro_x', 'gyro_y', 'gyro_z',
                     'mag_x', 'mag_y', 'mag_z', 'illumination', 'pressure', 'temperature',
                     'activity_class', 'label']

    # normalize data
    cols = ["acc_x", "acc_y", "acc_z"]
    df[cols] = z_score_normalization(df[cols])
    # df[cols] = df[cols] / GRAVITATIONAL_ACCELERATION
    cols_1 = ["gyro_x", "gyro_y", "gyro_z"]
    df[cols_1] = z_score_normalization(df[cols_1])

    # calculate
    tmpdata = df[needed_column]
    # nan = 0
    tmpdata['labelid'] = tmpdata['label'].apply(lambda x: label_str2num[x] if x in label_str_list else 0)
    tmpdata['velocity'] = tmpdata.apply(calculate_velocity, axis=1)
    tmpdata['timestamp'] = tmpdata['timestamp'].map(
        lambda x: pd.to_datetime(datetime.strptime(x, date_format)) + np.timedelta64(9,
                                                                                     'h'))  # adjust time zone to Japan
    tmpdata['filename'] = filename

    # Convert numpy.datetime64 to Unix timestamp (seconds since 1970-01-01 00:00:00 UTC)
    tmpdata['time'] = tmpdata.timestamp.map(lambda x: (x - np.datetime64('1970-01-01T00:00:00')) / np.timedelta64(1, 'us'))
    tmpdata['domainid'] = tmpdata['filename'].map(string_to_value)
    return tmpdata

def preprocess_datasets(cfg, trainingsetfolder_full):
    """
    This file preprocess raw sensor data and store new files into 'labeled_data' folder
    #------by xia---------
    the csv file contains every sample filename and the label positions (by human)
    """

    AnnotationData = []
    data_path = Path(os.path.join(cfg["project_path"], "labeled-data"))
    files = cfg["file_sets"].keys()

    # get filenames
    filenames = []
    for file in files:
        _, filename, _ = _robust_path_split(file)
        filenames.append(filename)
    string_to_value = {string: index + 1 for index, string in enumerate(filenames)}

    for file in files:
        _, filename, _ = _robust_path_split(file)
        file_path = os.path.join(
            data_path / filename, f'CollectedData_{cfg["scorer"]}.pkl'
        )
        # reading raw data here...
        try:
            if os.path.isfile(file_path):
                with open(file_path, 'rb') as f:
                    data = pickle.load(f)
            else:
                data = read_process_csv(file, filename, string_to_value)  # return dataframe
                with open(file_path, 'wb') as f:
                     pickle.dump(data, f)
            conversioncode.guarantee_multiindex_rows(data)
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
    # bodyparts = cfg["bodyparts"]
    # AnnotationData = AnnotationData.reindex(
    #     bodyparts, axis=1, level=AnnotationData.columns.names.index("bodyparts")
    # )

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


def create_training_dataset(
    config,
    # num_shuffles=1,
    # Shuffles=None,
    windows2linux=False,
    userfeedback=False,
    trainIndices=None,
    testIndices=None,
    net_type=None,
    augmenter_type=None,
    posecfg_template=None,
    superanimal_name="",
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
    cfg = auxiliaryfunctions.read_config(config)
    # dlc_root_path = auxiliaryfunctions.get_deepview_path()


   # remove if multianimal
   #  scorer = cfg["scorer"]  # part of project name, string
    project_path = cfg["project_path"]
    # Create path for training sets & store data there. Path: training_datasets/iteration_0/..
    trainingsetfolder = auxiliaryfunctions.get_training_set_folder(
        cfg
    )  # Create folder for above path. Path concatenation OS platform independent
    auxiliaryfunctions.attempt_to_make_folder(
        Path(os.path.join(project_path, str(trainingsetfolder))), recursive=True
    )

    # preprocess data
    # print('preprocessing data using min-max norm...')

    Data = preprocess_datasets(
        cfg,
        Path(os.path.join(project_path, trainingsetfolder)),
    )
    if Data is None:
        print('No data preprocessed.')
        return
    # Data = Data[scorer]  # extract labeled data, dataframe


    # 2. load模型训练参数，目前可以不写

    # loading & linking pretrained models
    if net_type is None:  # loading & linking pretrained models
        net_type = cfg.get("default_net_type", "resnet_50")
    else:
        if (
            "resnet" in net_type
            or "CNN_AE" in net_type
            # or "efficientnet" in net_type
            # or "dlcrnet" in net_type
        ):
            pass
        else:
            raise ValueError("Invalid network type:", net_type)


        ################################################################################
        # Creating file structure for training &
        # Test files as well as pose_yaml files (containing training and testing information)
        #################################################################################
        modelfoldername = auxiliaryfunctions.get_model_folder(cfg)
        auxiliaryfunctions.attempt_to_make_folder(
            Path(config).parents[0] / modelfoldername, recursive=True
        )
        auxiliaryfunctions.attempt_to_make_folder(
            str(Path(config).parents[0] / modelfoldername) + "/train"
        )
        auxiliaryfunctions.attempt_to_make_folder(
            str(Path(config).parents[0] / modelfoldername) + "/test"
        )

        path_train_config = str(
            os.path.join(
                cfg["project_path"],
                Path(modelfoldername),
                "train",
                "model_cfg.yaml",
            )
        )
        path_test_config = str(
            os.path.join(
                cfg["project_path"],
                Path(modelfoldername),
                "test",
                "model_cfg.yaml",
            )
        )
        # str(cfg['proj_path']+'/'+Path(modelfoldername) / 'test'  /  'pose_cfg.yaml')
        # bodyparts = ['label1', 'label2', 'label3']
        # dlcparent_path = auxiliaryfunctions.get_deepview_path()
        # model_path = auxfun_models.check_for_weights(
        #     net_type, Path(dlcparent_path))

        # Make training file! 读文件路径，是存到training datasets里的两个文件
        trainingsetfolder = auxiliaryfunctions.get_training_set_folder(cfg)
        (
            datafilename,
            metadatafilename,
        ) = auxiliaryfunctions.get_data_and_metadata_filenames(
            trainingsetfolder, cfg
        )
        items2change = {
            "dataset": datafilename,
            # "metadataset": metadatafilename,
            # "num_joints": len(bodyparts),
            # "all_joints": [[i] for i in range(len(bodyparts))],
            # "all_joints_names": [str(bpt) for bpt in bodyparts],
            "init_weights": '',
            "project_path": str(cfg["project_path"]),
            "net_type": net_type,
            "dataset_type": augmenter_type,
        }

        items2drop = {}
        if augmenter_type == "scalecrop":
            # these values are dropped as scalecrop
            # doesn't have rotation implemented
            items2drop = {"rotation": 0, "rotratio": 0.0}
        # Also drop maDLC smart cropping augmentation parameters
        for key in ["pre_resize", "crop_size", "max_shift", "crop_sampling"]:
            items2drop[key] = None

        dvparent_path = auxiliaryfunctions.get_deepview_path()
        defaultconfigfile = os.path.join(dvparent_path, "model_cfg.yaml")
        trainingdata = MakeTrain_yaml(
            items2change, path_train_config, defaultconfigfile, items2drop
        )

        keys2save = [
            "dataset",
            # "num_joints",
            # "all_joints",
            # "all_joints_names",
            "net_type",
            "init_weights",
            "global_scale",
            "location_refinement",
            "locref_stdev",
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
