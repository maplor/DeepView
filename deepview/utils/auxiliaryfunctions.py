
import os
from glob import glob
import typing
import pickle
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import ruamel.yaml.representer
import yaml
from ruamel.yaml import YAML


def create_config_template(multianimal=False):
    """
    Creates a template for config.yaml file. This specific order is preserved while saving as yaml file.
    """
    yaml_str = """\
    # Project definitions (do not edit)
        Task:
        scorer:
        date:
        multianimalproject:
        identity:
        \n
    # Project path (change when moving around)
        project_path:
        \n
    # Annotation data set configuration (and individual video cropping parameters)
        video_sets:
        bodyparts:
        \n
    # Fraction of video to start/stop when extracting frames for labeling/refinement
        start:
        stop:
        numframes2pick:
        \n
    # Plotting configuration
        skeleton:
        skeleton_color:
        pcutoff:
        dotsize:
        alphavalue:
        colormap:
        \n
    # Training,Evaluation and Analysis configuration
        TrainingFraction:
        iteration:
        default_net_type:
        default_augmenter:
        snapshotindex:
        batch_size:
        \n
    # Cropping Parameters (for analysis and outlier frame detection)
        cropping:
    #if cropping is true for analysis, then set the values here:
        x1:
        x2:
        y1:
        y2:
        \n
    # Refinement configuration (parameters from annotation dataset configuration also relevant in this stage)
        corner2move2:
        move2corner:
        """

    ruamelFile = YAML()
    cfg_file = ruamelFile.load(yaml_str)
    return cfg_file, ruamelFile

def write_config(configname, cfg):
    """
    Write structured config file.
    """
    with open(configname, "w") as cf:
        cfg_file, ruamelFile = create_config_template(
            cfg.get("multianimalproject", False)
        )
        for key in cfg.keys():
            cfg_file[key] = cfg[key]

        # Adding default value for variable skeleton and skeleton_color for backward compatibility.
        if not "skeleton" in cfg.keys():
            cfg_file["skeleton"] = []
            cfg_file["skeleton_color"] = "black"
        ruamelFile.dump(cfg_file, cf)

def read_config(configname):
    """
    Reads structured config file defining a project.
    """
    ruamelFile = YAML()
    path = Path(configname)
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                cfg = ruamelFile.load(f)
                curr_dir = os.path.dirname(configname)
                if cfg["project_path"] != curr_dir:
                    cfg["project_path"] = curr_dir
                    write_config(configname, cfg)
        except Exception as err:
            if len(err.args) > 2:
                if (
                    err.args[2]
                    == "could not determine a constructor for the tag '!!python/tuple'"
                ):
                    with open(path, "r") as ymlfile:
                        cfg = yaml.load(ymlfile, Loader=yaml.SafeLoader)
                        write_config(configname, cfg)
                else:
                    raise

    else:
        raise FileNotFoundError(
            "Config file is not found. Please make sure that the file exists and/or that you passed the path of the config file correctly!"
        )
    return cfg

def get_sup_model_folder(cfg, modelprefix=""):
    Task = cfg["Task"]
    date = cfg["date"]
    iterate = "iteration-" + str(cfg["iteration"])
    return Path(
        modelprefix,
        "sup-models",
        iterate,
        Task
        + date
    )

def get_sup_model_yaml(cfg, modelprefix=""):
    Task = cfg["Task"]
    date = cfg["date"]
    iterate = "iteration-" + str(cfg["iteration"])
    return Path(
        modelprefix,
        "sup-models",
        iterate,
        Task
        + date,
        'train',
        'model_cfg.yaml'
    )

def get_unsup_model_folder(cfg, modelprefix=""):
    Task = cfg["Task"]
    date = cfg["date"]
    iterate = "iteration-" + str(cfg["iteration"])
    return Path(
        modelprefix,
        "unsup-models",
        iterate,
        Task
        + date
    )

def grab_files_in_folder(folder, ext=".csv", relative=True):
    """Return the paths of files with extension *ext* present in *folder*."""
    for file in os.listdir(folder):
        # if file.endswith(ext):
        yield file if relative else os.path.join(folder, file)

def grab_files_in_folder_deep(folder, ext=".csv", relative=True):
    """Return the paths of files with extension *ext* present in *folder*."""
    all_files = []
    for path, subdir, files in os.walk(folder):
        for file in glob(os.path.join(path, ext)):
            all_files.append(file)
    return all_files

def get_deepview_path():
    """Get path of where deeplabcut is currently running"""
    import importlib.util
    return os.path.split(importlib.util.find_spec("deepview").origin)[0]

def read_plainconfig(configname):
    if not os.path.exists(configname):
        raise FileNotFoundError(
            f"Config {configname} is not found. Please make sure that the file exists."
        )
    with open(configname) as file:
        return YAML().load(file)

def write_plainconfig(configname, cfg):
    with open(configname, "w") as file:
        YAML().dump(cfg, file)

## Various functions to get filenames, foldernames etc. based on configuration parameters.
def get_unsupervised_set_folder():
    """get folder for all sensor data used for unsupervised learning"""
    # iterate = "iteration-" + str(cfg["iteration"])
    return Path(
        os.path.join("unsupervised-datasets", "allDataSet")
    )

def get_training_set_folder(cfg):
    """Training Set folder for config file based on parameters"""
    Task = cfg["Task"]
    date = cfg["date"]
    # iterate = "iteration-" + str(cfg["iteration"])
    return Path(
        'training-datasets',
        Task
        + date
    )

def get_evaluation_folder(cfg, modelprefix=""):
    Task = cfg["Task"]
    date = cfg["date"]
    iterate = "iteration-" + str(cfg["iteration"])
    if "eval_prefix" in cfg:
        eval_prefix = cfg["eval_prefix"]
    else:
        eval_prefix = "evaluation-results"
    return Path(
        modelprefix,
        eval_prefix,
        iterate,
        Task
        + date
    )

def get_labeled_data_folder(cfg):
    """Training Set folder for config file based on parameters"""
    # Task = cfg["Task"]
    # filenames = str(cfg["iteration"])  # todo 改成filenames （list）
    file_paths = []
    # 目前只能选择一个文件夹下的data
    for f in cfg['file_sets']:
        filename = os.path.split(f)[-1][:-4]
        tmp = Path(
            os.path.join(cfg['project_path'], "labeled-data", filename, "CollectedData_" + cfg["scorer"] +'.pkl')
        )
        file_paths.append(tmp)
    return file_paths

def get_data_and_metadata_filenames(trainingsetfolder, cfg):
    # Filename for metadata and data relative to project path for corresponding parameters
    metadatafn = os.path.join(
        str(trainingsetfolder),
        "CollectedData_"
        + cfg["scorer"]
        # + "_"
        # + str(int(trainFraction * 100))
        # + "shuffle"
        # + str(shuffle)
        + ".pkl",
    )
    datafn = os.path.join(
        str(trainingsetfolder),
        "CollectedData_"
        +cfg["scorer"]
        # + cfg["scorer"]
        # + str(int(100 * trainFraction))
        # + "shuffle"
        # + str(shuffle)
        + ".csv",
    )
    return metadatafn, datafn

def attempt_to_make_folder(foldername, recursive=True):
    """Attempts to create a folder with specified name. Does nothing if it already exists."""
    try:
        os.path.isdir(foldername)
    except TypeError:  # https://www.python.org/dev/peps/pep-0519/
        foldername = os.fspath(
            foldername
        )  # https://github.com/DeepLabCut/DeepLabCut/issues/105 (windows)

    if os.path.isdir(foldername):
        pass
    else:
        if recursive:
            os.makedirs(foldername)
        else:
            os.mkdir(foldername)



def create_folders_from_string(folder_string):
    # Split the input string into individual folder names
    folder_names = folder_string.split('/')

    # Initialize the base path
    base_path = ''

    # Iterate through each folder name and create the corresponding folder
    for folder_name in folder_names:
        if len(folder_name) == 0:
            continue
        # Update the base path for each iteration
        base_path = os.path.join(base_path, folder_name)

        # Check if the folder already exists; if not, create it
        if not os.path.exists(base_path):
            os.mkdir(base_path)
            print(f"Created folder: {base_path}")
        else:
            print(f"Folder already exists: {base_path}")

# # Example usage:
# folder_string = "root_folder/subfolder1/subfolder2"
# create_folders_from_string(folder_string)

def get_param_from_path(model_path):
    '''
    example of model_path="CNN_AE_epoch9_datalen180_accx-accy-accz"
    extract and return: model name, data length, data columns
    '''
    filename = str(Path(model_path).name)
    # (1) Extract model name
    model_name = filename.split("_epoch")[0]

    # (2) Extract data length
    datalen_index = filename.find("datalen") + len("datalen")
    underscore_index = filename.find("_", datalen_index)
    data_length = int(filename[datalen_index:underscore_index])

    # (3) Extract column names
    column_names = filename.split("_")[-1].split(".")[0]
    column_names_list = column_names.split('-') if '-' in column_names else [column_names]

    return model_name, data_length, column_names_list
