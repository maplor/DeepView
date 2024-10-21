#
# DeepLabCut Toolbox (deeplabcut.org)
# © A. & M.W. Mathis Labs
# https://github.com/DeepLabCut/DeepLabCut
#
# Please see AUTHORS for contributors.
# https://github.com/DeepLabCut/DeepLabCut/blob/master/AUTHORS
#
# Licensed under GNU Lesser General Public License v3.0
#

# 所有yaml参数在这里初始化并写入。。。

import os
import shutil
import sqlite3
import warnings
from pathlib import Path
from datetime import datetime as dt

from deepview.utils import auxiliaryfunctions

from deepview import DEBUG
from deepview.utils.auxfun_files import fileReader
from deepview.utils.auxiliaryfunctions import get_db_folder


def create_new_project(
    project,
    experimenter,
    files,
    working_directory=None,
    copy_videos=False,
    filetype="",
    # multianimal=False,
):
    r"""Create the necessary folders and files for a new project.

    Creating a new project involves creating the project directory, sub-directories and
    a basic configuration file. The configuration file is loaded with the default
    values. Change its parameters to your projects need.

    Parameters
    ----------
    project : string
        The name of the project.

    experimenter : string
        The name of the experimenter.

    videos : list[str]
        A list of strings representing the full paths of the videos to include in the
        project. If the strings represent a directory instead of a file, all videos of
        ``filetype`` will be imported.

    working_directory : string, optional
        The directory where the project will be created. The default is the
        ``current working directory``.

    copy_videos : bool, optional, Default: False.
        If True, the videos are copied to the ``videos`` directory. If False, symlinks
        of the videos will be created in the ``project/videos`` directory; in the event
        of a failure to create symbolic links, videos will be moved instead.

    multianimal: bool, optional. Default: False.
        For creating a multi-animal project (introduced in DLC 2.2)

    Returns
    -------
    str
        Path to the new project configuration file.

    Examples
    --------

    Linux/MacOS:

    >>> deeplabcut.create_new_project(
            project='reaching-task',
            experimenter='Linus',
            videos=[
                '/data/videos/mouse1.avi',
                '/data/videos/mouse2.avi',
                '/data/videos/mouse3.avi'
            ],
            working_directory='/analysis/project/',
        )
    >>> deeplabcut.create_new_project(
            project='reaching-task',
            experimenter='Linus',
            files=['/data/raw'],
            filetype='.csv',
        )

    Windows:

    >>> deeplabcut.create_new_project(
            'reaching-task',
            'Bill',
            [r'C:\yourusername\rig-95\Videos\reachingvideo1.avi'],
            copy_videos=True,
        )

    Users must format paths with either:  r'C:\ OR 'C:\\ <- i.e. a double backslash \ \ )
    """

    months_3letter = {
        1: "Jan",
        2: "Feb",
        3: "Mar",
        4: "Apr",
        5: "May",
        6: "Jun",
        7: "Jul",
        8: "Aug",
        9: "Sep",
        10: "Oct",
        11: "Nov",
        12: "Dec",
    }

    date = dt.today()
    month = months_3letter[date.month]
    day = date.day
    d = str(month[0:3] + str(day))
    date = dt.today().strftime("%Y-%m-%d")
    if working_directory is None:
        working_directory = "."
    wd = Path(working_directory).resolve()
    project_name = "{pn}-{exp}-{date}".format(pn=project, exp=experimenter, date=date)
    project_path = wd / project_name

    # Create project and sub-directories
    if not DEBUG and project_path.exists():
        print('Project "{}" already exists!'.format(project_path))
        return os.path.join(str(project_path), "config.yaml")
    file_path = project_path / "raw-data"
    label_path = project_path / "labeled-data"
    shuffles_path = project_path / "training-datasets"
    # results_path = project_path / "dview-models"
    unsupervised_path = project_path / "unsupervised-datasets"
    for p in [file_path, label_path, shuffles_path, unsupervised_path, unsupervised_path / "allDataSet"]:
        p.mkdir(parents=True, exist_ok=DEBUG)
        print('Created "{}"'.format(p))

    # Add all videos in the folder. Multiple folders can be passed in a list, similar to the video files. Folders and video files can also be passed!
    fids = []
    for i in files:
        # Check if it is a folder
        if os.path.isdir(i):
            fids_in_dir = [
                os.path.join(i, vp) for vp in os.listdir(i) if vp.endswith(filetype)
            ]
            fids = fids + fids_in_dir
            if len(fids_in_dir) == 0:
                print("No files found in", i)
                print(
                    "Perhaps change the filetype, which is currently set to:",
                    filetype,
                )
            else:
                files = fids
                print(
                    len(fids_in_dir),
                    " files from the directory",
                    i,
                    "were added to the project.",
                )
        else:
            if os.path.isfile(i):
                fids = fids + [i]
            files = fids

    files = [Path(vp) for vp in files]
    dirs = [label_path / Path(i.stem) for i in files]
    for p in dirs:
        """
        Creates directory under data
        """
        p.mkdir(parents=True, exist_ok=True)

    destinations = [file_path.joinpath(vp.name) for vp in files]
    if copy_videos:
        print("Copying the files")
        for src, dst in zip(files, destinations):
            shutil.copy(
                os.fspath(src), os.fspath(dst)
            )  # https://www.python.org/dev/peps/pep-0519/
    else:
        # creates the symlinks of the video and puts it in the videos directory.
        print("Attempting to create a symbolic link of the file ...")
        for src, dst in zip(files, destinations):
            if dst.exists() and not DEBUG:
                raise FileExistsError("File {} exists already!".format(dst))
            try:
                src = str(src)
                dst = str(dst)
                os.symlink(src, dst)
                print("Created the symlink of {} to {}".format(src, dst))
            except OSError:
                try:
                    import subprocess

                    subprocess.check_call("mklink %s %s" % (dst, src), shell=True)
                except (OSError, subprocess.CalledProcessError):
                    print(
                        "Symlink creation impossible (exFat architecture?): "
                        "copying the video instead."
                    )
                    shutil.copy(os.fspath(src), os.fspath(dst))
                    print("{} copied to {}".format(src, dst))
            files = destinations

    if copy_videos:
        files = destinations  # in this case the *new* location should be added to the config file

    # adds the video list to the config.yaml file
    file_sets = {}
    for file in files:
        print(file)
        try:
            # For windows os.path.realpath does not work and does not link to the real video. [old: rel_video_path = os.path.realpath(video)]
            rel_video_path = str(Path.resolve(Path(file)))
        except:
            rel_video_path = os.readlink(str(file))

        try:
            vid = fileReader(rel_video_path)
            file_sets[rel_video_path] = {"crop": ", ".join(map(str, vid.get_bbox()))}
        except IOError:
            warnings.warn("Cannot open the file! Skipping to the next one...")
            os.remove(file)  # Removing the video or link from the project

    if not len(file_sets):
        # Silently sweep the files that were already written.
        shutil.rmtree(project_path, ignore_errors=True)
        warnings.warn(
            "No valid files were found. The project was not created... "
            "Verify the files and re-create the project."
        )
        return "nothingcreated"

    # Set values to config file:
    cfg_file, ruamelFile = auxiliaryfunctions.create_config_template()
    # cfg_file["multianimalproject"] = False
    # cfg_file["bodyparts"] = ["bodypart1", "bodypart2", "bodypart3", "objectA"]
    # cfg_file["skeleton"] = [["bodypart1", "bodypart2"], ["objectA", "bodypart3"]]
    cfg_file["default_augmenter"] = "default"
    cfg_file["default_net_type"] = "AE_CNN"

    # common parameters:
    cfg_file["Task"] = project
    cfg_file["scorer"] = experimenter
    cfg_file["file_sets"] = file_sets
    cfg_file["project_path"] = str(project_path)
    cfg_file["date"] = d
    # cfg_file["cropping"] = False
    cfg_file["start"] = 0
    cfg_file["stop"] = 1
    # cfg_file["numframes2pick"] = 20
    cfg_file["TrainingFraction"] = [0.95]
    cfg_file["iteration"] = 0
    cfg_file["snapshotindex"] = -1
    cfg_file["x1"] = 0
    cfg_file["x2"] = 640
    cfg_file["y1"] = 277
    cfg_file["y2"] = 624
    cfg_file[
        "batch_size"
    ] = 64  # batch size during inference (video - analysis); see https://www.biorxiv.org/content/early/2018/10/30/457242
    cfg_file["corner2move2"] = (50, 50)
    cfg_file["move2corner"] = True
    # cfg_file["skeleton_color"] = "black"
    cfg_file["pcutoff"] = 0.6
    cfg_file["dotsize"] = 12  # for plots size of dots
    cfg_file["alphavalue"] = 0.7  # for plots transparency of markers
    cfg_file["colormap"] = "rainbow"  # for plots type of colormap
    cfg_file["label_dict"] = {'stationary': 0,
                              'preening': 0,
                              'bathing': 1,
                              'flight_take_off': 2,
                              'flight_cruising': 3,
                              'foraging_dive': 4,
                              'surface_seizing': 5,
                              'unknown': -1}  # labels

    cfg_file['sensor_dict'] = {
        'acceleration': ['acc_x', 'acc_y', 'acc_z'],
        'gyroscope': ['gyro_x', 'gyro_y', 'gyro_z'],
        'magnitude': ['mag_x', 'mag_y', 'mag_z'],
        'gps': ['latitude', 'longitude'],
        'illumination': ['illumination'],
        'pressure': ['pressure'],
        'temperature': ['temperature']
    }

    projconfigfile = os.path.join(str(project_path), "config.yaml")
    # Write dictionary to yaml  config file
    auxiliaryfunctions.write_config(projconfigfile, cfg_file)

    db_dir = os.path.join(str(project_path), "db")
    db_path = os.path.join(db_dir, "database.db")
    # 确保数据库目录存在
    os.makedirs(db_dir, exist_ok=True)
    conn = sqlite3.connect(db_path)

    cursor = conn.cursor()


    cursor.execute('''CREATE TABLE IF NOT EXISTS animals (
                    animal_tag TEXT
                    )''')

    cursor.execute('''CREATE TABLE IF NOT EXISTS labels (
                label_id INTEGER PRIMARY KEY,
                    animal_tag TEXT,
                stt_timestamp TEXT,
                    stp_timestamp TEXT,
                activity TEXT,
                    location TEXT,
                    notes TEXT,
                    label_name TEXT
                    )''')

    cursor.execute('''CREATE TABLE IF NOT EXISTS videos (
                    animal_tag TEXT,
                video_stt DATETIME, 
                    video_stp DATETIME,
                framerate INTEGER,
                frame_count INTEGER,
                video_id INTEGER
                    )''')


    cursor.execute('''CREATE TABLE IF NOT EXISTS raw_data (
                logger_id TEXT,
                    animal_tag TEXT,
                datetime DATETIME,
                timestamp TEXT,
                    unixtime INTEGER,
                    latitude REAL,
                    longitude REAL,
                    acc_x REAL,
                    acc_y REAL,
                    acc_z REAL,
                    gyro_x REAL,
                    gyro_y REAL,
                    gyro_z REAL,
                    mag_x REAL,
                    mag_y REAL,
                    mag_z REAL,
                illumination REAL,
                pressure REAL,
                GPS_velocity REAL,
                GPS_bearing REAL,
                temperature REAL,
                label_id TEXT,
                label TEXT,
                label_flag INTEGER
                    )''')

    # 创建索引
    cursor.execute('''CREATE INDEX IF NOT EXISTS raw_data_index ON raw_data (logger_id, timestamp)''')
    cursor.execute('''CREATE INDEX IF NOT EXISTS videos_index ON videos (animal_tag, video_stt, video_stp)''')


    conn.commit()
    conn.close()

    print('Generated "{}"'.format(project_path / "config.yaml"))
    print(
        "\nA new project with name %s is created at %s and a configurable file (config.yaml) is stored there. Change the parameters in this file to adapt to your project's needs.\n Once you have changed the configuration file, use the function 'extract_frames' to select frames for labeling.\n. [OPTIONAL] Use the function 'add_new_videos' to add new videos to your project (at any stage)."
        % (project_name, str(wd))
    )
    return projconfigfile
