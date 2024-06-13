

import os
from pathlib import Path


def return_train_network_path(config, trainingsetindex=0, modelprefix=""):
    """Returns the training and test pose config file names as well as the folder where the snapshot is
    Parameters
    ----------
    config : string
        Full path of the config.yaml file as a string.

    shuffle: int
        Integer value specifying the shuffle index to select for training.

    trainingsetindex: int, optional
        Integer specifying which TrainingsetFraction to use. By default the first (note that TrainingFraction is a list in config.yaml).

    Returns the triple: trainposeconfigfile, testposeconfigfile, snapshotfolder
    """
    from deepview.utils import auxiliaryfunctions

    cfg = auxiliaryfunctions.read_config(config)
    modelfoldername = auxiliaryfunctions.get_model_folder(
        cfg["TrainingFraction"][trainingsetindex], cfg, modelprefix=modelprefix
    )
    trainposeconfigfile = Path(
        os.path.join(
            cfg["project_path"], str(modelfoldername), "train", "pose_cfg.yaml"
        )
    )
    testposeconfigfile = Path(
        os.path.join(cfg["project_path"], str(modelfoldername), "test", "pose_cfg.yaml")
    )
    snapshotfolder = Path(
        os.path.join(cfg["project_path"], str(modelfoldername), "train")
    )

    return trainposeconfigfile, testposeconfigfile, snapshotfolder


def train_network(
    sensor_dict,
    progress_update,
    config,
    select_filenames,
    net_type,
    lr,
    batch_size,
    num_epochs,
    data_len,
    data_column
):

    # if allow_growth:
    #     os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

    # reload logger.
    import importlib
    import logging

    importlib.reload(logging)
    logging.shutdown()

    from deepview.utils import auxiliaryfunctions

    # tf.compat.v1.reset_default_graph()
    start_path = os.getcwd()

    # Read file path for pose_config file. >> pass it on
    cfg = auxiliaryfunctions.read_config(config)
    modelfoldername = auxiliaryfunctions.get_unsup_model_folder(cfg)
    poseconfigfile = Path(
        os.path.join(
            cfg["project_path"], str(modelfoldername), "model_cfg.yaml"
        )
    )
    if not poseconfigfile.is_file():
        print("The training datafile ", poseconfigfile, " is not present.")
        print(
            "Probably, the training dataset for this specific shuffle index was not created."
        )
        print(
            "Try with a different trainingsetfraction or use function 'create_training_dataset' to create a new trainingdataset."
            # "Try with a different shuffle/trainingsetfraction or use function 'create_training_dataset' to create a new trainingdataset with this shuffle index."
        )

    try:
        # remove if/elif for multianimal and animalzoo, keep simplist one
        from deepview.clustering_pytorch.core.train import train

        print("Selecting single-animal trainer")
        train(
            sensor_dict,
            progress_update,
            str(poseconfigfile),
            select_filenames,
            net_type=net_type,
            lr=lr,
            batch_size=batch_size,
            num_epochs=num_epochs,
            data_len=data_len,
            data_column=data_column
        )  # pass on path and file name for pose_cfg.yaml!

    except BaseException as e:
        raise e
    finally:
        os.chdir(str(start_path))
    print(
        "The network is now trained and ready to evaluate. Use the function 'evaluate_network' to evaluate the network."
    )
