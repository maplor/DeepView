

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
    config,
    select_filenames='',
    net_type='CNN_AE',
    lr=0.0005,
    batch_size=32,
    num_epochs=10,
    data_len=180,
    data_column=['acc_x']

):
    """Trains the network with the labels in the training dataset.

        Parameters
        ----------
        config : string
            Full path of the config.yaml file as a string.

        shuffle: int, optional, default=1
            Integer value specifying the shuffle index to select for training.

        trainingsetindex: int, optional, default=0
            Integer specifying which TrainingsetFraction to use.
            Note that TrainingFraction is a list in config.yaml.

        max_snapshots_to_keep: int or None
            Sets how many snapshots are kept, i.e. states of the trained network. Every
            saving iteration many times a snapshot is stored, however only the last
            ``max_snapshots_to_keep`` many are kept! If you change this to None, then all
            are kept.
            See: https://github.com/DeepLabCut/DeepLabCut/issues/8#issuecomment-387404835

        displayiters: optional, default=None
            This variable is actually set in ``pose_config.yaml``. However, you can
            overwrite it with this hack. Don't use this regularly, just if you are too lazy
            to dig out the ``pose_config.yaml`` file for the corresponding project. If
            ``None``, the value from there is used, otherwise it is overwritten!

        saveiters: optional, default=None
            This variable is actually set in ``pose_config.yaml``. However, you can
            overwrite it with this hack. Don't use this regularly, just if you are too lazy
            to dig out the ``pose_config.yaml`` file for the corresponding project.
            If ``None``, the value from there is used, otherwise it is overwritten!

        maxiters: optional, default=None
            This variable is actually set in ``pose_config.yaml``. However, you can
            overwrite it with this hack. Don't use this regularly, just if you are too lazy
            to dig out the ``pose_config.yaml`` file for the corresponding project.
            If ``None``, the value from there is used, otherwise it is overwritten!

        allow_growth: bool, optional, default=True.
            For some smaller GPUs the memory issues happen. If ``True``, the memory
            allocator does not pre-allocate the entire specified GPU memory region, instead
            starting small and growing as needed.
            See issue: https://forum.image.sc/t/how-to-stop-running-out-of-vram/30551/2

        gputouse: optional, default=None
            Natural number indicating the number of your GPU (see number in nvidia-smi).
            If you do not have a GPU put None.
            See: https://nvidia.custhelp.com/app/answers/detail/a_id/3751/~/useful-nvidia-smi-queries

        autotune: bool, optional, default=False
            Property of TensorFlow, somehow faster if ``False``
            (as Eldar found out, see https://github.com/tensorflow/tensorflow/issues/13317).

        keepdeconvweights: bool, optional, default=True
            Also restores the weights of the deconvolution layers (and the backbone) when
            training from a snapshot. Note that if you change the number of bodyparts, you
            need to set this to false for re-training.

        modelprefix: str, optional, default=""
            Directory containing the deeplabcut models to use when evaluating the network.
            By default, the models are assumed to exist in the project folder.

        superanimal_name: str, optional, default =""
            Specified if transfer learning with superanimal is desired

        superanimal_transfer_learning: bool, optional, default = False.
            If set true, the training is transfer learning (new decoding layer). If set false,
    and superanimal_name is True, then the training is fine-tuning (reusing the decoding layer)

        Returns
        -------
        None

        Examples
        --------
        To train the network for first shuffle of the training dataset

        >>> deeplabcut.train_network('/analysis/project/reaching-task/config.yaml')

        To train the network for second shuffle of the training dataset

        >>> deeplabcut.train_network(
                '/analysis/project/reaching-task/config.yaml',
                shuffle=2,
                keepdeconvweights=True,
            )
    """
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
            cfg["project_path"], str(modelfoldername), "train", "model_cfg.yaml"
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
        # cfg_dlc = auxiliaryfunctions.read_plainconfig(poseconfigfile)

        # remove if/elif for multianimal and animalzoo, keep simplist one
        from deepview.clustering_pytorch.core.train import train

        print("Selecting single-animal trainer")
        train(
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
