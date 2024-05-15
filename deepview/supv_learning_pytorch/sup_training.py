

import torch
import numpy as np
from pathlib import Path
import os

from deepview.supv_learning_pytorch.utils.utils import (
    DatasetLogbot2,
)
from deepview.supv_learning_pytorch.models.dcl import (
    DeepConvLstmV3
)
from deepview.supv_learning_pytorch.core.trainer import train

from deepview.utils.auxiliaryfunctions import (
    get_sup_model_yaml,
    read_config,
)

from torch.utils.data import DataLoader, Dataset


import pickle

def replace_str2int(label_dict, array):
    # Step 3: replace None with -1, to remove the nonclass data
    new_array = np.where(np.isnan(array[0]), -1, array)

    # Step 4: Create a function that performs the replacement
    def replace_string_with_int(value):
        return label_dict.get(value, value)  # Return the value itself if not found in the dictionary

    # Step 5: Vectorize the function using numpy.vectorize
    vectorized_replace = np.vectorize(replace_string_with_int)

    # Step 6: Apply the vectorized function to the numpy array
    int_array = vectorized_replace(new_array)

    return int_array

def prepare_data(label_dict, files, filenames):
    batch_size = 128
    window_len = 90

    samples, targets = [], []
    for file in files:
        for select_f in filenames:
            if select_f in file:
                with open(file, 'rb') as f:
                    # todo: 以后可以自由选择columns
                    tmpdata = pickle.load(f)
                    samples.append(tmpdata[['acc_x', 'acc_y', 'acc_z']])
                    targets.append(tmpdata['label'])

    concat_samples = np.concatenate(samples)
    concat_targets = np.concatenate(targets)

    # transform labels
    concat_targets = replace_str2int(label_dict, concat_targets)

    # convert data to window of data
    data_dim = concat_samples.shape[-1]
    restlen = concat_samples.shape[0] % window_len
    rest_samples = concat_samples[0:-restlen]
    rest_samples = rest_samples.reshape(-1, window_len, data_dim)

    rest_targets = concat_targets[0:-restlen]
    rest_targets = rest_targets.reshape(-1, window_len, 1)

    # dataloader
    data_dataset = DatasetLogbot2(
        samples=rest_samples,
        labels=rest_targets,
    )
    data_loader = DataLoader(data_dataset,
                             batch_size=batch_size,
                             shuffle=False,
                             drop_last=False)
    return data_loader, data_dim

def train_sup_network(
        root,
        allfiles,
        train_filenames,
        test_filenames
):
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('device using %s' % DEVICE)

    root_cfg = read_config(root.config)
    project_folder = root.project_folder
    sup_path = os.path.join(project_folder,
                            get_sup_model_yaml(root_cfg)
                            )
    config = read_config(sup_path)

    label_dict = config['label_dict']
    train_loader, data_dim = prepare_data(label_dict, allfiles, train_filenames)
    test_loader, _ = prepare_data(label_dict, allfiles, test_filenames)

    model = DeepConvLstmV3(in_ch=data_dim, num_classes=5)
    model.to(DEVICE)

    # initialize the optimizer and loss
    if config["train"]["optimizer"] == "Adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config["train"]["lr"],
            weight_decay=config["train"]["weight_decay"]
        )
    elif config["train"]["optimizer"] == "SGD":
        optimizer = torch.optim.SGD(
            model.parameters(),
            # lr=cfg.train.lr,
            lr=0.01,
            momentum=0.9,
            weight_decay=0
        )

    # loss function
    criterion = torch.nn.CrossEntropyLoss()

    # run training, and validate result using validation data
    best_model, df_log = train(
        model,
        optimizer,
        criterion,
        train_loader,
        test_loader,
        DEVICE,
        config
    )

    # save the training log
    path = Path(sup_path, "log.csv")
    df_log.to_csv(path, index=False)
    return
