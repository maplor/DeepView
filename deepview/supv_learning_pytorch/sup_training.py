# This Python file uses the following encoding: utf-8
import torch
import numpy as np
from pathlib import Path
import os
import pandas as pd

from deepview.supv_learning_pytorch.utils.utils import (
    DatasetLogbot2,
)
from deepview.supv_learning_pytorch.models.dcl import (
    DeepConvLstmV3
)
from deepview.supv_learning_pytorch.core.trainer import train

from deepview.utils.auxiliaryfunctions import (
    get_sup_model_yaml,
    get_sup_folder,
    # get_base_yaml,
    read_config,
)

from deepview.generate_training_dataset.trainingsetmanipulation import read_process_csv

from torch.utils.data import DataLoader, Dataset

import pickle

from deepview.utils import (
    auxiliaryfunctions,
)

def replace_str2int(label_dict, array):
    new_array = np.full((len(array),), 0, dtype=int)
    for idx, i in enumerate(array):
        if type(i) == str:
            new_array[idx] = label_dict[i]
        else:
            new_array[idx] = -1
    # # Step 3: replace None with -1, to remove the nonclass data
    # new_array = np.nan_to_num(array, nan=-1)
    # new_array = np.where(np.isnan(array), -1, array)
    # # Step 4: Create a function that performs the replacement
    # def replace_string_with_int(value):
    #     return label_dict.get(value, value)  # Return the value itself if not found in the dictionary
    #
    # # Step 5: Vectorize the function using numpy.vectorize
    # vectorized_replace = np.vectorize(replace_string_with_int)

    # # Step 6: Apply the vectorized function to the numpy array
    # int_array = vectorized_replace(new_array)

    return new_array


def remove_unlabeled_data(samples, labels):
    labels = labels.astype(float)

    # 找到包含 NaN 值的行的索引
    nan_row_indices = np.where(labels==-1)[0]

    # 删除空行
    new_labels = np.delete(labels, nan_row_indices, axis=0)
    new_samples = np.delete(samples, nan_row_indices, axis=0)

    return new_samples, new_labels

def remove_labels_not_in_trainset(samples, labels, train_labels):
    labels = labels.astype(float)
    # 找到包含 NaN 值的行的索引
    nan_row_indices = np.where(labels == -1)[0]
    # 删除空行
    labels = np.delete(labels, nan_row_indices, axis=0)
    samples = np.delete(samples, nan_row_indices, axis=0)

    nan_row_indices = np.where(labels not in train_labels)[0]
    labels = np.delete(labels, nan_row_indices, axis=0)
    samples = np.delete(samples, nan_row_indices, axis=0)

    return samples, labels

def prepare_data(root, batch_size, window_len, label_dict, filenames, DEVICE, train_labels=[], iftrain=False):
    # batch_size= 128
    # window_len = 90
    unsup_folder = auxiliaryfunctions.get_unsupervised_set_folder()
    edit_label_folder = auxiliaryfunctions.get_edit_data_folder()
    samples, targets = [], []
    for select_f in filenames:
        if select_f.endswith('.pkl'):
            with open(os.path.join(root.project_folder, unsup_folder, select_f), 'rb') as f:
                tmpdata = pickle.load(f)
        elif select_f.endswith('.csv'):
            tmpdata = pd.read_csv(os.path.join(root.project_folder, edit_label_folder, select_f))
        else:  # todo output error
            with open(select_f, 'rb') as f:
                tmpdata = pickle.load(f)
        samples.append(tmpdata[['acc_x', 'acc_y', 'acc_z']])
        targets.append(tmpdata['label'])

    if (len(samples)==0) or (len(targets)==0):
        print('Please select files for training and test.')
        return None, None, None

    concat_samples = np.concatenate(samples)
    concat_targets = np.concatenate(targets)

    # transform labels
    concat_targets = replace_str2int(label_dict, concat_targets)


    # remove unlabeled data and check if still data remain
    if iftrain:
        concat_samples, concat_targets = remove_unlabeled_data(concat_samples, concat_targets)
        train_labels = np.unique(concat_targets)
    else:
        concat_samples, concat_targets = remove_labels_not_in_trainset(concat_samples, concat_targets, train_labels)
    if len(concat_targets) == 0:
        return None, None, None

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
        device=DEVICE
    )
    data_loader = DataLoader(data_dataset,
                             batch_size=batch_size,
                             shuffle=False,
                             drop_last=False)
    return data_loader, data_dim, train_labels


def train_sup_network(
        root,
        net_type,
        learning_rate,
        batch_size,
        windowlen,
        max_iter,
        allfiles,
        train_filenames,
        test_filenames
):
    # DEVICE = 'cpu'
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device using %s' % DEVICE)

    root_cfg = read_config(root.config)
    project_folder = root.project_folder
    sup_path = os.path.join(project_folder,
                            get_sup_folder(root_cfg)
                            )

    label_dict = root_cfg['label_dict']
    train_loader, data_dim, train_labels = prepare_data(root, batch_size, windowlen,
                                                  label_dict, train_filenames,
                                                  DEVICE, train_labels=[], iftrain=True)
    if train_loader == None:
        print('No labels in training set, try other files again.')
        return

    test_loader, _, _ = prepare_data(root, batch_size, windowlen,
                                     label_dict, test_filenames,
                                     DEVICE, train_labels, iftrain=False)

    if test_loader == None:
        print('No labels in test sets, try other files again.')
        return

    uniquelabels = np.unique(list(label_dict.values()))
    num_classes = len(uniquelabels[uniquelabels >= 0])
    if net_type == 'DeepConvLSTM':
        model = DeepConvLstmV3(in_ch=data_dim, num_classes=num_classes)
    else:
        print('no model available')
    model.to(DEVICE)

    # initialize the optimizer and loss
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=learning_rate
    )
    # if config["train"]["optimizer"] == "Adam":
    #     optimizer = torch.optim.Adam(
    #         model.parameters(),
    #         lr=learning_rate,
    #         weight_decay=config["train"]["weight_decay"]
    #     )
    # elif config["train"]["optimizer"] == "SGD":
    #     optimizer = torch.optim.SGD(
    #         model.parameters(),
    #         # lr=cfg.train.lr,
    #         lr=learning_rate,
    #         momentum=0.9,
    #         weight_decay=0
    #     )

    # loss function
    criterion = torch.nn.CrossEntropyLoss()

    # run training, and validate result using validation data
    best_model = train(
        model,
        max_iter,
        optimizer,
        criterion,
        train_loader,
        test_loader,
        DEVICE,
        sup_path
    )

    # # save the training log
    # path = Path(sup_path, "log.csv")
    # df_log.to_csv(path, index=False)
    return
