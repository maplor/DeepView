
# DeepLabCut Toolbox (deeplabcut.org)
# © A. & M.W. Mathis Labs
# https://github.com/DeepLabCut/DeepLabCut
#
# Please see AUTHORS for contributors.
# https://github.com/DeepLabCut/DeepLabCut/blob/master/AUTHORS
#
# Adapted from DeeperCut by Eldar Insafutdinov
# https://github.com/eldar/pose-tensorflow
#
# Licensed under GNU Lesser General Public License v3.0
#


import warnings
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pickle
import pandas as pd
from pathlib import Path

from deepview.clustering_pytorch.nnet.util import (
    gen_aug,
)


class DatasetFactory:
    _datasets = dict()

    @classmethod
    def register(cls, type_):
        def wrapper(dataset):
            if type_ in cls._datasets:
                warnings.warn("Overwriting existing dataset {}.")
            cls._datasets[type_] = dataset
            return dataset

        return wrapper

    @classmethod
    def create(cls, cfg):
        dataset_type = cfg["dataset_type"]
        dataset = cls._datasets.get(dataset_type)
        if dataset is None:
            raise ValueError(f"Unsupported dataset of type {dataset_type}")
        return dataset(cfg)


# --------step1_SSL.py, by xia----------

def sliding_window(data, len_sw):
    # input is a segment of data
    # output is data, timestamp, domain(filename), label
    # sampling rate = 25Hz
    # window size = 900, step = window size/2
    # for umineko data, 目前只处理有标签的data，用slidewin取segment时，保证最后一块segment一定取到。
    # 同时，以每个单独的标签（起止时间）为单位，不要把其他时间的不同label的segment混在一起。

    # batch_size = 512
    # len_sw = 90
    step = int(len_sw / 2)

    if isinstance(data, pd.DataFrame):
        data1 = data.copy()
        datanp = data1.values
    else:
        datanp = data.copy()

    # generate batch of data by overlapping the training set
    data_batch = []
    for idx in range(0, datanp.shape[0] - len_sw - step, step):  # step10
        data_batch.append(datanp[idx: idx + len_sw, :])
    data_batch.append(datanp[-1 - len_sw: -1, :])  # last batch
    xlist = np.stack(data_batch, axis=0)  # [B, Len90, dim6]
    # [samples, timestamps, labels] = xlist
    # x_win_train = xlist.reshape((batch_size, xlist.shape[1], xlist.shape[-1]))  # [B, Len, dim]
    # print(" ..after sliding window: train inputs {0}".format(xlist.shape))
    return xlist


def process_pressure_sensor(df, columns):
    if 'pressure' in columns:
        df['pressure'] = df['pressure'] - 1013.25
    return df


def prep_dataset_umineko(data_path,
                         filenames,
                         len_sw,
                         data_column,
                         labeled_flag=False):
    extend_column = data_column.copy()

    extend_column.extend(['label_flag'])
    extend_column.extend(['unixtime'])
    extend_column.extend(['label_id'])

    datalist, timestamps, labels, timestr, label_flags = [], [], [], [], []
    for filename in filenames:
        full_file_path = os.path.join(data_path, filename)
        if os.path.exists(full_file_path):
            print('path to load pkl data: ' + data_path)
            with open(Path(full_file_path), 'rb') as f:
                data = pickle.load(f)  # 2634 labeled segments (dataframe)

            if labeled_flag:  # only use labeled data
                data = data[data['label_flag'] == 1]

            # process the pressure column by reducing 1013.25hPa
            data = process_pressure_sensor(data, extend_column)
            tmp = sliding_window(data[extend_column],
                                 len_sw)  # temp:['acc_x', 'acc_y', 'acc_z', 'timestamp', 'labelid', 'domain']
            if tmp.shape[1] != len_sw:
                continue
            datalist.append(tmp[:, :, :-3])
            label_flags.append(tmp[:, :, -3:-2])
            timestamps.append(tmp[:, :, -2:-1])
            labels.append(tmp[:, :, -1:])

    # if not labeled_flag:
    #     return datalist, timestamps, labels
    # else:
    return datalist, timestamps, labels, label_flags


def prep_dataset_umineko_single(data_path):
    with open(data_path, 'rb') as f:
        datalist = pickle.load(f)  # 2634 labeled segments (dataframe)

    print('generating segment batch data...')
    # sliding window length
    len_sw = 90
    data, timestamps, labels, domains, timestr = [], [], [], [], []
    for d in [datalist]:
        tmp = sliding_window(d[['acc_x', 'acc_y', 'acc_z', 'time', 'labelid', 'domainid']],
                             len_sw)  # temp:['acc_x', 'acc_y', 'acc_z', 'timestamp', 'labelid', 'domain']
        if tmp.shape[1] != len_sw:
            continue
        data.append(tmp[:, :, :3])
        timestamps.append(tmp[:, :, 3:4])
        labels.append(tmp[:, :, 4:5])
        domains.append(tmp[:, :, 5:6])
        timestr.append(tmp[:, :, 6:7])
        # todo, 在这里增加数据column

    return data, timestamps, labels, domains, timestr


class base_loader(Dataset):
    def __init__(self, samples, labels, domains, timestamps, timestr):
        self.samples = torch.tensor(samples.astype(float))  # acceleration data [x,y,z]
        # self.samples = torch.from_numpy(samples.astype(float))  # acceleration data [x,y,z]
        # self.timestamps = timestamps  # timestamp of each sample
        self.timestamps = torch.tensor(timestamps.astype(float))  # timestamp of each sample
        self.labels = torch.tensor(labels.astype(int))  # activity label of the sensor segment
        self.domains = torch.tensor(domains.astype(int))  # filename of the data belongs to
        self.timestr = timestr  # filename of the data belongs to

    def __getitem__(self, index):
        sample, label, domain, timestamp = self.samples[index], self.labels[index], self.domains[index], \
            self.timestamps[index]
        timestr = self.timestr[index]
        return sample, timestamp, label, domain, timestr

    def __len__(self):
        return len(self.samples)


class data_loader_umineko(Dataset):
    def __init__(self, samples, labels, timestamps, augment=False, device='cpu', label_flags=[]):
        self.augment = augment
        if self.augment:
            self.aug_sample1 = gen_aug(samples, 't_warp')
            self.aug_sample1 = torch.tensor(self.aug_sample1).to(device)  # t_warp, out.shape=batch64,width3,height900
            self.aug_sample2 = gen_aug(samples, 'negate')  # negate
            self.aug_sample2 = torch.tensor(self.aug_sample2).to(device)
        else:
            self.samples = torch.tensor(samples).to(device)  # check data type
        self.labels = torch.tensor(labels)  # check data type
        self.timestamps = torch.tensor(timestamps.astype(float)).to(device)
        self.label_flags = torch.tensor(label_flags.astype(float)).to(device)


    def __getitem__(self, index):
        target = self.labels[index]
        timestamp = self.timestamps[index]
        label_flag = self.label_flags[index]
        if self.augment == False:
            sample = self.samples[index]
            # if len(self.label_flags) == 0:
            #     return sample, timestamp, target
            # else:
            return sample, timestamp, target, label_flag
        else:
            sample1 = self.aug_sample1[index]
            sample2 = self.aug_sample2[index]
            # if len(self.label_flags) == 0:
            #     return sample1, sample2, timestamp, target
            # else:
            return sample1, sample2, timestamp, target, label_flag

    def __len__(self):
        return len(self.labels)


def generate_dataloader(data, target, timestamps, batch_size=512, augment=False,
                        device='cpu', label_flags=[]):
    # batch_size = 512
    # dataloader
    train_set_r = data_loader_umineko(data, target, timestamps, augment,
                                      device=device, label_flags=label_flags)
    train_loader_r = DataLoader(train_set_r, batch_size=batch_size,
                                shuffle=False, drop_last=False)
    return train_loader_r


def prepare_all_data(data_path,
                     select_filenames,
                     data_len,
                     data_column,
                     batch_size,
                     augment=False,
                     device='cpu',
                     labeled_flag=False):
    '''
    主要是训练集，需要全部数据做训练。读取数据
    :param data_path:
    :param data_path_new:
    :return:
    '''

    data, timestamps, labels, label_flags = prep_dataset_umineko(data_path,
                                                                 select_filenames,
                                                                 data_len,
                                                                 data_column,
                                                                 labeled_flag=labeled_flag)  # return dict: key=filename, value=[data, timestamps, labels]

    # concatenate list
    data_b = np.concatenate(data, axis=0)  # [B, Len, dim]
    timestamp_b = np.concatenate(timestamps, axis=0)  # [B, Len, dim]
    label_b = np.concatenate(labels, axis=0)  # [B, Len, dim]

    label_flags_b = np.concatenate(label_flags, axis=0)  # [B, Len, dim]
    train_loader = generate_dataloader(data_b, label_b, timestamp_b, batch_size, augment, device, label_flags_b)

    # get model input channel
    num_channel = data_b.shape[-1]
    return train_loader, num_channel  # return train_loader


def prepare_all_data_with_flag(data_path,
                               select_filenames,
                               data_len,
                               data_column,
                               batch_size,
                               augment=False,
                               device='cpu',
                               labeled_flag=False):
    '''
    主要是训练集，需要全部数据做训练。读取数据
    相比较上一个函数多输出了label_flag,表示label是否原来被填充
    :return:
    '''
    # if args.seabird_name == 'umineko':
    data, timestamps, labels = prep_dataset_umineko(data_path,
                                                    select_filenames,
                                                    data_len,
                                                    data_column,
                                                    labeled_flag=labeled_flag)  # return dict: key=filename, value=[data, timestamps, labels]

    # concatenate list
    data_b = np.concatenate(data, axis=0)  # [B, Len, dim]
    timestamp_b = np.concatenate(timestamps, axis=0)  # [B, Len, dim]
    label_b = np.concatenate(labels, axis=0)  # [B, Len, dim]

    # train_loader = generate_dataloader_overlap(data, labels, domains, timestamps)
    train_loader = generate_dataloader(data_b, label_b, timestamp_b, batch_size, augment, device)

    # get model input channel
    num_channel = data_b.shape[-1]
    return train_loader, num_channel  # return train_loader


def prepare_single_data(data_path):
    data, timestamps, labels, domains, timestr = prep_dataset_umineko_single(
        data_path)  # return dict: key=filename, value=[data, timestamps, labels]
    data_b = np.concatenate(data, axis=0)  # [B, Len, dim]
    timestamp_b = np.concatenate(timestamps, axis=0)  # [B, Len, dim]
    label_b = np.concatenate(labels, axis=0)  # [B, Len, dim]
    domain_b = np.concatenate(domains, axis=0)  # [B, Len, dim]

    # train_loader = generate_dataloader_overlap(data, labels, domains, timestamps)
    train_loader = generate_dataloader(data_b, label_b, domain_b, timestamp_b)

    return train_loader  # return train_loader


def prepare_unsup_dataset(segments, data_column):
    data_list, timestamp_list, label_list = [], [], []
    for seg in segments:
        data_list.append(seg[data_column].values)
        timestamp_list.append(seg['datetime'].values)
        # label_list.append(seg['label'].values)
    timestamp_b = np.array(timestamp_list)  # [B, Len, dim]
    # label_b = np.array(label_list)  # [B, Len, dim]
    data_b = np.array(data_list)  # [B, Len, dim]
    # check data shape
    if len(timestamp_b.shape) == 2:
        timestamp_b = timestamp_b.reshape(timestamp_b.shape[0], timestamp_b.shape[1], 1)

    # sudo label
    array_shape = (data_b.shape[0], data_b.shape[1], 1)
    label_b = np.ones(array_shape)
    train_loader = generate_dataloader(data_b, label_b, timestamp_b,
                                       batch_size=512, augment=False,
                                        device='cpu', label_flags=label_b[:,0,0])
    return train_loader
