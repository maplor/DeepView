# This Python file uses the following encoding: utf-8
import os
# import numpy as np
import numpy as np
from torch.utils.data import Dataset, DataLoader
# from torchvision import transforms
import torch
import pickle
import pickle as cp
# from data_preprocess.data_preprocess_utils import get_sample_weights, train_test_val_split
from sklearn.model_selection import train_test_split
# from data_preprocess.base_loader import *
# import pandas as pd
import matplotlib.pyplot as plt

import gc
import random
from operator import itemgetter
# from utils import mask_simi_series, filter_simi_by_threshold
# from motif_processing.peak_func import calculate_test
from scipy import signal
from datetime import datetime
import pandas as pd


GRAVITATIONAL_ACCELERATION = 9.80665

def convert_unit_system(args, df):
    """
    The unit systems of output CSVs (E4-preprocessing/zip_to_csv.csv) are follows.
    # if logi dataset, use another cols...
    * acc: [m/s^2] (raw value[G] are mutiplied with 9.80665)
    * gyro: [dps]
    * quat: [none]
    """
    # ACC: [m / s ^ 2] = > [G]

    if args.dataset== 'logi':
        cols = ['LW_x', 'LW_y', 'LW_z']
        # cols = ['LW_x', 'LW_y', 'LW_z', 'RW_x', 'RW_y', 'RW_z']
    elif args.dataset=='openpack':
        cols = ["acc_x", "acc_y", "acc_z"]
    elif args.dataset=='neji':
        cols = ["x", "y", "z"]
    else:
        print('In time_series_preparation.py, convert_unit_system function, data column not matching.')
        NotImplementedError
    # allcols = list(df.columns.values)
    # cols = list(set(allcols) - set(['time']))
    df[cols] = df[cols] / GRAVITATIONAL_ACCELERATION

    return df

def prep_dataset_test():
    # load already proposed data, Maekawa sensei gave me a part of data,
    # I separated the data according to date: 20200822,23,24
    # xxday[['timestamp', 'AccX', 'AccY', 'AccZ', 'Lat(GPS)', 'Lon(GPS)', 'Alt(GPS)', 'Status(GPS)',
    #           'DevTemp', 'MagX', 'MagY', 'MagZ', 'Illum', 'AtmPress', 'WatPress',
    #           'WatTemp', 'Depth', 'Alt(WatPressSensor)']]
    # sampling rate = 30Hz

    root_path = r'D:\Code\animalResearch\seabird'
    datapath = os.path.join(root_path, 'firstday1.pkl')
    # data = pd.read_pickle(datapath)
    with open(datapath, 'rb') as f:
        data = pickle.load(f)

    # only use timestamp and accelrometer
    data1 = data[['AccX', 'AccY', 'AccZ']].copy()
    datanp = data1.values
    # data = data[['timestamp', 'AccX', 'AccY', 'AccZ']]

    return datanp



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
        data1 = data[['acc_x', 'acc_y', 'acc_z', 'timestamp', 'labelid', 'domainid']].copy()
        datanp = data1.values
    else:
        datanp = data.copy()

    # generate batch of data by overlapping the training set
    data_batch = []
    for idx in range(0, datanp.shape[0] - len_sw - step, step): #step10
        data_batch.append(datanp[idx: idx+len_sw, :])
    data_batch.append(datanp[-1 - len_sw: -1, :])  # last batch
    xlist = np.stack(data_batch, axis=0)  # [B, Len90, dim6]
    # [samples, timestamps, labels] = xlist
    # x_win_train = xlist.reshape((batch_size, xlist.shape[1], xlist.shape[-1]))  # [B, Len, dim]
    # print(" ..after sliding window: train inputs {0}".format(xlist.shape))
    return xlist


def prep_dataset_umineko_file():
    # ['logger_id', 'animal_tag', 'timestamp', 'acc_x', 'acc_y', 'acc_z',
    #        'latitude', 'longitude', 'gyro_x', 'gyro_y', 'gyro_z', 'mag_x', 'mag_y',
    #        'mag_z', 'illumination', 'pressure', 'temperature', 'labelid',
    #        'velocity']
    # sampling rate: look back to \seabird\export\masterLabelsByOtsuka\animal_id.csv

    root_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
    # the pickle file is a dict, key is filename, value is dataframe of the file
    datapath = os.path.join(root_path, 'umineko_labeled_data.pkl')

    with open(datapath, 'rb') as f:
        datadict = pickle.load(f)

    # get filenames (one corresponds to one bird's one day data)
    filenames = list(datadict.keys())
    for fn in filenames:
        tmpdata = datadict[fn]
        data1 = tmpdata[['acc_x', 'acc_y', 'acc_z']]
        timestamps = tmpdata.timestamp
        labels = tmpdata.labelid
        datadict[fn] = [data1.values, timestamps.values, labels.values]

    return datadict


def prep_dataset_umineko(args):
    # ['logger_id', 'animal_tag', 'timestamp', 'acc_x', 'acc_y', 'acc_z',
    #        'latitude', 'longitude', 'gyro_x', 'gyro_y', 'gyro_z', 'mag_x', 'mag_y',
    #        'mag_z', 'illumination', 'pressure', 'temperature', 'labelid',
    #        'velocity']
    # sampling rate: look back to \seabird\export\masterLabelsByOtsuka\animal_id.csv

    root_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir))


    if False:
        # the pickle file is a list of dataframes, each dataframe is a segment with a label
        datapath = os.path.join(root_path, 'umineko_labeled_data_segment.pkl')
        print('path to load pkl data: ' + datapath)
        with open(datapath, 'rb') as f:
            datalist = pickle.load(f)  # 2634 labeled segments (dataframe)

        print('generating segment batch data...')
        # sliding window length
        len_sw = 90
        data, timestamps, labels, domains = [], [], [], []
        for d in datalist:
            tmp = sliding_window(d, len_sw)  # temp:['acc_x', 'acc_y', 'acc_z', 'timestamp', 'labelid', 'domain']
            if tmp.shape[1] != len_sw:
                continue
            data.append(tmp[:,:,:3])
            timestamps.append(tmp[:,:,3:4])
            labels.append(tmp[:,:,4:5])
            domains.append(tmp[:,:,5:6])

        with open(os.path.join(root_path, 'umineko_label_segment_batch.pkl'), 'wb') as f:
            pickle.dump([data, timestamps, labels, domains], f)

    print('load segment batch data...')
    with open(os.path.join(root_path, args.batch_data), 'rb') as f:
        [data, timestamps, labels, domains] = pickle.load(f)  # return list of batch(,,)
        # print(timestamps.shape)  # acc+gyro (15961, 90, 1)
    # set args.num_channel
    args.num_channel = data[0].shape[-1]
    return data, timestamps, labels, domains



def generate_dataloader_overlap(data, target, domains, timestamps):
    # sampling rate = 25Hz
    # window size = 900, step = window size/2
    # for umineko data, 目前只处理有标签的data，用slidewin取segment时，保证最后一块segment一定取到。
    # 同时，以每个单独的标签（起止时间）为单位，不要把其他时间的不同label的segment混在一起。

    batch_size = 512
    len_sw = 90
    step = int(len_sw / 2)

    # generate batch of data by overlapping the training set
    data_batch = []
    for idx in range(0, data.shape[0] - len_sw - step, step): #step10
        data_batch.append(data[idx: idx+len_sw, :])
    xlist = np.stack(data_batch, axis=0)  # [B, Len, dim]
    # [samples, timestamps, labels] = xlist
    x_win_train = xlist.reshape((batch_size, xlist.shape[1], xlist.shape[-1]))  # [B, Len, dim]
    print(" ..after sliding window: train inputs {0}".format(x_win_train.shape))

    # dataloader
    train_set_r = data_loader_umineko(x_win_train, target, domains, timestamps)
    train_loader_r = DataLoader(train_set_r, batch_size=batch_size,
                                shuffle=False, drop_last=False)

    return train_loader_r

def generate_dataloader(data, target, domains, timestamps):
    batch_size = 512
    # dataloader
    train_set_r = data_loader_umineko(data, target, domains, timestamps)
    train_loader_r = DataLoader(train_set_r, batch_size=batch_size,
                                shuffle=False, drop_last=False)
    return train_loader_r


def prepare_data_file(args):
    # if args.seabird_name == 'test':
        # data = prep_dataset_test()
        # train_loader = generate_dataloader_overlap(data)
        # train_loader_r = [train_loader]
    if args.seabird_name == 'umineko':
        datadict = prep_dataset_umineko(args)  # return dict: key=filename, value=[data, timestamps, labels]
        train_loader_r = []
        for idx, (filename, values) in enumerate(datadict.items()):
            data, timestamps, target = values
            train_loader = generate_dataloader_overlap(data, target, filename, timestamps)
            # train_loader = generate_dataloader_nolap(data)
            train_loader_r.append(train_loader)

    else:
        print('TODO prepare_data')

    return train_loader_r  # return train_loader list


def prepare_data(args):
    if args.seabird_name == 'umineko':
        data, timestamps, labels, domains = prep_dataset_umineko(args)  # return dict: key=filename, value=[data, timestamps, labels]

        # concatenate list
        data_b = np.concatenate(data, axis=0)  # [B, Len, dim]
        timestamp_b = np.concatenate(timestamps, axis=0)  # [B, Len, dim]
        label_b = np.concatenate(labels, axis=0)  # [B, Len, dim]
        domain_b = np.concatenate(domains, axis=0)  # [B, Len, dim]

        # train_loader = generate_dataloader_overlap(data, labels, domains, timestamps)
        train_loader = generate_dataloader(data_b, label_b, domain_b, timestamp_b)

    return [train_loader]  # return train_loader list


# class base_loader(Dataset):
#     def __init__(self, samples):
#         self.samples = samples
#
#     def __getitem__(self, index):
#         sample = self.samples[index]
#         return sample
#
#     def __len__(self):
#         return len(self.samples)
#
#
# class data_loader(base_loader):
#     def __init__(self, samples):
#         super(data_loader, self).__init__(samples)
#
#     def __getitem__(self, index):
#         sample = self.samples[index]
#         return sample


class base_loader(Dataset):
    def __init__(self, samples, labels, domains, timestamps):
        self.samples = torch.from_numpy(samples.astype(np.float))  # acceleration data [x,y,z]
        self.timestamps = torch.tensor(timestamps)  # timestamp of each sample
        self.labels = torch.tensor(labels)  # activity label of the sensor segment
        self.domains = torch.tensor(domains)  # filename of the data belongs to

    def __getitem__(self, index):
        sample, label, domain, timestamp = self.samples[index], self.labels[index], self.domains[index], self.timestamps[index]
        return sample, timestamp, label, domain

    def __len__(self):
        return len(self.samples)


class data_loader_umineko(base_loader):
    def __init__(self, samples, labels, domains, timestamps):
        super(data_loader_umineko, self).__init__(samples, labels, domains, timestamps)
        self.samples = torch.tensor(samples)
        self.labels = torch.tensor(labels)
        self.domains = torch.tensor(domains)
        self.timestamps = torch.tensor(timestamps)

    def __getitem__(self, index):
        sample, target, domain = self.samples[index], self.labels[index], self.domains[index]
        timestamp = self.timestamps[index]
        return sample, timestamp, target, domain
