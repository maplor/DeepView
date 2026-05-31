import glob
import os
import pickle
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset


MAX_INSTS = 100000000


def setup_train_val_test_animal_id_list(all_animal_id_list, test_animal_id_list):
    '''
    :param all_animal_id_list: file names of umineko or omi.. dataset. One file is one id
    in this function, separate all data indices into train/val/test
    TODO, after modifying the supervised learning tabs with 复选框, get list from the tabs
    '''

    # train_animal_id_list = cfg.dataset.labelled.animal_id_list.all.copy()
    train_animal_id_list = all_animal_id_list.copy()
    # remove test data
    for i in range(len(test_animal_id_list)):
        if test_animal_id_list[i] in train_animal_id_list:
            train_animal_id_list.remove(test_animal_id_list[i])

    # randomly select a file from train list as validation list
    val_animal_id_list = [random.choice(train_animal_id_list)]
    # remove validation data
    # val_animal_id_list = ["DUMMY"]
    for i in range(len(val_animal_id_list)):
        if val_animal_id_list[i] in train_animal_id_list:
            train_animal_id_list.remove(val_animal_id_list[i])
    return train_animal_id_list, val_animal_id_list, test_animal_id_list


def setup_dataloaders_supervised_learning(cfg,
                                          train_animal_id_list,
                                          val_animal_id_list,
                                          test_animal_id_list,
                                          train=True,
                                          train_balanced=True):

    (
        train_loader, val_loader, test_loader
    ) = prep_dataloaders_for_supervised_learning(cfg,
                                                 train_animal_id_list,
                                                 val_animal_id_list,
                                                 test_animal_id_list)

    return train_loader, val_loader, test_loader


def prep_dataloaders_for_supervised_learning(cfg,
                                             train_animal_id_list,
                                            val_animal_id_list,
                                            test_animal_id_list,
                                            test_only=False):

    train_val_split = True  # otsuka: return true or false
    train_data_ratio = 0.8  # otsuka config_dl.yaml
    batch_size = 128  # otsuka config_dl.yaml
    shuffle = True  # otsuka config_dl.yaml

    # # Added for experiment 07 (Experiment S1) data augmentation parameter grid search
    # da_param1 = cfg.dataset.da_param1
    # da_param2 = cfg.dataset.da_param2
    # da_param1 = None if da_param1 in ["None", "none", None, False, 0] else da_param1
    # da_param2 = None if da_param2 in ["None", "none", None, False, 0] else da_param2

    if cfg.dataset.species == 'om':
        species_dir = 'omizunagidori'
    elif cfg.dataset.species == 'um':
        species_dir = 'umineko'
    # species_dir = 'omizunagidori'

    train_loader_paths = []
    val_loader_paths = []
    test_loader_paths = []

    npz_format_data_dir = Path(cfg.path.dataset.npz_format_data_dir.labelled_data)
    # log.info(f"npz_format_data_dir: {npz_format_data_dir}")
    # npz_search_path = Path(
    #     npz_format_data_dir,
    #     species_dir,
    #     "**/*.npz"
    # )
    npz_search_path = Path(
        npz_format_data_dir,
        species_dir,
        "**/*.pkl"
    )
    # log.info(f"npz_search_path: {npz_search_path}")
    npz_file_path_list = sorted(glob.glob(str(npz_search_path)))

    if cfg.debug == True:
        npz_file_path_list = random.sample(npz_file_path_list, 2000)
    # log.info(f'N of npz files (instances): {len(npz_file_path_list)}')

    # -----------
    # Test data
    # -----------
    for idx, npz_file_path in enumerate(npz_file_path_list):
        animal_id = os.path.basename(os.path.dirname(npz_file_path))
        if animal_id in test_animal_id_list:
            test_loader_paths.append(npz_file_path)

    # test dataset & dataloader
    # load data and get label
    # npz = np.load(self.paths[index], allow_pickle=True)
    samples, targets = [], []
    for tp in test_loader_paths:
        with open(tp, 'rb') as f:
            tmp_dict = pickle.load(f)
            samples.append(tmp_dict["sample"])
            targets.append(tmp_dict["target"])
    test_dataset = DatasetLogbot2(
        samples=np.concatenate(samples),
        labels=np.concatenate(targets),
        paths=test_loader_paths,
        # augmentation=False,  # Do not apply DA for test dataset
        da_type=cfg.dataset.da_type,
        # da_param1=da_param1,
        # da_param2=da_param2,
        in_ch=cfg.dataset.in_ch
    )
    # test_loader = DataLoaderNpz2(
    #     dataset=test_dataset,
    #     batch_size=batch_size,
    #     shuffle=False,
    #     drop_last=False
    # )
    test_loader = DataLoader(test_dataset,
                                 batch_size=batch_size,
                                 shuffle=False,
                                 drop_last=False)

    if test_only == True:
        return None, None, test_loader

    # ---------------------
    # Train and val data
    # ---------------------

    for idx, npz_file_path in enumerate(npz_file_path_list):
        animal_id = os.path.basename(os.path.dirname(npz_file_path))
        if animal_id in train_animal_id_list:
            train_loader_paths.append(npz_file_path)
        if animal_id in val_animal_id_list:
            val_loader_paths.append(npz_file_path)

    if train_val_split == True:
        train_val_loader_paths = train_loader_paths + val_loader_paths
        dataset_size = len(train_val_loader_paths)
        train_count = int(dataset_size * train_data_ratio)
        val_count = dataset_size - train_count
        train_loader_paths_, val_loader_paths_ = train_test_split(
            train_val_loader_paths,
            test_size=val_count,
            train_size=train_count,
            random_state=0,  # otsuka config_dl.yaml cfg.seed,
            shuffle=True,
            stratify=None)
    else:
        train_loader_paths_ = train_loader_paths
        val_loader_paths_ = val_loader_paths

    # train dataset & dataloader
    samples, targets = [], []
    for tp in train_loader_paths_:
        with open(tp, 'rb') as f:
            tmp_dict = pickle.load(f)
            samples.append(tmp_dict["sample"])
            targets.append(tmp_dict["target"])
    train_dataset = DatasetLogbot2(
        samples=np.concatenate(samples),
        labels=np.concatenate(targets),
        paths=train_loader_paths_,
        # augmentation=cfg.train.data_augmentation,
        da_type=cfg.dataset.da_type,
        # da_param1=da_param1,
        # da_param2=da_param2,
        in_ch=cfg.dataset.in_ch
    )
    # train_loader = DataLoaderNpz2(
    #     dataset=train_dataset,
    #     batch_size=batch_size,
    #     shuffle=shuffle,
    #     drop_last=True
    # )
    train_loader = DataLoader(train_dataset,
                   batch_size=batch_size,
                   shuffle=shuffle,
                   drop_last=True)

    # val dataset & dataloader
    samples, targets = [], []
    for tp in val_loader_paths_:
        with open(tp, 'rb') as f:
            tmp_dict = pickle.load(f)
            samples.append(tmp_dict["sample"])
            targets.append(tmp_dict["target"])
    val_dataset = DatasetLogbot2(
        samples=np.concatenate(samples),
        labels=np.concatenate(targets),
        paths=val_loader_paths_,
        # augmentation=False,  # Do not apply DA for validation dataset
        da_type=cfg.dataset.da_type,
        # da_param1=da_param1,
        # da_param2=da_param2,
        in_ch=cfg.dataset.in_ch
    )
    # val_loader = DataLoaderNpz2(
    #     dataset=val_dataset,
    #     batch_size=batch_size,
    #     shuffle=True,
    #     drop_last=False
    # )
    val_loader = DataLoader(val_dataset,
                   batch_size=batch_size,
                   shuffle=True,
                   drop_last=False)

    return train_loader, val_loader, test_loader


# dataloaders
class BaseDataset(Dataset):
    def __init__(self, samples, labels):
        self.samples = samples
        self.labels = labels

    def __getitem__(self, index):
        sample, target = self.samples[index], self.labels[index]
        return sample, target

    def __len__(self):
        return len(self.samples)


# class DatasetLogbot2(BaseDataset):
#     def __init__(self,
#                  samples=None,
#                  labels=None,
#                  paths=None,
#                  # augmentation=False,
#                  da_type='random',
#                  # da_param1=None,  # None -> default params
#                  # da_param2=None,
#                  in_ch=3):
#         super(DatasetLogbot2, self).__init__(samples, labels)
#         # self.paths = paths
#         # self.augmentation = augmentation
#         # self.da_type = da_type
#         # self.da_param1 = da_param1
#         # self.da_param2 = da_param2
#         # self.in_ch = in_ch
#
#     def __getitem__(self, index):
#         # if self.samples is None:
#         #     # load data and get label
#         #     npz = np.load(self.paths[index], allow_pickle=True)
#         #     sample = npz["X"]
#         #     target = npz["label_id"]
#         # else:
#         sample, target = self.samples[index], self.labels[index]
#
#         # check the shape of sample
#         # print(f"sample.shape: {sample.shape}") # sample.shape: (1, 50, 3)
#         # print(target.shape)
#         # otsuka code: here is data augmentation
#
#         if isinstance(sample, np.ndarray):
#             sample = torch.from_numpy(sample)
#         return sample, target
#
#     def __len__(self):
#         return len(self.samples)
#         # return len(self.paths)
#     #
#     # def load(self, MAX_INSTS=MAX_INSTS):
#     # def unload(self):


class DatasetLogbot2(Dataset):
    def __init__(self, samples, labels, device='cuda'):
        self.samples = torch.from_numpy(samples.astype(float))  # activity label of the sensor segment
        self.labels = torch.tensor(np.array(labels).astype(int))  # filename of the data belongs to
        # self.label = torch.tensor(label)  # filename of the data belongs to
        self.samples = self.samples.to(device=device, non_blocking=True, dtype=torch.float)
        self.labels = self.labels.to(device=device, non_blocking=True, dtype=torch.int)

    def __getitem__(self, index):
        labels, samples = self.labels[index], self.samples[index]
        return samples, labels

    def __len__(self):
        return len(self.labels)