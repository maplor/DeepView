
import os
import torch
import glob
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import matplotlib.ticker as tck
import seaborn as sns
import torch
import random
import pickle
from pathlib import Path
# from torch.utils.data import DataLoader
import sklearn
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    jaccard_score
)
from sklearn.manifold import TSNE
import pandas as pd


# https://discuss.pytorch.org/t/pytocrh-way-for-one-hot-encoding-multiclass-target-variable/68321
def to_one_hot(y, n_classes):
    '''
    y: shape(BS, T, 1)
    y_onehot: shape(BS, T, 1, n_classes)
    '''
    y_onehot = torch.nn.functional.one_hot(y, num_classes=n_classes)
    return y_onehot


def mixup_process(x, y, lam):
    '''
    x: shape(BS, CH, T, 1)
    y (y_onehot): shape(BS, T, 1, n_classes)
    lam: shape()
    '''
    # print(f"mixup_alpha = {mixup_alpha}")
    batch_size = x.size()[0]
    y = y.to(torch.float32)
    indices = np.random.permutation(batch_size)
    x_shuffled = x[indices]
    y_shuffled = y[indices]

    # expand lambda
    # log.debug(f"x.shape: {x.shape}")
    # log.debug(f"x.dim(): {x.dim()}")
    # log.debug(f"y.shape: {y.shape}")
    # log.debug(f"y.dim(): {y.dim()}")
    x_last_dim_idx = int(x.dim() - 1)
    y_last_dim_idx = int(y.dim() - 1)
    lam_x = lam.expand_as(x.transpose(0, x_last_dim_idx))
    lam_x = lam_x.transpose(0, x_last_dim_idx)
    lam_y = lam.expand_as(y.transpose(0, y_last_dim_idx))
    lam_y = lam_y.transpose(0, y_last_dim_idx)
    # lam_x = lam.expand_as(x.transpose(0, 3))
    # lam_x = lam_x.transpose(0, 3)
    # lam_y = lam.expand_as(y.transpose(0, 3))
    # lam_y = lam_y.transpose(0, 3)

    # mixup
    x_mixed = x * lam_x + x_shuffled * (1 - lam_x)
    y_mixed = y * lam_y + y_shuffled * (1 - lam_y)

    return x_mixed, y_mixed

def get_label_species(cfg):
    label_species = cfg.dataset.species
    return label_species


def convert_torch_labels(targets, label_species):
    if label_species == "om":
        targets = torch.where(targets == 200, 0, targets)  # stationary
        targets = torch.where(targets == 201, 0, targets)  # preening
        targets = torch.where(targets == 300, 1, targets)  # bathing
        targets = torch.where(targets == 400, 2, targets)  # flight_take_off
        targets = torch.where(targets == 401, 3, targets)  # flight_cruising
        targets = torch.where(targets == 501, 4, targets)  # foraging_dive
        targets = torch.where(targets == 502, 5, targets)  # surface_seizing (dipping)
    elif label_species == "um":
        targets = torch.where(targets == 100, 0, targets)  # ground_stationary
        targets = torch.where(targets == 101, 1, targets)  # ground_active
        targets = torch.where(targets == 200, 0, targets)  # stationary
        targets = torch.where(targets == 201, 0, targets)  # preening
        targets = torch.where(targets == 300, 2, targets)  # bathing
        targets = torch.where(targets == 301, 2, targets)  # bathing
        targets = torch.where(targets == 400, 3, targets)  # flying_active
        targets = torch.where(targets == 401, 4, targets)  # flying_passive
        targets = torch.where(targets == 500, 5, targets)  # foraging
        targets = torch.where(targets == 501, 5, targets)  # foraging_poss
        targets = torch.where(targets == 502, 5, targets)  # foraging_fish
        targets = torch.where(targets == 503, 5, targets)  # foraging_fish_poss
        # targets = torch.where(targets==510, 6, targets) # foraging_insect
        # targets = torch.where(targets==511, 6, targets) # foraging_insect_poss
        # targets = torch.where(targets==520, 7, targets) # foraging_something

    return targets


def return_species_jp_name(cfg):
    if cfg.dataset.species == "om":
        species_jp_name = "omizunagidori"
    elif cfg.dataset.species == "um":
        species_jp_name = "umineko"
    else:
        raise Exception(f"cfg.dataset: {cfg.dataset} is unknonw dataset.")

    return species_jp_name

def generate_class_labels_for_vis(species):
    if species == "omizunagidori":
        class_label = [
            'Stationary',
            'Bathing',
            'Take-off',
            'Cruising Flight',
            'Foraging Dive',
            'Dipping'
        ]
    elif species == "umineko":
        class_label = [
            'Stationary',
            'Ground Active',
            'Bathing',
            'Active Flight',
            'Passive Flight',
            'Foraging'
        ]
    return class_label

def plot_confusion_matrix(y_gt, y_pred, cfg, figsize=(9, 7)):
    species_jp_name = return_species_jp_name(cfg)
    class_labels = generate_class_labels_for_vis(species_jp_name)

    labels_int = np.arange(0, len(class_labels), 1).tolist()

    cm = confusion_matrix(y_gt, y_pred, labels=labels_int)

    df_cm = pd.DataFrame(data=cm, index=class_labels, columns=class_labels)
    print(df_cm)

    fig = plt.figure(figsize=figsize)
    group_counts = ["{0:0.0f}".format(value) for value in cm.flatten()]
    group_precision_scores = (df_cm / np.sum(df_cm)).values.flatten()
    group_percentages = ["{0:.2f}".format(value)
                         for value in group_precision_scores]
    annot_labels = [
        f"{v1}\n({v2})" for v1,
        v2 in zip(
            group_counts,
            group_percentages)]
    annot_labels = np.asarray(annot_labels).reshape(len(labels_int), len(labels_int))
    ax = sns.heatmap(
        df_cm / np.sum(df_cm),
        # df_cm,
        vmin=0, vmax=1.0,
        square=True, cbar=True, annot=annot_labels, fmt='',
        cmap='Blues'
    )
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.xlabel("Prediction", fontsize=14, rotation=0, labelpad=10)
    plt.ylabel("Ground Truth", fontsize=14, labelpad=10)
    ax.set_ylim(len(cm), 0)
    fig.tight_layout()
    plt.show()
    plt.close()

    fig_cm = fig

    return cm, df_cm, fig_cm

def plot_window_ax(ax, X, label, npz_file_name):
    '''
    X: numpy array
    '''
    acc_x = X[0].transpose(1, 0)[0]
    acc_y = X[0].transpose(1, 0)[1]
    acc_z = X[0].transpose(1, 0)[2]
    window_size = len(X[0])
    data_number = list(range(1, window_size + 1, 1))
    # color_list = ['#EE6677', '#228833', '#4477AA']
    color_list = ['#D81B60', '#FFC107', '#1E88E5']
    ax = sns.lineplot(ax=ax, x=data_number, y=acc_x, label="x", color=color_list[0])
    ax = sns.lineplot(ax=ax, x=data_number, y=acc_y, label="y", color=color_list[1])
    ax = sns.lineplot(ax=ax, x=data_number, y=acc_z, label="z", color=color_list[2])
    if npz_file_name is None:
        print("No title")
    else:
        if label is None:
            ax.set_title(f"{npz_file_name}", pad=10)
        else:
            ax.set_title(f"{npz_file_name} | label_id: {(int(label))}", pad=10)
    ax.set_xlabel("t")
    ax.set_ylabel("g")
    ax.set_xticks(np.arange(0, 51, 10), fontsize=18)
    ax.set_yticks(np.arange(-4.0, 4.2, 2.0))
    xticklabels = ['{:,.0f}'.format(x) for x in np.arange(0, 51, 10.0)]
    yticklabels = ['{:,.1f}'.format(x) for x in np.arange(-4.0, 4.1, 2.0)]
    ax.set_xticklabels(xticklabels)
    ax.set_yticklabels(yticklabels)
    ax.set_xlim(-3, 53)
    ax.set_ylim(-4.5, 4.5)
    ax.legend(ncol=3)
    ax.yaxis.set_minor_locator(tck.AutoMinorLocator(n=2))
    ax.grid(axis='both', which='major', alpha=0.5)
    ax.grid(axis='y', which='minor', alpha=0.5)
    # plt.show()
    # plt.close()

    return ax


def generate_test_score_df(cfg, y_gt, y_pred, test_animal_id):
    model_name = cfg.model.model_name
    species_jp_name = return_species_jp_name(cfg)
    class_labels = generate_class_labels_for_vis(species_jp_name)
    labels_int = np.arange(0, len(class_labels), 1).tolist()

    category = ["Macro", "Weighted"]
    category.extend(class_labels)

    model = np.array([model_name] * len(category))
    test_id = np.array([test_animal_id] * len(category))

    species = np.array([cfg.dataset.species] * len(category))

    # macro average
    precision_macro = precision_score(y_gt, y_pred, average="macro")
    recall_macro = recall_score(y_gt, y_pred, average="macro")
    f1_macro = f1_score(y_gt, y_pred, average="macro")
    IoU_macro = jaccard_score(y_gt, y_pred, average="macro")

    # weighted average
    precision_weighted = precision_score(y_gt, y_pred, average="weighted")
    recall_weighted = recall_score(y_gt, y_pred, average="weighted")
    f1_weighted = f1_score(y_gt, y_pred, average="weighted")
    IoU_weighted = jaccard_score(y_gt, y_pred, average="weighted")

    # scores for each class
    precision_scores = precision_score(y_gt, y_pred, average=None, labels=labels_int)
    recall_scores = recall_score(y_gt, y_pred, average=None, labels=labels_int)
    f1_scores = f1_score(y_gt, y_pred, average=None, labels=labels_int)
    IoU_scores = jaccard_score(y_gt, y_pred, average=None, labels=labels_int)

    # df for storing results
    data_dict = {
        'Model': model,
        'Category': category,
        'Test_ID': test_id,
        'Species': species,
        'Precision': np.append(
            np.append(
                precision_macro,
                precision_weighted),
            precision_scores),
        'Recall': np.append(
            np.append(
                recall_macro,
                recall_weighted),
            recall_scores),
        'F1': np.append(
            np.append(
                f1_macro,
                f1_weighted),
            f1_scores),
        'IoU': np.append(
            np.append(
                IoU_macro,
                IoU_weighted),
            IoU_scores),
    }

    df = pd.DataFrame(data=data_dict)

    return df

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

    # if train == True:
    #     if hasattr(train_loader, "load"):
    #         # print("Loading train_loader: ")
    #         train_loader = train_loader.load()
    #     if hasattr(val_loader, "load"):
    #         # print("Loading val_loader: ")
    #         val_loader = val_loader.load()
        # if train_balanced == True:
        #     train_dataset = train_loader.dataset
        #     train_loader_balanced = setup_balanced_dataloader(train_dataset, cfg)
        #     return train_loader_balanced, val_loader, test_loader
        # else:
        #     return train_loader, val_loader, test_loader
    # else:
    #     if hasattr(test_loader, "load"):
    #         # print("Loading test_loader: ")
    #         test_loader = test_loader.load()
    #     return train_loader, val_loader, test_loader
    return train_loader, val_loader, test_loader

def prep_dataloaders_for_supervised_learning(cfg,
                                             train_animal_id_list,
                                            val_animal_id_list,
                                            test_animal_id_list,
                                            test_only=False):
    # train_animal_id_list = cfg.dataset.labelled.animal_id_list.train
    # val_animal_id_list = cfg.dataset.labelled.animal_id_list.val
    # test_animal_id_list = cfg.dataset.labelled.animal_id_list.test
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

class DatasetLogbot2(BaseDataset):
    def __init__(self,
                 samples=None,
                 labels=None,
                 paths=None,
                 # augmentation=False,
                 da_type='random',
                 # da_param1=None,  # None -> default params
                 # da_param2=None,
                 in_ch=3):
        super(DatasetLogbot2, self).__init__(samples, labels)
        self.paths = paths
        # self.augmentation = augmentation
        self.da_type = da_type
        # self.da_param1 = da_param1
        # self.da_param2 = da_param2
        self.in_ch = in_ch

    def __getitem__(self, index):
        # if self.samples is None:
        #     # load data and get label
        #     npz = np.load(self.paths[index], allow_pickle=True)
        #     sample = npz["X"]
        #     target = npz["label_id"]
        # else:
        sample, target = self.samples[index], self.labels[index]

        # check the shape of sample
        # print(f"sample.shape: {sample.shape}") # sample.shape: (1, 50, 3)
        # print(target.shape)
        # otsuka code: here is data augmentation

        if isinstance(sample, np.ndarray):
            sample = torch.from_numpy(sample)
        return sample, target

    def __len__(self):
        return len(self.samples)
        # return len(self.paths)
    #
    # def load(self, MAX_INSTS=MAX_INSTS):
    # def unload(self):


# class DataLoaderNpz2(DataLoader):
#     def __init__(self, dataset, batch_size=64, shuffle=False, drop_last=True):
#         super(
#             DataLoaderNpz2,
#             self).__init__(
#             dataset,
#             batch_size=batch_size,
#             shuffle=shuffle,
#             drop_last=drop_last)
#         self.shuffle = shuffle
#
#     def load(self, MAX_INSTS=MAX_INSTS):
#         dataset = self.dataset.load(MAX_INSTS)
#         return DataLoaderNpz2(dataset,
#                               batch_size=self.batch_size,
#                               shuffle=self.shuffle,
#                               drop_last=self.drop_last)
#
#     def unload(self):
#         self.dataset.unload()