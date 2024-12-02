import numpy as np
import os
import matplotlib.pyplot as plt
import pickle
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn

from scipy.interpolate import interp1d
import math
from datetime import datetime
from tqdm import tqdm
from sklearn.manifold import TSNE
import torch.nn.functional as F
# import pickle
import copy

# get label id from label string
label_dict = {
    'ground_stationary': 0,
    'stationary': 0,
    'preening': 0,
    'bathing': 1,
    'bathing_poss': 1,
    'flight_take_off': 2,
    'flight_cruising': 3,
    'flying_active': 4,
    'flying_passive': 4,
    'foraging': 5,
    'poss_foraging': 5,
    'foraging_fish_poss': 6,
    'foraging_insect_poss': 7,
    'foraging_insect': 7,
    'foraging_non-fish': 8,
    'foraging_poss': 9,
    'foraging_dive': 10,
    'surface_seizing': 11,
    'body_shaking': 12,
    'ground_active': 13,
    'unknown': 14,  #-1
}


back_label_path = r'D:\logbot-data\BioTaggerData\masterLabelsByOtsuka\animal_id.csv'
back_label_pd = pd.read_csv(back_label_path)
def get_backid_samplerate(result):
    [species, year, tag] = result
    # filtering
    filtered_rows = back_label_pd[(back_label_pd['species'] == species) &
                                  (back_label_pd['animal_tag'] == tag) &
                                  (back_label_pd['year'] == year)]

    # Get the values from column C for the filtered rows
    result_values = [0, 0]
    result_values[0] = filtered_rows['back'].values[0]
    result_values[1] = filtered_rows['acc_sampling_rate'].values[0]
    return result_values

def majority_value(arr):
    majority = []
    for row in arr:
        values, counts = np.unique(row, return_counts=True)
        majority.append(values[np.argmax(counts)])
    return np.array(majority)

def AE_eval_time_series(train_loader, model, device, memotimes=30):
    model.eval()

    representation_list = []
    sample_list, timestamp_list, label_list, pred_list, timestr_list, flag_list = [], [], [], [], [], []
    for i, (sample, label) in enumerate(train_loader):

        if i > memotimes:  # cpu memory not enough
            continue
        sample = sample.to(device=device, non_blocking=True, dtype=torch.float)
        # input of autoencoder will be 3D, the backbone is 1d-cnn
        x_encoded, output = model(sample)  # x_encoded.shape=batch512,outchannel128,len13
        # print(type(output))
        # x_encoded, output = model(input_).view(b, 2, -1)  # output.shape=b,2,128, split the first dim into 2 parts
        tmp_representation = x_encoded.detach().cpu().numpy()
        representation_list.append(tmp_representation)
        sample_list.append(sample.detach().cpu().numpy())
        label_list.append(label.detach().cpu().numpy())
        pred_list.append(output.detach().cpu().numpy())

    return representation_list, sample_list, pred_list, label_list


def Classify_eval_time_series(train_loader, model, device):
    model.eval()

    representation_list = []
    sample_list, timestamp_list, label_list, pred_list, timestr_list, flag_list = [], [], [], [], [], []
    for i, (sample, label) in enumerate(train_loader):
        sample = sample.to(device=device, non_blocking=True, dtype=torch.float)

        # input of autoencoder will be 3D, the backbone is 1d-cnn
        x_encoded, output = model(sample)  # x_encoded.shape=batch512,outchannel128,len13
        # x_encoded = model.feature_extractor(sample)
        # print(type(output))
        # x_encoded, output = model(input_).view(b, 2, -1)  # output.shape=b,2,128, split the first dim into 2 parts
        tmp_representation = x_encoded.detach().cpu().numpy()
        representation_list.append(tmp_representation)
        sample_list.append(sample.detach().cpu().numpy())
        label_list.append(label.detach().cpu().numpy())
        pred_list.append(output.detach().cpu().numpy())

    return representation_list, sample_list, pred_list, label_list


def plot_reconstruction_result(representation_list, sample_list, pred_list, label_list):
    # tsne latent representation to shape=(2, len) PCA降维到形状为 (2, len)
    repre_concat = np.concatenate(representation_list)
    repre_reshape = repre_concat.reshape(repre_concat.shape[0], -1)

    sample_concat = np.concatenate(sample_list)
    # sample_reshape = sample_concat.reshape(-1, 3)
    sample_concat = sample_concat.transpose(0, 2, 1)
    sample_reshape = sample_concat.reshape(-1, sample_concat.shape[-1])

    pred_concat = np.concatenate(pred_list)
    # pred_reshape = pred_concat.reshape(-1, 3)
    pred_concat = pred_concat.transpose(0, 2, 1)
    pred_reshape = pred_concat.reshape(-1, pred_concat.shape[-1])

    label_concat = np.concatenate(label_list)
    label_concat_vote = majority_value(label_concat)
    # label_concat_vote.shape

    start, end = 0, -1
    col = 0
    fig, axes = plt.subplots(3, 1, figsize=(8, 6))
    axes[0].plot(sample_reshape[start:end, col], 'r', label='groundtruthY')
    axes[0].plot(pred_reshape[start:end, col], 'b-.', label='predictY')
    axes[0].set_title('Autoencoder_Reconstruct_Umineko2018_back1_accel_axisy')
    axes[0].set_xlabel('timestamp')
    axes[0].set_ylabel('ACC[G]')
    axes[0].legend()

    axes[1].plot(sample_reshape[start:end, col], 'r', label='groundtruthY')
    # axes[1].plot(pred_reshape[10000:length, 1], 'b-.', label='predictY')
    # axes[1].set_title('ResNet_SSL_pretrained_Reconstruct_Umineko2018_back1')
    axes[1].set_xlabel('timestamp')
    axes[1].set_ylabel('ACC[G]')
    axes[1].legend()

    # axes[2].plot(sample_reshape[10000:length, 1], 'r', label='groundtruthY')
    axes[2].plot(pred_reshape[start:end, col], 'b-.', label='predictY')
    # axes[2].set_title('ResNet_SSL_pretrained_Reconstruct_Umineko2018_back1')
    axes[2].set_xlabel('timestamp')
    axes[2].set_ylabel('ACC[G]')
    axes[2].legend()

    # Adjust layout
    plt.tight_layout()
    # Show the figure
    plt.show()
    print('')
    return

def get_info_from_csv(p):
    # 使用 os.path.basename() 获取路径中的最后一个文件名
    file_name = os.path.basename(p)
    # file_name
    # 去掉扩展名
    file_name_without_extension = file_name.rsplit('.', 1)[0]

    # 按下划线分割
    parts = file_name_without_extension.split('_')

    # 提取所需的信息
    # 假设你需要的固定模式是：<name><year>_<other>_<code>_<id>
    name_year = parts[0]  # 如 'Umineko2022'
    code = parts[-2]  # 如 'LB09'

    # 将 'Umineko2022' 分成 'Umineko' 和 '2022'
    name = ''.join(filter(str.isalpha, name_year))  # 提取字母部分
    year = ''.join(filter(str.isdigit, name_year))  # 提取数字部分

    # 将提取的信息放入数组
    result = [name.lower(), int(year), code]
    return result

def format_timestamp(df):
    # if 'datetime' not in df.columns:
    s = df['timestamp'].str.replace('T', ' ').str.replace('Z', '')
    # df = df.drop('timestamp', axis=1)
    s_datetime = pd.to_datetime(s)  # to datetime64[ns]
    df.insert(loc=0, column='datetime', value=s_datetime)
    # round at 1 millisecond
    df['datetime'] = df['datetime'].dt.round('1L')
    # unixtime
    unixtime = df['datetime'].apply(lambda t: t.timestamp())
    df.insert(loc=1, column='unixtime', value=unixtime)
    return df

def check_if_has_str(listdata):
    # check if a list contains string values
    for l in listdata:
        if type(l) == str:
            return True
    return False

def resampling(df, intermediate_sampling_rate=100, output_sampling_rate=25):
    if np.sum(df['unixtime'].duplicated()) > 1:
        # print(len(df[df['unixtime'].duplicated()]))
        df.drop_duplicates(subset='datetime', keep=False, inplace=True)
        # print(len(df[df['unixtime'].duplicated()]))
        print("duplicated index detected -> duplicates removed")
    else:
        print("No duplicates")

    # Generate original time indices
    original_time = np.arange(len(df)) / intermediate_sampling_rate
    # Generate new time indices
    new_length = int(len(df) * output_sampling_rate / intermediate_sampling_rate)
    new_time = np.arange(new_length) / output_sampling_rate
    # Create a new DataFrame to store the resampled values
    resampled_df = pd.DataFrame(index=new_time)
    for column in df.columns:
        # Step 1: Extract unique characters from the column
        unique_chars = df[column].unique()

        if check_if_has_str(unique_chars):

            # Step 2: Create a dictionary that maps each character to a unique integer
            char_to_int = {char: i for i, char in enumerate(unique_chars, start=1)}
            # Step 3: Use the dictionary to replace the characters with their corresponding integers
            df[column] = df[column].map(char_to_int)

            # Create interpolation function for each column
            interp_func = interp1d(original_time, df[column].values, kind='linear', fill_value='extrapolate')
            # Generate new values at the new sampling rate
            resampled_df[column] = interp_func(new_time)

            # Step 4: Create an inverse mapping dictionary
            int_to_char = {v: k for k, v in char_to_int.items()}
            # Step 5: Use the inverse mapping dictionary to convert the integers back to the original characters
            resampled_df[column] = resampled_df[column].map(int_to_char)
        else:
            # Create interpolation function for each column
            interp_func = interp1d(original_time, df[column].values, kind='linear', fill_value='extrapolate')
            # Generate new values at the new sampling rate
            resampled_df[column] = interp_func(new_time)

    resampled_df.drop(columns=['index'], inplace=True)
    return resampled_df

def sliding_window(data, len_sw, step=300):
    # input is a segment of data
    # output is data, timestamp, domain(filename), label
    # sampling rate = 25Hz
    # window size = 900, step = window size/2
    # for umineko data, 目前只处理有标签的data，用slidewin取segment时，保证最后一块segment一定取到。
    # 同时，以每个单独的标签（起止时间）为单位，不要把其他时间的不同label的segment混在一起。

    # batch_size = 512
    # len_sw = 90
    # step = int(len_sw / 2)

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

def process_gps(df):
    # identify if gps exists
    # if exists, calculate velocity and angle
    gps_len = len(df)
    df_columns = df.columns
    if ('latitude' in df_columns) or\
        ('longitude' in df_columns):
        # Extract rows where both latitude and longitude are not NaN
        df_non_nan = df.dropna(subset=['latitude', 'longitude'])

        # get sampling rate of GPS signal,newlen*oldHz/oldlen得到GPShz，所以这里传newlen
        gps_len = len(df_non_nan)

        # Calculate differences, handling NaN by filling with zeros
        df_non_nan['lat_diff'] = np.radians(df_non_nan['latitude'].diff())
        df_non_nan['lon_diff'] = np.radians(df_non_nan['longitude'].diff())

        # Convert latitude to radians, handling NaN by filling with zeros
        df_non_nan['lat1'] = np.radians(df_non_nan['latitude'].shift())
        df_non_nan['lat2'] = np.radians(df_non_nan['latitude'])

        # Calculate time difference in seconds
        df_non_nan['timestamp'] = pd.to_datetime(df_non_nan['timestamp'])
        df_non_nan['time_diff'] = df_non_nan['timestamp'].diff().dt.total_seconds()

        # Haversine formula
        a = (np.sin(df_non_nan['lat_diff'] / 2) ** 2 +
             np.cos(df_non_nan['lat1']) * np.cos(df_non_nan['lat2']) * np.sin(df_non_nan['lon_diff'] / 2) ** 2)
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        R = 6371000  # Earth radius in meters
        df_non_nan['distance'] = R * c
        # Calculate velocity (m/s)
        df_non_nan['GPS_velocity'] = df_non_nan['distance'] / df_non_nan['time_diff']

        # Calculate bearing
        x = np.sin(df_non_nan['lon_diff']) * np.cos(df_non_nan['lat2'])
        y = (np.cos(df_non_nan['lat1']) * np.sin(df_non_nan['lat2']) -
             np.sin(df_non_nan['lat1']) * np.cos(df_non_nan['lat2']) * np.cos(df_non_nan['lon_diff']))
        initial_bearing = np.arctan2(x, y)
        initial_bearing = np.degrees(initial_bearing)
        df_non_nan['GPS_bearing'] = (initial_bearing + 360) % 360

        # Merge velocity and bearing back to the original dataframe
        df = df.merge(df_non_nan[['GPS_velocity', 'GPS_bearing']], left_index=True, right_index=True, how='left')

    return df, gps_len




class data_loader_umineko(Dataset):
    def __init__(self, samples, labels, device='cpu'):
        self.samples = torch.tensor(samples).to(device)  # check data type
        self.labels = torch.tensor(labels)  # check data type

    def __getitem__(self, index):
        target = self.labels[index]
        sample = self.samples[index]
        return sample, target

    def __len__(self):
        return len(self.labels)

def adjust_learning_rate(lr, optimizer, epoch, p_scheduler, p_epochs):
    # lr = 0.0001

    if p_scheduler == 'cosine':
        lr_decay_rate = 0.1
        eta_min = lr * (lr_decay_rate ** 3)
        lr = eta_min + (lr - eta_min) * (1 + math.cos(math.pi * epoch / p_epochs)) / 2

    # elif p['scheduler'] == 'step':
    #     steps = np.sum(epoch > np.array(p['scheduler_kwargs']['lr_decay_epochs']))
    #     if steps > 0:
    #         lr = lr * (p['scheduler_kwargs']['lr_decay_rate'] ** steps)

    elif p_scheduler == 'constant':
        lr = lr

    else:
        raise ValueError('Invalid learning rate schedule {}'.format(p_scheduler))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
class MSEloss(nn.Module):
    def __init__(self):
        super(MSEloss, self).__init__()

    def forward(self, input, target):
        '''
        input: raw sensor data
        target: reconstructed sensor data
        the mse loss makes the target data to be similar to the input data
        '''
        loss = nn.MSELoss()
        output = loss(input, target)
        return output


def get_optimizer(p_opti, model):
    params = model.parameters()
    if p_opti == 'sgd':
        optimizer = torch.optim.SGD(params,
                                    nesterov=False,
                                    weight_decay=0.0001,
                                    momentum=0.9,
                                    lr=0.0001)

    elif p_opti == 'adam':
        optimizer = torch.optim.Adam(params,
                                     weight_decay=0.0001,
                                     lr=0.01)
    else:
        raise ValueError('Invalid optimizer {}'.format(p_opti))

    return optimizer


class AverageMeter(object):
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

def AE_train_time_series(train_loader, model, criterion, optimizer):
    """
    Train according to the scheme from SimCLR
    https://arxiv.org/abs/2002.05709
    """
    losses = AverageMeter('Loss', ':.4e')

    model.train()

    for i, (sample, label) in enumerate(train_loader):
        # aug_sample1 = gen_aug(sample, 't_warp')  # t_warp, out.shape=batch64,width3,height900
        # reshape data by adding channel to 1, and transpose height and width
        # sample = sample.to(device=device, non_blocking=True, dtype=torch.float)
        sample = sample.to(dtype=torch.float)

        # input of autoencoder will be 3D, the backbone is 1d-cnn
        x_encoded, output = model(sample)  # x_encoded.shape=batch512,outchannel128,len13
        loss = criterion(sample, output)
        losses.update(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    #     if i % 10 == 0:
    #         print('loss of the ' + str(epoch) + '-th training epoch is :' + losses.__str__())
    # print('loss of the ' + str(epoch) + '-th training epoch is :' + losses.__str__())
    return losses

def AE_train_time_series_resnet(train_loader, model, criterion, optimizer, epoch, scheduler, device='cuda'):
    """
    Train according to the scheme from SimCLR
    https://arxiv.org/abs/2002.05709
    """
    # losses = AverageMeter('Loss', ':.4e')
    losses = []

    model.train()
    warm_up_step = 100
    for i, (sample, label) in enumerate(train_loader):
        # aug_sample1 = gen_aug(sample, 't_warp')  # t_warp, out.shape=batch64,width3,height900
        # reshape data by adding channel to 1, and transpose height and width
        # sample = sample.to(device=device, non_blocking=True, dtype=torch.float)
        sample = sample.to(device=device, dtype=torch.float)

        # input of autoencoder will be 3D, the backbone is 1d-cnn
        x_encoded, output = model(sample)  # x_encoded.shape=batch512,outchannel128,len13
        loss = criterion(sample, output)
        losses.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if (epoch >= warm_up_step):
    # if (epoch >= warm_up_step) and (scheduler.optimizer.param_groups[0]["lr"] >0.00002):
        # scheduler.step()
        scheduler.step(np.average(losses))
        print(
            f'Epoch {epoch + 1}, Learning Rate: {scheduler.optimizer.param_groups[0]["lr"]}')
    #     if i % 10 == 0:
    #         print('loss of the ' + str(epoch) + '-th training epoch is :' + losses.__str__())
    # print('loss of the ' + str(epoch) + '-th training epoch is :' + losses.__str__())
    return losses

def Classify_train_time_series_resnet(train_loader, model, criterion, optimizer, epoch, scheduler, device='cuda'):
    """
    Train according to the scheme from SimCLR
    https://arxiv.org/abs/2002.05709
    """
    losses = AverageMeter('Loss', ':.4e')

    model.train()
    warm_up_step = 5
    for i, (sample, label) in enumerate(train_loader):
        # aug_sample1 = gen_aug(sample, 't_warp')  # t_warp, out.shape=batch64,width3,height900
        # reshape data by adding channel to 1, and transpose height and width
        # sample = sample.to(device=device, non_blocking=True, dtype=torch.float)
        sample = sample.to(device=device, dtype=torch.float)
        label = label.to(device=device, dtype=torch.long)

        # input of autoencoder will be 3D, the backbone is 1d-cnn
        # sample = sample.transpose(0, 2, 1)
        feature, output = model(sample)  # x_encoded.shape=batch512,outchannel128,len13
        # print(output.shape)
        # print(type(label[:,0]))
        loss = criterion(output, label[:,0])
        losses.update(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch >= warm_up_step):
        scheduler.step()
    #     if i % 10 == 0:
    #         print('loss of the ' + str(epoch) + '-th training epoch is :' + losses.__str__())
    # print('loss of the ' + str(epoch) + '-th training epoch is :' + losses.__str__())
    return losses


def load_weights(
    weight_path, model, my_device="cpu", name_start_idx=2, is_dist=False
):
    # only need to change weights name when the
    # model is trained in a distributed manner

    pretrained_dict = torch.load(weight_path, map_location=my_device)
    pretrained_dict_v2 = copy.deepcopy(
        pretrained_dict
    )  # v2 has the right para names

    if is_dist:
        for key in pretrained_dict:
            para_names = key.split(".")
            new_key = ".".join(para_names[name_start_idx:])
            pretrained_dict_v2[new_key] = pretrained_dict_v2.pop(key)

    model_dict = model.state_dict()

    # 1. filter out unnecessary keys such as the final linear layers
    #    we don't want linear layer weights either
    pretrained_dict = {
        k: v
        for k, v in pretrained_dict_v2.items()
        if k in model_dict and k.split(".")[0] != "classifier"
    }

    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)

    # 3. load the new state dict
    model.load_state_dict(model_dict)
    print("%d Weights loaded" % len(pretrained_dict))

def freeze_feature_extractor(model):
    # Check if parameters are frozen
    for name, param in model.feature_extractor.named_parameters():
        print(f"{name} requires_grad: {param.requires_grad}")
    return model

class Downsample(nn.Module):
    r"""Downsampling layer that applies anti-aliasing filters.
    For example, order=0 corresponds to a box filter (or average downsampling
    -- this is the same as AvgPool in Pytorch), order=1 to a triangle filter
    (or linear downsampling), order=2 to cubic downsampling, and so on.
    See https://richzhang.github.io/antialiased-cnns/ for more details.
    """

    def __init__(self, channels=None, factor=2, order=1):
        super(Downsample, self).__init__()
        assert factor > 1, "Downsampling factor must be > 1"
        self.stride = factor
        self.channels = channels
        self.order = order

        # Figure out padding and check params make sense
        # The padding is given by order*(factor-1)/2
        # so order*(factor-1) must be divisible by 2
        total_padding = order * (factor - 1)
        assert total_padding % 2 == 0, (
            "Misspecified downsampling parameters."
            "Downsampling factor and order must be such "
            "that order*(factor-1) is divisible by 2"
        )
        self.padding = int(order * (factor - 1) / 2)

        box_kernel = np.ones(factor)
        kernel = np.ones(factor)
        for _ in range(order):
            kernel = np.convolve(kernel, box_kernel)
        kernel /= np.sum(kernel)
        kernel = torch.Tensor(kernel)
        self.register_buffer(
            "kernel", kernel[None, None, :].repeat((channels, 1, 1))
        )

    def forward(self, x):
        return F.conv1d(
            x,
            self.kernel,
            stride=self.stride,
            padding=self.padding,
            groups=x.shape[1],
        )

#-----------------------Resnet------------------------------------------

class ResBlock(nn.Module):
    r""" Basic bulding block in Resnets:

       bn-relu-conv-bn-relu-conv
      /                         \
    x --------------------------(+)->

    """

    def __init__(
        self, in_channels, out_channels, kernel_size=5, stride=1, padding=2
    ):

        super(ResBlock, self).__init__()

        self.bn1 = nn.BatchNorm1d(in_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            bias=False,
            padding_mode="circular",
        )
        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            bias=False,
            padding_mode="circular",
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        x = self.relu(self.bn1(x))
        x = self.conv1(x)
        x = self.relu(self.bn2(x))
        x = self.conv2(x)

        x = x + identity

        return x

class Reconstructor(nn.Module):
    def __init__(self, input_size=512, len_sw=300):
        super().__init__()
        self.len_sw = len_sw
        self.decoder = nn.Sequential(
            nn.Linear(input_size, 2048),
            nn.Mish(),
            nn.Linear(2048, 1024),
            nn.Mish(),
            nn.Linear(1024, self.len_sw * 3),
            # nn.PReLU()
            # nn.Softmax()
        )

    def forward(self, x):
        x = self.decoder(x)
        x = x.reshape(x.shape[0], -1, self.len_sw)  # batch,dim,len
        return x


class ReconstructorConv(nn.Module):
    def __init__(self, input_size=512, len_sw=300):
        super().__init__()
        self.len_sw = len_sw
        # self.decoder = nn.Sequential(
        #     nn.Linear(input_size, 2048),
        #     nn.Mish(),
        #     nn.Linear(2048, 1024),
        #     nn.Mish(),
        #     nn.Linear(1024, self.len_sw * 3),
        #     nn.Mish()
        # )

        self.Conv1 = nn.Conv1d(1, 16, 3, stride=1)
        self.bn1 = nn.BatchNorm1d(16)
        self.drop = nn.Dropout(0.5)
        self.ac1 = nn.Mish()
        self.Conv2 = nn.Conv1d(16, 3, 3, stride=2)
        self.bn2 = nn.BatchNorm1d(3)
        self.drop2 = nn.Dropout(0.5)
        self.ac2 = nn.Mish()
        self.linear1 = nn.Linear(510, self.len_sw)
        self.ac3 = nn.Mish()

    def forward(self, x):  # batch, 1024
        # x = self.decoder(x)
        # x = x.reshape(x.shape[0], -1, self.len_sw)  # batch,dim,len
        x = x.unsqueeze(1)
        x = self.Conv1(x)
        x = self.bn1(x)
        x = self.drop(x)
        x = self.ac1(x)

        x = self.Conv2(x)
        x = self.bn2(x)
        x = self.drop2(x)
        x = self.ac2(x)

        # x = x.reshape(x.shape[0], self.len_sw, -1)
        x = self.linear1(x)
        x = self.ac3(x)
        # x = x.reshape(x.shape[0], -1, self.len_sw)

        return x

class Resnet(nn.Module):
    r"""The general form of the architecture can be described as follows:

    x->[Conv-[ResBlock]^m-BN-ReLU-Down]^n->y

    In other words:

            bn-relu-conv-bn-relu-conv                        bn-
           /                         \                      /
    x->conv --------------------------(+)-bn-relu-down-> conv ----

    """

    def __init__(
        self,
        output_size=1,
        n_channels=3,
            len_sw=300,
        is_eva=False,
        resnet_version=1,
        epoch_len=10,
        is_mtl=False,
        is_simclr=False,
            is_reconst=False
    ):
        super(Resnet, self).__init__()

        # Architecture definition. Each tuple defines
        # a basic Resnet layer Conv-[ResBlock]^m]-BN-ReLU-Down
        # isEva: change the classifier to two FC with ReLu
        # For example, (64, 5, 1, 5, 3, 1) means:
        # - 64 convolution filters
        # - kernel size of 5
        # - 1 residual block (ResBlock)
        # - ResBlock's kernel size of 5
        # - downsampling factor of 3
        # - downsampling filter order of 1
        # In the below, note that 3*3*5*5*4 = 900 (input size)
        if resnet_version == 1:
            if epoch_len == 5:
                cgf = [
                    (64, 5, 2, 5, 2, 2),
                    (128, 5, 2, 5, 2, 2),
                    (256, 5, 2, 5, 3, 1),
                    (256, 5, 2, 5, 3, 1),
                    (512, 5, 0, 5, 3, 1),
                ]
            elif epoch_len == 10:
                cgf = [
                    (64, 5, 2, 5, 2, 2),
                    (128, 5, 2, 5, 2, 2),
                    (256, 5, 2, 5, 5, 1),
                    (512, 5, 2, 5, 5, 1),
                    (1024, 5, 0, 5, 3, 1),
                ]
            else:
                cgf = [
                    (64, 5, 2, 5, 3, 1),
                    (128, 5, 2, 5, 3, 1),
                    (256, 5, 2, 5, 5, 1),
                    (512, 5, 2, 5, 5, 1),
                    (1024, 5, 0, 5, 4, 0),
                ]
        else:
            cgf = [
                (64, 5, 2, 5, 3, 1),
                (64, 5, 2, 5, 3, 1),
                (128, 5, 2, 5, 5, 1),
                (128, 5, 2, 5, 5, 1),
                (256, 5, 2, 5, 4, 0),
            ]  # smaller resnet
        in_channels = n_channels
        feature_extractor = nn.Sequential()
        for i, layer_params in enumerate(cgf):
            (
                out_channels,
                conv_kernel_size,
                n_resblocks,
                resblock_kernel_size,
                downfactor,
                downorder,
            ) = layer_params
            feature_extractor.add_module(
                f"layer{i+1}",
                Resnet.make_layer(
                    in_channels,
                    out_channels,
                    conv_kernel_size,
                    n_resblocks,
                    resblock_kernel_size,
                    downfactor,
                    downorder,
                ),
            )
            in_channels = out_channels

        self.feature_extractor = feature_extractor
        self.is_mtl = is_mtl

        # Classifier input size = last out_channels in previous layer
        if is_eva:
            self.classifier = EvaClassifier(
                input_size=out_channels, output_size=output_size
            )
        elif is_mtl:
            self.aot_h = Classifier(
                input_size=out_channels, output_size=output_size
            )
            self.scale_h = Classifier(
                input_size=out_channels, output_size=output_size
            )
            self.permute_h = Classifier(
                input_size=out_channels, output_size=output_size
            )
            self.time_w_h = Classifier(
                input_size=out_channels, output_size=output_size
            )
        elif is_simclr:
            self.classifier = ProjectionHead(
                input_size=out_channels, encoding_size=output_size
            )
        elif is_reconst:
            self.classifier = Reconstructor(
                input_size=out_channels,
                len_sw=len_sw,
            )

        weight_init(self)

    @staticmethod
    def make_layer(
        in_channels,
        out_channels,
        conv_kernel_size,
        n_resblocks,
        resblock_kernel_size,
        downfactor,
        downorder=1,
    ):
        r""" Basic layer in Resnets:

        x->[Conv-[ResBlock]^m-BN-ReLU-Down]->

        In other words:

                bn-relu-conv-bn-relu-conv
               /                         \
        x->conv --------------------------(+)-bn-relu-down->

        """

        # Check kernel sizes make sense (only odd numbers are supported)
        assert (
            conv_kernel_size % 2
        ), "Only odd number for conv_kernel_size supported"
        assert (
            resblock_kernel_size % 2
        ), "Only odd number for resblock_kernel_size supported"

        # Figure out correct paddings
        conv_padding = int((conv_kernel_size - 1) / 2)
        resblock_padding = int((resblock_kernel_size - 1) / 2)

        modules = [
            nn.Conv1d(
                in_channels,
                out_channels,
                conv_kernel_size,
                1,
                conv_padding,
                bias=False,
                padding_mode="circular",
            )
        ]

        for i in range(n_resblocks):
            modules.append(
                ResBlock(
                    out_channels,
                    out_channels,
                    resblock_kernel_size,
                    1,
                    resblock_padding,
                )
            )

        modules.append(nn.BatchNorm1d(out_channels))
        modules.append(nn.ReLU(True))
        modules.append(Downsample(out_channels, downfactor, downorder))

        return nn.Sequential(*modules)

    def forward(self, x):
        feats = self.feature_extractor(x)

        if self.is_mtl:
            aot_y = self.aot_h(feats.view(x.shape[0], -1))
            scale_y = self.scale_h(feats.view(x.shape[0], -1))
            permute_y = self.permute_h(feats.view(x.shape[0], -1))
            time_w_h = self.time_w_h(feats.view(x.shape[0], -1))
            return aot_y, scale_y, permute_y, time_w_h
        else:
            y = self.classifier(feats.view(x.shape[0], -1))
            return feats, y
        return y

#-----------------------Resnet------------------------------------------

#-----------------------autoencoder------------------------------------------
class Encoder3d(nn.Module):
    def __init__(self):
        super(Encoder3d, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.bn1 = nn.BatchNorm1d(64)

        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        self.bn2 = nn.BatchNorm1d(128)

        self.conv3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool1d(kernel_size=2)
        self.bn3 = nn.BatchNorm1d(256)

        # Calculate the flattened size after all convolutions and pooling
        self.flattened_size = 3072  # Adjust this to match the output of the last pooling layer
        self.fc = nn.Linear(self.flattened_size, 256)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.bn1(x)
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.bn2(x)
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = self.bn3(x)
        x = self.pool3(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)  # Output size will be (batch_size, 1024)
        return x


class Decoder3d(nn.Module):
    def __init__(self):
        super(Decoder3d, self).__init__()
        self.fc = nn.Linear(256, 3072)  # Adjust size if necessary
        self.unflatten = nn.Unflatten(1, (256, 12))

        self.conv_trans1 = nn.ConvTranspose1d(in_channels=256, out_channels=128, kernel_size=3, padding=1)
        self.upsample1 = nn.Upsample(scale_factor=2)
        self.bn1 = nn.BatchNorm1d(128)

        self.conv_trans2 = nn.ConvTranspose1d(in_channels=128, out_channels=64, kernel_size=3, padding=1)
        self.upsample2 = nn.Upsample(scale_factor=2)
        self.bn2 = nn.BatchNorm1d(64)

        self.conv_trans3 = nn.ConvTranspose1d(in_channels=64, out_channels=3, kernel_size=3, padding=1)
        self.upsample3 = nn.Upsample(scale_factor=2)
        self.bn3 = nn.BatchNorm1d(3)

        self.linear = nn.Linear(96, 100)
    def forward(self, x):
        x = F.relu(self.fc(x))
        x = self.unflatten(x)
        x = F.relu(self.conv_trans1(x))
        x = self.bn1(x)
        x = self.upsample1(x)
        x = F.relu(self.conv_trans2(x))
        x = self.bn2(x)
        x = self.upsample2(x)
        x = F.mish(self.conv_trans3(x))
        x = self.bn3(x)
        x = self.upsample3(x)
        x = self.linear(x)
        return x


class Autoencoder3d(nn.Module):
    def __init__(self, is_reconst=True, is_classify=False):
        super(Autoencoder3d, self).__init__()
        self.feature_extractor = Encoder3d()

        self.is_reconst = is_reconst
        self.is_classify = is_classify
        if self.is_reconst:
            self.decoder = Decoder3d()
        elif self.is_classify:
            self.classify = EvaClassifier(output_size=15)
        else:
            print('error: no module in Autoencoder3d.')

        weight_init(self)

    def forward(self, x):
        feature = self.feature_extractor(x)
        # out = self.decoder(feature)
        if self.is_reconst:
            out = self.decoder(feature)
        elif self.is_classify:
            out = self.classify(feature)
        else:
            out = self.decoder(feature)
            print('error: no module in Autoencoder3d.')
        return feature, out


class Encoder3d_transf(nn.Module):
    def __init__(self):
        super(Encoder3d_transf, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=300, out_channels=150, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.batchnorm1 = nn.BatchNorm1d(150)

        self.conv2 = nn.Conv1d(in_channels=150, out_channels=75, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        self.batchnorm2 = nn.BatchNorm1d(75)

        self.conv3 = nn.Conv1d(in_channels=75, out_channels=75, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool1d(kernel_size=2)
        self.batchnorm3 = nn.BatchNorm1d(75)

        # Calculate the flattened size after all convolutions and pooling
        self.flattened_size = 75 * 3  # Adjust this to match the output of the last pooling layer
        self.fc = nn.Linear(self.flattened_size, 1024)

    def forward(self, x):
        x = x.permute(0,2,1)
        x = self.batchnorm1(F.relu(self.conv1(x)))
        # x = self.pool1(x)
        x = self.batchnorm2(F.relu(self.conv2(x)))
        # x = self.pool2(x)
        x = self.batchnorm3(F.relu(self.conv3(x)))
        # x = self.pool3(x)
        x = x.view(x.size(0), -1)  # Flatten
        # x = self.fc(x)  # Output size will be (batch_size, 1024)
        return x


class Decoder3d_transf(nn.Module):
    def __init__(self):
        super(Decoder3d_transf, self).__init__()
        # self.fc = nn.Linear(1024, 256 * 37)  # Adjust size if necessary
        self.unflatten = nn.Unflatten(1, (75, 3))

        self.conv_trans1 = nn.ConvTranspose1d(in_channels=75, out_channels=75, kernel_size=3, padding=1)
        self.upsample1 = nn.Upsample(scale_factor=2)
        self.batchnorm1 = nn.BatchNorm1d(75)

        self.conv_trans2 = nn.ConvTranspose1d(in_channels=75, out_channels=150, kernel_size=3, padding=1)
        self.upsample2 = nn.Upsample(scale_factor=2)
        self.batchnorm2 = nn.BatchNorm1d(150)

        self.conv_trans3 = nn.ConvTranspose1d(in_channels=150, out_channels=300, kernel_size=3, padding=1)
        self.upsample3 = nn.Upsample(scale_factor=2)
        self.batchnorm3 = nn.BatchNorm1d(300)

        # self.linear = nn.Linear(296, 300)
    def forward(self, x):
        # x = F.relu(self.fc(x))
        x = self.unflatten(x)
        x = self.batchnorm1(F.relu(self.conv_trans1(x)))
        # x = self.upsample1(x)
        x = self.batchnorm2(F.relu(self.conv_trans2(x)))
        # x = self.upsample2(x)
        x = self.batchnorm3(F.mish((self.conv_trans3(x))))
        # x = self.upsample3(x)
        # x = self.linear(x)
        x = x.permute(0,2,1)
        return x

class Autoencoder3d_transf(nn.Module):
    def __init__(self, is_reconst=True, is_classify=False):
        super(Autoencoder3d_transf, self).__init__()
        self.feature_extractor = Encoder3d_transf()

        self.is_reconst = is_reconst
        self.is_classify = is_classify
        if self.is_reconst:
            self.decoder = Decoder3d_transf()
        elif self.is_classify:
            self.classify = EvaClassifier(output_size=15)
        else:
            print('error: no module in Autoencoder3d.')

        weight_init(self)

    def forward(self, x):
        feature = self.feature_extractor(x)
        # out = self.decoder(feature)
        if self.is_reconst:
            out = self.decoder(feature)
        elif self.is_classify:
            out = self.classify(feature)
        else:
            out = self.decoder(feature)
            print('error: no module in Autoencoder3d.')
        return feature, out

class Encoder2d(nn.Module):
    def __init__(self):
        super(Encoder2d, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=2, out_channels=32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool1d(kernel_size=2)

        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool1d(kernel_size=2)

        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool1d(kernel_size=2)

        # Calculate the flattened size after all convolutions and pooling
        self.flattened_size = 128 * 37  # Adjust this to match the output of the last pooling layer
        self.fc = nn.Linear(self.flattened_size, 1024)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = self.pool3(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)  # Output size will be (batch_size, 1024)
        return x


class Decoder2d(nn.Module):
    def __init__(self):
        super(Decoder2d, self).__init__()
        self.fc = nn.Linear(1024, 128 * 37)  # Adjust size if necessary
        self.unflatten = nn.Unflatten(1, (128, 37))

        self.conv_trans1 = nn.ConvTranspose1d(in_channels=128, out_channels=64, kernel_size=3, padding=1)
        self.upsample1 = nn.Upsample(scale_factor=2)

        self.conv_trans2 = nn.ConvTranspose1d(in_channels=64, out_channels=32, kernel_size=3, padding=1)
        self.upsample2 = nn.Upsample(scale_factor=2)

        self.conv_trans3 = nn.ConvTranspose1d(in_channels=32, out_channels=2, kernel_size=3, padding=1)
        self.upsample3 = nn.Upsample(scale_factor=2)

        self.linear = nn.Linear(296, 300)
    def forward(self, x):
        x = F.relu(self.fc(x))
        x = self.unflatten(x)
        x = F.relu(self.conv_trans1(x))
        x = self.upsample1(x)
        x = F.relu(self.conv_trans2(x))
        x = self.upsample2(x)
        x = torch.sigmoid(self.conv_trans3(x))
        x = self.upsample3(x)
        x = self.linear(x)
        return x


class Autoencoder2d(nn.Module):
    def __init__(self):
        super(Autoencoder2d, self).__init__()
        self.feature_extractor = Encoder2d()
        self.decoder = Decoder2d()

        weight_init(self)

    def forward(self, x):
        feature = self.feature_extractor(x)
        out = self.decoder(feature)
        return feature, out


class Encoder1d(nn.Module):
    def __init__(self):
        super(Encoder1d, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool1d(kernel_size=2)

        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool1d(kernel_size=2)

        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool1d(kernel_size=2)

        # Calculate the flattened size after all convolutions and pooling
        self.flattened_size = 64 * 37  # Adjust this to match the output of the last pooling layer
        self.fc = nn.Linear(self.flattened_size, 1024)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = self.pool3(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)  # Output size will be (batch_size, 1024)
        return x


class Decoder1d(nn.Module):
    def __init__(self):
        super(Decoder1d, self).__init__()
        self.fc = nn.Linear(1024, 64 * 37)  # Adjust size if necessary
        self.unflatten = nn.Unflatten(1, (64, 37))

        self.conv_trans1 = nn.ConvTranspose1d(in_channels=64, out_channels=32, kernel_size=3, padding=1)
        self.upsample1 = nn.Upsample(scale_factor=2)

        self.conv_trans2 = nn.ConvTranspose1d(in_channels=32, out_channels=16, kernel_size=3, padding=1)
        self.upsample2 = nn.Upsample(scale_factor=2)

        self.conv_trans3 = nn.ConvTranspose1d(in_channels=16, out_channels=1, kernel_size=3, padding=1)
        self.upsample3 = nn.Upsample(scale_factor=2)

        self.linear = nn.Linear(296, 300)
    def forward(self, x):
        x = F.relu(self.fc(x))
        x = self.unflatten(x)
        x = F.relu(self.conv_trans1(x))
        x = self.upsample1(x)
        x = F.relu(self.conv_trans2(x))
        x = self.upsample2(x)
        x = torch.sigmoid(self.conv_trans3(x))
        x = self.upsample3(x)
        x = self.linear(x)
        return x


class Autoencoder1d(nn.Module):
    def __init__(self):
        super(Autoencoder1d, self).__init__()
        self.feature_extractor = Encoder1d()
        self.decoder = Decoder1d()

        weight_init(self)

    def forward(self, x):
        feature = self.feature_extractor(x)
        out = self.decoder(feature)
        return feature, out
#-----------------------autoencoder------------------------------------------

def weight_init(self, mode="fan_out", nonlinearity="relu"):

    for m in self.modules():

        if isinstance(m, (nn.Conv1d, nn.Linear, nn.ConvTranspose1d)):
            nn.init.kaiming_normal_(
                m.weight, mode=mode, nonlinearity=nonlinearity
            )

        elif isinstance(m, (nn.BatchNorm1d)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

class EvaClassifier(nn.Module):
    def __init__(self, input_size=1024, nn_size=512, output_size=2):
        super(EvaClassifier, self).__init__()
        self.linear1 = torch.nn.Linear(input_size, nn_size)
        self.linear2 = torch.nn.Linear(nn_size, output_size)

    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        return x

class Classifier(nn.Module):
    def __init__(self, input_size=1024, output_size=2):
        super(Classifier, self).__init__()
        self.linear1 = torch.nn.Linear(input_size, output_size)

    def forward(self, x):
        y_pred = self.linear1(x)
        return y_pred


class ProjectionHead(nn.Module):
    def __init__(self, input_size=1024, nn_size=256, encoding_size=100):
        super(ProjectionHead, self).__init__()
        self.linear1 = torch.nn.Linear(input_size, nn_size)
        self.linear2 = torch.nn.Linear(nn_size, encoding_size)

    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        return x