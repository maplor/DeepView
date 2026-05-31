




################################## Imports###################################
import os
import pandas as pd
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import Dataset, DataLoader
import plotly.express as px
from umap import UMAP
from sklearn.cluster import KMeans
from scipy.stats import entropy
from sklearn.semi_supervised import LabelSpreading
################################## Imports###################################




################################## read data #################################

labelcategory_dict = {
    'Breathing': 'Breathing',
    'Catching': 'Feeding',
    'Chewing on movement': 'Feeding',
    'Chewing stationary': 'Feeding',
    'Escape': 'Unknown',
    'Flipper beat': 'Unknown',
    'Foraging': 'Unknown',
    'Gliding ascent': 'Gliding',
    'Gliding descent': 'Gliding',
    'Grabbing on movement': 'Feeding',
    'Grabbing stationary': 'Feeding',
    'Interaction': 'Unknown',
    'Left U-turn': 'Swimming',
    'New video': 'Unknown', #??
    'Obstacle': 'Unknown',
    'Prospection': 'Swimming',
    'Pursuit': 'Unknown',
    'Resting': 'Resting',
    'Resting active': 'Resting',
    'Resting in flow': 'Resting',
    'Right U-turn': 'Swimming',
    'Scratching': 'Scratching',
    'Shaking': 'Unknown',
    'Shaking head': 'Unknown',
    'Stay in surface': 'Stay in surface',
    'Swimming 1 ascent': 'Swimming',
    'Swimming 1 descent': 'Swimming',
    'Swimming 1 horizontally': 'Swimming',
    'Swimming ascent': 'Swimming',
    'Swimming descent': 'Swimming',
    'Swimming fast descent': 'Swimming',
    'Swimming fast horizontally': 'Swimming',
    'Swimming horizontally': 'Swimming',
    'Swimming in place': 'Swimming',
    'Swimming on the bottom': 'Swimming',
    'Watching': 'Swimming',
    'Catching jellyfish': 'Feeding',
    'Resting watching': 'Resting',
    'Sand': 'Unknown',
    'Scratching camera': 'Scratching',
    'Swimming fast ascent': 'Swimming',
    'Chewing jellyfish': 'Feeding',
    'Landing': 'Unknown',
    'Scratching head': 'Scratching',
    'Stepping back': 'Swimming',
    'Regurgitating': 'Unknown',
    'Time .': 'Unknown', #'rest_passive',#??
    'Grabbing the wall': 'Feeding',
    'Hunting jellyfish': 'Unknown',
    'Unknown': 'Unknown',
}

label_dict = {
    'Resting': 0,
    'Swimming': 1,
    'Stay in surface': 2,
    'Gliding': 3,
    'Feeding': 4,
    'Scratching': 5,
    'Breathing': 6,
    'Unknown': -1,
}


def read_sensor_data():
    df_list = []
    with open(r'D:\code\DeepView\deepview\calculate_results\data\turtle\turtle.pkl', 'rb') as f:
        while True:
            try:
                item = pickle.load(f)
                df_list.append(item)
            except EOFError:
                break
    df_all = pd.concat(df_list, ignore_index=True)
    df_all['category'] = df_all['Label'].map(labelcategory_dict)
    df_all['label_id'] = df_all['category'].map(label_dict)
    df_all['label_id'] = df_all['label_id'].fillna(-2)

    # # 删除无标签的数据
    # selected_df = selected_df[selected_df.label_id != -2]
    return df_all

def gaussian_std(X):
    mean_val = np.mean(X.astype(float), axis=0)
    std_val = np.std(X.astype(float), axis=0)
    X_standardized = (X - mean_val) / np.maximum(std_val, 10 ** -5)
    return X_standardized, mean_val, std_val

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

def find_majority_minority(label_b):
    unique_labels, counts = np.unique(label_b, return_counts=True)

    if len(unique_labels) == 1:
        print(f"Only one label present: {unique_labels[0]}")
        return unique_labels[0], unique_labels[0]  # 只有一个类别，返回相同的值

    # 计算多数类和少数类
    majority_label = unique_labels[np.argmax(counts)]
    minority_label = unique_labels[np.argmin(counts)]

    return majority_label, minority_label

def majority_value(arr):
    if isinstance(arr, torch.Tensor):
        arr = arr.detach().cpu().numpy()
    majority = []
    for row in arr:
        values, counts = np.unique(row, return_counts=True)
        majority.append(values[np.argmax(counts)])
    return np.array(majority)


# sensor_types = ['accelerometer', 'gyroscope', 'magnetometer',
#                'pressure', 'temperature', 'light', 'sound', 'depth']  # GUI上选中的传感器类型
sensor_types = ['accelerometer', 'Depth']

all_df = read_sensor_data()
selected_df = all_df[all_df.label_id > -1]
fill_selected_df = selected_df.fillna(method='ffill').fillna(method='bfill')

sensor_list = []
# todo: 传感器名字和对应的数据列名字有差别，现在是写死的，后续想办法改成从传感器类型中获取对应的列名
for sensor in sensor_types:
    if sensor == 'accelerometer':
        selected_np = fill_selected_df[['AccX', 'AccY', 'AccZ']].values
        tmp_acc_stand, mean_val, std_val = gaussian_std(selected_np)
        sensor_list.append(tmp_acc_stand)
    elif sensor == 'gyroscope':
        selected_np = fill_selected_df[['gyr_x', 'gyr_y', 'gyr_z']].values
        tmp_gyro_stand, mean_val, std_val = gaussian_std(selected_np)
        sensor_list.append(tmp_gyro_stand)
    elif sensor == 'magnetometer':
        selected_np = fill_selected_df[['mag_x', 'mag_y', 'mag_z']].values
        tmp_mag_stand, mean_val, std_val = gaussian_std(selected_np)
        sensor_list.append(tmp_mag_stand)
    elif sensor in ['temperature', 'Depth']:
        selected_np = fill_selected_df[sensor].values
        tmp_sensor_stand, mean_val, std_val = gaussian_std(selected_np)
        sensor_list.append(tmp_sensor_stand.reshape(-1,1))
    elif sensor in ['pressure']:
        selected_np = fill_selected_df[sensor].values
        data = selected_np - 1013.25  # standard pressure
        tmp_sensor_stand, mean_val, std_val = gaussian_std(data)
    elif sensor == 'GPS':
        fill_selected_df, gps_len = process_gps(fill_selected_df)
        selected_np = fill_selected_df[sensor].values
        tmp_sensor_stand, mean_val, std_val = gaussian_std(selected_np)
        sensor_list.append(tmp_sensor_stand)
    else:
        print('no sensor data for: ' + sensor + ', please check the sensor type.')


# concatenate all sensor data and labels
label_list = fill_selected_df['label_id'].values.reshape(-1,1)
sensor_list.append(label_list)
allselected_np = np.concatenate(sensor_list, axis=1)

# 将数据分为滑动窗口，每段数据长50，对应latent space里面的一个点
len_sw = 50  # 数据长度，默认50
tmp_b = sliding_window(allselected_np, len_sw, len_sw)
# concatenate list
data_b = np.transpose(tmp_b[:, :, :-1], (0, 2, 1))  # [B, Len, dim-1] -> [B, dim-1, Len]
label_b = tmp_b[:, :, -1]  # [B, Len]

majority_label, minority_label = find_majority_minority(label_b)

# 将数据分为8:2，其中2为测试集
# todo: 将来，数据不一定都有label，X_labeled为有标签的数据，X_unlabeled为无标签的数据，没有X_test
vote_label = majority_value(label_b)
X_train_full, X_test, y_train_full, y_test = (
    train_test_split(data_b, label_b,
                     test_size=0.2, stratify=vote_label,
                     # random_state=42
                     ))

# 从X_train_full中随机选择1%的数据作为X_labeled，其余为X_unlabeled
X_labeled, X_unlabeled, y_labeled, y_unlabeled = (
    train_test_split(X_train_full,
                     y_train_full,
                     test_size=0.99,
                     # random_state=42
                     ))

# 确保X_labeled和X_unlabeled的大小
print("X_labeled shape:", X_labeled.shape)
print("X_unlabeled shape:", X_unlabeled.shape)
print("X_test shape:", X_test.shape)
################################## read data #################################


################################## init framework ##########################
'''
1 - initFramework()
传入传感器类型，生成模型框架
'''
device =  'cuda:0' if torch.cuda.is_available() else 'cpu'
# sensor_types = ['accelerometer', 'gyroscope', 'magnetometer',
#                'pressure', 'temperature', 'light', 'sound', 'depth']

nclass = int(max(list(label_dict.values())) + 1)

class Encoder3d4(nn.Module):
    def __init__(self):
        super(Encoder3d4, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=3,
                               out_channels=64,
                               kernel_size=3,
                               padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(kernel_size=2,
                                  stride=2,
                                  return_indices=True)

        self.conv2 = nn.Conv1d(in_channels=64,
                               out_channels=128,
                               kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(kernel_size=2,
                                  stride=2,
                                  return_indices=True)

        self.conv3 = nn.Conv1d(in_channels=128,
                               out_channels=128,
                               kernel_size=3,
                               padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        self.pool3 = nn.MaxPool1d(kernel_size=2,
                                  stride=2,
                                  return_indices=True)

        self.flattened_size = 128 * 6  # 12  # Adjust this based on input size and pooling
        self.fc = nn.Linear(self.flattened_size, 64)

    def forward(self, x):
        # input: batch, channel, length
        # x = input.permute(0, 2, 1)
        size1 = x.size()
        x, idx1 = self.pool1(F.elu(self.bn1(self.conv1(x))))
        size2 = x.size()
        x, idx2 = self.pool2(F.elu(self.bn2(self.conv2(x))))
        size3 = x.size()
        x, idx3 = self.pool3(F.elu(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)  # Flatten: batch, 128*6
        # x = self.fc(x)  # Latent space
        return x, [idx1, idx2, idx3], [size1, size2, size3]

class Encoder2d(nn.Module):
    def __init__(self):
        super(Encoder2d, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=2, out_channels=64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(kernel_size=2,
                                  stride=2,
                                  return_indices=True)

        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(kernel_size=2,
                                  stride=2,
                                  return_indices=True)

        self.conv3 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        self.pool3 = nn.MaxPool1d(kernel_size=2,
                                  stride=2,
                                  return_indices=True)

        # Calculate the flattened size after all convolutions and pooling
        self.flattened_size = 128 * 6  # Adjust this to match the output of the last pooling layer
        self.fc = nn.Linear(self.flattened_size, 64)

    def forward(self, x):
        # input: batch, channel, length
        size1 = x.size()
        x, idx1 = self.pool1(F.elu(self.bn1(self.conv1(x))))
        size2 = x.size()
        x, idx2 = self.pool2(F.elu(self.bn2(self.conv2(x))))
        size3 = x.size()
        x, idx3 = self.pool3(F.elu(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)  # Flatten: batch, 128*6
        # x = self.fc(x)  # Latent space
        return x, [idx1, idx2, idx3], [size1, size2, size3]

class Encoder1d(nn.Module):
    def __init__(self):
        super(Encoder1d, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(kernel_size=2,
                                  stride=2,
                                  return_indices=True)

        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(kernel_size=2,
                                  stride=2,
                                  return_indices=True)

        self.conv3 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        self.pool3 = nn.MaxPool1d(kernel_size=2,
                                  stride=2,
                                  return_indices=True)

        # Calculate the flattened size after all convolutions and pooling
        self.flattened_size = 128 * 6  # Adjust this to match the output of the last pooling layer
        self.fc = nn.Linear(self.flattened_size, 64)

    def forward(self, x):
        # input: batch, channel, length
        size1 = x.size()
        x, idx1 = self.pool1(F.elu(self.bn1(self.conv1(x))))
        size2 = x.size()
        x, idx2 = self.pool2(F.elu(self.bn2(self.conv2(x))))
        size3 = x.size()
        x, idx3 = self.pool3(F.elu(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)  # Flatten: batch, 128*6
        # x = self.fc(x)  # Latent space
        return x, [idx1, idx2, idx3], [size1, size2, size3]

class FlexibleNN(nn.Module):
    def __init__(self,
                 input_dim=128 * 6,
                 sensor_types=['accelerometer', 'GPS', 'Depth'],  # GUI选择的传感器
                 number_classes=5):
        super(FlexibleNN, self).__init__()
        self.sensor_type_list = sensor_types
        self.input_dim = input_dim
        self.encoder_dict = nn.ModuleDict()

        for modality in sensor_types:
            if modality in ['accelerometer', 'gyroscope', 'magnetometer']:
                self.encoder_dict[modality] = Encoder3d4()
            elif modality in ['GPS']:
                self.encoder_dict[modality] = Encoder2d()
            elif modality in ['pressure', 'temperature', 'light', 'sound', 'Depth']:
                self.encoder_dict[modality] = Encoder1d()
            else:
                raise ValueError(f"Unsupported sensor type in model: {modality}")

        # shared projection layer before contrast/classification
        self.linear = nn.Linear(self.input_dim*len(sensor_types), 32)

        self.projector = nn.Sequential(
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 16)
        )

        self.classifier = nn.Sequential(
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, number_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, data, if_contrast=True):
        start_idx = 0
        features = []

        for modality in self.sensor_type_list:
            if modality in ['accelerometer', 'gyroscope', 'magnetometer']:
                segment = data[:, start_idx:start_idx+3, :]  # acc: 3 channels
                fea, _, _ = self.encoder_dict[modality](segment)
                features.append(fea)
                start_idx += 3
            elif modality in ['GPS']:
                segment = data[:, start_idx:start_idx+2, :]  # GPS: 2 channels
                fea, _, _ = self.encoder_dict[modality](segment)
                features.append(fea)
                start_idx += 2
            elif modality in ['pressure', 'temperature', 'light', 'sound', 'Depth']:
                segment = data[:, start_idx:start_idx+1, :]  # depth: 1 channel
                fea, _, _ = self.encoder_dict[modality](segment)
                features.append(fea)
                start_idx += 1

        concat_fea = torch.cat(features, dim=1)
        concat_fea = self.linear(concat_fea)

        if if_contrast:
            output = self.projector(concat_fea)
        else:
            output = self.classifier(concat_fea)

        return output, concat_fea

class SupContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.3):
        super(SupContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        """
        Args:
            features: Tensor of shape [batch_size, feature_dim], normalized embeddings.
            labels: Tensor of shape [batch_size], ground truth labels for the samples.
        Returns:
            loss: Supervised contrastive loss value.
        """
        features = F.normalize(features, dim=1)
        similarity_matrix = torch.matmul(features, features.T) / self.temperature

        # 数值稳定处理
        similarity_matrix = similarity_matrix - torch.max(similarity_matrix, dim=1, keepdim=True)[0]

        # Positive and negative masks
        # labels = labels.to(device=features.device)
        labels1 = labels.unsqueeze(1)
        mask = torch.eye(similarity_matrix.size(0), dtype=torch.bool, device=features.device)
        positive_mask = (labels1 == labels1.T) & ~mask

        exp_sim = torch.exp(similarity_matrix)
        numerator = exp_sim * positive_mask
        denominator = exp_sim * ~mask

        numerator_sum = numerator.sum(dim=1) + 1e-8
        denominator_sum = denominator.sum(dim=1) + 1e-8

        valid_mask = numerator_sum > 0  # Skip samples with no positive pairs
        loss = -torch.log(numerator_sum[valid_mask] / denominator_sum[valid_mask])
        loss = loss.mean()

        # # cluster center loss
        # all_loss = combined_loss(latent, labels, loss, lambda_intra=1.0, lambda_inter=10)

        return loss


model = FlexibleNN(sensor_types=sensor_types, number_classes=nclass)
model = model.to(device)
classify_criterion = nn.CrossEntropyLoss()
supContrast_criterion = SupContrastiveLoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)


################################## init framework ##########################


################################## create latent ##########################
'''
params:

- model [require] [Model]
- data [require]
- pthPath [option] [string]
- supervisedCount [option] [int]
- contrastiveCount [option] [int]

根据传入参数，分别执行：
1、根据权重生成 latent
2、根据次数，生成latent，并且保存权重文件
'''

batch_size = 512
sampling_method = 'entropy'
SupCount = 1  # supcontrastive learning
warmup = 20  # warmup epoch of supcontrastive learning
name_label = '_%s_Contrast%s_warm%s' % (
    sampling_method, str(SupCount), str(warmup))

label_colors = {
    0: "#5470C6",  # 深蓝 - Attack
    1: "#91CC75",  # 绿色 - Investigation
    2: "#FAC858",  # 金黄 - Mount
    3: "#EE6666",  # 红色 - Category 3
    4: "#73C0DE",  # 天蓝 - Category 4
    5: 'brown',     # 棕色
    6: 'pink',      # 粉色
    7: 'cyan',      # 青色
    8: 'magenta',   # 品红
    9: 'lime',      # 青柠色
    10: 'teal',     # 蓝绿色
    11: 'violet',   # 紫罗兰色
    12: 'gold',     # 金色
    13: 'coral',    # 珊瑚色
    14: 'salmon'    # 三文鱼色
}

labeldict_findstr = {
                     0: 'ground_stationary',
                     1: 'stationary',
                     2: 'bathing',
                     3: 'flying_active',
                     4: 'flying_passive'}

labeldict_findstr_omizu = {
                     0: 'stationary',
                     1: 'bathing',
                     2: 'flying',
                     3: 'foraging'}

labeldict_findstr_turtle = {
0:'Resting in flow',
1:'Resting',
2:'Swimming horizontally',
3:'Stay in surface',
4:'Swimming descent',
5:'Swimming ascent',
6:'other',
}

class data_loader_umineko(Dataset):
    def __init__(self, samples, labels1d, labels2d, device='cpu'):
        self.samples = torch.tensor(samples).to(device)  # check data type
        self.labels1d = torch.LongTensor(labels1d).to(device)  # check data type

        self.labels2d = torch.tensor(labels2d)  # check data type

    def __getitem__(self, index):
        target2d = self.labels2d[index]
        target1d = self.labels1d[index]
        sample = self.samples[index]
        return sample, target1d, target2d

    def __len__(self):
        return len(self.labels1d)

def load_model_weights(model, pth_path, device='cpu'):
    """
    将 .pth 文件中的权重加载到提供的模型实例中
    """
    state_dict = torch.load(pth_path, map_location=device)
    model.load_state_dict(state_dict)
    print(f"✅ 模型权重已加载自: {pth_path}")
    return model

def save_model_weights(model, save_path):
    """
    保存模型参数（state_dict）到指定路径
    """
    torch.save(model.state_dict(), save_path)
    print(f"✅ 模型权重已保存到: {save_path}")

def AE_eval_time_series(train_loader, model, device, memotimes=30):
    model.eval()

    representation_list = []
    sample_list, timestamp_list, label_list, pred_list, timestr_list, flag_list = [], [], [], [], [], []
    for i, (sample, _, label2d) in enumerate(train_loader):

        # if i > memotimes:  # cpu memory not enough
        #     continue
        sample = sample.to(device=device, non_blocking=True, dtype=torch.float)
        # input of autoencoder will be 3D, the backbone is 1d-cnn
        x_encoded, output = model(sample)  # x_encoded.shape=batch512,outchannel128,len13
        # print(type(output))
        # x_encoded, output = model(input_).view(b, 2, -1)  # output.shape=b,2,128, split the first dim into 2 parts
        tmp_representation = x_encoded.detach().cpu().numpy()
        representation_list.append(tmp_representation)
        sample_list.append(sample.detach().cpu().numpy())
        label_list.append(label2d.detach().cpu().numpy())
        pred_list.append(output.detach().cpu().numpy())

    return representation_list, sample_list, pred_list, label_list

def freeze_encoders(model):
    for name, module in model.named_modules():
        if "encoder" in name:
            for param in module.parameters():
                param.requires_grad = False
    for param in model.linear.parameters():
        param.requires_grad = False
    for param in model.projector.parameters():
        param.requires_grad = False
    for param in model.classifier.parameters():
        param.requires_grad = True

def unfreeze_encoders(model):
    for name, module in model.named_modules():
        if "encoder" in name:
            for param in module.parameters():
                param.requires_grad = True
    for param in model.linear.parameters():
        param.requires_grad = True
    for param in model.projector.parameters():
        param.requires_grad = True
    for param in model.classifier.parameters():
        param.requires_grad = False

def unfreeze_all(model):
    for name, module in model.named_modules():
        if "encoder" in name:
            for param in module.parameters():
                param.requires_grad = True
    for param in model.linear.parameters():
        param.requires_grad = True
    for param in model.projector.parameters():
        param.requires_grad = True
    for param in model.classifier.parameters():
        param.requires_grad = True

def train_model(model, loader, criterion, optimizer, epochs=500, device='cpu', if_contrast=True):
    model.train()
    avg_loss = []
    for epoch in range(epochs):
        losses = []
        for batch in loader:
            data, labels, _ = batch
            # label_vote = majority_value(labels)
            # label_vote = torch.from_numpy(label_vote)
            data = data.to(device=device, dtype=torch.float)
            labels = labels.to(device=device, dtype=torch.long)
            # label_vote = label_vote.to(device=device, dtype=torch.long)
            # outputs: supcontrast output; feature: feature extractor output
            outputs, fea = model(data, if_contrast)
            loss = criterion(outputs, labels)
            losses.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        avg_loss.append(np.mean(losses))
        # print(f"Epoch {epoch + 1}, Loss: {avg_loss[-1]}")
    return model, avg_loss


def plot_scatter_omizu(data_umap, label_str, iteration, name):
    fig_2d = px.scatter(
        data_umap, x=0, y=1,
        color=label_str,
        labels={'activity': 'activity'},
        color_discrete_map={'stationary': label_colors[0],
                            'bathing': label_colors[1],
                            'flying': label_colors[2],
                            'foraging': label_colors[3],
                            }
    )
    # Reduce marker size for all points
    for trace in fig_2d.data:
        trace.marker.size = 6  # Adjust the size to your preference (e.g., 6)
    # 更新 x 和 y 轴的标题
    fig_2d.update_layout(
        xaxis_title="UMAP 1",
        yaxis_title="UMAP 2",
        legend_title="Color: Activity" # None  # 取消图例标题
    )
    # Update transparency for traces where activity is '-2.0'
    fig_2d.for_each_trace(lambda trace: trace.update(marker=dict(opacity=0.5)) if trace.name == '-2.0' else ())
    fig_2d.update_traces(marker=dict(line=dict(width=0)))  # remove boundary of point
    fig_2d.write_html(r'D:\code\DeepView\deepview\calculate_results\data\omizunagidori\figures\test_%s_activeSup_epoch_%s_omizu.html' % (name, str(iteration)))

    return

def plot_scatter_umi(data_umap, label_str, iteration, name):
    fig_2d = px.scatter(
        data_umap, x=0, y=1,
        color=label_str,
        labels={'activity': 'activity'},
        color_discrete_map={'ground_stationary': label_colors[0],
                            'stationary': label_colors[1],
                            'bathing': label_colors[2],
                            'flying_active': label_colors[3],
                            'flying_passive': label_colors[4],
                            'foraging': label_colors[5],
                            }
    )
    # Reduce marker size for all points
    for trace in fig_2d.data:
        trace.marker.size = 6  # Adjust the size to your preference (e.g., 6)
    # 更新 x 和 y 轴的标题
    fig_2d.update_layout(
        xaxis_title="UMAP 1",
        yaxis_title="UMAP 2",
        legend_title="Color: Activity" # None  # 取消图例标题
    )
    # Update transparency for traces where activity is '-2.0'
    fig_2d.for_each_trace(lambda trace: trace.update(marker=dict(opacity=0.5)) if trace.name == '-2.0' else ())
    fig_2d.update_traces(marker=dict(line=dict(width=0)))  # remove boundary of point
    fig_2d.write_html(r'D:\code\DeepView\deepview\calculate_results\data\umineko\figures\test_%s_activeSup_epoch_%s.html' % (name, str(iteration)))

    return

def plot_scatter_turtle(data_umap, label_str, iteration, name):
    fig_2d = px.scatter(
        data_umap, x=0, y=1,
        color=label_str,
        labels={'activity': 'activity'},
        color_discrete_map={'surface_behavior': label_colors[0],
                            'social_selfdirected': label_colors[1],
                            'rest_passive': label_colors[2],
                            'locomotion': label_colors[3],
                            'feeding_related': label_colors[4],
                            'exploration_environment': label_colors[5],
                            }
    )
    # Reduce marker size for all points
    for trace in fig_2d.data:
        trace.marker.size = 6  # Adjust the size to your preference (e.g., 6)
    # 更新 x 和 y 轴的标题
    fig_2d.update_layout(
        xaxis_title="UMAP 1",
        yaxis_title="UMAP 2",
        legend_title="Color: Activity" # None  # 取消图例标题
    )
    # Update transparency for traces where activity is '-2.0'
    fig_2d.for_each_trace(lambda trace: trace.update(marker=dict(opacity=0.5)) if trace.name == '-2.0' else ())
    fig_2d.update_traces(marker=dict(line=dict(width=0)))  # remove boundary of point
    fig_2d.write_html(r'D:\code\DeepView\deepview\calculate_results\data\turtle\figures\test_%s_activeSup_epoch_%s.html' % (name, str(iteration)))

    return

def vis_scatter_label_2d(representation_list, label_list,
                         iteration, name, plot_flag = False, is_omizu=False):
    label_concat = np.concatenate(label_list)
    repre_concat = np.concatenate(representation_list)
    repre_reshape = repre_concat.reshape(repre_concat.shape[0], -1).astype(float)

    umap_2d = UMAP(n_components=2)
    data_umap = umap_2d.fit_transform(repre_reshape)  # 耗时！！！

    label_concat_vote = majority_value(label_concat)
    if 'omizu' in is_omizu:
        label_concat_vote_str = [labeldict_findstr_omizu[i] for i in label_concat_vote]
    elif is_omizu == 'turtle':
        label_concat_vote_str = [labeldict_findstr_turtle[i] for i in label_concat_vote]
    elif is_omizu == 'umineko':
        label_concat_vote_str = [labeldict_findstr[i] for i in label_concat_vote]
    else:
        print('no label to string dictionary')
    if plot_flag:
        # plot ground truth labels
        if 'omizu' in is_omizu:
            plot_scatter_omizu(data_umap, label_concat_vote_str, iteration, name + '_gt')
        elif 'umineko' == is_omizu:
            plot_scatter_umi(data_umap, label_concat_vote_str, iteration, name+'_gt')
        elif is_omizu == 'turtle':
            plot_scatter_turtle(data_umap, label_concat_vote_str, iteration, name+'_gt')
        else:
            print('no plot for this dataset')
    return data_umap

def create_latent(plot_loader,
                  model,
                  dataset='turtle',
                  device='cpu',
                  name_label='_test',
                  sensor_types=sensor_types):

    repres_list, sample_list, pred_list, label_list = \
        AE_eval_time_series(plot_loader, model, device)
    data_umap = (
        vis_scatter_label_2d(repres_list,
                             label_list, 0,
                             str(sensor_types) + name_label,
                             plot_flag=True,
                             is_omizu=dataset))

    # # 保存潜在空间表示
    # save_model_weights(model, r'D:\code\DeepView\deepview\calculate_results\data\turtle\pths\model_turtle_%s.pth' % name_label)
    return data_umap


## 创建数据加载器
def create_loader(data_b, label_b, batch_size=512, shuffle=False, device='cpu'):
    """
    创建数据加载器
    :param data_b: 输入数据
    :param label_b: 标签
    :param batch_size: 批处理大小
    :param device: 设备
    :return: DataLoader对象
    """
    major_label_b = majority_value(label_b)
    dataset = data_loader_umineko(data_b.astype(float),
                                  major_label_b.astype(int),
                                  label_b.astype(int),
                                  device=device)
    return DataLoader(dataset,
                      batch_size=batch_size,
                      shuffle=shuffle,
                      drop_last=False)

## initial plot
plot_loader = create_loader(data_b, label_b,
                               batch_size=512,
                               shuffle=False,
                               device=device)
data_umap = create_latent(plot_loader,
                  model,
                  dataset='turtle',
                  device=device,
                  name_label=name_label,
                  sensor_types=sensor_types)



if len(X_unlabeled) > 0:  # 数据中必须存在label的条件
    labeled_loader = create_loader(X_labeled, y_labeled,
                                batch_size=512,
                                shuffle=True,
                                device=device)

    # contrastive learning
    # epoch = 50
    contrastiveCount = 10
    unfreeze_encoders(model)
    for i in range(contrastiveCount):
        model, _ = train_model(model, labeled_loader, supContrast_criterion, optimizer,
                           epochs=50, device=device, if_contrast=True)

    # supervised learning
    supervisedCount = 50
    freeze_encoders(model)  # only classifier is trainable
    model, avg_loss1 = train_model(model, labeled_loader, classify_criterion, optimizer,
                                   epochs=supervisedCount, device=device, if_contrast=False)
    unfreeze_all(model)  # only projector is NOT trainable
    model, avg_loss2 = train_model(model, labeled_loader, classify_criterion, optimizer,
                                   epochs=supervisedCount, device=device, if_contrast=False)

    data_umap = create_latent(plot_loader,
                  model,
                  dataset='turtle',
                  device=device,
                  name_label=name_label,
                  sensor_types=sensor_types)

################################## create latent ##########################


################################## label propagation ##########################
# # all_labels_before = np.concatenate(unlabel_list)  # 新数据还没有标记
# all_labels_before = label_b  # 一列
#
# # Apply Label Spreading for semi-supervised label propagation
# label_spread_model = LabelSpreading(kernel='knn', n_neighbors=10, alpha=0.2)  # Graph-based propagation
# label_spread_model.fit(data_umap, all_labels_before)
#
# # Get the predicted labels for all data
# propagated_labels = label_spread_model.transduction_

def generate_propagated_labels(data_umap, label_b):
    # 我们将已经标记的样本标签设置为大于0，其余样本为 -1（未标记）
    all_labels_before = majority_value(label_b)  # 最终为一列

    # Apply Label Spreading for semi-supervised label propagation
    label_spread_model = LabelSpreading(kernel='knn', n_neighbors=10, alpha=0.2)  # Graph-based propagation
    label_spread_model.fit(data_umap, all_labels_before)

    # Get the predicted labels for all data
    propagated_labels = label_spread_model.transduction_
    return propagated_labels

propagated_labels = generate_propagated_labels(data_umap, label_b)

################################## label propagation ##########################



################################## sampling ##########################

select_size = 100

def least_confidence(probabilities, n_samples):
    '''最小置信度方法选择模型对其预测最不自信的样本，通常通过计算每个样本的最大预测概率来实现'''
    # 选择最大置信度的样本
    max_confidence = np.max(probabilities, axis=1)
    # 选择最小置信度的样本
    selected_indices = np.argsort(max_confidence)[:n_samples]
    return selected_indices

def min_margin(probabilities, n_samples):
    '''最小边际方法选择模型对其预测最不自信的样本，通常通过计算每个样本的最大预测概率来实现'''
    # 计算最大边际
    margins = np.partition(probabilities, -2, axis=1)[:, -1] - np.partition(probabilities, -1, axis=1)[:, -2]
    # 选择最小边际的样本
    selected_indices = np.argsort(margins)[:n_samples]
    return selected_indices

# 计算样本不确定性（基于熵, 还有least_confidence and min_margin可以选择）
'''
对于需要快速响应的在线学习任务，可以选择最小置信度；而在需要更全面评估不确定性的情况下，最大熵可能更合适。
'''
def uncertainty_sampling(X_labeled, y_labeled,
                         X_unlabeled, y_unlabeled,
                         model, update_size, device, choice=0):
    model.eval()
    with torch.no_grad():
        # 计算未标注数据和已标注数据的嵌入
        _, unlabeled_embeddings = model(torch.tensor(X_unlabeled, device=device, dtype=torch.float32),
                                        if_contrast=True)
        probs = model.classifier(unlabeled_embeddings)

        if choice==0:
            uncertainty = entropy(probs.detach().cpu().numpy().T)  # 计算每个样本的熵
            # 选择贡献度最大的样本
            selected_indices = np.argsort(uncertainty)[-update_size:]
        elif choice ==1 :
            selected_indices = least_confidence(probs.detach().cpu().numpy(), update_size)
        else:
            selected_indices = min_margin(probs.detach().cpu().numpy(), update_size)

        selected_samples = X_unlabeled[selected_indices]
        selected_labels = y_unlabeled[selected_indices]
        ## 更新标注集和未标注数据池
        X_labeled = np.vstack([X_labeled, selected_samples])
        y_labeled = np.concatenate([y_labeled, selected_labels], axis=0)
        X_unlabeled = np.delete(X_unlabeled, selected_indices, axis=0)
        y_unlabeled = np.delete(y_unlabeled, selected_indices, axis=0)
        print(f"Labeled samples: {len(X_labeled)}")

        return X_labeled, y_labeled, X_unlabeled, y_unlabeled, (selected_samples, selected_labels)


def random_sampling(X_labeled, y_labeled,
                    X_unlabeled, y_unlabeled, update_size):
    """
    在未标注数据集中随机选取一定数量的样本更新到标注集中。

    参数:
    --------
    X_labeled : np.ndarray
        已标注数据特征 (N_labeled, D)
    y_labeled : np.ndarray
        已标注数据标签 (N_labeled, )
    X_unlabeled : np.ndarray
        未标注数据特征 (N_unlabeled, D)
    y_unlabeled : np.ndarray
        未标注数据标签 (N_unlabeled, )
    model : torch.nn.Module
        模型实例，随机采样不依赖模型，但保留接口以保持格式一致
    update_size : int
        本轮迭代需要从未标注集中选取的样本数量
    device : torch.device
        设备信息 (cpu or cuda)，随机采样中也不会用到，但保留以统一函数签名

    返回:
    --------
    X_labeled, y_labeled, X_unlabeled, y_unlabeled, selected_labels
      - 更新后的已标注数据与未标注数据
      - selected_labels: 本轮迭代所选数据的标签 (便于后续统计等)
    """

    # 随机选取 update_size 个索引
    selected_indices = np.random.choice(len(X_unlabeled), update_size, replace=False)

    # 根据索引取对应样本及标签
    selected_samples = X_unlabeled[selected_indices]
    selected_labels = y_unlabeled[selected_indices]

    # 更新 X_labeled, y_labeled
    X_labeled = np.vstack([X_labeled, selected_samples])
    y_labeled = np.concatenate([y_labeled, selected_labels], axis=0)

    # 从未标注集中剔除已选样本
    X_unlabeled = np.delete(X_unlabeled, selected_indices, axis=0)
    y_unlabeled = np.delete(y_unlabeled, selected_indices, axis=0)

    print(f"Labeled samples: {len(X_labeled)}")

    return X_labeled, y_labeled, X_unlabeled, y_unlabeled, (selected_samples, selected_labels)

def random_undersampling(X_labeled, y_labeled,
                         X_unlabeled, y_unlabeled,
                         majority_label, update_size, undersample_ratio=1.0):
    """
    在已标注数据集中，对多数类样本进行随机欠采样，以达到类别平衡。

    参数:
    --------
    X_labeled : np.ndarray
        已标注数据特征 (N_labeled, D)
    y_labeled : np.ndarray
        已标注数据标签 (N_labeled, )
    majority_label : int or str
        多数类的类别标签
    undersample_ratio : float
        欠采样比例，表示欠采样后，多数类样本相对于少数类的比例。

    返回:
    --------
    X_labeled, y_labeled, X_unlabeled, y_unlabeled, selected_labels
      - 更新后的已标注数据与未标注数据
      - selected_labels: 本轮迭代被移除的数据的标签 (便于后续统计等)
    """

    # 随机选取 update_size 个索引
    selected_indices = np.random.choice(len(X_unlabeled), update_size, replace=False)

    # 根据索引取对应样本及标签
    selected_samples = X_unlabeled[selected_indices]
    selected_labels = y_unlabeled[selected_indices]
    # 从未标注集中剔除已选样本
    X_unlabeled = np.delete(X_unlabeled, selected_indices, axis=0)
    y_unlabeled = np.delete(y_unlabeled, selected_indices, axis=0)

    # 统计类别样本数量
    selected_labels_batch = majority_value(selected_labels)
    unique, counts = np.unique(selected_labels_batch, return_counts=True)
    class_counts = dict(zip(unique, counts))

    if majority_label not in class_counts:
        # 更新 X_labeled, y_labeled
        X_labeled = np.vstack([X_labeled, selected_samples])
        y_labeled = np.concatenate([y_labeled, selected_labels], axis=0)
        # 从未标注集中剔除已选样本
        X_unlabeled = np.delete(X_unlabeled, selected_indices, axis=0)
        y_unlabeled = np.delete(y_unlabeled, selected_indices, axis=0)
        print(f"Error: 指定的多数类 {majority_label} 在数据集中不存在！")
        return X_labeled, y_labeled, X_unlabeled, y_unlabeled, np.array([])

    # 计算少数类样本数量
    minority_count = min(class_counts.values())

    # 计算多数类需要保留的样本数量
    majority_count_target = int(minority_count * undersample_ratio)

    # 获取多数类样本的索引
    majority_indices = np.where(selected_labels_batch == majority_label)[0]

    # 随机选择需要保留的多数类样本
    selected_majority_indices = np.random.choice(majority_indices, majority_count_target, replace=False)

    # 计算被移除的样本
    removed_indices = np.setdiff1d(majority_indices, selected_majority_indices)

    # 获取所有少数类样本的索引
    minority_indices = np.where(selected_labels_batch != majority_label)[0]

    # 组合新数据索引
    selected_indices = np.concatenate([selected_majority_indices, minority_indices])

    # 生成新的标注数据
    X_selected_new = selected_samples[selected_indices]
    y_selected_new = selected_labels[selected_indices]

    # 更新 X_labeled, y_labeled
    X_labeled_new = np.vstack([X_labeled, X_selected_new])
    y_labeled_new = np.concatenate([y_labeled, y_selected_new], axis=0)

    print(
        f"Under-sampling completed: Majority class reduced from {class_counts[majority_label]} → {majority_count_target}"
    )
    print(f"Labeled samples: {len(X_labeled_new)}")

    return X_labeled_new, y_labeled_new, X_unlabeled, y_unlabeled, (X_selected_new, y_selected_new)

def data_augmentation_oversampling(X_labeled, y_labeled,
                                   X_unlabeled, y_unlabeled,
                                   minority_label, update_size,
                                   augmentation_factor=2):
    """
    通过数据增强（Data Augmentation）对少数类样本进行过采样。

    参数:
    ----------
    X_labeled : np.ndarray
        已标注数据特征 (N_labeled, T) (时间序列或其他数据)
    y_labeled : np.ndarray
        已标注数据标签 (N_labeled,)
    minority_label : int or str
        需要增强的少数类标签
    augmentation_factor : int
        每个少数类样本生成多少个新的样本（倍数）

    返回:
    ----------
    X_labeled, y_labeled, X_unlabeled, y_unlabeled, selected_labels
      - 更新后的已标注数据与未标注数据
      - selected_labels: 本轮迭代生成的新样本的标签 (便于后续统计等)
    """

    # augmentation_factor = random.choice([1, 2, 3, 4])

    # 随机选取 update_size 个索引
    selected_indices = np.random.choice(len(X_unlabeled), update_size, replace=False)

    # 根据索引取对应样本及标签
    selected_samples = X_unlabeled[selected_indices]
    selected_labels = y_unlabeled[selected_indices]
    # 从未标注集中剔除已选样本
    X_unlabeled = np.delete(X_unlabeled, selected_indices, axis=0)
    y_unlabeled = np.delete(y_unlabeled, selected_indices, axis=0)

    # 获取少数类样本
    y_labeled_batch = majority_value(selected_labels)
    minority_indices = np.where(y_labeled_batch == minority_label)[0]

    X_minority = selected_samples[minority_indices]

    # 存储合成的新样本
    X_synthetic = []
    y_synthetic = []
    if len(X_minority) == 0:
        X_new_labeled = selected_samples
        y_new_labeled = selected_labels
    else:
        for x, y in zip(X_minority, minority_indices):
            for _ in range(augmentation_factor):
                # 对单个少数类样本进行数据增强
                x_augmented = augment_sample(x)
                X_synthetic.append(x_augmented)
                y_synthetic.append(selected_labels[y])

        # 转换为 NumPy 数组
        X_synthetic = np.array(X_synthetic)
        y_synthetic = np.array(y_synthetic)
        # y_synthetic = selected_labels[minority_indices]

        X_new_labeled = np.vstack([X_synthetic, selected_samples])
        y_new_labeled = np.concatenate([y_synthetic, selected_labels])

    # 更新 X_labeled, y_labeled
    X_labeled = np.vstack([X_labeled, X_new_labeled])
    y_labeled = np.concatenate([y_labeled, y_new_labeled], axis=0)

    # # 由于数据增强的样本不属于原始未标注数据，X_unlabeled 和 y_unlabeled 为空
    # selected_labels = y_synthetic  # 记录新增的样本标签

    print(f"Data Augmentation Oversampling completed: Added {len(X_synthetic)} new samples for label {minority_label}")
    print(f"Labeled samples: {len(X_labeled)}")

    return X_labeled, y_labeled, X_unlabeled, y_unlabeled, (X_new_labeled, y_new_labeled)

def augment_sample(x):
    """
    针对单个样本进行数据增强（适用于时序数据）。

    增强策略：
    1. 加噪声（随机高斯噪声）
    2. 时移（随机向前或向后平移）
    3. 放缩（信号幅度缩放）
    4. 翻转（适用于对称信号）

    参数:
    ----------
    x : np.ndarray
        输入的单个样本 (T,)

    返回:
    ----------
    x_augmented : np.ndarray
        增强后的样本
    """
    x_aug = x.copy()

    # 1. **加高斯噪声** (Noise Injection)
    if np.random.rand() < 0.5:  # 50% 概率应用
        noise = np.random.normal(loc=0, scale=0.02 * np.std(x), size=x.shape)
        x_aug += noise

    # 2. **时移 (Time Shifting)**
    if np.random.rand() < 0.5:
        shift = np.random.randint(-3, 3)  # 在 [-3, 3] 之间随机平移
        x_aug = np.roll(x_aug, shift)

    # 3. **放缩 (Scaling)**
    if np.random.rand() < 0.5:
        scale_factor = np.random.uniform(0.9, 1.1)  # 在 [0.9, 1.1] 之间缩放
        x_aug *= scale_factor

    # 4. **翻转 (Flipping)**
    if np.random.rand() < 0.3:  # 30% 概率应用
        x_aug = -x_aug

    return x_aug

def representative_sampling(X_labeled, y_labeled,
                                   X_unlabeled, y_unlabeled,
                                   model, update_size, device):
    '''代表性采样选择那些能代表整个数据分布的样本，通常使用聚类算法（如K-means）来确定样本的代表性。'''
    # 使用K-means聚类
    # 先获得latent representation
    model.eval()
    with torch.no_grad():
        # 计算未标注数据和已标注数据的嵌入
        _, unlabeled_embeddings = model(torch.tensor(X_unlabeled, device=device, dtype=torch.float32),
                                        if_contrast=True)
    data = unlabeled_embeddings.detach().cpu().numpy()
    # 再选择new data
    kmeans = KMeans(n_clusters=update_size, n_init=10).fit(data)
    labels = kmeans.labels_
    # 初始化一个数组来存储每个聚类的最近点索引
    selected_indices = np.zeros(kmeans.n_clusters, dtype=int)
    # 遍历每个聚类
    for cluster_index in range(kmeans.n_clusters):
        # 获取当前聚类中所有点的索引
        cluster_points_indices = np.where(labels == cluster_index)[0]
        # 获取这些点的坐标
        cluster_points = data[cluster_points_indices]
        # 计算这些点到当前聚类中心的距离
        distances = np.linalg.norm(cluster_points -
                                   kmeans.cluster_centers_[cluster_index], axis=1)
        # 找到距离最近的点的索引
        closest_point_index_within_cluster = np.argmin(distances)
        # 转换为原始数据集中的索引
        selected_indices[cluster_index] = cluster_points_indices[closest_point_index_within_cluster]

    # 根据索引取对应样本及标签
    selected_samples = X_unlabeled[selected_indices]
    selected_labels = y_unlabeled[selected_indices]
    # 从未标注集中剔除已选样本
    X_unlabeled = np.delete(X_unlabeled, selected_indices, axis=0)
    y_unlabeled = np.delete(y_unlabeled, selected_indices, axis=0)

    # 更新 X_labeled, y_labeled
    X_labeled = np.vstack([X_labeled, selected_samples])
    y_labeled = np.concatenate([y_labeled, selected_labels], axis=0)

    # # 由于数据增强的样本不属于原始未标注数据，X_unlabeled 和 y_unlabeled 为空
    # selected_labels = y_synthetic  # 记录新增的样本标签
    print(f"Labeled samples: {len(X_labeled)}")

    return X_labeled, y_labeled, X_unlabeled, y_unlabeled, (selected_samples, selected_labels)

def data_sampling(sampling_method, X_labeled, y_labeled,
                  X_unlabeled, y_unlabeled,
                  model, update_size,
                  majority_label=0, minority_label=0, device='cpu'):
    '''
    selected_data = (newX_labeled, newy_labeled),表示新添加的标注数据
    '''
    if sampling_method == "random":
        X_labeled_new, y_labeled_new, X_unlabeled_new, y_unlabeled_new, selected_data = (
            random_sampling(X_labeled, y_labeled,
                            X_unlabeled, y_unlabeled, update_size))
    elif sampling_method == "entropy":
        X_labeled_new, y_labeled_new, X_unlabeled_new, y_unlabeled_new, selected_data = (
            uncertainty_sampling(X_labeled, y_labeled,
                                 X_unlabeled, y_unlabeled,
                                 model, update_size, device))
    elif sampling_method == "least_confidence":
        X_labeled_new, y_labeled_new, X_unlabeled_new, y_unlabeled_new, selected_data = (
            uncertainty_sampling(X_labeled, y_labeled,
                                 X_unlabeled, y_unlabeled,
                                 model, update_size, device, choice=1))
    elif sampling_method == "min_margin":
        X_labeled_new, y_labeled_new, X_unlabeled_new, y_unlabeled_new, selected_data = (
            uncertainty_sampling(X_labeled, y_labeled,
                                 X_unlabeled, y_unlabeled,
                                 model, update_size, device,choice=2))
    elif sampling_method == "underRandom":
        X_labeled_new, y_labeled_new, X_unlabeled_new, y_unlabeled_new, selected_data = (
            random_undersampling(X_labeled, y_labeled,
                                 X_unlabeled, y_unlabeled,
                                 majority_label, update_size, undersample_ratio=1.0))
    elif sampling_method == "overAugment":
        X_labeled_new, y_labeled_new, X_unlabeled_new, y_unlabeled_new, selected_data\
            = data_augmentation_oversampling(
                                            X_labeled, y_labeled,
                                            X_unlabeled, y_unlabeled,
                                            minority_label, update_size,
                                            augmentation_factor=2)
    elif sampling_method == "repreSamp":
        X_labeled_new, y_labeled_new, X_unlabeled_new, y_unlabeled_new, selected_data\
            = representative_sampling(
                                    X_labeled, y_labeled,
                                    X_unlabeled, y_unlabeled,
                                    model, update_size, device)
    else:
        raise ValueError(f"Unknown sampling method: {sampling_method}")
    return X_labeled_new, y_labeled_new, X_unlabeled_new, y_unlabeled_new, selected_data

X_labeled, y_labeled, X_unlabeled, y_unlabeled, selected_data = (
            data_sampling(sampling_method, X_labeled, y_labeled,
                          X_unlabeled, y_unlabeled,
                          model, select_size,
                          majority_label=majority_label,
                          minority_label=minority_label,
                          device=device))

print('')
################################## sampling ##########################



