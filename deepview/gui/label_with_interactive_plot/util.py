"""
深度学习半监督学习框架 - 动物行为分析系统
====================================================

本系统专为分析动物传感器数据而设计，特别针对海洋动物（如海龟）的行为识别。
系统采用深度学习结合半监督学习的方法，利用少量标注数据和大量无标注数据，
实现高效的动物行为自动分类。

主要功能模块：
1. 数据读取和预处理 - 处理传感器时序数据
2. 神经网络模型构建 - 多传感器数据融合的深度学习模型
3. 半监督学习训练 - 对比学习 + 监督学习结合
4. 主动学习采样 - 智能选择有价值的样本进行标注
5. 可视化分析 - UMAP降维和交互式可视化

日期：2025年7月15日
"""

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
import faiss
from scipy.sparse import csr_matrix, diags
from sklearn.preprocessing import LabelBinarizer

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore
from PySide6.QtWidgets import QApplication, QToolTip
from typing import Optional  # Python 3.9 兼容：为 Optional[str] 提供类型
################################## Imports###################################




################################## 数据读取和预处理 #################################

# 标签映射字典：将原始行为标签映射到统一的行为类别
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

    # Omiz
    'stationary': 'stationary',
    'preening': 'stationary',
    'bathing': 'bathing',
    'flight_take_off': 'flying',
    'flight_cruising': 'flying',
    'foraging_dive': 'foraging',
    'surface_seizing': 'foraging',

}

# 将行为类别映射为数值标签用于机器学习
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


label_dict_omizu = {
    'stationary': 0,
    'preening': 1,
    'bathing': 2,
    'flight_take_off': 3,
    'flight_cruising': 4,
    'foraging_dive': 5,
    'surface_seizing': 6
}

# # 不同数据集的标签映射字典
# labeldict_findstr = {
#     0: 'ground_stationary',
#     1: 'stationary',
#     2: 'bathing',
#     3: 'flying_active',
#     4: 'flying_passive'
# }

# labeldict_findstr_omizu = {
#     0: 'stationary',
#     1: 'bathing',
#     2: 'flying',
#     3: 'foraging'
# }

# labeldict_findstr_turtle = {
# 0:'Resting in flow',
# 1:'Resting',
# 2:'Swimming horizontally',
# 3:'Stay in surface',
# 4:'Swimming descent',
# 5:'Swimming ascent',
# 6:'other',
# }

def read_sensor_data(pkl_path: str = r'./test/turtle.pkl'):
    """
    读取并预处理传感器数据
    
    功能：
    1. 从pickle文件中读取传感器数据
    2. 将原始行为标签映射为统一类别
    3. 处理无标签数据
    
    Returns:
        pd.DataFrame: 包含传感器数据和标签的DataFrame
    """
    df_list = []
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"Required data file not found: {pkl_path}. Please ensure the file exists in the working directory.")
    
    # 读取pickle文件中的所有数据片段
    with open(pkl_path, 'rb') as f:
        while True:
            try:
                item = pickle.load(f)
                df_list.append(item)
            except EOFError:
                break
    
    # 合并所有数据片段
    df_all = pd.concat(df_list, ignore_index=True)
    
    # 标签映射和处理
    df_all['category'] = df_all['Label'].map(labelcategory_dict)
    df_all['label_id'] = df_all['category'].map(label_dict)
    df_all['label_id'] = df_all['label_id'].fillna(-2)  # 用-2表示未知标签

    return df_all






def read_sensor_data_from_memory(pkl_objects):
    """
    直接接收已经通过 pickle.load(...) 得到的对象并完成合并与标签映射。
    使用的是copy的DataFrame，避免修改原始数据。
    支持：
      1) 单个 DataFrame
      2) list / tuple / 其它可迭代 (其元素为 DataFrame)
    参数:
        pkl_objects: DataFrame 或 可迭代的多个 DataFrame
    返回:
        DataFrame: 处理后的数据
    """
    if isinstance(pkl_objects, pd.DataFrame):
        df_all = pkl_objects.copy()
    elif isinstance(pkl_objects, (list, tuple)):
        if len(pkl_objects) == 0:
            raise ValueError("pkl_objects 为空。")
        df_all = pd.concat(pkl_objects, ignore_index=True)
    else:
        try:
            tmp_list = list(pkl_objects)
            if len(tmp_list) == 0:
                raise ValueError("pkl_objects 可迭代为空。")
            df_all = pd.concat(tmp_list, ignore_index=True)
        except Exception as e:
            raise TypeError(
                "pkl_objects 类型不支持，请传入 DataFrame 或 (list/tuple/可迭代) 且元素为 DataFrame。"
            ) from e
        
    # 标签映射和处理
    df_all['label_id'] = df_all['label_id'].fillna(-2)  # 用-2表示未知标签

    # 统一将 label_id 转为数值；支持 bytes → 小端无符号整数
    # if 'label_id' not in df_all.columns:
    #     df_all['label_id'] = -2
    # else:
    #     def _parse_label_id(v):
    #         try:
    #             # NaN
    #             if pd.isna(v):
    #                 return -2
    #         except Exception:
    #             pass

    #         # bytes/bytearray/numpy bytes_
    #         if isinstance(v, (bytes, bytearray, np.bytes_)):
    #             b = bytes(v)
    #             if len(b) == 0:
    #                 return -2
    #             try:
    #                 # 典型格式：b'\x03\x00\x00\x00\x00\x00\x00\x00' → 3
    #                 return int.from_bytes(b, byteorder='little', signed=False)
    #             except Exception:
    #                 return -2

    #         # 数值
    #         if isinstance(v, (np.integer, int)):
    #             return int(v)
    #         if isinstance(v, (np.floating, float)):
    #             return int(v) if np.isfinite(v) else -2

    #         # 其他转字符串再尝试
    #         s = str(v).strip()
    #         if s == '':
    #             return -2
    #         try:
    #             return int(s)
    #         except Exception:
    #             try:
    #                 return int(float(s))
    #             except Exception:
    #                 return -2

    #     df_all['label_id'] = df_all['label_id'].apply(_parse_label_id).astype('int64')

    return df_all


def gaussian_std(X):
    """
    对数据进行高斯标准化
    
    Args:
        X: 输入数据矩阵
        
    Returns:
        tuple: 标准化后的数据, 均值, 标准差
    """
    mean_val = np.mean(X.astype(float), axis=0)
    std_val = np.std(X.astype(float), axis=0)
    X_standardized = (X - mean_val) / np.maximum(std_val, 10 ** -5)
    return X_standardized, mean_val, std_val

def process_gps(df):
    """
    处理GPS数据并计算速度和方向
    
    功能：
    1. 检测GPS数据的存在
    2. 使用Haversine公式计算距离
    3. 计算速度和方向角
    
    Args:
        df: 包含GPS数据的DataFrame
        
    Returns:
        tuple: 处理后的DataFrame, GPS数据长度
    """
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
    """
    使用滑动窗口方法处理时序数据
    
    功能：
    1. 将连续的时序数据分割成固定长度的窗口
    2. 支持重叠采样，提高数据利用率
    3. 确保最后一个窗口也被包含
    
    Args:
        data: 输入数据（DataFrame或numpy数组）
        len_sw: 滑动窗口长度
        step: 步长（默认为窗口长度的一半）
        
    Returns:
        np.ndarray: 分窗后的数据，形状为[批次数, 窗口长度, 特征维度]
    """
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
    """
    查找数据集中的多数类和少数类标签
    
    Args:
        label_b: 标签数组
        
    Returns:
        tuple: (多数类标签, 少数类标签)
    """
    unique_labels, counts = np.unique(label_b, return_counts=True)

    if len(unique_labels) == 1:
        print(f"Only one label present: {unique_labels[0]}")
        return unique_labels[0], unique_labels[0]  # 只有一个类别，返回相同的值

    # 计算多数类和少数类
    majority_label = unique_labels[np.argmax(counts)]
    minority_label = unique_labels[np.argmin(counts)]

    return majority_label, minority_label

def majority_value(arr):
    """
    计算每行数据的众数（最频繁出现的值）
    
    Args:
        arr: 输入数组，可以是torch.Tensor或numpy数组
        
    Returns:
        np.ndarray: 每行的众数组成的数组
    """
    if isinstance(arr, torch.Tensor):
        arr = arr.detach().cpu().numpy()
    majority = []
    for row in arr:
        values, counts = np.unique(row, return_counts=True)
        majority.append(values[np.argmax(counts)])
    return np.array(majority)


def build_window_spans(n_rows, len_sw, step):
    """
    复用 sliding_window 的窗口策略，输出每个窗口覆盖的行区间（含端点）。
    返回: np.ndarray, shape=(num_windows, 2), 每行是 (start_inclusive, end_inclusive)
    注意：最后一个窗口对应 [-1-len_sw : -1]，即 [n-1-len_sw, n-2]
    """
    spans = []
    for idx in range(0, n_rows - len_sw - step, step):
        spans.append((idx, idx + len_sw - 1))
    if n_rows >= len_sw + 1:
        spans.append((n_rows - 1 - len_sw, n_rows - 2))
    return np.array(spans, dtype=int)

def apply_window_labels_to_all_df(df, spans, win_labels, colname):
    """
    将窗口级别标签写回 df 的行级列 colname。
    spans: shape (B,2) 的 (start,end) 含端点
    win_labels: shape (B,) 的窗口标签（如 propagated_labels 或窗口多数票）
    """
    if colname not in df.columns:
        df[colname] = np.nan
    for (s, e), lab in zip(spans, win_labels):
        if 0 <= s <= e < len(df):
            df.loc[s:e, colname] = lab
    return df




# # 选择使用的传感器类型（可配置）
# sensor_types = ['accelerometer', 'Depth']

################################## 数据读取和预处理 #################################


################################## 深度学习模型框架 ##########################
'''
模型构建流程：
1. 根据传入的传感器类型，生成对应的多模态深度学习模型
2. 支持1D、2D、3D传感器数据的卷积编码
3. 集成对比学习和监督学习功能
'''
# 设备配置
device =  'cuda:0' if torch.cuda.is_available() else 'cpu'
# sensor_types = ['accelerometer', 'gyroscope', 'magnetometer',
#                'pressure', 'temperature', 'light', 'sound', 'depth']

# 计算类别数量
nclass = int(max(list(label_dict.values())) + 1)

class Encoder3d4(nn.Module):
    """
    三维传感器数据编码器（适用于加速度计、陀螺仪、磁力计等3轴传感器）
    
    网络结构：
    - 3个卷积层+池化层
    - 批量归一化和ELU激活
    - 最终输出768维特征向量
    """
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

        self.flattened_size = 128 * 6  # 计算展平后的维度
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
    """
    二维传感器数据编码器（适用于GPS等2轴数据）
    
    网络结构：与Encoder3d4类似，但输入通道数为2
    """
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
    """
    一维传感器数据编码器（适用于温度、深度、压力等单轴传感器）
    
    网络结构：与Encoder3d4类似，但输入通道数为1
    """
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
    """
    灵活的多模态神经网络
    
    功能：
    1. 根据传感器类型自动构建对应的编码器
    2. 支持对比学习和监督学习两种模式
    3. 多传感器数据融合和特征提取
    
    Args:
        input_dim: 每个编码器的输出维度
        sensor_types: 传感器类型列表
        number_classes: 分类类别数
    """
    def __init__(self,
                 input_dim=128 * 6,
                 sensor_types=['accelerometer', 'GPS', 'Depth'],  # GUI选择的传感器
                 number_classes=5):
        super(FlexibleNN, self).__init__()
        self.sensor_type_list = sensor_types
        self.input_dim = input_dim
        self.encoder_dict = nn.ModuleDict()

        # 根据传感器类型构建对应的编码器
        for modality in sensor_types:
            if modality in ['acceleration','accelerometer', 'gyroscope', 'magnetometer','magnitude']:
                self.encoder_dict[modality] = Encoder3d4()
            elif modality in ['GPS']:
                self.encoder_dict[modality] = Encoder2d()
            elif modality in ['pressure', 'temperature', 'light', 'sound', 'Depth']:
                self.encoder_dict[modality] = Encoder1d()
            else:
                raise ValueError(f"Unsupported sensor type in model: {modality}")

        # 共享投影层，用于对比学习和分类
        self.linear = nn.Linear(self.input_dim*len(sensor_types), 32)

        # 对比学习投影头
        self.projector = nn.Sequential(
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 16)
        )

        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, number_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, data, if_contrast=True):
        """
        前向传播
        
        Args:
            data: 输入数据 [batch_size, total_channels, sequence_length]
            if_contrast: 是否用于对比学习（True）或分类（False）
            
        Returns:
            output: 对比学习特征或分类结果
            concat_fea: 融合后的特征向量
        """
        start_idx = 0
        features = []

        # 按传感器类型分别处理数据
        for modality in self.sensor_type_list:
            if modality in ['accelerometer', 'acceleration', 'gyroscope', 'magnetometer','magnitude']:
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

        # 特征融合
        concat_fea = torch.cat(features, dim=1)
        concat_fea = self.linear(concat_fea)

        # 根据模式选择输出
        if if_contrast:
            output = self.projector(concat_fea)  # 对比学习
        else:
            output = self.classifier(concat_fea)  # 分类

        return output, concat_fea

class SupContrastiveLoss(nn.Module):
    """
    监督对比学习损失函数
    
    功能：
    1. 拉近同类样本在特征空间中的距离
    2. 推远不同类样本的距离
    3. 提升特征表示的判别能力
    
    Args:
        temperature: 温度参数，控制对比学习的强度
    """
    def __init__(self, temperature=0.3):
        super(SupContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        """
        Args:
            features: 特征向量 [batch_size, feature_dim]，已归一化
            labels: 样本标签 [batch_size]
        Returns:
            loss: 监督对比学习损失值
        """
        features = F.normalize(features, dim=1)
        similarity_matrix = torch.matmul(features, features.T) / self.temperature

        # 数值稳定处理
        similarity_matrix = similarity_matrix - torch.max(similarity_matrix, dim=1, keepdim=True)[0]

        # 构建正样本和负样本掩码
        labels1 = labels.unsqueeze(1)
        mask = torch.eye(similarity_matrix.size(0), dtype=torch.bool, device=features.device)
        positive_mask = (labels1 == labels1.T) & ~mask

        exp_sim = torch.exp(similarity_matrix)
        numerator = exp_sim * positive_mask
        denominator = exp_sim * ~mask

        numerator_sum = numerator.sum(dim=1) + 1e-8
        denominator_sum = denominator.sum(dim=1) + 1e-8

        valid_mask = numerator_sum > 0  # 跳过没有正样本对的样本
        loss = -torch.log(numerator_sum[valid_mask] / denominator_sum[valid_mask])
        loss = loss.mean()

        # # cluster center loss
        # all_loss = combined_loss(latent, labels, loss, lambda_intra=1.0, lambda_inter=10)

        return loss


# # 初始化模型和优化器
# model = FlexibleNN(sensor_types=sensor_types, number_classes=nclass)
# model = model.to(device)
# classify_criterion = nn.CrossEntropyLoss()  # 分类损失
# supContrast_criterion = SupContrastiveLoss()  # 对比学习损失
# optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)


################################## 深度学习模型框架 ##########################


################################## 潜在空间生成和可视化 ##########################
'''
功能模块：
1. 生成数据的潜在空间表示
2. 训练深度学习模型
3. 可视化结果

主要参数：
- model: 深度学习模型
- data: 训练数据
- pthPath: 模型权重保存路径（可选）
- supervisedCount: 监督学习训练轮数（可选）
- contrastiveCount: 对比学习训练轮数（可选）

根据传入参数分别执行：
1. 根据权重生成潜在空间表示
2. 根据训练轮数生成潜在空间表示，并保存权重文件
'''

# # 训练配置参数
# batch_size = 512
# sampling_method = 'entropy'  # 采样方法：entropy, random, least_confidence等
# SupCount = 1  # 监督对比学习轮数
# warmup = 20  # 对比学习预热轮数
# name_label = '_%s_Contrast%s_warm%s' % (
#     sampling_method, str(SupCount), str(warmup))

# 可视化颜色配置
label_colors = {
    0: "#5470C6",  # 深蓝 - Resting
    1: "#91CC75",  # 绿色 - Swimming
    2: "#FAC858",  # 金黄 - Stay in surface
    3: "#EE6666",  # 红色 - Gliding
    4: "#73C0DE",  # 天蓝 - Feeding
    5: 'brown',     # 棕色 - Scratching
    6: 'pink',      # 粉色 - Breathing
    7: 'cyan',      # 青色
    8: 'magenta',   # 品红
    9: 'lime',      # 青柠色
    10: 'teal',     # 蓝绿色
    11: 'violet',   # 紫罗兰色
    12: 'gold',     # 金色
    13: 'coral',    # 珊瑚色
    14: 'salmon'    # 三文鱼色
}


# 统一各数据集: int → 名称
DATASET_LABEL_NAME_MAP = {
    'turtle': {
        0: 'Resting',
        1: 'Swimming',
        2: 'Stay in surface',
        3: 'Gliding',
        4: 'Feeding',
        5: 'Scratching',
        6: 'Breathing'
    },
    'umineko': {
        0: 'ground_stationary',
        1: 'stationary',
        2: 'bathing',
        3: 'flying_active',
        4: 'flying_passive',
        5: 'foraging'
    },
    'omizu': {  # 若 is_omizu in 名称
        0: 'stationary',
        1: 'bathing',
        2: 'flying',
        3: 'foraging'
    }
}

# 新增: 针对不同数据集的颜色映射（与各 plot_scatter_* 的 color_discrete_map 对齐）
# 名称 → 颜色（与各 plot_scatter_* color_discrete_map 完全一致）
def get_dataset_color_map(dataset_name: str) -> dict:
    dn = dataset_name.lower()
    if 'omizu' in dn:
        return {
            'stationary': label_colors[0],
            'bathing': label_colors[1],
            'flying': label_colors[2],
            'foraging': label_colors[3],
        }
    if dn == 'umineko':
        return {
            'ground_stationary': label_colors[0],
            'stationary': label_colors[1],
            'bathing': label_colors[2],
            'flying_active': label_colors[3],
            'flying_passive': label_colors[4],
            'foraging': label_colors[5],
        }
    if dn == 'turtle':
        return {
            'Resting in flow': label_colors[0],
            'Swimming': label_colors[1],
            'Stay in surface': label_colors[2],
            'Gliding': label_colors[3],
            'Feeding': label_colors[4],
            'Scratching': label_colors[5],
            'Breathing': label_colors[6],
        }
    return {}




def get_label_name(dataset_name: str, label_int: int) -> str:
    dn = dataset_name.lower()
    key = 'omizu' if 'omizu' in dn else dn
    mapping = DATASET_LABEL_NAME_MAP.get(key, {})
    return mapping.get(label_int, 'Unknown')

def get_color_for_label(dataset_name: str, label_int: int) -> str:
    name = get_label_name(dataset_name, label_int)
    cmap = get_dataset_color_map(dataset_name)
    return cmap.get(name, "#999999")

def build_brushes(dataset_name: str, label_int_array):
    import pyqtgraph as pg
    colors = []
    brushes = []
    for lid in label_int_array:
        c = get_color_for_label(dataset_name, int(lid))
        colors.append(c)
        brushes.append(pg.mkBrush(c))
    return colors, brushes

def _get_plotly_color_logic(dataset_name: str):
    """
    不修改原有 plotly 绘图函数的前提下，复刻它们内部的颜色逻辑，
    以便 PyQtGraph 使用与 plotly 最终一致的颜色。
    """
    ds = dataset_name.lower()
    if 'omizu' in ds:
        # 与 plot_scatter_omizu 内部 color_discrete_map 完全一致
        label_map_dict = labeldict_findstr_omizu  # int → label str
        color_discrete_map = {
            'stationary': label_colors[0],
            'bathing': label_colors[1],
            'flying': label_colors[2],
            'foraging': label_colors[3],
        }
    elif ds == 'umineko':
        label_map_dict = labeldict_findstr
        color_discrete_map = {
            'ground_stationary': label_colors[0],
            'stationary': label_colors[1],
            'bathing': label_colors[2],
            'flying_active': label_colors[3],
            'flying_passive': label_colors[4],
            'foraging': label_colors[5],
        }
    elif ds == 'turtle':
        # 注意：当前 plot_scatter_turtle 使用的 color_discrete_map 只包含 7 类
        # 而 vis_scatter_label_2d 对 turtle 使用的是 labeldict_findstr_turtle
        # （包含 Resting in flow / Swimming ascent 等，与 plotly 的 map 不匹配）
        # 为保持“逻辑一致”而不修改 plotly 代码：若出现不在 color_discrete_map 的标签，
        # 需要使用 plotly 默认序列顺序补色。
        label_map_dict = labeldict_findstr_turtle
        color_discrete_map = {
            'Resting': label_colors[0],
            'Swimming': label_colors[1],
            'Stay in surface': label_colors[2],
            'Gliding': label_colors[3],
            'Feeding': label_colors[4],
            'Scratching': label_colors[5],
            'Breathing': label_colors[6],
        }
    else:
        label_map_dict = {}
        color_discrete_map = {}
    return label_map_dict, color_discrete_map

def map_labels_to_plotly_colors(label_int_array, dataset_name: str):
    """
    复刻 plotly.express.scatter 的颜色分配策略：
    - 已在 color_discrete_map 指定的类别使用指定颜色
    - 其它类别按首次出现顺序使用默认 qualitative.Plotly 序列
    """
    from plotly.express.colors import qualitative as qcolors
    default_seq = qcolors.Plotly  # 默认颜色序列
    label_map_dict, cdm = _get_plotly_color_logic(dataset_name)

    # 将 int → label str
    label_str_list = [label_map_dict.get(int(lid), f"UNK_{lid}") for lid in label_int_array]

    # 找出需要默认配色的标签（保持首次出现顺序）
    seen = set()
    unknown_labels_ordered = []
    for name in label_str_list:
        if name not in cdm and name not in seen:
            seen.add(name)
            unknown_labels_ordered.append(name)

    # 为未映射标签分配默认序列颜色
    seq_idx = 0
    for name in unknown_labels_ordered:
        cdm[name] = default_seq[seq_idx % len(default_seq)]
        seq_idx += 1

    # 返回与每个样本对应的颜色列表
    color_list = [cdm[name] for name in label_str_list]
    return label_str_list, color_list, cdm

# 不同数据集的标签映射字典
labeldict_findstr = {
    0: 'ground_stationary',
    1: 'stationary',
    2: 'bathing',
    3: 'flying_active',
    4: 'flying_passive'
}

labeldict_findstr_omizu = {
    0: 'stationary',
    1: 'bathing',
    2: 'flying',
    3: 'foraging'
}

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
    """
    自定义数据集类，用于PyTorch DataLoader
    
    功能：
    1. 封装传感器数据和标签
    2. 支持批量加载和GPU加速
    3. 同时处理1D和2D标签
    
    Args:
        samples: 传感器数据样本
        labels1d: 1D标签（众数标签）
        labels2d: 2D标签（原始时序标签）
        device: 计算设备
    """
    def __init__(self, samples, labels1d, labels2d, device='cpu'):
        self.samples = torch.tensor(samples).to(device)
        self.labels1d = torch.LongTensor(labels1d).to(device)
        self.labels2d = torch.tensor(labels2d)

    def __getitem__(self, index):
        target2d = self.labels2d[index]
        target1d = self.labels1d[index]
        sample = self.samples[index]
        return sample, target1d, target2d

    def __len__(self):
        return len(self.labels1d)

def load_model_weights(model, pth_path, device='cpu'):
    """
    从.pth文件加载模型权重
    
    Args:
        model: 模型实例
        pth_path: 权重文件路径
        device: 计算设备
        
    Returns:
        加载权重后的模型
    """
    state_dict = torch.load(pth_path, map_location=device)
    model.load_state_dict(state_dict)
    print(f"✅ 模型权重已加载自: {pth_path}")
    return model

def save_model_weights(model, save_path):
    """
    保存模型参数到指定路径
    
    Args:
        model: 要保存的模型
        save_path: 保存路径
    """
    torch.save(model.state_dict(), save_path)
    print(f"✅ 模型权重已保存到: {save_path}")

def AE_eval_time_series(train_loader, model, device, memotimes=30):
    """
    评估模型并提取潜在空间表示
    
    功能：
    1. 将模型设置为评估模式
    2. 前向传播获取特征表示
    3. 收集样本、标签和预测结果
    
    Args:
        train_loader: 数据加载器
        model: 训练好的模型
        device: 计算设备
        memotimes: 内存限制（批次数）
        
    Returns:
        tuple: (特征表示列表, 样本列表, 预测列表, 标签列表)
    """
    model.eval()

    representation_list = []
    sample_list, timestamp_list, label_list, pred_list, timestr_list, flag_list = [], [], [], [], [], []
    
    for i, (sample, _, label2d) in enumerate(train_loader):
        sample = sample.to(device=device, non_blocking=True, dtype=torch.float)
        # 获取潜在空间表示
        x_encoded, output = model(sample)
        
        # 收集结果
        tmp_representation = x_encoded.detach().cpu().numpy()
        representation_list.append(tmp_representation)
        sample_list.append(sample.detach().cpu().numpy())
        label_list.append(label2d.detach().cpu().numpy())
        pred_list.append(output.detach().cpu().numpy())

    return representation_list, sample_list, pred_list, label_list

def freeze_encoders(model):
    """
    冻结编码器参数，只训练分类器
    
    用途：在监督学习阶段，固定特征提取部分，只优化分类层
    """
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
    """
    解冻编码器参数，冻结分类器
    
    用途：在对比学习阶段，训练特征提取部分，不训练分类层
    """
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
    """
    解冻所有模型参数
    
    用途：端到端训练所有网络层
    """
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
    """
    训练深度学习模型
    
    功能：
    1. 支持对比学习和监督学习两种模式
    2. 批量训练和梯度更新
    3. 损失跟踪和监控
    
    Args:
        model: 要训练的模型
        loader: 数据加载器
        criterion: 损失函数
        optimizer: 优化器
        epochs: 训练轮数
        device: 计算设备
        if_contrast: 是否为对比学习模式
        
    Returns:
        tuple: (训练后的模型, 平均损失列表)
    """
    model.train()
    avg_loss = []
    
    for epoch in range(epochs):
        losses = []
        for batch in loader:
            data, labels, _ = batch
            data = data.to(device=device, dtype=torch.float)
            labels = labels.to(device=device, dtype=torch.long)
            
            # 前向传播
            outputs, fea = model(data, if_contrast)
            loss = criterion(outputs, labels)
            losses.append(loss.item())

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        avg_loss.append(np.mean(losses))
        # print(f"Epoch {epoch + 1}, Loss: {avg_loss[-1]}")
    return model, avg_loss


def plot_scatter_omizu(data_umap, label_str, iteration, name):
    """
    为Omizunagidori数据集绘制散点图
    
    Args:
        data_umap: UMAP降维后的2D数据
        label_str: 标签字符串列表
        iteration: 迭代次数
        name: 图表名称
    """
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
    fig_2d.write_html(f'./test/test_{name}_activeSup_epoch_{iteration}.html')
    return

def plot_scatter_umi(data_umap, label_str, iteration, name):
    """
    为Umineko数据集绘制散点图
    
    Args:
        data_umap: UMAP降维后的2D数据
        label_str: 标签字符串列表
        iteration: 迭代次数
        name: 图表名称
    """
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
    fig_2d.write_html(f'./test/test_{name}_activeSup_epoch_{iteration}.html')
    return

def plot_scatter_turtle(data_umap, label_str, iteration, name):
    """
    为海龟数据集绘制散点图
    
    Args:
        data_umap: UMAP降维后的2D数据
        label_str: 标签字符串列表
        iteration: 迭代次数
        name: 图表名称
    """
    fig_2d = px.scatter(
        data_umap, x=0, y=1,
        color=label_str,
        labels={'activity': 'activity'},
        color_discrete_map={'Resting': label_colors[0],
                            'Swimming': label_colors[1],
                            'Stay in surface': label_colors[2],
                            'Gliding': label_colors[3],
                            'Feeding': label_colors[4],
                            'Scratching': label_colors[5],
                            'Breathing': label_colors[6],
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
    fig_2d.write_html(f'./test/test_{name}_activeSup_epoch_{iteration}.html')
    return

def vis_scatter_label_2d(representation_list, label_list,
                         iteration, name, plot_flag = False, is_omizu=False, label_colors=None):
    """
    可视化潜在空间的2D散点图
    
    功能：
    1. 使用UMAP降维到2D空间
    2. 根据数据集类型选择合适的标签映射
    3. 生成交互式散点图
    
    Args:
        representation_list: 潜在空间表示列表
        label_list: 标签列表
        iteration: 迭代次数
        name: 图表名称
        plot_flag: 是否生成图表
        is_omizu: 数据集类型标识
        
    Returns:
        np.ndarray: UMAP降维后的2D数据
    """
    label_concat = np.concatenate(label_list)
    repre_concat = np.concatenate(representation_list)
    repre_reshape = repre_concat.reshape(repre_concat.shape[0], -1).astype(float)

    # UMAP降维 - 这是计算密集的步骤！
    umap_2d = UMAP(n_components=2)
    data_umap = umap_2d.fit_transform(repre_reshape)

    # 获取众数标签并转换为字符串
    label_concat_vote = majority_value(label_concat)
    if 'omizu' in is_omizu:
        label_concat_vote_str = [labeldict_findstr_omizu[i] for i in label_concat_vote]
    elif is_omizu == 'turtle':
        label_concat_vote_str = [labeldict_findstr_turtle[i] for i in label_concat_vote]
    elif is_omizu == 'umineko':
        label_concat_vote_str = [labeldict_findstr[i] for i in label_concat_vote]
    else:
        print('no label to string dictionary')
        
    # 生成可视化图表
    # if plot_flag:
    #     if 'omizu' in is_omizu:
    #         plot_scatter_omizu(data_umap, label_concat_vote_str, iteration, name + '_gt')
    #     elif 'umineko' == is_omizu:
    #         plot_scatter_umi(data_umap, label_concat_vote_str, iteration, name+'_gt')
    #     elif is_omizu == 'turtle':
    #         plot_scatter_turtle(data_umap, label_concat_vote_str, iteration, name+'_gt')
    #     else:
    #         print('no plot for this dataset')

    pw_scatter = None
    if plot_flag:
        pw_scatter = plot_scatter_pg(data_umap, label_concat_vote,
                                     iteration, name + '_gt',
                                     dataset_name=is_omizu,
                                     show=True, save=False, label_colors=label_colors)

    return data_umap, pw_scatter


def test_plot_scatter_pg(data_umap):
    """
    测试绘制 UMAP 2D 散点图的功能
    """
    label_concat_vote_str = [labeldict_findstr_omizu[i] for i in label_concat_vote]

    # 调用绘图函数
    plot_scatter_pg(data_umap, label_concat_vote_str, iteration=0, name='test_plot')



# def plot_scatter_pg(data_umap, label_str, iteration, name,
#                     dataset_name='turtle',
#                     point_size=6,
#                     show=True,
#                     save=False,
#                     save_width=1000):
#     """
#     使用 pyqtgraph 绘制 UMAP 2D 散点。
#     默认只显示窗口，不再保存文件；若需保存设 save=True。

#     args:
#         data_umap: ndarray [N,2]
#         label_str: 长度 N 的标签字符串列表
#         iteration: 迭代次数
#         name: 图表名称
#         dataset_name: 数据集名称
#         point_size: 点大小
#         show: 是否显示图表
#         save: 是否保存图表
#         save_width: 保存图表的宽度

#     """
#     import pyqtgraph as pg
#     from pyqtgraph.Qt import QtCore
#     from PySide6.QtWidgets import QApplication, QToolTip
#     app = _build_qapp()

#     cmap_fixed = get_dataset_color_map(dataset_name)
#     dynamic_map = {}
#     next_color_idx = 0
#     default_cycle = list(label_colors.values())

#     colors_for_points = []
#     for lbl in label_str:
#         if lbl in cmap_fixed:
#             colors_for_points.append(cmap_fixed[lbl])
#         else:
#             if lbl not in dynamic_map:
#                 dynamic_map[lbl] = default_cycle[next_color_idx % len(default_cycle)]
#                 next_color_idx += 1
#             colors_for_points.append(dynamic_map[lbl])

#     pw = pg.PlotWidget(title=f"{dataset_name} UMAP (iteration={iteration})")
#     pw.setLabel('bottom', 'UMAP 1')
#     pw.setLabel('left', 'UMAP 2')
#     # pw.showGrid(x=True, y=True, alpha=0.3)

#     # 显示鼠标悬停事件
#     class HoverScatter(pg.ScatterPlotItem):
#         def __init__(self, *a, **kw):
#             super().__init__(*a, **kw)
#             self.setAcceptHoverEvents(True)
#         def hoverEvent(self, ev):
#             if ev.isExit():
#                 return
#             pts = self.pointsAt(ev.pos())
#             # 修复: 避免 “ValueError: The truth value of an empty array is ambiguous”
#             if pts is None or len(pts) == 0:
#                 return
#             pt = pts[0]
#             dat = pt.data()
#             if not dat:
#                 return
#             txt = f"索引: {dat.get('index')}\n标签: {dat.get('label')}"
#             QToolTip.showText(ev.screenPos().toPoint(), txt)

#     spots = []
#     for i, (xy, lbl, color_hex) in enumerate(zip(data_umap, label_str, colors_for_points)):
#         brush_color = pg.mkColor(color_hex)
#         if lbl == '-2.0':
#             brush_color = brush_color.lighter(170)
#         spots.append({
#             'pos': (float(xy[0]), float(xy[1])),
#             'brush': pg.mkBrush(brush_color),
#             'pen': None,
#             'size': point_size,
#             'data': {'index': i, 'label': lbl}
#         })

#     scatter = HoverScatter()
#     scatter.addPoints(spots)
#     pw.addItem(scatter)

#     # TODO 添加图例，图例颜色有bug需要修复
#     # legend = pw.addLegend()
#     # added = set()
#     # for lbl, color_hex in zip(label_str, colors_for_points):
#     #     if lbl in added:
#     #         continue
#     #     dummy = pg.PlotDataItem(pen=None, symbol='o', symbolBrush=color_hex, symbolSize=point_size+2)
#     #     legend.addItem(dummy, lbl)
#     #     added.add(lbl)

#     if save:
#         # 可选：仍支持保存
#         from pyqtgraph.exporters import ImageExporter
#         exporter = ImageExporter(pw.plotItem)
#         exporter.parameters()['width'] = save_width
#         save_path = f'./test/test_{name}_activeSup_epoch_{iteration}.png'
#         exporter.export(save_path)
#         print(f"✅ 已保存: {save_path}")

#     if show:
#         pw.show()
#         app.processEvents()
    
#     app.exec()

#     return pw, scatter


# 显示鼠标悬停事件
class HoverScatter(pg.ScatterPlotItem):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.setAcceptHoverEvents(True)
    def hoverEvent(self, ev):
        if ev.isExit():
            return
        pts = self.pointsAt(ev.pos())
        # 修复: 避免 “ValueError: The truth value of an empty array is ambiguous”
        if pts is None or len(pts) == 0:
            return
        pt = pts[0]
        dat = pt.data()
        if not dat:
            return
        txt = f"索引: {dat.get('index')}\n标签: {dat.get('label')}"
        QToolTip.showText(ev.screenPos().toPoint(), txt)


def plot_scatter_pg_old(data_umap, label_str, iteration, name,
                    dataset_name='turtle',
                    point_size=6,
                    show=True,
                    save=False,
                    save_width=1000):
    """
    使用 pyqtgraph 绘制 UMAP 2D 散点。
    默认只显示窗口，不再保存文件；若需保存设 save=True。

    args:
        data_umap: ndarray [N,2]
        label_str: 长度 N 的标签字符串列表
        iteration: 迭代次数
        name: 图表名称
        dataset_name: 数据集名称
        point_size: 点大小
        show: 是否显示图表
        save: 是否保存图表
        save_width: 保存图表的宽度

    """


    cmap_fixed = get_dataset_color_map(dataset_name)
    dynamic_map = {}
    next_color_idx = 0
    default_cycle = list(label_colors.values())

    colors_for_points = []
    for lbl in label_str:
        if lbl in cmap_fixed:
            colors_for_points.append(cmap_fixed[lbl])
        else:
            if lbl not in dynamic_map:
                dynamic_map[lbl] = default_cycle[next_color_idx % len(default_cycle)]
                next_color_idx += 1
            colors_for_points.append(dynamic_map[lbl])

    spots = []
    for i, (xy, lbl, color_hex) in enumerate(zip(data_umap, label_str, colors_for_points)):
        brush_color = pg.mkColor(color_hex)
        if lbl == '-2.0':
            brush_color = brush_color.lighter(170)
        spots.append({
            'pos': (float(xy[0]), float(xy[1])),
            'brush': pg.mkBrush(brush_color),
            'pen': None,
            'size': point_size,
            'data': {'index': i, 'label': lbl}
        })

    scatter = HoverScatter()
    scatter.addPoints(spots)

    return scatter


def plot_scatter_pg(data_umap, label_np, iteration, name,
                    dataset_name='turtle',
                    point_size=6,
                    show=True,
                    save=False,
                    save_width=1000,
                    label_colors=None):
    """
    使用 pyqtgraph 绘制 UMAP 2D 散点。
    默认只显示窗口，不再保存文件；若需保存设 save=True。

    args:
        data_umap: ndarray [N,2]
        label_np: 长度 N 的标签数组（可能为 np.int64 等）
        iteration: 迭代次数
        name: 图表名称
        dataset_name: 数据集名称
        point_size: 点大小
        show: 是否显示图表
        save: 是否保存图表
        save_width: 保存图表的宽度
        label_colors: 标签→颜色映射（支持：int键、str键或标签名键）
    """
    # 允许 label_colors 为空时使用数据集默认颜色表
    cmap_fixed = label_colors if label_colors else get_dataset_color_map(dataset_name)

    # 回退颜色序列（当映射中找不到时使用）
    fallback_cycle = [
        '#91cc75', '#5470c6', '#fac858', '#ee6666',
        '#73c0de', '#3ba272', '#fc8452', '#9a60b4',
        '#ea7ccc'
    ]
    default_cycle = list(cmap_fixed.values()) or fallback_cycle

    colors_for_points = []
    labels_py = []

    # TODO 优化：当前实现对每个点都尝试多种键，效率较低
    # TODO 需要将窗口级别标签写回 df 的行级列 colname,来获取开始结束时间
    for lbl in label_np:
        # 转为纯 Python 标量，避免 np.int64 等类型带来的键不匹配
        lbl_py = lbl.item() if hasattr(lbl, 'item') else lbl
        lid_int = int(lbl_py)
        # 直接索引列表default_cycle的值
        color_hex = default_cycle[lid_int]
        labels_py.append(lbl_py)


        # 构造候选键：原值、字符串、整数、以及“数据集名映射后的标签名”
        # candidates = [lbl_py, str(lbl_py)]
        # try:
        #     lid_int = int(lbl_py)
        #     candidates.append(lid_int)
        #     # 尝试用数据集映射把 int → 标签名
        #     candidates.append(get_label_name(dataset_name, lid_int))
        # except Exception:
        #     pass

        # color_hex = None
        # for key in candidates:
        #     if key in cmap_fixed:
        #         color_hex = cmap_fixed[key]
        #         break

        if color_hex is None:
            # 找不到映射时，安全回退到循环色板
            try:
                idx = int(lbl_py) % len(default_cycle)
            except Exception:
                idx = 0
            color_hex = default_cycle[idx]

        colors_for_points.append(color_hex)

    spots = []
    for i, (xy, lbl_py, color_hex) in enumerate(zip(data_umap, labels_py, colors_for_points)):
        brush_color = pg.mkColor(color_hex)
        # 未知类高亮（支持 -2、-2.0、'-2'）
        if lbl_py in (-2, -2.0, '-2'):
            brush_color = brush_color.lighter(170)
        spots.append({
            'pos': (float(xy[0]), float(xy[1])),
            'brush': pg.mkBrush(brush_color),
            'pen': None,
            'size': point_size,
            'data': {'index': i, 'label': lbl_py}
        })

    scatter = HoverScatter()
    scatter.addPoints(spots)
    return scatter




def create_latent(plot_loader,
                  model,
                  dataset='turtle',
                  device='cpu',
                  name_label='_test',
                  sensor_types=None,
                  label_colors=None):
    """
    创建潜在空间表示和可视化
    
    功能：
    1. 提取模型的潜在空间表示
    2. 进行UMAP降维
    3. 生成可视化图表
    
    Args:
        plot_loader: 数据加载器
        model: 训练好的模型
        dataset: 数据集名称
        device: 计算设备
        name_label: 标签名称
        sensor_types: 传感器类型
        
    Returns:
        np.ndarray: UMAP降维后的数据
    """
    # 提取潜在空间表示
    repres_list, sample_list, pred_list, label_list = \
        AE_eval_time_series(plot_loader, model, device)
    
    # 生成2D可视化
    data_umap, pw_scatter = (
        vis_scatter_label_2d(repres_list,
                             label_list, 0,
                             str(sensor_types) + name_label,
                             plot_flag=True,
                             is_omizu=dataset,
                             label_colors=label_colors
                             )
    )
    # data_umap = (
    #     vis_scatter_label_2d(repres_list,
    #                          label_list, 0,
    #                          str(sensor_types) + name_label,
    #                          plot_flag=False,
    #                          is_omizu=dataset))

    return data_umap, pw_scatter

def create_loader(data_b, label_b, batch_size=512, shuffle=False, device='cpu'):
    """
    创建PyTorch数据加载器
    
    Args:
        data_b: 输入数据
        label_b: 标签数据
        batch_size: 批处理大小
        shuffle: 是否打乱数据
        device: 计算设备
        
    Returns:
        DataLoader: PyTorch数据加载器对象
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

# def generate_propagated_labels(data_umap, label_b):
#     # 我们将已经标记的样本标签设置为大于0，其余样本为 -1（未标记）
#     all_labels_before = majority_value(label_b)  # 最终为一列

#     # Apply Label Spreading for semi-supervised label propagation
#     label_spread_model = LabelSpreading(kernel='knn', n_neighbors=10, alpha=0.2)  # Graph-based propagation
#     label_spread_model.fit(data_umap, all_labels_before)

#     # Get the predicted labels for all data
#     propagated_labels = label_spread_model.transduction_
#     return propagated_labels



def label_propagation(X, y, k=10, max_iter=30, tol=1e-4):
    """
    Label Propagation with approximate KNN graph (HNSW) and hard constraint
    X: 特征矩阵 [n_samples, n_features]
    y: 标签数组，未标记为 -1
    k: 每个样本KNN邻居数
    max_iter: 最大迭代轮数
    tol: 收敛容忍度（两次更新的变化）
    """

    n_samples = X.shape[0]
    X = X.astype(np.float32)

    # Step 1: 构建近似KNN图（用 HNSW 提速）
    d = X.shape[1]
    index = faiss.IndexHNSWFlat(d, 32)
    index.hnsw.efSearch = 64
    faiss.omp_set_num_threads(8)
    index.add(X)
    D, I = index.search(X, k + 1)
    D = D[:, 1:]
    I = I[:, 1:]

    # Step 2: 构建稀疏邻接矩阵 W（RBF核）
    rows = np.repeat(np.arange(n_samples), k)
    cols = I.flatten()
    weights = np.exp(-D.flatten())
    W = csr_matrix((weights, (rows, cols)), shape=(n_samples, n_samples))

    # Step 3: 对称归一化传播矩阵 P = D^{-1/2} W D^{-1/2}
    d_sqrt_inv = 1.0 / np.sqrt(W.sum(axis=1).A1 + 1e-12)
    D_inv_sqrt = diags(d_sqrt_inv)
    P = D_inv_sqrt @ W @ D_inv_sqrt

    # Step 4: 初始化标签矩阵 Y
    label_mask = (y != -1)
    lb = LabelBinarizer()
    Y_labeled = lb.fit_transform(y[label_mask])
    n_classes = Y_labeled.shape[1]

    Y = np.zeros((n_samples, n_classes))
    Y[label_mask] = Y_labeled

    # Step 5: Label Propagation（固定label点）
    for _ in range(max_iter):
        Y_prev = Y.copy()
        Y = P @ Y
        Y[label_mask] = Y_labeled  # 保持有标签点不变

        if np.linalg.norm(Y - Y_prev) < tol:
            break

    # Step 6: 输出预测
    y_pred = np.argmax(Y, axis=1)
    y_pred[label_mask] = y[label_mask]  # 恢复原始标签

    return y_pred, Y
	
def generate_propagated_labels(data_umap, label_b, maxiter = 30):
    """
    使用自定义标签传播生成新的标签
    Args:
        data_umap: UMAP降维后的2D数据
        label_b: 原始标签数组
        maxiter: 最大迭代次数
    Returns:
        np.ndarray: 传播后的标签数组
    """
    # 我们将已经标记的样本标签设置为大于0，其余样本为 -1（未标记）
    all_labels_before = majority_value(label_b)  # 最终为一列
    
    y_pred, Y_current = label_propagation(data_umap, all_labels_before,
                                                 k=10, max_iter=maxiter)

    return y_pred

# # # 使用标签传播生成新的标签
# propagated_labels = generate_propagated_labels(data_umap, label_b)


################################## label propagation ##########################



################################## 主动学习采样策略 ##########################
"""
主动学习采样策略模块

包含多种采样方法：
1. 不确定性采样 - 基于模型预测的不确定性
2. 随机采样 - 随机选择样本
3. 代表性采样 - 基于聚类的代表性样本选择
4. 数据增强过采样 - 针对少数类的数据增强
5. 随机欠采样 - 针对多数类的数据平衡
"""

# 采样配置参数
select_size = 100

def least_confidence(probabilities, n_samples):
    """
    最小置信度采样策略
    
    选择模型对其预测最不自信的样本，通过计算每个样本的最大预测概率实现。
    
    Args:
        probabilities: 模型预测概率 [n_samples, n_classes]
        n_samples: 要选择的样本数量
        
    Returns:
        np.ndarray: 选中样本的索引
    """
    # 计算最大置信度
    max_confidence = np.max(probabilities, axis=1)
    # 选择最小置信度的样本
    selected_indices = np.argsort(max_confidence)[:n_samples]
    return selected_indices

def min_margin(probabilities, n_samples):
    """
    最小边际采样策略
    
    选择模型预测中第一和第二高概率之间差距最小的样本。
    
    Args:
        probabilities: 模型预测概率 [n_samples, n_classes]
        n_samples: 要选择的样本数量
        
    Returns:
        np.ndarray: 选中样本的索引
    """
    # 计算边际（第一高概率与第二高概率的差值）
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
    """
    不确定性采样策略（支持多种不确定性度量）
    
    根据模型预测的不确定性选择最有价值的样本进行标注。
    支持三种不确定性度量：熵、最小置信度、最小边际。
    
    Args:
        X_labeled: 已标注数据特征
        y_labeled: 已标注数据标签
        X_unlabeled: 未标注数据特征
        y_unlabeled: 未标注数据标签
        model: 训练好的模型
        update_size: 要选择的样本数量
        device: 计算设备
        choice: 不确定性度量方法 (0:熵, 1:最小置信度, 2:最小边际)
        
    Returns:
        tuple: 更新后的已标注数据、未标注数据和选中的样本
    """
    model.eval()
    with torch.no_grad():
        # 计算未标注数据的预测概率
        _, unlabeled_embeddings = model(torch.tensor(X_unlabeled, device=device, dtype=torch.float32),
                                        if_contrast=True)
        probs = model.classifier(unlabeled_embeddings)

        # 根据选择的方法计算不确定性
        if choice==0:
            # 使用熵作为不确定性度量
            uncertainty = entropy(probs.detach().cpu().numpy().T)
            selected_indices = np.argsort(uncertainty)[-update_size:]
        elif choice ==1 :
            # 使用最小置信度
            selected_indices = least_confidence(probs.detach().cpu().numpy(), update_size)
        else:
            # 使用最小边际
            selected_indices = min_margin(probs.detach().cpu().numpy(), update_size)

        # 选择样本并更新数据集
        selected_samples = X_unlabeled[selected_indices]
        selected_labels = y_unlabeled[selected_indices]
        
        # 更新标注集和未标注数据池
        X_labeled = np.vstack([X_labeled, selected_samples])
        y_labeled = np.concatenate([y_labeled, selected_labels], axis=0)
        X_unlabeled = np.delete(X_unlabeled, selected_indices, axis=0)
        y_unlabeled = np.delete(y_unlabeled, selected_indices, axis=0)
        
        print(f"Labeled samples: {len(X_labeled)}")
        # return X_labeled, y_labeled, X_unlabeled, y_unlabeled, (selected_samples, selected_labels)
        return X_labeled, y_labeled, X_unlabeled, y_unlabeled, (selected_samples, selected_labels, selected_indices)


def random_sampling(X_labeled, y_labeled,
                    X_unlabeled, y_unlabeled, update_size):
    """
    随机采样策略
    
    在未标注数据集中随机选取一定数量的样本更新到标注集中。
    作为基线方法，用于比较其他采样策略的效果。

    Args:
        X_labeled: 已标注数据特征 (N_labeled, D)
        y_labeled: 已标注数据标签 (N_labeled, )
        X_unlabeled: 未标注数据特征 (N_unlabeled, D)
        y_unlabeled: 未标注数据标签 (N_unlabeled, )
        update_size: 本轮迭代需要从未标注集中选取的样本数量

    Returns:
        tuple: 更新后的已标注数据、未标注数据和选中的样本
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
    # return X_labeled, y_labeled, X_unlabeled, y_unlabeled, (selected_samples, selected_labels)
    return X_labeled, y_labeled, X_unlabeled, y_unlabeled, (selected_samples, selected_labels, selected_indices)

def random_undersampling(X_labeled, y_labeled,
                         X_unlabeled, y_unlabeled,
                         majority_label, update_size, undersample_ratio=1.0):
    """
    随机欠采样策略
    
    在已标注数据集中，对多数类样本进行随机欠采样，以达到类别平衡。
    用于处理类别不平衡问题。

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
        # 如果多数类不存在，直接添加所有样本
        X_labeled = np.vstack([X_labeled, selected_samples])
        y_labeled = np.concatenate([y_labeled, selected_labels], axis=0)
        print(f"Error: 指定的多数类 {majority_label} 在数据集中不存在！")
        return X_labeled, y_labeled, X_unlabeled, y_unlabeled, np.array([])

    # 计算需要保留的多数类样本数量
    minority_count = min(class_counts.values())
    majority_count_target = int(minority_count * undersample_ratio)

    # 进行欠采样
    majority_indices = np.where(selected_labels_batch == majority_label)[0]
    selected_majority_indices = np.random.choice(majority_indices, majority_count_target, replace=False)
    minority_indices = np.where(selected_labels_batch != majority_label)[0]
    
    # 组合新数据索引
    selected_indices = np.concatenate([selected_majority_indices, minority_indices])
    X_selected_new = selected_samples[selected_indices]
    y_selected_new = selected_labels[selected_indices]

    # 更新标注数据
    X_labeled_new = np.vstack([X_labeled, X_selected_new])
    y_labeled_new = np.concatenate([y_labeled, y_selected_new], axis=0)

    print(f"Under-sampling completed: Majority class reduced from {class_counts[majority_label]} → {majority_count_target}")
    print(f"Labeled samples: {len(X_labeled_new)}")

    # return X_labeled_new, y_labeled_new, X_unlabeled, y_unlabeled, (X_selected_new, y_selected_new)
    return X_labeled_new, y_labeled_new, X_unlabeled, y_unlabeled, (X_selected_new, y_selected_new, selected_indices)

def data_augmentation_oversampling(X_labeled, y_labeled,
                                   X_unlabeled, y_unlabeled,
                                   minority_label, update_size,
                                   augmentation_factor=2):
    """
    数据增强过采样策略
    
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
    selected_samples = X_unlabeled[selected_indices]
    selected_labels = y_unlabeled[selected_indices]
    
    # 从未标注集中剔除已选样本
    X_unlabeled = np.delete(X_unlabeled, selected_indices, axis=0)
    y_unlabeled = np.delete(y_unlabeled, selected_indices, axis=0)

    # 获取少数类样本并进行数据增强
    y_labeled_batch = majority_value(selected_labels)
    minority_indices = np.where(y_labeled_batch == minority_label)[0]
    X_minority = selected_samples[minority_indices]

    X_synthetic = []
    y_synthetic = []
    
    if len(X_minority) == 0:
        X_new_labeled = selected_samples
        y_new_labeled = selected_labels
    else:
        # 对每个少数类样本进行增强
        for x, y in zip(X_minority, minority_indices):
            for _ in range(augmentation_factor):
                x_augmented = augment_sample(x)
                X_synthetic.append(x_augmented)
                y_synthetic.append(selected_labels[y])

        # 合并原始样本和增强样本
        X_synthetic = np.array(X_synthetic)
        y_synthetic = np.array(y_synthetic)
        X_new_labeled = np.vstack([X_synthetic, selected_samples])
        y_new_labeled = np.concatenate([y_synthetic, selected_labels])

    # 更新标注数据
    X_labeled = np.vstack([X_labeled, X_new_labeled])
    y_labeled = np.concatenate([y_labeled, y_new_labeled], axis=0)

    # # 由于数据增强的样本不属于原始未标注数据，X_unlabeled 和 y_unlabeled 为空
    # selected_labels = y_synthetic  # 记录新增的样本标签

    print(f"Data Augmentation Oversampling completed: Added {len(X_synthetic)} new samples for label {minority_label}")
    print(f"Labeled samples: {len(X_labeled)}")

    # return X_labeled, y_labeled, X_unlabeled, y_unlabeled, (X_new_labeled, y_new_labeled)
    return X_labeled, y_labeled, X_unlabeled, y_unlabeled, (X_new_labeled, y_new_labeled, selected_indices)

def augment_sample(x):
    """
    针对单个时序样本进行数据增强
    
    增强策略包括：
    1. 加噪声（随机高斯噪声）
    2. 时移（随机向前或向后平移）
    3. 放缩（信号幅度缩放）
    4. 翻转（适用于对称信号）

    Args:
        x: 输入的单个样本 (channels, time_steps)

    返回:
    ----------
    x_augmented : np.ndarray
        增强后的样本
    """
    x_aug = x.copy()

    # 1. 加高斯噪声 (Noise Injection)
    if np.random.rand() < 0.5:  # 50% 概率应用
        noise = np.random.normal(loc=0, scale=0.02 * np.std(x), size=x.shape)
        x_aug += noise

    # 2. 时移 (Time Shifting)
    if np.random.rand() < 0.5:
        shift = np.random.randint(-3, 3)  # 在 [-3, 3] 之间随机平移
        x_aug = np.roll(x_aug, shift, axis=-1)  # 在时间维度上平移

    # 3. 放缩 (Scaling)
    if np.random.rand() < 0.5:
        scale_factor = np.random.uniform(0.9, 1.1)  # 在 [0.9, 1.1] 之间缩放
        x_aug *= scale_factor

    # 4. 翻转 (Flipping)
    if np.random.rand() < 0.3:  # 30% 概率应用
        x_aug = -x_aug

    return x_aug

def representative_sampling(X_labeled, y_labeled,
                                   X_unlabeled, y_unlabeled,
                                   model, update_size, device):
    """
    代表性采样策略
    
    选择那些能代表整个数据分布的样本，使用K-means聚类算法来确定样本的代表性。
    每个聚类选择最接近聚类中心的样本。

    Args:
        X_labeled: 已标注数据特征
        y_labeled: 已标注数据标签
        X_unlabeled: 未标注数据特征
        y_unlabeled: 未标注数据标签
        model: 训练好的模型
        update_size: 要选择的样本数量
        device: 计算设备

    Returns:
        tuple: 更新后的已标注数据、未标注数据和选中的样本
    """
    model.eval()
    with torch.no_grad():
        # 获取潜在空间表示
        _, unlabeled_embeddings = model(torch.tensor(X_unlabeled, device=device, dtype=torch.float32),
                                        if_contrast=True)
    data = unlabeled_embeddings.detach().cpu().numpy()
    
    # 使用K-means聚类
    kmeans = KMeans(n_clusters=update_size, n_init=10).fit(data)
    labels = kmeans.labels_
    
    # 选择每个聚类中最接近聚类中心的样本
    selected_indices = np.zeros(kmeans.n_clusters, dtype=int)
    for cluster_index in range(kmeans.n_clusters):
        cluster_points_indices = np.where(labels == cluster_index)[0]
        cluster_points = data[cluster_points_indices]
        distances = np.linalg.norm(cluster_points - kmeans.cluster_centers_[cluster_index], axis=1)
        closest_point_index_within_cluster = np.argmin(distances)
        selected_indices[cluster_index] = cluster_points_indices[closest_point_index_within_cluster]

    # 更新数据集
    selected_samples = X_unlabeled[selected_indices]
    selected_labels = y_unlabeled[selected_indices]
    X_unlabeled = np.delete(X_unlabeled, selected_indices, axis=0)
    y_unlabeled = np.delete(y_unlabeled, selected_indices, axis=0)
    X_labeled = np.vstack([X_labeled, selected_samples])
    y_labeled = np.concatenate([y_labeled, selected_labels], axis=0)

    print(f"Labeled samples: {len(X_labeled)}")
    # return X_labeled, y_labeled, X_unlabeled, y_unlabeled, (selected_samples, selected_labels)
    return X_labeled, y_labeled, X_unlabeled, y_unlabeled, (selected_samples, selected_labels, selected_indices)

def data_sampling(sampling_method, X_labeled, y_labeled,
                  X_unlabeled, y_unlabeled,
                  model, update_size,
                  majority_label=0, minority_label=0, device='cpu'):
    """
    统一的数据采样接口
    
    根据指定的采样方法调用相应的采样策略。
    
    Args:
        sampling_method: 采样方法名称
        X_labeled: 已标注数据特征
        y_labeled: 已标注数据标签
        X_unlabeled: 未标注数据特征
        y_unlabeled: 未标注数据标签
        model: 训练好的模型
        update_size: 要选择的样本数量
        majority_label: 多数类标签
        minority_label: 少数类标签
        device: 计算设备
        
    Returns:
        tuple: 更新后的数据集和选中的样本 (selected_data = (newX_labeled, newy_labeled))
    """
    if sampling_method == "random":
        return random_sampling(X_labeled, y_labeled, X_unlabeled, y_unlabeled, update_size)
    elif sampling_method == "entropy":
        return uncertainty_sampling(X_labeled, y_labeled, X_unlabeled, y_unlabeled, model, update_size, device)
    elif sampling_method == "least_confidence":
        return uncertainty_sampling(X_labeled, y_labeled, X_unlabeled, y_unlabeled, model, update_size, device, choice=1)
    elif sampling_method == "min_margin":
        return uncertainty_sampling(X_labeled, y_labeled, X_unlabeled, y_unlabeled, model, update_size, device, choice=2)
    elif sampling_method == "underRandom":
        return random_undersampling(X_labeled, y_labeled, X_unlabeled, y_unlabeled, majority_label, update_size, undersample_ratio=1.0)
    elif sampling_method == "overAugment":
        return data_augmentation_oversampling(X_labeled, y_labeled, X_unlabeled, y_unlabeled, minority_label, update_size, augmentation_factor=2)
    elif sampling_method == "repreSamp":
        return representative_sampling(X_labeled, y_labeled, X_unlabeled, y_unlabeled, model, update_size, device)
    else:
        raise ValueError(f"Unknown sampling method: {sampling_method}")





def prepare_dataset(
    # pkl_path: str = r'./test/turtle.pkl',
    pkl_objects: list[pd.DataFrame] = None,
    sensor_dict: dict[str, list[str]] = None,
    sensor_types: list[str] = None,
    len_sw: int = 50,
    test_size: float = 0.2,
    labeled_ratio: float = 0.01
):
    """
    从数据文件到可训练/可评估/可采样的数据准备全流程。
    args:
      pkl_objects: 从pickle文件中读取的DataFrame列表
      sensor_dict: 各传感器对应的列名字典
      sensor_types: 选择的传感器类型列表
      len_sw: 滑动窗口长度
      test_size: 测试集比例
      labeled_ratio: 有标签数据比例
    返回：
      data_b, label_b: 整体滑窗后的张量 [B, C, T] 与标签窗口 [B, T]
      X_labeled, y_labeled, X_unlabeled, y_unlabeled, X_test, y_test
      majority_label, minority_label
    """
    if sensor_types is None:
        # 与原默认保持一致
        sensor_types = ['accelerometer']

    # all_df = read_sensor_data(pkl_path)
    all_df = read_sensor_data_from_memory(pkl_objects)

    selected_df = all_df[all_df.label_id > -1]
    fill_selected_df = selected_df.ffill().bfill()

    # 组各传感器
    sensor_list = []
    for sensor in sensor_types:
        if sensor == 'accelerometer':
            selected_np = fill_selected_df[sensor_dict['accelerometer']].values
            tmp_acc_stand, mean_val, std_val = gaussian_std(selected_np)
            sensor_list.append(tmp_acc_stand)
        elif sensor == 'acceleration':
            selected_np = fill_selected_df[sensor_dict['acceleration']].values
            tmp_acc_stand, mean_val, std_val = gaussian_std(selected_np)
            sensor_list.append(tmp_acc_stand)
        elif sensor == 'gyroscope':
            selected_np = fill_selected_df[sensor_dict['gyroscope']].values
            tmp_gyro_stand, mean_val, std_val = gaussian_std(selected_np)
            sensor_list.append(tmp_gyro_stand)
        elif sensor == 'magnetometer':
            selected_np = fill_selected_df[sensor_dict['magnetometer']].values
            tmp_mag_stand, mean_val, std_val = gaussian_std(selected_np)
            sensor_list.append(tmp_mag_stand)
        elif sensor == 'magnitude':
            selected_np = fill_selected_df[sensor_dict['magnitude']].values
            tmp_mag_stand, mean_val, std_val = gaussian_std(selected_np)
            sensor_list.append(tmp_mag_stand.reshape(-1,1))
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


    # 拼接标签
    label_list_np = fill_selected_df['label_id'].values.reshape(-1, 1)
    sensor_list.append(label_list_np)
    allselected_np = np.concatenate(sensor_list, axis=1)

    # 滑窗
    tmp_b = sliding_window(allselected_np, len_sw, len_sw)

    # 记录窗口到原始行的映射
    window_spans = build_window_spans(len(allselected_np), len_sw, len_sw)  # shape (B,2)

    data_b = np.transpose(tmp_b[:, :, :-1], (0, 2, 1))  # [B, C, T]
    label_b = tmp_b[:, :, -1]                            # [B, T]

    majority_label, minority_label = find_majority_minority(label_b)

    # 8:2 切分；labeled_ratio 用于从训练集中再切 1% 做有标签
    # vote_label = majority_value(label_b)
    # X_train_full, X_test, y_train_full, y_test = train_test_split(
    #     data_b, label_b, test_size=test_size, stratify=vote_label
    # )
    # X_labeled, X_unlabeled, y_labeled, y_unlabeled = train_test_split(
    #     X_train_full, y_train_full, test_size=(1.0 - labeled_ratio)
    # )

    # 用“索引”来分割，保证我们能把窗口索引一路带下去
    win_idx_all = np.arange(data_b.shape[0])
    vote_label = majority_value(label_b)

    idx_train_full, idx_test = train_test_split(
        win_idx_all, test_size=0.2, stratify=vote_label
    )
    X_train_full, X_test = data_b[idx_train_full], data_b[idx_test]
    y_train_full, y_test = label_b[idx_train_full], label_b[idx_test]
    spans_train_full, spans_test = window_spans[idx_train_full], window_spans[idx_test]

    # 从 train_full 中再划分 labeled/unlabeled，同步带上索引与 spans
    idx_labeled_rel, idx_unlabeled_rel = train_test_split(
        np.arange(len(idx_train_full)),
        test_size=0.99,
    )
    idx_labeled = idx_train_full[idx_labeled_rel]
    idx_unlabeled = idx_train_full[idx_unlabeled_rel]

    X_labeled, X_unlabeled = X_train_full[idx_labeled_rel], X_train_full[idx_unlabeled_rel]
    y_labeled, y_unlabeled = y_train_full[idx_labeled_rel], y_train_full[idx_unlabeled_rel]
    spans_labeled, spans_unlabeled = spans_train_full[idx_labeled_rel], spans_train_full[idx_unlabeled_rel]


    return {
        'data_b': data_b, 'label_b': label_b,
        'X_labeled': X_labeled, 'y_labeled': y_labeled,
        'X_unlabeled': X_unlabeled, 'y_unlabeled': y_unlabeled,
        'X_test': X_test, 'y_test': y_test,
        'majority_label': majority_label, 'minority_label': minority_label,
        'sensor_types': sensor_types,
        'window_spans': window_spans,
        'idx_labeled': idx_labeled,
        'idx_unlabeled': idx_unlabeled
    }

def build_model_and_optim(sensor_types: list[str], nclass: int, device: str):
    """
    构建模型、损失与优化器，供外部类复用。
    """
    model = FlexibleNN(sensor_types=sensor_types, number_classes=nclass).to(device)
    classify_criterion = nn.CrossEntropyLoss()
    supContrast_criterion = SupContrastiveLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
    return model, classify_criterion, supContrast_criterion, optimizer

class DeepViewPipeline:
    """
    对外统一可复用 API。别的类只需持有本类实例并调用方法。
    """
    # def __init__(self, device: str | None = None):
    def __init__(self, device: Optional[str] = None):  # Python 3.9 不支持 str | None
        self.device = device or ('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.data = None
        self.model = None
        self.classify_criterion = None
        self.supContrast_criterion = None
        self.optimizer = None

    def load_and_split(self, pkl_objects=None, sensor_dict=None, sensor_types=None, len_sw=50, test_size=0.2, labeled_ratio=0.01):
        self.data = prepare_dataset(
            pkl_objects=pkl_objects,
            sensor_dict=sensor_dict,
            sensor_types=sensor_types,
            len_sw=len_sw,
            test_size=test_size,
            labeled_ratio=labeled_ratio
        )
        return self.data

    def build_model(self, nclass):
        assert self.data is not None, "请先调用 load_and_split()"
        self.model, self.classify_criterion, self.supContrast_criterion, self.optimizer = \
            build_model_and_optim(self.data['sensor_types'], nclass, self.device)
        return self.model

    def train(self, contrast_rounds=10, contrast_epoch_each=50, supervised_epochs=50, batch_size=512):
        '''
        args:
            contrast_rounds: 对比学习的轮数
            contrast_epoch_each: 每轮对比学习的 epoch 数
            supervised_epochs: 监督学习的 epoch 数
            batch_size: 批量大小
        '''
        assert self.model is not None and self.data is not None, "请先调用 build_model() 与 load_and_split()"
        labeled_loader = create_loader(
            self.data['X_labeled'], self.data['y_labeled'],
            batch_size=batch_size, shuffle=True, device=self.device
        )
        # 对比学习阶段
        unfreeze_encoders(self.model)
        for _ in range(contrast_rounds):
            self.model, _ = train_model(self.model, labeled_loader, self.supContrast_criterion, self.optimizer,
                                        epochs=contrast_epoch_each, device=self.device, if_contrast=True)
        # 监督阶段（仅分类器）
        freeze_encoders(self.model)
        self.model, _ = train_model(self.model, labeled_loader, self.classify_criterion, self.optimizer,
                                    epochs=supervised_epochs, device=self.device, if_contrast=False)
        # 联合微调
        unfreeze_all(self.model)
        self.model, _ = train_model(self.model, labeled_loader, self.classify_criterion, self.optimizer,
                                    epochs=supervised_epochs, device=self.device, if_contrast=False)
        return self.model

    def create_latent_html(self, dataset_name='turtle', batch_size=512, name_label='_pipeline', label_colors=None):
        assert self.model is not None and self.data is not None, "请先调用 build_model() 与 load_and_split()"
        plot_loader = create_loader(
            self.data['data_b'], self.data['label_b'],
            batch_size=batch_size, shuffle=False, device=self.device
        )
        data_umap, pw_scatter = create_latent(
            plot_loader, self.model, dataset=dataset_name,
            device=self.device, name_label=name_label,
            sensor_types=self.data['sensor_types'], label_colors=label_colors
        )
        return data_umap, pw_scatter

    def getSamplingLabel(self, method='entropy', update_size=100):
        '''
        根据指定的采样方法，从未标注数据中选择样本进行标注。推荐新label
        '''
        assert self.model is not None and self.data is not None, "请先调用 build_model() 与 load_and_split()"
        X_labeled, y_labeled, X_unlabeled, y_unlabeled, selected_data = data_sampling(
            method,
            self.data['X_labeled'], self.data['y_labeled'],
            self.data['X_unlabeled'], self.data['y_unlabeled'],
            self.model, update_size,
            majority_label=self.data['majority_label'],
            minority_label=self.data['minority_label'],
            device=self.device
        )
        # 覆盖回内部状态，便于后续继续训练/继续采样
        # self.data['X_labeled'], self.data['y_labeled'] = X_labeled, y_labeled
        # self.data['X_unlabeled'], self.data['y_unlabeled'] = X_unlabeled, y_unlabeled

        return selected_data











# 特征 = 输入数据（如传感器数值）
# 标签 = 输出类别（如行为类型）
################################## 主动学习采样策略 ##########################



















################################## 主程序入口和完整运行流程 ##########################




if __name__ == "__main__":
    import sys
    from PySide6.QtWidgets import QApplication, QToolTip
    from PySide6 import QtCore
    import pyqtgraph as pg
    class Mainlabel():
        def __init__(self, root, cfg) -> None:
            self.label_propagated_data = None  # 保存传播后的标签
            self.data = None  # 原始数据 DataFrame
            self.label_dict = label_dict  # 标签字典
            self.label_colors = label_colors  # 标签颜色字典
            self.pipeline = DeepViewPipeline()
            self.label_propagated_data = None  # 传播后的标签

        # 构建第二个图Label Propagation的markData TODO: 优化性能
        def build_second_mark_data(self, label_propagated_data_df):
            """
            将传播后的标签(np.ndarray 或 list)转换为 ECharts markArea 数据并下发到前端。
            要求:
            - len(labels) == len(self.data)
            - 标签 > 0 视为有效；<=0 或 NaN 跳过不绘制
            """
            label_propagated_data = label_propagated_data_df['propagated_label']
            
            # 保存原始数据，避免多次调用时被覆盖
            self.label_propagated_data = label_propagated_data

            try:
                if self.data is None or len(self.data) == 0:
                    print("update_propagated_labels: 后端没有可用的数据 DataFrame。")
                    return

                arr = np.asarray(label_propagated_data)
                if arr.ndim != 1:
                    print(f"update_propagated_labels: 期望一维数组，收到 ndim={arr.ndim}")
                    return
                if len(arr) != len(self.data):
                    print(f"update_propagated_labels: 长度不一致, labels={len(arr)} vs data={len(self.data)}")
                    return
                
                # color_for: label_colors 的 key 为 label 名，lbl 为对应的 id（int），0-based index
                def color_for(lbl: int) -> str:
                    try:
                        vals = list(self.label_colors.values())
                        if not vals:
                            return "rgba(0,0,0,0.28)"
                        idx = int(lbl)
                        if idx < 0:
                            return "rgba(0,0,0,0.28)"
                        # 使用 0-based 索引；越界时循环取模
                        return vals[idx % len(vals)]
                    except Exception:
                        return "rgba(0,0,0,0.28)"

                label_str_list = list(self.label_dict.keys())

                markData = []
                curr_label = None
                start_i = None


                def flush_segment(end_i, lbl):
                    if start_i is None or end_i is None:
                        return
                    # 位置索引用 iloc，避免非 RangeIndex 时出错
                    start_ts = self.data.iloc[start_i]["timestamp"]
                    end_ts = self.data.iloc[end_i]["timestamp"]
                    markData.append([
                        {
                            "name": f"{label_str_list[lbl] if 0 < lbl <= len(label_str_list) else lbl}",
                            "labelId": int(lbl),
                            "xAxis": start_ts,
                            "itemStyle": {"color": color_for(int(lbl))}
                        },
                        {"xAxis": end_ts}
                    ])

                for i, lbl in enumerate(arr):
                    # 仅正整数作为有效标签
                    valid = False
                    if lbl is not None and not (isinstance(lbl, float) and np.isnan(lbl)):
                        try:
                            valid = int(lbl) > 0
                        except Exception:
                            valid = False

                    if not valid:
                        if curr_label is not None:
                            flush_segment(i - 1, curr_label)
                            curr_label, start_i = None, None
                        continue

                    lbl = int(lbl)
                    if curr_label is None:
                        curr_label, start_i = lbl, i
                    elif lbl != curr_label:
                        flush_segment(i - 1, curr_label)
                        curr_label, start_i = lbl, i

                if curr_label is not None:
                    flush_segment(len(arr) - 1, curr_label)
            except Exception as e:
                print(f"update_propagated_labels: 发生错误: {e}")

            return markData




        def get_sampling_label(self, method, update_size):
            method = self.sampling_method_combobox.currentText()
            text_val = self.recommended_count_input.text().strip()
            try:
                update_size = int(text_val) if text_val else 10
            except ValueError:
                update_size = 10

            X_labeled, y_labeled, X_unlabeled, y_unlabeled, selected_data = self.pipeline.getSamplingLabel(
                sampling_method=method,
                select_size=update_size,
                majority_label=None,
                minority_label=None
            )

            
        # 构建第三个图推荐新标签Sampling的markData
        def build_third_mark_data(self, selected_data_df):
            """
            将推荐的新标签索引列表转换为 ECharts markArea 数据并下发到前端。
            要求:
            - 使用 selected_data_df['selected_label'] 非空的行作为“被推荐”的区间
            - 区间 name = label_str_list[int(label_id_多数票)]
            """
            if self.data is None or len(self.data) == 0:
                print("build_third_mark_data: 后端没有可用的数据 DataFrame。")
                return []

            if 'selected_label' not in selected_data_df.columns:
                print("build_third_mark_data: 传入的 DataFrame 缺少列 'selected_label'。")
                return []

            label_str_list = list(self.label_dict.keys())
            sel_col = selected_data_df['selected_label']

            # 取出“被推荐”的行索引（selected_label 非空）
            sel_idx = sel_col[sel_col.notna()].index.tolist()
            if not sel_idx:
                print("build_third_mark_data: 没有任何被推荐的行。")
                return []

            sel_idx.sort()

            markData = []
            seg_start = sel_idx[0]
            prev = sel_idx[0]

            def flush_segment(s, e):
                if s is None or e is None:
                    return
                # 位置索引用 iloc，避免非 RangeIndex 时出错
                start_ts = self.data.iloc[s]["timestamp"]
                end_ts = self.data.iloc[e]["timestamp"]

                # 该区间内的标签多数票
                seg_labels = sel_col.loc[s:e].dropna().astype(float).astype(int).values
                if len(seg_labels) > 0:
                    vals, counts = np.unique(seg_labels, return_counts=True)
                    label_id = int(vals[np.argmax(counts)])
                    if 0 <= label_id < len(label_str_list):
                        name_str = label_str_list[label_id]
                    else:
                        name_str = str(label_id)
                else:
                    name_str = "Selected"

                markData.append([
                    {
                        "name": name_str,
                        "xAxis": start_ts,
                        "labelId": int(label_id) if 'label_id' in locals() else -1,
                        "itemStyle": {"color": "rgba(255, 165, 0, 0.6)"}
                    },
                    {"xAxis": end_ts}
                ])

            # 按连续行索引拼区间
            for i in sel_idx[1:]:
                if i == prev + 1:
                    prev = i
                else:
                    flush_segment(seg_start, prev)
                    seg_start = i
                    prev = i
            flush_segment(seg_start, prev)

            return markData

    print("=== 动物行为分析深度学习系统 ===")
    print("请确保数据文件路径正确，并根据需要调整超参数")
    print("系统将依次执行：数据预处理 → 模型训练 → 可视化 → 主动学习")
    print("=====================================")
    colorPalette = ['#91cc75', '#5470c6', '#fac858', '#ee6666',
                            '#73c0de', '#3ba272', '#fc8452', '#9a60b4',
                            '#ea7ccc', '#fff018', '#6800ff', '#4bb0ff',
                            '#1bff00', '#09ffdb']
    
    
    label_colors = {}
    for i, label in enumerate(label_dict.keys()):
        # 使用取余运算符来循环使用颜色
        color_index = i % len(colorPalette)
        label_colors[label] = colorPalette[color_index]

    print("标签颜色映射：", label_colors)

    label_list_str = list(label_dict.keys())
    print("标签列表：", label_list_str)

    # 训练配置参数
    batch_size = 512
    sampling_method = 'entropy'  # 采样方法：entropy, random, least_confidence等
    SupCount = 1  # 监督对比学习轮数
    warmup = 20  # 对比学习预热轮数
    contrast_epoch_each = 50  # 对比学习每轮的训练次数
    supervised_epochs = 50  # 监督训练次数
    contrast_rounds = 10  # 对比学习轮数
    sensor_dict = {
            # 'accelerometer': ['AccX', 'AccY', 'AccZ'],
            # 'gyroscope': ['GyrX', 'GyrY', 'GyrZ'],
            'accelerometer': ['acc_x', 'acc_y', 'acc_z'],
            'acceleration': ['acc_x', 'acc_y', 'acc_z'],
            'gyroscope': ['gyro_x', 'gyro_y', 'gyro_z'],
            'magnetometer': ['mag_x', 'mag_y', 'mag_z'],
            'temperature': ['temperature'],
            'pressure': ['pressure'],
            'GPS': ['GPS']
        }

    name_label = '_%s_Contrast%s_warm%s' % (
        sampling_method, str(SupCount), str(warmup))
    


    pipeline = DeepViewPipeline()
    dataset_name = 'turtle'  # 统一使用变量，便于颜色逻辑共享

    # pkl_path = r'./test/turtle.pkl'
    pkl_path = r"C:\Users\user\Desktop\fast-ttt-2024-10-11\unsupervised-datasets\allDataSet\Omizunagidori2018_raw_data_9B36360_lb0005_25Hz.pkl"
    # 读取pkl文件
    df_list = []
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"Required data file not found: {pkl_path}. Please ensure the file exists in the working directory.")
    
    # 读取pickle文件中的所有数据片段
    with open(pkl_path, 'rb') as f:
        while True:
            try:
                item = pickle.load(f)
                df_list.append(item)
            except EOFError:
                break
    
    # 合并所有数据片段
    df_all = pd.concat(df_list, ignore_index=True)
    
    # 1) 数据加载与切分
    data_state = pipeline.load_and_split(
        pkl_objects=df_all,
        sensor_types=['acceleration','gyroscope'],
        # sensor_types=['accelerometer','gyroscope'],
        sensor_dict=sensor_dict,
        len_sw=50, test_size=0.2, labeled_ratio=0.01
    )
    nclass = int(max(list(label_dict.values())) + 1)
    # 2) 构建模型并训练
    pipeline.build_model(nclass=nclass)
    pipeline.train(contrast_rounds=10, contrast_epoch_each=50, supervised_epochs=50, batch_size=batch_size)
    # 3) 潜在空间与可视化 (HTML)
    data_umap, pw_scatter = pipeline.create_latent_html(dataset_name=dataset_name, name_label='_entropy_Contrast1_warm20', label_colors=label_colors)

    # # 取窗口众数标签


    app = QApplication.instance() or QApplication(sys.argv)
    pw = pg.PlotWidget(title="UMAP散点图 (PyQtGraph)")
    pw.setWindowTitle("UMAP散点图 (PyQtGraph)")
    pw.setLabel('bottom', 'UMAP 1')
    pw.setLabel('left', 'UMAP 2')
    pw.show()



    pw.addItem(pw_scatter)

    

    # 4) 推荐新label
    selected_data = pipeline.getSamplingLabel(method='entropy', update_size=100)
    selected_samples, selected_labels_win, selected_indices_rel = selected_data
    # 相对未标注集合 -> 绝对“全量窗口”索引
    selected_win_idx_abs = pipeline.data['idx_unlabeled'][selected_indices_rel]
    # 用窗口多数票作为行级标签写回（可按需修改为别的定义）
    selected_win_labels = majority_value(selected_labels_win)
    selected_spans_abs = pipeline.data['window_spans'][selected_win_idx_abs]
    all_df = apply_window_labels_to_all_df(df_all, selected_spans_abs, selected_win_labels, colname='selected_label')

    mainlabel = Mainlabel(None, None)
    mainlabel.data = df_all  
    mainlabel.label_dict = label_dict
    mainlabel.label_colors = label_colors
    mainlabel.pipeline = pipeline

    markData_sampling = mainlabel.build_third_mark_data(all_df)  # selected_data 是一个元组 (X_selected, y_selected)
    



    propagated_labels = generate_propagated_labels(data_umap, pipeline.data['label_b'])
    # propagated_labels = generate_propagated_labels(data_umap, label_b)
    # 将窗口级 propagated_labels 写回 all_df（逐行）
    all_df = apply_window_labels_to_all_df(df_all, pipeline.data['window_spans'], propagated_labels, colname='propagated_label')

    print("标签传播结果示例：", propagated_labels[:10])
    markData_propagated = mainlabel.build_second_mark_data(all_df)


    print(f"propagated_label 覆盖窗口数: {len(propagated_labels)}, 写回行数: {all_df['propagated_label'].notna().sum()}")
    print(f"selected_label 覆盖窗口数: {len(selected_win_idx_abs)}, 写回行数: {all_df['selected_label'].notna().sum()}")



    app.exec()

    



