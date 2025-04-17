import pickle

import numpy as np
import random

import pandas as pd
import torch
from deepview.calculate_results.data.umineko.umineko_data import (
    label_dict,
)
from deepview.calculate_results.models.utils import (
    sliding_window,
)
from deepview.calculate_results.models.utils import (
    gaussian_std,
    read_sensor_data,
    process_sensor_np,
    set_random_seed,
)

# def set_random_seed(seed):
#     # Set seed for Python's random module
#     random.seed(seed)
#
#     # Set seed for NumPy
#     np.random.seed(seed)
#
#     # Set seed for PyTorch
#     torch.manual_seed(seed)
#
#     # If using CUDA, set seed for GPU as well
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed(seed)
#         torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
#

# Set a fixed random seed
seed_value = 2025
set_random_seed(seed_value)

labeldict_findstr = {-2: 'unknown',
                     0: 'ground_stationary',
                     1: 'stationary',
                     2: 'bathing',
                     3: 'flying_active',
                     4: 'flying_passive',
                     5: 'foraging'}

label_colors = {
    0: 'red',       # 红色
    1: 'blue',      # 蓝色
    2: 'green',     # 绿色
    3: 'orange',    # 橙色
    4: 'purple',    # 紫色
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


# def read_sensor_data():
#     raw_data, labeled_data = [], []
#     for year in ['2018', '2022']:
#         dp = r'D:\code\DeepView\deepview\calculate_results\data\umineko_%s.npy'
#         data = np.load(dp % year, allow_pickle=True).item()
#         # Access the individual components
#         raw_ = data['raw_data']
#         labeled_ = data['labeled_data']
#         if year == '2018':
#             df_raw_2018 = raw_
#             df_2018 = labeled_
#         # elif year=='2019':
#         #     df_raw_2019 = raw_
#         #     df_2019 = labeled_
#         elif year == '2022':
#             df_raw_2022 = raw_
#             df_2022 = labeled_
#         else:
#             print('Error: year not found')
#             break
#
#     selected_df = pd.concat([df_raw_2018, df_raw_2022], ignore_index=True)
#     selected_df['filename'] = selected_df['year'].astype(int).astype(str) + '_' + selected_df['animal_tag']
#     animal_tag_list = ['2018_LB07', '2018_LB08', '2018_LB09', '2018_LB10', '2018_LB11', '2018_LB12', '2018_LB13',
#                        '2022_LB02', '2022_LB03', '2022_LB08', '2022_LB09']
#     selected_df = selected_df[selected_df['filename'].isin(animal_tag_list)]
#
#     selected_df['label_id'] = selected_df['label'].map(label_dict)
#     selected_df['label_id'] = selected_df['label_id'].fillna(-2)
#
#     # 删除无标签的数据
#     selected_df = selected_df[selected_df.label_id != -2]
#     return selected_df
#
#
# def gaussian_std(X):
#     mean_val = np.mean(X.astype(float), axis=0)
#     std_val = np.std(X.astype(float), axis=0)
#     X_standardized = (X - mean_val) / np.maximum(std_val, 10 ** -5)
#     return X_standardized
#
#
# def up_sample_data():
#     selected_df = read_sensor_data()
#     # 选择需要的列
#     selected_df = selected_df[['acc_x', 'acc_y', 'acc_z', 'pressure', 'label_id']]
#     fill_selected_df = selected_df.fillna(method='ffill').fillna(method='bfill')  # pressure has nan values
#     selected_np = fill_selected_df.values
#
#     # 分别预处理两种传感器数据
#     acc_np = fill_selected_df[['acc_x', 'acc_y', 'acc_z']].values
#     tmp_acc_stand = gaussian_std(acc_np)
#
#     fill_selected_df['pressure'] = fill_selected_df['pressure'] - 1013.25  # standard pressure
#     data = fill_selected_df['pressure'].values
#
#     # 计算均值和标准差
#     mean = np.mean(data)
#     std_dev = np.std(data)
#     # 识别超过3个标准差的异常值
#     outlier_mask = np.abs(data - mean) > 1 * std_dev
#     # 创建数据副本
#     data_replaced = np.copy(data)
#
#     # 用周围最近的正常值替换异常值
#     for i in range(len(data)):
#         if outlier_mask[i]:
#             # 查找前一个正常值
#             j = i - 1
#             # 查找后一个正常值
#             k = i + 1
#
#             # 找到前一个正常值
#             while j >= 0 and outlier_mask[j]:
#                 j -= 1
#             # 找到后一个正常值
#             while k < len(data) and outlier_mask[k]:
#                 k += 1
#
#             # 用最近的正常值替换异常值
#             if j >= 0 and (k >= len(data) or i - j <= k - i):
#                 data_replaced[i] = data[j]
#             elif k < len(data):
#                 data_replaced[i] = data[k]
#     tmp_press_stand = gaussian_std(data_replaced)
#
#     selected_np[:, :3] = tmp_acc_stand
#     selected_np[:, -2] = tmp_press_stand
#     return selected_np

def generate_batch_data(selected_np, len_sw=50):
    tmp_b = sliding_window(selected_np, len_sw, len_sw)
    # concatenate list
    data_b = np.transpose(tmp_b[:, :, :-1], (0, 2, 1))  # [B, Len, dim-1] -> [B, dim-1, Len]
    label_b = tmp_b[:, :, -1]  # [B, Len]
    return data_b, label_b

all_df = read_sensor_data()
# 删除无标签的数据
selected_df = all_df[all_df.label_id != -2]
columns = ['acc_x', 'acc_y', 'acc_z', 'pressure', 'label_id']
selected_np = process_sensor_np(selected_df, columns)

len_sw = 50
data_b, label_b = generate_batch_data(selected_np, len_sw=len_sw)

# 将数据保存到文件
with open('data.pkl', 'wb') as file:  # 'wb'表示以二进制写入模式打开文件
    pickle.dump({'data': data_b, 'label': label_b}, file)
# with open('data.pkl', 'rb') as file:  # 'rb'表示以二进制读取模式打开文件
#     d_dict = pickle.load(file)
#     data_b = d_dict['data']
#     label_b = d_dict['label']