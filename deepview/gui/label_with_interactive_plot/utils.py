
import os
import pickle
import torch
import pandas as pd
import numpy as np
from deepview.utils.auxiliaryfunctions import (
    get_unsupervised_set_folder,
    get_unsup_model_folder,
    read_config,
)
# 从sklearn.manifold导入TSNE
from sklearn.manifold import TSNE

# 获取潜在特征
# get latent feature
from deepview.clustering_pytorch.nnet.common_config import get_model
from deepview.clustering_pytorch.datasets.factory import prepare_unsup_dataset
from deepview.clustering_pytorch.nnet.train_utils import AE_eval_time_series, simclr_eval_time_series

# 从.pkl文件获取数据的方法
def get_data_from_pkl(filename, cfg, dataChanged=[]):
    # 获取无监督数据集文件夹路径
    unsup_data_path = get_unsupervised_set_folder()
    # 构建文件路径
    datapath = os.path.join(cfg["project_path"], unsup_data_path, filename)
    # 打开.pkl文件
    with open(datapath, 'rb') as f:
        # 加载数据
        data = pickle.load(f)

        # 将UNIX时间戳转换为ISO 8601格式
        # data['timestamp'] = pd.to_datetime(data['unixtime'],
        #                                    unit='s').dt.strftime('%Y-%m-%dT%H:%M:%S.%fZ').str[:-4] + 'Z'
        # data['index'] = data.index  # Add an index column
    # todo 不应该放在这里
    if dataChanged != []:
        dataChanged.emit(data)
    return data, dataChanged



def featureExtraction(root, data, data_length, column_names, model_path, model_name):
    # 特征提取：找到数据帧中的列名
    # preprocessing: find column names in dataframe
    # new_column_names = find_data_columns(sensor_dict, column_names)

    segments, start_indices = split_dataframe(data, data_length)
    end_indices = [i + data_length for i in start_indices]


    train_loader = prepare_unsup_dataset(segments, column_names)

    config = root.config
    cfg = read_config(config)
    unsup_model_path = get_unsup_model_folder(cfg)
    full_model_path = os.path.join(cfg["project_path"], unsup_model_path, model_path)

    if 'AE_CNN' in model_path.upper():
        p_setup = 'autoencoder'
    elif 'simclr' in model_path.upper():
        p_setup = 'simclr'
    else:
        raise ValueError("Invalid model type")

    model = get_model(p_backbone=model_name, p_setup=p_setup, num_channel=len(column_names))

    if torch.cuda.is_available():
        model.load_state_dict(torch.load(full_model_path, weights_only=False))
    else:
        model.load_state_dict(torch.load(full_model_path, weights_only=False, map_location=torch.device('cpu')))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    if p_setup == 'autoencoder':
        representation_list, _, _, _, _ = \
            AE_eval_time_series(train_loader, model, device)
    elif p_setup == 'simclr':
        representation_list, _ = simclr_eval_time_series(train_loader, model, device)

    # tsne latent representation to shape=(2, len) PCA降维到形状为 (2, len)
    repre_concat = np.concatenate(representation_list)
    repre_reshape = repre_concat.reshape(repre_concat.shape[0], -1)
    repre_tsne = reduce_dimension_with_tsne(repre_reshape)

    return start_indices, end_indices, repre_tsne


def find_data_columns(sensor_dict, column_names):
    new_column_names = []
    for column_name in column_names:
        real_names = sensor_dict[column_name]  # 获取每个列名对应的实际列名
        new_column_names.extend(real_names)  # 将实际列名添加到新的列名列表中
    return new_column_names


def split_dataframe(df, segment_size):
    '''
    将DataFrame按segment_size分割成多个片段
    split the dataframe into segments based on segment_size
    '''
    segments = []
    start_indices = []
    num_segments = len(df) // segment_size

    for i in range(num_segments):
        start_index = i * segment_size
        end_index = start_index + segment_size
        segment = df.iloc[start_index:end_index]
        segments.append(segment)
        start_indices.append(start_index)

    # 处理最后一个片段，如果其大小不足一个segment_size
    # Handle the last segment which may not have the full segment size
    start_index = num_segments * segment_size
    last_segment = df.iloc[start_index:]
    if len(last_segment) == segment_size:
        segments.append(last_segment)
        start_indices.append(start_index)

    return segments, start_indices


def reduce_dimension_with_tsne(array, method='tsne'):
    # tsne or pca
    tsne = TSNE(n_components=2)  # 创建TSNE对象，降维到2维
    reduced_array = tsne.fit_transform(array)  # 对数组进行降维
    return reduced_array
