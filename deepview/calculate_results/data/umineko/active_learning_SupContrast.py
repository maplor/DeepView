import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
# import numpy as np
import random
import torch
import pickle
from tqdm import tqdm
from torch import optim
# import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from umap import UMAP
import plotly.express as px

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# import cupy as cp
# from cuml.cluster import DBSCAN as cuDBSCAN


from deepview.calculate_results.data.umineko.umineko_data import (
    read_umineko_path,
    extract_data_from_year_back,
    label_dict,
)

from deepview.calculate_results.models.utils import (
    sliding_window,
    data_loader_umineko,
    MSEloss_weighted,
    MSEloss,
    NTXentloss,
    # torch,
    AE_eval_time_series,
    AE_train_time_series_resnet,
    # np,
    # tqdm
    Autoencoder3d4,  # debug purpose
    Autoencoder4d,
    CrossModelAutoencoderContrastiveModel,
    Autoencoder1d,
    plot_reconstruction_result,
    majority_value,
)


def set_random_seed(seed):
    # Set seed for Python's random module
    random.seed(seed)

    # Set seed for NumPy
    np.random.seed(seed)

    # Set seed for PyTorch
    torch.manual_seed(seed)

    # If using CUDA, set seed for GPU as well
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU setups


# Set a fixed random seed
seed_value = 2025
set_random_seed(seed_value)


def gaussian_std(X):
    mean_val = np.mean(X.astype(float), axis=0)
    std_val = np.std(X.astype(float), axis=0)
    X_standardized = (X - mean_val) / np.maximum(std_val, 10 ** -5)
    return X_standardized


def dbscan_plot(data):
    # 数据标准化
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    # 应用DBSCAN
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    predicted_labels = dbscan.fit_predict(data_scaled)

    # 评估聚类效果
    # 注意：轮廓系数在标签数量少于2的情况下不可用
    if len(set(predicted_labels)) > 1:
        silhouette_avg = silhouette_score(data_scaled, predicted_labels)
        print(f"Silhouette Score: {silhouette_avg}")
    else:
        print("Silhouette Score无法计算，因为聚类结果只有一个簇或全部为噪声。")

    # 可视化聚类结果
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 绘制聚类结果
    unique_labels = set(predicted_labels)
    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))

    for label, col in zip(unique_labels, colors):
        if label == -1:
            # 黑色用于噪声点
            col = 'k'

        class_member_mask = (predicted_labels == label)

        xyz = data[class_member_mask]
        ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=[col], label=f'Cluster {label}')

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    ax.legend()
    plt.show()
    return


def plot_func(representation_list, sample_list, label_list, pred_list, epoch, name):
    label_concat = np.concatenate(label_list)
    label_concat_vote = majority_value(label_concat)
    repre_concat = np.concatenate(representation_list)
    repre_reshape = repre_concat.reshape(repre_concat.shape[0], -1).astype(float)

    if len(pred_list) != 0:  # clip model without reconstruct data
        # Step 1: Concatenate each element along the batch dimension
        concatenated_segments = [np.concatenate(segment.transpose(0, 2, 1), axis=0) for segment in sample_list]
        sample_reshape = np.concatenate(concatenated_segments, axis=0)

        concatenated_segments = [np.concatenate(segment.transpose(0, 2, 1), axis=0) for segment in pred_list]
        pred_reshape = np.concatenate(concatenated_segments, axis=0)

        start = 0
        end = -1
        axis_dict = {0: 'accx', 1: 'accy', 2: 'accz', 3: 'pressure'}
        sensordim = 3
        # fig, axes = plt.subplots(4 * 3, 1, figsize=(12, 24))
        # for col in range(4):
        #     axes[0 + sensordim * col].plot(sample_reshape[start:end, col], 'r', label='groundtruth%s' % axis_dict[col])
        #     axes[0 + sensordim * col].plot(pred_reshape[start:end, col], 'b-', label='predict%s' % axis_dict[col])
        #     axes[0 + sensordim * col].set_title('Autoencoder_Reconstruct_Umineko_%s' % (axis_dict[col]))
        #     axes[0 + sensordim * col].set_xlabel('timestamp')
        #     axes[0 + sensordim * col].set_ylabel('value')
        #     axes[0 + sensordim * col].legend(loc="right")
        #
        #     axes[1 + sensordim * col].plot(sample_reshape[start:end, col], 'r', label='groundtruth%s' % axis_dict[col])
        #     axes[1 + sensordim * col].set_xlabel('timestamp')
        #     axes[1 + sensordim * col].set_ylabel('value')
        #     axes[1 + sensordim * col].legend(loc="right")
        #
        #     axes[2 + sensordim * col].plot(pred_reshape[start:end, col], 'b-', label='predict%s' % axis_dict[col])
        #     axes[2 + sensordim * col].set_xlabel('timestamp')
        #     axes[2 + sensordim * col].set_ylabel('value')
        #     axes[2 + sensordim * col].legend(loc="right")
        #
        # # Adjust layout
        # plt.tight_layout()
        # plt.title('%s_reconstruct_epoch_%s.png' % (name, str(epoch)))
        # plt.savefig('%s_reconstruct_epoch_%s.png' % (name, str(epoch)))
        # plt.close('all')

    umap_3d = UMAP(n_components=2)
    proj_3d_gyro = umap_3d.fit_transform(repre_reshape)
    labeldict_findstr = {-2: 'unknown',
                         0: 'ground_stationary',
                         1: 'stationary',
                         2: 'bathing',
                         3: 'flying_active',
                         4: 'flying_passive',
                         5: 'foraging'}

    label_concat_vote_str = [labeldict_findstr[i] for i in label_concat_vote]

    fig_3d = px.scatter(
        proj_3d_gyro, x=0, y=1,
        color=label_concat_vote_str,
        labels={'color': 'activity'},
        color_discrete_map={'unknown': 'lightgrey'},
    )
    # Reduce marker size for all points
    for trace in fig_3d.data:
        trace.marker.size = 6  # Adjust the size to your preference (e.g., 6)

    # Update transparency for traces where activity is '-2.0'
    fig_3d.for_each_trace(lambda trace: trace.update(marker=dict(opacity=0.5)) if trace.name == '-2.0' else ())
    fig_3d.update_traces(marker=dict(line=dict(width=0)))  # remove boundary of point
    fig_3d.write_html('%s_reconstruct_epoch_%s.html' % (name, str(epoch)))

    # 数据标准化
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(proj_3d_gyro)

    # 应用DBSCAN
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    predicted_labels = dbscan.fit_predict(data_scaled)

    # 评估聚类效果
    # 注意：轮廓系数在标签数量少于2的情况下不可用
    if len(set(predicted_labels)) > 1:
        silhouette_avg = silhouette_score(data_scaled, predicted_labels)
        print(f"Silhouette Score: {silhouette_avg}")
    else:
        print("Silhouette Score无法计算，因为聚类结果只有一个簇或全部为噪声。")

    fig_3d_dbscan = px.scatter_3d(
        proj_3d_gyro, x=0, y=1,
        color=predicted_labels,
        labels={'color': 'activity'},
        color_discrete_map={'unknown': 'lightgrey'},
    )
    # Reduce marker size for all points
    for trace in fig_3d_dbscan.data:
        trace.marker.size = 6  # Adjust the size to your preference (e.g., 6)

    # Update transparency for traces where activity is '-2.0'
    fig_3d_dbscan.for_each_trace(lambda trace: trace.update(marker=dict(opacity=0.5)) if trace.name == '-2.0' else ())
    fig_3d_dbscan.update_traces(marker=dict(line=dict(width=0)))  # remove boundary of point
    fig_3d_dbscan.write_html('%s_reconstruct_epoch_%s_DBSCAN.html' % (name, str(epoch)))

    return


def read_sensor_data():
    raw_data, labeled_data = [], []
    for year in ['2018', '2022']:
        dp = r'D:\code\DeepView\deepview\calculate_results\data\umineko_%s.npy'
        data = np.load(dp % year, allow_pickle=True).item()
        # Access the individual components
        raw_ = data['raw_data']
        labeled_ = data['labeled_data']
        if year == '2018':
            df_raw_2018 = raw_
            df_2018 = labeled_
        # elif year=='2019':
        #     df_raw_2019 = raw_
        #     df_2019 = labeled_
        elif year == '2022':
            df_raw_2022 = raw_
            df_2022 = labeled_
        else:
            print('Error: year not found')
            break

    selected_df = pd.concat([df_raw_2018, df_raw_2022], ignore_index=True)
    selected_df['filename'] = selected_df['year'].astype(int).astype(str) + '_' + selected_df['animal_tag']
    animal_tag_list = ['2018_LB07', '2018_LB08', '2018_LB09', '2018_LB10', '2018_LB11', '2018_LB12', '2018_LB13',
                       '2022_LB02', '2022_LB03', '2022_LB08', '2022_LB09']
    selected_df = selected_df[selected_df['filename'].isin(animal_tag_list)]

    selected_df['label_id'] = selected_df['label'].map(label_dict)
    selected_df['label_id'] = selected_df['label_id'].fillna(-2)

    # 删除无标签的数据
    selected_df = selected_df[selected_df.label_id != -2]
    return selected_df


# selected_df = read_sensor_data()
# # 选择需要的列
# selected_df = selected_df[['acc_x', 'acc_y', 'acc_z', 'pressure', 'label_id']]
# fill_selected_df = selected_df.fillna(method='ffill').fillna(method='bfill')  # pressure has nan values
# selected_np = fill_selected_df.values
#
# # 分别预处理两种传感器数据
# acc_np = fill_selected_df[['acc_x', 'acc_y', 'acc_z']].values
# tmp_acc_stand = gaussian_std(acc_np)
#
# fill_selected_df['pressure'] = fill_selected_df['pressure'] - 1013.25  # standard pressure
# data = fill_selected_df['pressure'].values
#
# # 计算均值和标准差
# mean = np.mean(data)
# std_dev = np.std(data)
# # 识别超过3个标准差的异常值
# outlier_mask = np.abs(data - mean) > 1 * std_dev
# # 创建数据副本
# data_replaced = np.copy(data)
#
# # 用周围最近的正常值替换异常值
# for i in range(len(data)):
#     if outlier_mask[i]:
#         # 查找前一个正常值
#         j = i - 1
#         # 查找后一个正常值
#         k = i + 1
#
#         # 找到前一个正常值
#         while j >= 0 and outlier_mask[j]:
#             j -= 1
#         # 找到后一个正常值
#         while k < len(data) and outlier_mask[k]:
#             k += 1
#
#         # 用最近的正常值替换异常值
#         if j >= 0 and (k >= len(data) or i - j <= k - i):
#             data_replaced[i] = data[j]
#         elif k < len(data):
#             data_replaced[i] = data[k]
# tmp_press_stand = gaussian_std(data_replaced)
#
# selected_np[:, :3] = tmp_acc_stand
# selected_np[:, -2] = tmp_press_stand
#
len_sw = 50
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# tmp_b = sliding_window(selected_np, len_sw, len_sw)
# # concatenate list
# data_b = np.transpose(tmp_b[:, :, :-1], (0, 2, 1))  # [B, Len, dim-1] -> [B, dim-1, Len]
# label_b = tmp_b[:, :, -1]  # [B, Len]

# ======= 1. 数据生成 ======= #
# 将数据保存到文件
# with open('data.pkl', 'wb') as file:  # 'wb'表示以二进制写入模式打开文件
#     pickle.dump({'data': data_b, 'label': label_b}, file)
with open('data.pkl', 'rb') as file:  # 'rb'表示以二进制读取模式打开文件
    d_dict = pickle.load(file)
    data_b = d_dict['data']
    label_b = d_dict['label']
# 将数据分为8:2，其中2为测试集
X_train_full, X_test, y_train_full, y_test = train_test_split(data_b, label_b, test_size=0.2, random_state=42)

# 初始化标注数据集（1%）和未标注数据池（99%）
initial_size = int(0.01 * len(X_train_full))
X_labeled = X_train_full[:initial_size]
y_labeled = y_train_full[:initial_size]
X_unlabeled = X_train_full[initial_size:]
y_unlabeled = y_train_full[initial_size:]


# ======= 2. 模型与 Supervised Contrastive Learning ======= #
class SimpleNN(nn.Module):
    def __init__(self, input_dim=128 * 6, embedding_dim=128, number_classes=6):
        super(SimpleNN, self).__init__()
        self.acc_encoder = Autoencoder3d4().feature_extractor
        self.pre_encoder = Autoencoder1d().feature_extractor

        # self.projector = nn.Sequential(
        #     nn.Linear(input_dim * 2, 256),
        #     nn.ReLU(),
        #     nn.Linear(256, embedding_dim)
        # )
        self.linear = nn.Linear(input_dim*2, 32)
        self.projector = nn.Sequential(
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 16)
        )

        # self.classifier = nn.Sequential(
        #     nn.Linear(input_dim * 2, 256),
        #     nn.ReLU(),
        #     nn.Linear(256, number_classes),
        #     nn.Softmax(dim=1),
        # )
        self.classifier = nn.Sequential(
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, number_classes),
            nn.Softmax(dim=1),
        )

    def init_extractors(self):
        press_model_path = r'D:\code\DeepView\deepview\calculate_results\data\umineko\AE_reconstruct_epoch499_datalen50_pressure.pth'
        accel_model_path = r'D:\code\DeepView\deepview\calculate_results\data\umineko\AE_reconstruct_epoch500_datalen50_accel.pth'
        if torch.cuda.is_available():
            self.pre_encoder.load_state_dict(torch.load(press_model_path,
                                                        weights_only=False))
            self.acc_encoder.load_state_dict(torch.load(accel_model_path,
                                                        weights_only=False))
        else:
            self.pre_encoder.load_state_dict(torch.load(press_model_path,
                                                        weights_only=False,
                                                        map_location=torch.device('cpu')))
            self.acc_encoder.load_state_dict(torch.load(accel_model_path,
                                                        weights_only=False,
                                                        map_location=torch.device('cpu')))

    def forward(self, data, if_contrast=True):
        accel = data[:, :3, :]
        press = data[:, -1:, :]
        acc_fea, _, _ = self.acc_encoder(accel)
        pre_fea, _, _ = self.pre_encoder(press)
        concat_fea = torch.concat([acc_fea, pre_fea], dim=1)
        concat_fea = self.linear(concat_fea)
        if if_contrast:
            output = self.projector(concat_fea)  # output(batch, 128)
        else:
            output = self.classifier(concat_fea)
        return output, concat_fea


class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.3):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, latent, labels):
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

        # cluster center loss
        all_loss = combined_loss(latent, labels, loss, lambda_intra=1.0, lambda_inter=10)

        return all_loss


# ======= 调整不同类别latent space的距离 ======= #
def compute_clustered_distances(features, labels, eps=0.5, min_samples=5):
    """
    计算类内距离和类间距离，考虑同类别的多个簇情况。
    Args:
        features: Tensor, shape (N, D) - 样本特征
        labels: Tensor, shape (N,) - 样本标签
        eps: float - DBSCAN的半径参数
        min_samples: int - DBSCAN的最小点数参数
    Returns:
        intra_distances: 平均类内距离 (list of floats)
        inter_distances: 平均类间距离 (list of floats)
    """

    unique_labels = labels.unique()
    intra_distances = []
    inter_distances = []

    # 转换为 NumPy 格式以便使用 DBSCAN
    features_np = features.detach().cpu().numpy()
    labels_np = labels.detach().cpu().numpy()

    # 存储每个簇的中心点
    cluster_centers = []

    for label in unique_labels:
        # 获取当前类别的特征
        class_features = features_np[labels_np == label.item()]

        # 对当前类别进行聚类（DBSCAN）
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(class_features)
        cluster_labels = clustering.labels_

        # visualize_clusters(class_features, cluster_labels)
        # print('')

        # 计算类内距离
        for cluster_id in set(cluster_labels):
            if cluster_id == -1:  # 忽略噪声点
                continue
            cluster_points = class_features[cluster_labels == cluster_id]
            center = cluster_points.mean(axis=0)
            intra_dist = torch.norm(torch.tensor(cluster_points) - torch.tensor(center), dim=1).mean().item()
            intra_distances.append(intra_dist)
            cluster_centers.append((label.item(), center))  # 保存每个簇的中心点

    # 计算类间距离（簇中心之间的距离）
    for i, (label1, center1) in enumerate(cluster_centers):
        for j, (label2, center2) in enumerate(cluster_centers):
            if i >= j:  # 避免重复计算
                continue
            dist = torch.norm(torch.tensor(center1) - torch.tensor(center2)).item()
            if label1 == label2:
                intra_distances.append(dist)  # 同类簇之间的距离
            else:
                inter_distances.append(dist)  # 不同类别之间的距离

    return intra_distances, inter_distances


def compute_cluster_loss(data, labels, eps=1, min_samples=10):
    '''
    feature为tentor，使用sklearn获得cluster标签，但是使用feature计算，确保能grad
    return inter and intra losses
    '''
    cluster_labels = {}
    unique_labels = labels.unique().detach().cpu().numpy()
    for label in unique_labels:
        # 筛选出当前类型的数据
        type_data = data[labels == label]
        # 使用 DBSCAN 聚类
        numpy_data = type_data.detach().cpu().numpy()  # 转换为 numpy 计算聚类
        # dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        # clusters = dbscan.fit_predict(numpy_data)

        if numpy_data.shape[0] < 3:
            clusters = np.zeros((numpy_data.shape[0]))
        else:
            kmeans = KMeans(n_clusters=3)
            kmeans.fit(numpy_data)
            # centers = kmeans.cluster_centers_
            clusters = kmeans.labels_
        cluster_labels[label] = clusters

        # # plot
        # cluster_centers = []
        # for cc in set(clusters):
        #     cluster_centers.append((label, np.average(numpy_data[clusters == cc], axis=0)))
        # visualize_clusters(numpy_data, clusters, cluster_centers)
        # print('')
    # numpy_data = data.detach().cpu().numpy()  # plot
    # np_labels = labels.detach().cpu().numpy()  # plot
    # visualize_clusters(numpy_data, np_labels, [])

    # 计算相同类的cluster距离
    intra_centers = []
    for i in unique_labels:
        cluster_l = cluster_labels[i]

        # # plot
        # np_labels = labels.detach().cpu().numpy()  # plot
        # np_labels[np_labels != i] = -2
        # np_labels[np_labels == i] = cluster_l
        # cluster_centers = []
        # for cc in set(cluster_l):
        #     cluster_centers.append((i, np.average(numpy_data[np_labels != -2][cluster_l == cc], axis=0)))
        # visualize_clusters(numpy_data, np_labels, cluster_centers)

        if len(set(cluster_l)) > 1:
            for cidx1 in set(cluster_l):
                for cidx2 in set(cluster_l):
                    if cidx2 > cidx1:
                        center1 = data[labels == i][cluster_l == cidx1].mean()
                        center2 = data[labels == i][cluster_l == cidx2].mean()
                        intra_centers.append(abs(center1 - center2))

    # 计算不同类的cluster距离
    inter_centers = []
    for i in unique_labels:
        for j in unique_labels:
            if j > i:
                cluster_i = cluster_labels[i]
                cluster_j = cluster_labels[j]
                for cidx1 in set(cluster_i):
                    for cidx2 in set(cluster_j):
                        center1 = data[labels == i][cluster_i == cidx1].mean()
                        center2 = data[labels == j][cluster_j == cidx2].mean()
                        inter_centers.append(abs(center1 - center2))
    return inter_centers, intra_centers


def compute_cluster_loss_groundtruth(data, labels, eps=1, min_samples=10):
    '''
    用groundtruth label计算cluster center
    '''
    cluster_labels = {}
    unique_labels = labels.unique().detach().cpu().numpy()

    # 计算不同类的cluster距离
    inter_centers = []
    for i in unique_labels:
        for j in unique_labels:
            if j > i:
                center1 = data[labels == i].mean()
                center2 = data[labels == j].mean()
                inter_centers.append(abs(center1 - center2))
    return inter_centers, inter_centers


def combined_loss(features, labels, scl_loss, lambda_intra=1.0, lambda_inter=1.0):
    """
    结合 Supervised Contrastive Loss 和类内/类间距离正则化的总损失。

    Args:
        features: Tensor, shape (N, D) - 样本特征
        labels: Tensor, shape (N,) - 样本标签
        scl_loss_fn: 已实现的 Supervised Contrastive Loss 函数
        lambda_intra: float - 类内距离正则化的权重
        lambda_inter: float - 类间距离正则化的权重

    Returns:
        total_loss: 结合的总损失
    """
    # # 1. 计算原始 SCL 损失
    # scl_loss = scl_loss_fn(features, labels)

    # inter_losses, intra_losses = compute_cluster_loss(features, labels, eps=0.5, min_samples=5)
    inter_losses, intra_losses = compute_cluster_loss_groundtruth(features, labels, eps=0.5, min_samples=5)
    # inter_loss = sum(inter_losses)/len(inter_losses)  # between classes
    inter_loss = 1 / (torch.stack(inter_losses).min() + 1e-8)  # 越小越好
    intra_loss = sum(intra_losses) / len(intra_losses)  # within classes
    print(
        f"SupCon loss: {scl_loss.item()}, Intra-class Distance: {intra_loss.item()}, Inter-class Distance: {inter_loss.item()}")

    # 5. 结合总损失
    # total_loss = lambda_inter * intra_loss * 10
    total_loss = lambda_inter * inter_loss * 10
    # lambda_inter = 10
    # total_loss = lambda_intra * intra_loss + lambda_inter * inter_loss
    # total_loss = scl_loss + lambda_intra * intra_loss + lambda_inter * inter_loss

    return total_loss


def visualize_clusters(class_features, cluster_labels, cluster_centers, eps=0.5, ms=5):
    features = class_features
    labels = cluster_labels

    if len(cluster_centers) == 0:
        cluster_centers = [(0, features[0])]
    centers = []
    for c, center in cluster_centers:
        centers.append(center)
    centers = np.stack(centers)
    reducer = UMAP(n_components=2, random_state=42)
    features_2d = reducer.fit_transform(features)
    centers_2d = reducer.transform(centers)  # 将簇中心点映射到同一空间

    # 2. 绘制散点图
    plt.figure(figsize=(10, 8))

    # 根据 cluster_labels 绘制样本点
    scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=labels, cmap='viridis', s=30, alpha=0.7)
    plt.colorbar(scatter, label="Cluster Labels")

    # 绘制簇中心点
    plt.scatter(centers_2d[:, 0], centers_2d[:, 1], c='red', s=100, edgecolors='black', marker='X',
                label='Cluster Centers')

    # 图形细节
    plt.title("2D UMAP Projection with Cluster Labels" + str(eps) + "_" + str(ms))
    plt.xlabel("UMAP Dimension 1")
    plt.ylabel("UMAP Dimension 2")
    plt.legend()
    plt.grid(True)
    plt.show()


# ======= 3. 模型训练 ======= #
def train_scl_model(model, loader, criterion, optimizer, epochs=500, device='cpu', if_contrast=True):
    model.train()
    avg_loss = []
    for epoch in range(epochs):
        losses = []
        for batch in loader:
            data, labels = batch
            label_vote = majority_value(labels)
            label_vote = torch.from_numpy(label_vote)

            data = data.to(device=device, dtype=torch.float)
            label_vote = label_vote.to(device=device, dtype=torch.long)
            # outputs: supcontrast output; feature: feature extractor output
            outputs, features = model(data, if_contrast)
            # todo 需要考虑supcon时用output还是feature处理聚类中心
            loss = criterion(outputs, features, label_vote)
            losses.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        avg_loss.append(np.mean(losses))
        # print(f"Epoch {epoch + 1}, Loss: {avg_loss[-1]}")
    return model


def train_sup_model(model, loader, criterion, optimizer, epochs=500, device='cpu', if_contrast=True):
    model.train()
    avg_loss = []
    for epoch in range(epochs):
        losses = []
        for batch in loader:
            data, labels = batch
            label_vote = majority_value(labels)
            label_vote = torch.from_numpy(label_vote)

            data = data.to(device=device, dtype=torch.float)
            label_vote = label_vote.to(device=device, dtype=torch.long)
            # outputs: supcontrast output; feature: feature extractor output
            outputs, features = model(data, if_contrast)
            loss = criterion(outputs, label_vote)
            losses.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        avg_loss.append(np.mean(losses))
        # print(f"Epoch {epoch + 1}, Loss: {avg_loss[-1]}")
    return model


# ======= 4. 主动学习策略 ======= #

def compute_loss_contribution(model, unlabeled_data, labeled_data, labeled_labels):
    model.eval()
    with torch.no_grad():
        # 计算未标注数据和已标注数据的嵌入
        _, unlabeled_embeddings = model(torch.tensor(unlabeled_data, device=device, dtype=torch.float32),
                                        if_contrast=True)
        _, labeled_embeddings = model(torch.tensor(labeled_data, device=device, dtype=torch.float32), if_contrast=True)

        # 计算相似性矩阵 (unlabeled x labeled)
        similarities = torch.matmul(unlabeled_embeddings, labeled_embeddings.T)
        label_vote = majority_value(labeled_labels)
        label_vote = torch.from_numpy(label_vote)
        labeled_labels = label_vote.to(device=device, dtype=torch.long)

        loss_contributions = []
        for i, sim in enumerate(similarities):  # 遍历未标注样本
            # 获取与已标注数据的正负样本掩码
            pos_mask = (labeled_labels == labeled_labels[i % len(labeled_labels)]).float()
            neg_mask = 1 - pos_mask

            # 避免掩码全为 0
            if pos_mask.sum() == 0:
                pos_mask[torch.argmax(sim)] = 1
            if neg_mask.sum() == 0:
                neg_mask[torch.argmin(sim)] = 1

            # 归一化相似性以避免指数计算溢出
            sim = sim - sim.max()  # 数值稳定性优化

            # 正样本损失
            pos_sim = sim * pos_mask
            pos_loss = -torch.logsumexp(pos_sim, dim=-1) + torch.logsumexp(sim, dim=-1)  # 使用 logsumexp 避免显式指数计算

            # 负样本损失
            neg_sim = sim * neg_mask
            neg_loss = -torch.logsumexp(neg_sim, dim=-1) + torch.logsumexp(sim, dim=-1)

            # 合并正负样本损失
            total_loss = pos_loss + neg_loss
            loss_contributions.append(total_loss.item())

    return np.array(loss_contributions)


# 冻结特征提取器函数：stage2，只训练分类器
def freeze_encoders(model):
    for param in model.acc_encoder.parameters():
        param.requires_grad = False
    for param in model.pre_encoder.parameters():
        param.requires_grad = False
    for param in model.projector.parameters():
        param.requires_grad = False
    for param in model.classifier.parameters():
        param.requires_grad = True


# 解冻特征提取器函数：stage1，训练对比损失
def unfreeze_encoders(model):
    for param in model.acc_encoder.parameters():
        param.requires_grad = True
    for param in model.pre_encoder.parameters():
        param.requires_grad = True
    for param in model.projector.parameters():
        param.requires_grad = True
    for param in model.classifier.parameters():
        param.requires_grad = False


def unfreeze_all(model):
    for param in model.acc_encoder.parameters():
        param.requires_grad = True
    for param in model.pre_encoder.parameters():
        param.requires_grad = True
    for param in model.projector.parameters():
        param.requires_grad = True
    for param in model.classifier.parameters():
        param.requires_grad = True


def accuracy_class(predict_labels, true_labels):
    # 1. 获取唯一类别
    unique_classes = np.unique(true_labels)

    # 2. 计算每个类别的准确率
    class_accuracies = {}
    for cls in unique_classes:
        # 筛选出属于当前类别的样本
        class_indices = true_labels == cls
        # 计算预测正确的数量和总数量
        correct_predictions = np.sum(predict_labels[class_indices] == true_labels[class_indices])
        total_samples = np.sum(class_indices)
        # 计算准确率
        accuracy = correct_predictions / total_samples if total_samples > 0 else 0
        class_accuracies[cls] = accuracy

    # 输出每个类别的准确率
    for cls, acc in class_accuracies.items():
        print(f"Class {cls}: Accuracy = {acc:.2f}")


def evaluate_model(model, loader, cal_threshold=0.0, if_contrast=False):
    model.eval()
    correct, total = 0, 0
    threshold_acc, threshold_total = 0, 0
    data_list, predlabel_list, truelabel_list = [], [], []  # 保存新标签用于supCon训练
    with torch.no_grad():
        for data, labels in loader:
            data = data.to(device=device, dtype=torch.float)
            label_vote = majority_value(labels)
            label_vote = torch.from_numpy(label_vote)
            label_vote = label_vote.to(device=device, dtype=torch.long)
            predicts, features = model(data, if_contrast=False)
            # predictions = torch.argmax(predicts, dim=1)  # predicts(batch,numcls); predictions(batch)
            max_values, predictions = torch.max(predicts, dim=1)  # max_values用来挑选threshold
            correct += (predictions == label_vote).sum().item()

            ## 计算预测值大于cal_threshold前提下的准确率
            # 将 max_values 小于 cal_threshold 的位置上的 max_indices 赋值为 -1
            predictions[max_values < cal_threshold] = -1
            threshold_acc += (predictions == label_vote).sum().item()
            # 计算 predictions 中 -1 的个数
            count_neg_ones = (predictions == -1).sum().item()

            total += labels.size(0)
            threshold_total = total - count_neg_ones

            if if_contrast:
                ## 保存新标签和数据
                # 获取 predictions 中 -1 的位置
                pos_positions = torch.nonzero(predictions != -1).squeeze()  # squeeze() 去掉多余的维度
                if len(pos_positions) > 0:
                    new_data = data[pos_positions].detach().cpu().numpy()
                    data_list.append(new_data)

                    new_label = predictions[pos_positions].detach().cpu().numpy()
                    reshaped_array = new_label.reshape(new_label.shape[0], 1)  # 将形状从 (batch,) 转换为 (batch, 1)
                    reshaped_array = np.tile(reshaped_array, (1, labels.shape[1]))  # 复制到 (batch, 50)

                    t_label = label_vote[pos_positions].detach().cpu().numpy()
                    reshaped_t = t_label.reshape(t_label.shape[0], 1)  # 将形状从 (batch,) 转换为 (batch, 1)
                    reshaped_t = np.tile(reshaped_t, (1, labels.shape[1]))  # 复制到 (batch, 50)

                    predlabel_list.append(reshaped_array)
                    truelabel_list.append(reshaped_t)
                else:
                    data_list.append(data.detach().cpu().numpy())

                    new_label = predictions.detach().cpu().numpy()
                    reshaped_array = new_label.reshape(labels.shape[0], 1)  # 将形状从 (batch,) 转换为 (batch, 1)
                    reshaped_array = np.tile(reshaped_array, (1, labels.shape[1]))
                    predlabel_list.append(reshaped_array)

                    truelabel_list.append(labels.numpy())

    # 重新组合 data和label
    if if_contrast:
        concat_data = np.concatenate(data_list, axis=0)
        concat_label = np.concatenate(predlabel_list, axis=0)
        true_label = np.concatenate(truelabel_list, axis=0)
        return (correct / total, threshold_acc / threshold_total,
                concat_data, concat_label, true_label)

    return correct / total, threshold_acc / threshold_total


# ======= 5. 循环主动学习 ======= #
# 初始化模型和优化器
model = SimpleNN()
model = model.to(device)
criterion = ContrastiveLoss()
classify_criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)


def AE_eval_time_series(train_loader, model, device):
    model.eval()

    representation_list = []
    sample_list, timestamp_list, label_list, pred_list, timestr_list, flag_list = [], [], [], [], [], []
    for i, (sample, label) in enumerate(train_loader):
        sample = sample.to(device=device, non_blocking=True, dtype=torch.float)
        # input of autoencoder will be 3D, the backbone is 1d-cnn
        output, x_encoded = model(sample)  # x_encoded.shape=batch512,outchannel128,len13
        tmp_representation = x_encoded.detach().cpu().numpy()
        representation_list.append(tmp_representation)
        sample_list.append(sample.detach().cpu().numpy())
        label_list.append(label.detach().cpu().numpy())
        pred_list.append(output.detach().cpu().numpy())

    return representation_list, sample_list, pred_list, label_list


# 主动学习迭代
update_size = max(int(0.01 * len(X_train_full)), 2)  # 每次选择 1% 的样本
batch_size = 5000
iteration = 1

## 创建数据加载器
plot_dataset = data_loader_umineko(data_b.astype(float), label_b.astype(int))
plot_loader = DataLoader(plot_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
## initial plot
repres_list, sample_list, pred_list, label_list = \
    AE_eval_time_series(plot_loader, model, device)
plot_func(repres_list, sample_list, label_list, sample_list,  # not necessary to plot pred_list
          str(0), 'activerepresent')

while len(X_unlabeled) > 0:
    print(f"==============Iteration {iteration}===============")
    print(f"Remaining unlabeled samples: {len(X_unlabeled)}")

    # 创建数据加载器
    labeled_dataset = data_loader_umineko(X_labeled.astype(float), y_labeled.astype(int))
    labeled_loader = DataLoader(labeled_dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    ## stage 1: train supervised model, 获得分类器新参数
    freeze_encoders(model)
    model = train_sup_model(model, labeled_loader, classify_criterion, optimizer,
                            epochs=200, device=device, if_contrast=False)

    ## 用all_data - labeled_data 的数据测试，获得标签，用于supCon训练
    rest_data = np.concatenate([X_test, X_unlabeled], axis=0)
    rest_label = np.concatenate([y_test, y_unlabeled], axis=0)
    rest_dataset = data_loader_umineko(rest_data.astype(float), rest_label.astype(int))
    rest_loader = DataLoader(rest_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    accuracy, _, rest_datan, rest_labeln, rest_truelabel = evaluate_model(model, rest_loader,
                                                                          cal_threshold=0.6,
                                                                          if_contrast=True)
    accuracy_class(rest_labeln[:, 0], rest_truelabel[:, 0])

    ## 将上一步的测试标签和labeled_data的真实标签一起用作supCon的标签，positive label是真是标签+predict>0.8的标签
    all_data = np.concatenate([X_labeled, rest_datan], axis=0)
    all_label = np.concatenate([y_labeled, rest_labeln], axis=0)
    all_dataset = data_loader_umineko(all_data.astype(float), all_label.astype(int))
    all_loader = DataLoader(all_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    ## stage 2: train supCon model (所有数据都来训练)
    unfreeze_encoders(model)
    # todo
    model = train_scl_model(model, plot_loader, criterion, optimizer,
                            # model = train_scl_model(model, all_loader, criterion, optimizer,
                            epochs=50, device=device, if_contrast=True)

    ### plot
    repres_list, sample_list, pred_list, label_list = \
        AE_eval_time_series(plot_loader, model, device)
    plot_func(repres_list, sample_list, label_list, sample_list,  # not necessary to plot pred_list
              str(iteration), 'activerepresent')

    # ## stage 3: fine-tuneing. train supervised model (目前看效果不行)
    # unfreeze_all(model)
    # model = train_scl_model(model, labeled_loader, classify_criterion, optimizer,
    #                         epochs=50, device=device, if_contrast=False)

    ## 计算未标注数据（all-labeled-test）的对比损失贡献
    loss_contributions = compute_loss_contribution(model, X_unlabeled, X_labeled, y_labeled)

    ## 选择对比损失贡献最大的样本
    select_size = min(update_size, len(X_unlabeled))
    selected_indices = np.argsort(loss_contributions)[-select_size:]
    selected_samples = X_unlabeled[selected_indices]
    selected_labels = y_unlabeled[selected_indices]

    ## 更新标注集和未标注数据池
    X_labeled = np.vstack([X_labeled, selected_samples])
    y_labeled = np.concatenate([y_labeled, selected_labels], axis=0)
    X_unlabeled = np.delete(X_unlabeled, selected_indices, axis=0)
    y_unlabeled = np.delete(y_unlabeled, selected_indices, axis=0)
    print(f"Labeled samples: {len(X_labeled)}")

    # ======= 6. 模型测试 ======= #
    accuracy, _ = evaluate_model(model, labeled_loader)
    print(f"train Accuracy: {accuracy * 100:.2f}%")

    test_dataset = data_loader_umineko(X_test.astype(float), y_test.astype(int))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    accuracy, _ = evaluate_model(model, test_loader)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

    iteration += 1

print("All unlabeled samples have been labeled!")
