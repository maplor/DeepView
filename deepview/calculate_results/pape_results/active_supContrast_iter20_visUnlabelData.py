'''
active_supContrast.py
基于上面py代码，获得labeled data和model weight
然后对unlabeled data进行预测，获得latent
将latent对应的cluster画到sensor data上，并和true label比较
'''

# import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
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

import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, fowlkes_mallows_score, silhouette_score


import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# import cupy as cp
# from cuml.cluster import DBSCAN as cuDBSCAN
from sklearn.metrics import accuracy_score, f1_score


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

from scipy.stats import entropy


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

def gaussian_std(X):
    mean_val = np.mean(X.astype(float), axis=0)
    std_val = np.std(X.astype(float), axis=0)
    X_standardized = (X - mean_val) / np.maximum(std_val, 10 ** -5)
    return X_standardized


def kmeans_best_numCenters_elbow(X, init_num_centers=2, max_num_centers=10):
    # 计算不同聚类数下的 SSE
    sse = []
    k_values = range(init_num_centers, max_num_centers)
    for k in k_values:
        kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
        kmeans.fit(X)
        sse.append(kmeans.inertia_)

    # 返回最佳聚类数
    # 计算每两个点之间的差值
    diff = np.diff(sse)
    # 计算每两个差值之间的差值
    second_diff = np.diff(diff)
    # 找到第二个差值的最小值的位置
    elbow_point = np.argmin(second_diff) + 1  # 加1是因为我们计算了第二个差值
    print(f"Elbow point: {elbow_point + init_num_centers}")

    # 绘制肘部法则图
    plt.figure(figsize=(8, 6))
    plt.plot(k_values, sse, marker='o')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('SSE')
    plt.title('Elbow Method for Optimal k')
    plt.xticks(k_values)
    plt.grid()
    plt.show()
    return elbow_point + init_num_centers


def kmeans_best_numCenters_silhoutte(X, iteration, init_num_centers=2, max_num_centers=10):
    silhouette_scores = []
    k_values = range(init_num_centers, max_num_centers)
    for k in k_values:
        kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = kmeans.fit_predict(X)
        silhouette_scores.append(silhouette_score(X, labels))

    # 找到最佳聚类数
    optimal_k = k_values[np.argmax(silhouette_scores)]
    print(f"The optimal number of clusters is: {optimal_k}")

    return optimal_k

def plot_func(representation_list, sample_list, label_list, pred_list, count, name):
    label_concat = np.concatenate(label_list)
    label_concat_vote = majority_value(label_concat)
    repre_concat = np.concatenate(representation_list)
    repre_reshape = repre_concat.reshape(repre_concat.shape[0], -1).astype(float)

    umap_3d = UMAP(n_components=2)
    proj_3d_gyro = umap_3d.fit_transform(repre_reshape)

    label_concat_vote_str = [labeldict_findstr[i] for i in label_concat_vote]

    if (count == 1) or (count == 10) or (count == 20) or (count == -1):
        fig_3d = px.scatter(
            proj_3d_gyro, x=0, y=1,
            color=label_concat_vote_str,
            # color=label_concat_vote_str,
            labels={'activity': 'activity'},
            color_discrete_map={'ground_stationary': 'red',
                                'stationary': 'blue',
                                'bathing': 'green',
                                'flying_active': 'orange',
                                'flying_passive': 'purple',
                                'foraging': 'brown',
                                'unknown': 'lightgrey'
                                }
        )
        # Reduce marker size for all points
        for trace in fig_3d.data:
            trace.marker.size = 6  # Adjust the size to your preference (e.g., 6)

        # Update transparency for traces where activity is '-2.0'
        fig_3d.for_each_trace(lambda trace: trace.update(marker=dict(opacity=0.5)) if trace.name == '-2.0' else ())
        fig_3d.update_traces(marker=dict(line=dict(width=0)))  # remove boundary of point
        fig_3d.write_html('%s_activeSupBase_epoch_%s.html' % (name, str(count)))

    ############################################
    # plot cluster results
    n_clusters = kmeans_best_numCenters_silhoutte(proj_3d_gyro, iteration,
                                                  init_num_centers=6, max_num_centers=12)
    # Create a KMeans instance
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)

    # Fit the model and get cluster labels
    labels = kmeans.fit_predict(proj_3d_gyro)
    cus_color = [label_colors[i] for i in labels]

    if (count == 1) or (count == 10) or (count == 20) or (count == -1):
        fig_3d = px.scatter(
            proj_3d_gyro, x=0, y=1,
            color=cus_color,
            # color=label_concat_vote_str,
            labels={'color': 'cluster'},
            color_discrete_map={'unknown': 'lightgrey'},
        )
        # Reduce marker size for all points
        for trace in fig_3d.data:
            trace.marker.size = 6  # Adjust the size to your preference (e.g., 6)

        # Update transparency for traces where activity is '-2.0'
        fig_3d.for_each_trace(lambda trace: trace.update(marker=dict(opacity=0.5)) if trace.name == '-2.0' else ())
        fig_3d.update_traces(marker=dict(line=dict(width=0)))  # remove boundary of point
        fig_3d.write_html('%s_activeSupBase_epoch_%s_cluster.html' % (name, str(count)))

    ##############################
    # calculate score
    # 假设 y_true 是真实标签，labels 是聚类标签
    y_true = label_concat_vote_str  # 真实标签
    labels = labels  # 聚类标签

    # 计算调整兰德指数
    ari = adjusted_rand_score(y_true, labels)
    print(f"Adjusted Rand Index: {ari:.4f}")

    # 计算归一化互信息
    nmi = normalized_mutual_info_score(y_true, labels)
    print(f"Normalized Mutual Information: {nmi:.4f}")

    # 计算Fowlkes-Mallows指数
    fmi = fowlkes_mallows_score(y_true, labels)
    print(f"Fowlkes-Mallows Index: {fmi:.4f}")

    silhouette = silhouette_score(proj_3d_gyro, labels)
    print(f"Silhouette Score: {silhouette:.4f}")
    return ari, nmi, fmi, silhouette, labels


# ======= 1. 数据生成 ======= #
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
#         elif year == '2022':
#             df_raw_2022 = raw_
#             df_2022 = labeled_
#         else:
#             print('Error: year not found')
#             break
#
#     all_df = pd.concat([df_raw_2018, df_raw_2022], ignore_index=True)
#     all_df['filename'] = all_df['year'].astype(int).astype(str) + '_' + all_df['animal_tag']
#     animal_tag_list = ['2018_LB07', '2018_LB08', '2018_LB09', '2018_LB10', '2018_LB11', '2018_LB12', '2018_LB13',
#                        '2022_LB02', '2022_LB03', '2022_LB08', '2022_LB09']
#     all_df = all_df[all_df['filename'].isin(animal_tag_list)]
#
#     all_df['label_id'] = all_df['label'].map(label_dict)
#     all_df['label_id'] = all_df['label_id'].fillna(-2)
#
#     # 删除无标签的数据
#     selected_df = all_df[all_df.label_id != -2]
#     return all_df, selected_df
#
#
# all_df,  selected_df = read_sensor_data()
# # 选择需要的列
# selected_df = selected_df[['acc_x', 'acc_y', 'acc_z', 'pressure', 'label_id']]
# fill_selected_df = selected_df.fillna(method='ffill').fillna(method='bfill')  # pressure has nan values
# selected_np = fill_selected_df.values
# all_df = all_df[['acc_x', 'acc_y', 'acc_z', 'pressure', 'label_id']]
# fill_all_df = all_df.fillna(method='ffill').fillna(method='bfill')  # pressure has nan values
# all_np = fill_all_df.values
#
# # 分别预处理两种传感器数据
# acc_np = fill_selected_df[['acc_x', 'acc_y', 'acc_z']].values
# tmp_acc_stand = gaussian_std(acc_np)
# fill_selected_df['pressure'] = fill_selected_df['pressure'] - 1013.25  # standard pressure
# data = fill_selected_df['pressure'].values
#
# # 计算均值和标准差
# mean = np.mean(data)
# std_dev = np.std(data)
#
# # 用label data获得mean和std后， 用mean和std去处理所有数据
# # 分别预处理两种传感器数据
# acc_np = fill_all_df[['acc_x', 'acc_y', 'acc_z']].values
# tmp_acc_stand = gaussian_std(acc_np)
# fill_all_df['pressure'] = fill_all_df['pressure'] - 1013.25  # standard pressure
# data = fill_all_df['pressure'].values
#
# # 识别超过3个标准差的异常值
# outlier_mask = np.abs(data - mean) > 4 * std_dev
# # 创建数据副本
# data_replaced = np.copy(data)
# # 获取所有 False 对应的索引
# false_indices = np.where(outlier_mask == False)[0]
# for i in false_indices:
#     # 查找前一个正常值
#     j = i - 1
#     # 查找后一个正常值
#     k = i + 1
#
#     # 找到前一个正常值
#     while j >= 0 and outlier_mask[j]:
#         j -= 1
#     # 找到后一个正常值
#     while k < len(data) and outlier_mask[k]:
#         k += 1
#
#     # 用最近的正常值替换异常值
#     if j >= 0 and (k >= len(data) or i - j <= k - i):
#         data_replaced[i] = data[j]
#     elif k < len(data):
#         data_replaced[i] = data[k]
#
# tmp_press_stand = gaussian_std(data_replaced)
#
# all_np[:, :3] = tmp_acc_stand
# all_np[:, -2] = tmp_press_stand
#
# len_sw = 50
# tmp_b = sliding_window(all_np, len_sw, len_sw)
# # concatenate list
# data_b = np.transpose(tmp_b[:, :, :-1], (0, 2, 1))  # [B, Len, dim-1] -> [B, dim-1, Len]
# label_b = tmp_b[:, :, -1]  # [B, Len]
# vote_label = majority_value(label_b)

# ======= 2. 数据读取 ======= #

# with open('all_norm_data.pkl', 'wb') as file:  # 'wb'表示以二进制写入模式打开文件
#     pickle.dump({'data': data_b, 'label': label_b}, file)
with open('all_norm_data.pkl', 'rb') as file:  # 'rb'表示以二进制读取模式打开文件
    d_dict = pickle.load(file)
    data_b = d_dict['data']
    label_b = d_dict['label']
    vote_label = majority_value(label_b)


len_sw = 50
device = 'cuda:1' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
batch_size = 4000
all_dataset = data_loader_umineko(data_b.astype(float), vote_label)
all_loader = DataLoader(all_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

# 去掉labels为-2的数据
data_select = data_b[vote_label != -2]
label_select = vote_label[vote_label != -2]

# 将数据分为8:2，其中2为测试集
vote_label = majority_value(label_b)
X_train_full, X_test, y_train_full, y_test = train_test_split(data_select,
                                                              label_select,
                                                              test_size=0.2,
                                                              stratify=label_select,
                                                              random_state=42)

# 从X_train_full中随机选择1%的数据作为X_labeled，其余为X_unlabeled
X_labeled, X_unlabeled, y_labeled, y_unlabeled = train_test_split(X_train_full, y_train_full, test_size=0.99, random_state=42)

# 确保X_labeled和X_unlabeled的大小
print("X_labeled shape:", X_labeled.shape)
print("X_unlabeled shape:", X_unlabeled.shape)
print("X_test shape:", X_test.shape)


# ======= 2. 模型与 Supervised Contrastive Learning ======= #
class SimpleNN(nn.Module):
    def __init__(self, input_dim=128 * 6, embedding_dim=128, number_classes=6):
        super(SimpleNN, self).__init__()
        self.acc_encoder = Autoencoder3d4().feature_extractor
        self.pre_encoder = Autoencoder1d().feature_extractor

        self.linear = nn.Linear(input_dim*2, 32)
        self.projector = nn.Sequential(
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 16)
        )

        self.classifier = nn.Sequential(
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, number_classes),
            nn.Softmax(dim=1),
        )


    def forward(self, data, if_contrast=True):
        accel = data[:, :3, :]
        press = data[:, -1:, :]
        acc_fea, _, _ = self.acc_encoder(accel)
        pre_fea, _, _ = self.pre_encoder(press)
        concat_fea = torch.concat([acc_fea, pre_fea], dim=1)
        concat_fea = self.linear(concat_fea)
        # concat_fea = self.linear(acc_fea)
        if if_contrast:
            output = self.projector(concat_fea)  # output(batch, 128)
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


# ======= 3. 模型训练 ======= #

def train_model(model, loader, criterion, optimizer, epochs=500, device='cpu', if_contrast=True):
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
    return model, avg_loss

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


def evaluate_model(model, loader):
    # supervised learning
    model.eval()
    ground_truth, prediction = [], []
    data_list, predlabel_list, truelabel_list = [], [], []  # 保存新标签用于supCon训练
    with (torch.no_grad()):
        for data, labels in loader:
            data = data.to(device=device, dtype=torch.float)
            label_vote = majority_value(labels)
            label_vote = torch.from_numpy(label_vote)
            label_vote = label_vote.to(device=device, dtype=torch.long)
            predicts, features = model(data, if_contrast=False)
            max_values, predictions = torch.max(predicts, dim=1)  # max_values用来挑选threshold
            ground_truth.append(label_vote.detach().cpu().numpy())
            prediction.append(predictions.detach().cpu().numpy())
            data_list.append(data.detach().cpu().numpy())

    # 计算准确率
    accuracy = accuracy_score(np.concatenate(ground_truth, axis=0), np.concatenate(prediction, axis=0))
    # print(f'Accuracy: {accuracy:.2f}, Macro F1 Score: {macro_f1:.2f}, Micro F1 Score: {micro_f1:.2f}')
    # 计算 macro F1 分数
    macro_f1 = f1_score(np.concatenate(ground_truth, axis=0), np.concatenate(prediction, axis=0), average='macro')
    # print(f'Macro F1 Score: {macro_f1:.2f}')
    # 计算 micro F1 分数
    micro_f1 = f1_score(np.concatenate(ground_truth, axis=0), np.concatenate(prediction, axis=0), average='micro')
    # print(f'Micro F1 Score: {micro_f1:.2f}')
    print(f'Accuracy: {accuracy:.2f}, Macro F1 Score: {macro_f1:.2f}, Micro F1 Score: {micro_f1:.2f}')
    return accuracy, macro_f1, micro_f1, prediction, ground_truth

def evaluate_supContrast_model(model, loader, cal_threshold=0.0):
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
            max_values, predictions = torch.max(predicts, dim=1)  # max_values用来挑选threshold
            correct += (predictions == label_vote).sum().item()

            ## 计算预测值大于cal_threshold前提下的准确率
            # 将 max_values 小于 cal_threshold 的位置上的 max_indices 赋值为 -1
            predictions[max_values < cal_threshold] = -1
            threshold_acc += (predictions == label_vote).sum().item()
            # # 计算 predictions 中 -1 的个数
            # count_neg_ones = (predictions == -1).sum().item()

            ## 保存新标签和数据
            # 获取 predictions 中 -1 的位置
            pos_positions = torch.nonzero(predictions != -1).squeeze()  # squeeze() 去掉多余的维度
            if len(pos_positions) > 0:
                new_data = data[pos_positions].detach().cpu().numpy()
                data_list.append(new_data)

                new_label = predictions[pos_positions].detach().cpu().numpy()
                # reshaped_array = new_label.reshape(new_label.shape[0], 1)  # 将形状从 (batch,) 转换为 (batch, 1)
                # reshaped_array = np.tile(reshaped_array, (1, labels.shape[1]))  # 复制到 (batch, 50)

                t_label = label_vote[pos_positions].detach().cpu().numpy()
                # reshaped_t = t_label.reshape(t_label.shape[0], 1)  # 将形状从 (batch,) 转换为 (batch, 1)
                # reshaped_t = np.tile(reshaped_t, (1, labels.shape[1]))  # 复制到 (batch, 50)

                predlabel_list.append(new_label)
                truelabel_list.append(t_label)
            else:
                data_list.append(data.detach().cpu().numpy())

                new_label = predictions.detach().cpu().numpy()
                # reshaped_array = new_label.reshape(labels.shape[0], 1)  # 将形状从 (batch,) 转换为 (batch, 1)
                # reshaped_array = np.tile(reshaped_array, (1, labels.shape[1]))
                predlabel_list.append(new_label)

                truelabel_list.append(label_vote.detach().cpu().numpy())

    # 重新组合 data和label
    concat_data = np.concatenate(data_list, axis=0)
    concat_label = np.concatenate(predlabel_list, axis=0)
    true_label = np.concatenate(truelabel_list, axis=0)
    # 计算准确率
    accuracy = accuracy_score(true_label, concat_label)
    # print(f'Accuracy: {accuracy:.2f}, Macro F1 Score: {macro_f1:.2f}, Micro F1 Score: {micro_f1:.2f}')
    # 计算 macro F1 分数
    macro_f1 = f1_score(true_label, concat_label, average='macro')
    # print(f'Macro F1 Score: {macro_f1:.2f}')
    # 计算 micro F1 分数
    micro_f1 = f1_score(true_label, concat_label, average='micro')
    # print(f'Micro F1 Score: {micro_f1:.2f}')
    print(f'Accuracy: {accuracy:.2f}, Macro F1 Score: {macro_f1:.2f}, Micro F1 Score: {micro_f1:.2f}')
    return (accuracy, macro_f1, micro_f1,  # accuracy
            concat_data, concat_label, true_label)  # label


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


def plot_confusion_matrix(pred_label, truth_label, name):
    # # 将列表展开为一维数组
    # truth_label = np.concatenate(truth_label_list, axis=0)
    # pred_label = np.concatenate(pred_label_list, axis=0)

    # 计算混淆矩阵
    cm = confusion_matrix(truth_label, pred_label)

    # 获取完整的类别标签列表，确保所有类别都能显示
    class_names = sorted(set(labeldict_findstr.values()))  # 确保按顺序排列

    # 设置热图尺寸
    plt.figure(figsize=(10, 7))

    # 绘制热图，确保所有标签完整显示
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)

    # 添加标题和标签
    plt.title('Confusion Matrix Heatmap')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')

    # 调整刻度标签显示，防止标签重叠
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=45, va='center')

    # 保存热图到文件
    plt.savefig(f'heatmap_{name}.png', bbox_inches='tight')

    # 显示热图
    # plt.show()
    plt.close()
    return


# ======= 5. 循环主动学习 ======= #
# 初始化模型和优化器
model = SimpleNN()
model = model.to(device)
classify_criterion = nn.CrossEntropyLoss()
supContrast_criterion = SupContrastiveLoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)

# testmodel = SimpleNN()
# testmodel = testmodel.to(device)


# ---------- 主动学习，sample选择策略 ---------- #
# todo: 需要确定选出来的sample到底具有什么特点

def random_sampling(X_unlabeled, y_unlabeled, update_size):
    select_size = min(update_size, len(X_unlabeled))
    selected_indices = np.random.choice(len(X_unlabeled), select_size, replace=False)
    selected_samples = X_unlabeled[selected_indices]
    selected_labels = y_unlabeled[selected_indices]
    X_unlabeled = np.delete(X_unlabeled, selected_indices, axis=0)
    y_unlabeled = np.delete(y_unlabeled, selected_indices, axis=0)
    return selected_samples, selected_labels, X_unlabeled, y_unlabeled

def freeze_encoders(model):
    for param in model.acc_encoder.parameters():
        param.requires_grad = False
    for param in model.pre_encoder.parameters():
        param.requires_grad = False
    for param in model.linear.parameters():
        param.requires_grad = False
    for param in model.projector.parameters():
        param.requires_grad = False
    for param in model.classifier.parameters():
        param.requires_grad = True

def freeze_projector(model):
    for param in model.acc_encoder.parameters():
        param.requires_grad = True
    for param in model.pre_encoder.parameters():
        param.requires_grad = True
    for param in model.linear.parameters():
        param.requires_grad = True
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
    for param in model.linear.parameters():
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
    for param in model.linear.parameters():
        param.requires_grad = True
    for param in model.projector.parameters():
        param.requires_grad = True
    for param in model.classifier.parameters():
        param.requires_grad = True


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

def supContrast_contribution(X_labeled, y_labeled, X_unlabeled, y_unlabeled, model, update_size):
    # 计算每个样本的贡献度
    ## 计算未标注数据（all-labeled-test）的对比损失贡献
    loss_contributions = compute_loss_contribution(model, X_unlabeled, X_labeled, y_labeled)

    # 选择贡献度最大的样本
    selected_indices = np.argsort(loss_contributions)[-update_size:]
    selected_samples = X_unlabeled[selected_indices]
    selected_labels = y_unlabeled[selected_indices]
    ## 更新标注集和未标注数据池
    X_labeled = np.vstack([X_labeled, selected_samples])
    y_labeled = np.concatenate([y_labeled, selected_labels], axis=0)
    X_unlabeled = np.delete(X_unlabeled, selected_indices, axis=0)
    y_unlabeled = np.delete(y_unlabeled, selected_indices, axis=0)
    print(f"Labeled samples: {len(X_labeled)}")

    return X_labeled, y_labeled, X_unlabeled, y_unlabeled


# 计算样本不确定性（基于熵）
def uncertainty_sampling(X_labeled, y_labeled, X_unlabeled, y_unlabeled, model, update_size):
    model.eval()
    with torch.no_grad():
        # 计算未标注数据和已标注数据的嵌入
        _, unlabeled_embeddings = model(torch.tensor(X_unlabeled, device=device, dtype=torch.float32),
                                        if_contrast=True)
        probs = model.classifier(unlabeled_embeddings)

        uncertainty = entropy(probs.detach().cpu().numpy().T)  # 计算每个样本的熵

        # 选择贡献度最大的样本
        selected_indices = np.argsort(uncertainty)[-update_size:]
        selected_samples = X_unlabeled[selected_indices]
        selected_labels = y_unlabeled[selected_indices]
        ## 更新标注集和未标注数据池
        X_labeled = np.vstack([X_labeled, selected_samples])
        y_labeled = np.concatenate([y_labeled, selected_labels], axis=0)
        X_unlabeled = np.delete(X_unlabeled, selected_indices, axis=0)
        y_unlabeled = np.delete(y_unlabeled, selected_indices, axis=0)
        print(f"Labeled samples: {len(X_labeled)}")

        return X_labeled, y_labeled, X_unlabeled, y_unlabeled, selected_labels


# ---------- 主动学习，sample选择策略 ---------- #
SupCount = 1  # supcontrastive learning
warmup = 20  # warmup epoch of supcontrastive learning
# SupCount = 0  # base mode, no supcontrastive learning
# warmup = 0

name_label = '_entropy_Contrast%s_warm%s'%(str(SupCount), str(warmup))
# name_label = '_entropy_Contrast%s_freeze%s'%(str(SupCount), str(warmup))
sensor_type = 'AccelPress'
# 主动学习迭代
update_size = max(int(0.01 * len(X_train_full)), 2)  # 每次选择 1% 的样本
batch_size = 4000
iteration = 1

## 创建数据加载器
labelall_dataset = data_loader_umineko(data_select.astype(float), label_select.astype(int))
labelall_loader = DataLoader(labelall_dataset, batch_size=batch_size, shuffle=True, drop_last=False)

unlabeled_dataset = data_loader_umineko(data_b[:1000].astype(float), vote_label[:1000].astype(int))
unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=batch_size, shuffle=True, drop_last=False)


train_accuracy_list, train_macro_f1_list, train_micro_f1_list = [], [], []
test_accuracy_list, test_macro_f1_list, test_micro_f1_list = [], [], []
test_pred_label_lists, test_truth_label_lists = [], []
weight_list, loss_list, selected_labels_list = [], [], []
ari_list, nmi_list, fmi_list, silhouette_list = [], [], [], []
while len(X_unlabeled) > 0:
    print(f"==============Iteration {iteration}===============")
    print(f"Already labeled samples: {len(X_labeled)}")
    print(f"Remaining unlabeled samples: {len(X_unlabeled)}")

    # 创建数据加载器
    labeled_dataset = data_loader_umineko(X_labeled.astype(float), y_labeled.astype(int))
    labeled_loader = DataLoader(labeled_dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    ## stage 1: train supCon model (所有数据都来训练)
    unfreeze_encoders(model)
    if iteration < warmup:
        epoch = 500
        model, _ = train_model(model, labeled_loader, supContrast_criterion, optimizer,
                                      epochs=epoch, device=device, if_contrast=True)


    ## stage 2: train supervised model, 获得分类器新参数
    freeze_encoders(model)  # only classifier is trainable
    model, _ = train_model(model, labeled_loader, classify_criterion, optimizer,
                                  epochs=10, device=device, if_contrast=False)
    freeze_projector(model)  # only projector is NOT trainable
    model, avg_loss = train_model(model, labeled_loader, classify_criterion, optimizer,
                            epochs=50, device=device, if_contrast=False)
    loss_list.append(np.average(avg_loss))  # 记录50次epoch的平均loss

    ### plot
    repres_list, sample_list, pred_list, label_list = \
        AE_eval_time_series(labeled_loader, model, device)
    ari, nmi, fmi, silhouette = (
        plot_func(repres_list, sample_list, label_list, sample_list,  # not necessary to plot pred_list
              iteration, sensor_type+name_label))


    # ======= 6. 模型测试 ======= #
    print('-------Training----------')
    accuracy, macro_f1, micro_f1, pred_label_list, truth_label_list = evaluate_model(model, labeled_loader)
    train_accuracy_list.append(accuracy)
    train_macro_f1_list.append(macro_f1)
    train_micro_f1_list.append(micro_f1)

    test_dataset = data_loader_umineko(X_test.astype(float), y_test.astype(int))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    print('-------Test----------')
    (accuracy, macro_f1, micro_f1,
     rest_datan, pred_label_list, truth_label_list) = (
        evaluate_supContrast_model(model, test_loader, cal_threshold=0.5))
    test_accuracy_list.append(accuracy)
    test_micro_f1_list.append(micro_f1)
    test_macro_f1_list.append(macro_f1)
    test_pred_label_lists.append(pred_label_list)
    test_truth_label_lists.append(truth_label_list)

    # print('-------cluster results----------')
    ari_list.append(ari)
    nmi_list.append(nmi)
    fmi_list.append(fmi)
    silhouette_list.append(silhouette)

    # 保存当前模型权重到列表
    weight_list.append(model.state_dict())

    # # stage 3: 更新数据集
    select_size = min(update_size, len(X_unlabeled))

    ### uncertainty_sampling
    X_labeled, y_labeled, X_unlabeled, y_unlabeled, selected_labels = (
        uncertainty_sampling(X_labeled, y_labeled,
                             X_unlabeled, y_unlabeled,
                             model, select_size))  # data变化了
    selected_labels_list.append(selected_labels)

    # 每十次迭代，也就是增加10%的数据后，保存一次模型和heatmap
    # if iteration % 10 == 0:
    if (iteration == 1) or (iteration == 10) or (iteration == 20):
        plot_confusion_matrix(pred_label_list, truth_label_list,
                              name='%s_%s_pred_%s'%(sensor_type,name_label,str(iteration)))
    iteration += 1
    if iteration == warmup+1:
        break # 只保存前20次的结果

# ======= 7. 用labeled data和unlabeled data观察latent ======= #
# todo dataloader 需要同时将unlabel和label的数据传入
combinedata = np.concatenate([X_labeled, data_b[134000:139000]])
combinelabel = np.concatenate([y_labeled, vote_label[134000:139000]])
unlabeled_dataset = data_loader_umineko(combinedata.astype(float), combinelabel.astype(int))
unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=batch_size, shuffle=True, drop_last=False)

repres_list, sample_list, pred_list, label_list = \
    AE_eval_time_series(unlabeled_loader, model, device)
ari, nmi, fmi, silhouette, clusterlabels = (
    plot_func(repres_list, sample_list, label_list, sample_list,  # not necessary to plot pred_list
          -1, sensor_type+name_label+'_unlabeled'))

# sensor data visualization
dataseg = np.concatenate(data_b[134000:139000,:3,:].transpose(0,2,1), axis=0)
print(dataseg.shape)
labelseg = vote_label[134000:139000]

# Create a scatter plot
plt.figure(figsize=(12, 4))
for label, label_name in labeldict_findstr.items():
    if label == -2:
        continue
    indices = np.where(labelseg == label)[0]
    plt.scatter(indices, [label] * len(indices), s=1, label=label_name)
# Formatting the plot
plt.xlabel("Frame no.")
plt.ylabel("Behavior")
plt.title("Behavior Labels Over Time")
plt.yticks(ticks=list(labeldict_findstr.keys()), labels=list(labeldict_findstr.values()))
plt.legend(markerscale=5, fontsize=8, loc='upper right')
plt.savefig('label.png')

plt.figure(figsize=(12, 4))
plt.plot(dataseg)
plt.savefig('data.png')

# 保存结果
with open(sensor_type+'_%s_results.pkl'%name_label, 'wb') as f:
    result_dict = {
        'train_accuracy_list': train_accuracy_list,
        'train_macro_f1_list': train_macro_f1_list,
        'train_micro_f1_list': train_micro_f1_list,
        'test_accuracy_list': test_accuracy_list,
        'test_macro_f1_list': test_macro_f1_list,
        'test_micro_f1_list': test_micro_f1_list,
        'test_pred_label_lists': test_pred_label_lists,
        'test_truth_label_lists': test_truth_label_lists,
        'selected_labels_list': selected_labels_list,
        'weight_list': weight_list,
        'ari_list': ari_list,
        'nmi_list': nmi_list,
        'fmi_list': fmi_list,
        'silhouette_list': silhouette_list
    }

    pickle.dump(result_dict, f)



print("All unlabeled samples have been labeled!")
