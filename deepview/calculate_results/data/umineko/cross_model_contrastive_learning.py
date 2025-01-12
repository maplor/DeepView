'''
3. 跨模态对比学习 (Cross-Modal Contrastive Learning)
对比学习方法可以让模型在不依赖标签的情况下，从两个模态（加速度和压力数据）中学习到相似的特征，区分四种行为。

方法：

使用对比学习方法（如 SimCLR 或 MoCo）来训练加速度和压力传感器数据的嵌入空间。
对比学习的目标是让相似的样本（例如飞行状态下的加速度和压力数据）靠近，而不相似的样本（例如飞行和非飞行状态下的加速度数据）则远离。
通过对比学习，你可以获得一个联合表示，该表示能够将这四种行为类别区分开来。
优点：

无需标签，通过自监督方式学习联合表示，适用于未标注的数据集。
'''

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
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

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
    Autoencoder3d4, # debug purpose
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

def dbscan_plot():
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
        fig, axes = plt.subplots(4 * 3, 1, figsize=(12, 24))
        for col in range(4):
            axes[0 + sensordim * col].plot(sample_reshape[start:end, col], 'r', label='groundtruth%s' % axis_dict[col])
            axes[0 + sensordim * col].plot(pred_reshape[start:end, col], 'b-', label='predict%s' % axis_dict[col])
            axes[0 + sensordim * col].set_title('Autoencoder_Reconstruct_Umineko_%s' % (axis_dict[col]))
            axes[0 + sensordim * col].set_xlabel('timestamp')
            axes[0 + sensordim * col].set_ylabel('value')
            axes[0 + sensordim * col].legend(loc="right")

            axes[1 + sensordim * col].plot(sample_reshape[start:end, col], 'r', label='groundtruth%s' % axis_dict[col])
            axes[1 + sensordim * col].set_xlabel('timestamp')
            axes[1 + sensordim * col].set_ylabel('value')
            axes[1 + sensordim * col].legend(loc="right")

            axes[2 + sensordim * col].plot(pred_reshape[start:end, col], 'b-', label='predict%s' % axis_dict[col])
            axes[2 + sensordim * col].set_xlabel('timestamp')
            axes[2 + sensordim * col].set_ylabel('value')
            axes[2 + sensordim * col].legend(loc="right")

        # Adjust layout
        plt.tight_layout()
        plt.title('%s_reconstruct_epoch_%s.png'%(name, str(epoch)))
        plt.savefig('%s_reconstruct_epoch_%s.png'%(name, str(epoch)))
        plt.close('all')


    umap_3d = UMAP(n_components=3)
    proj_3d_gyro = umap_3d.fit_transform(repre_reshape)
    labeldict_findstr = {-2: 'unknown',
                         0: 'ground_stationary',
                         1: 'stationary',
                         2: 'bathing',
                         3: 'flying_active',
                         4: 'flying_passive',
                         5: 'foraging'}

    label_concat_vote_str = [labeldict_findstr[i] for i in label_concat_vote]

    fig_3d = px.scatter_3d(
        proj_3d_gyro, x=0, y=1, z=2,
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
    fig_3d.write_html('%s_reconstruct_epoch_%s.html'%(name, str(epoch)))

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
        proj_3d_gyro, x=0, y=1, z=2,
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
    fig_3d_dbscan.write_html('%s_reconstruct_epoch_%s_DBSCAN.html'%(name, str(epoch)))

    return

raw_data, labeled_data = [], []
for year in ['2018', '2022']:
    dp = r'D:\code\DeepView\deepview\calculate_results\data\umineko_%s.npy'
    data = np.load(dp % year, allow_pickle=True).item()
    # Access the individual components
    raw_ = data['raw_data']
    labeled_ = data['labeled_data']
    if year=='2018':
        df_raw_2018 = raw_
        df_2018 = labeled_
    # elif year=='2019':
    #     df_raw_2019 = raw_
    #     df_2019 = labeled_
    elif year=='2022':
        df_raw_2022 = raw_
        df_2022 = labeled_
    else:
        print('Error: year not found')
        break

selected_df = pd.concat([df_raw_2018, df_raw_2022], ignore_index=True)
selected_df['filename'] = selected_df['year'].astype(int).astype(str) + '_' + selected_df['animal_tag']
animal_tag_list = ['2018_LB07', '2018_LB08', '2018_LB09', '2018_LB10', '2018_LB11', '2018_LB12', '2018_LB13', '2022_LB02', '2022_LB03', '2022_LB08', '2022_LB09']
selected_df = selected_df[selected_df['filename'].isin(animal_tag_list)]

selected_df['label_id'] = selected_df['label'].map(label_dict)
selected_df['label_id'] = selected_df['label_id'].fillna(-2)

# 选择需要的列
selected_df = selected_df[['acc_x', 'acc_y', 'acc_z', 'pressure', 'label_id']]
fill_selected_df = selected_df.fillna(method='ffill').fillna(method='bfill')  # pressure has nan values
selected_np = fill_selected_df.values

# 分别预处理两种传感器数据
acc_np = fill_selected_df[['acc_x', 'acc_y', 'acc_z']].values
tmp_acc_stand = gaussian_std(acc_np)

fill_selected_df['pressure'] = fill_selected_df['pressure'] - 1013.25  # standard pressure
data = fill_selected_df['pressure'].values

# 计算均值和标准差
mean = np.mean(data)
std_dev = np.std(data)
# 识别超过3个标准差的异常值
outlier_mask = np.abs(data - mean) > 1 * std_dev
# 创建数据副本
data_replaced = np.copy(data)

# 用周围最近的正常值替换异常值
for i in range(len(data)):
    if outlier_mask[i]:
        # 查找前一个正常值
        j = i - 1
        # 查找后一个正常值
        k = i + 1

        # 找到前一个正常值
        while j >= 0 and outlier_mask[j]:
            j -= 1
        # 找到后一个正常值
        while k < len(data) and outlier_mask[k]:
            k += 1

        # 用最近的正常值替换异常值
        if j >= 0 and (k >= len(data) or i - j <= k - i):
            data_replaced[i] = data[j]
        elif k < len(data):
            data_replaced[i] = data[k]
tmp_press_stand = gaussian_std(data_replaced)

selected_np[:, :3] = tmp_acc_stand
selected_np[:, -2] = tmp_press_stand

# plt.figure()
# plt.plot(selected_np[:, -2])
# plt.savefig('pressure.png')
#
# plt.figure()
# plt.plot(selected_np[:, 0],label='x')
# plt.plot(selected_np[:, 1],label='Y')
# plt.plot(selected_np[:, 2],label='Z')
# plt.legend()
# plt.savefig('accel.png')
##########################################

len_sw = 50
device = 'cuda' if torch.cuda.is_available() else 'cpu'
tmp_b = sliding_window(selected_np, len_sw, len_sw)
# concatenate list
data_b = np.transpose(tmp_b[:, :, :-1], (0, 2, 1))  # [B, Len, dim-1] -> [B, dim-1, Len]
label_b = tmp_b[:, :, -1]  # [B, Len]


# concatenate之后是否和原始数据一样
# plt.figure()
# plt.plot(selected_np[:50*300, 3],label='label')
# lbb = []
# for i in range(300):
#     lbb.append(data_b[i, 3, :])
# lb = np.concatenate(lbb)
# # plt.plot(lb, label='labelbatch')
# plt.legend()
# plt.savefig('label.png')



batch_size = 2048
train_set = data_loader_umineko(data_b.astype(float), label_b.astype(int), device=device)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False, drop_last=False)


# # training 1： 将两个autoencoder合并训练，encoder output用对比学习。效果并不好
# ## model initialization
# model = CrossModelAutoencoderContrastiveModel()
# model = model.to(device)
# mse_loss = MSEloss()
# mse_loss = mse_loss.to(device)
# ntxent_loss = NTXentloss()
# ntxent_loss = ntxent_loss.to(device)
#
# learning_rate = 0.01
# optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-3)
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.01)
#
#
# # Training loop
# num_epochs = 1500
# warmup_epochs = 110
# avg_losses, avg_acc_loss, avg_pres_loss, avg_nt_loss = [], [], [], []
# for epoch in tqdm(range(num_epochs), total=num_epochs):
#     losses, acc_losses, pres_losses, nt_losses = [], [], [], []
#     sample_list, represent_list, pred_list, label_list = [], [], [], []
#     model.train()
#     for i, (sample, label) in enumerate(train_loader):
#         sample = sample.to(device=device, dtype=torch.float)
#         acc_feature, acc_out, press_feature, press_out = model(sample[:,:3,:], sample[:,-1:,:])
#         acc_loss = mse_loss(acc_out, sample[:,:3,:])
#         press_loss = mse_loss(press_out, sample[:,-1:,:])
#         contract_loss = ntxent_loss(acc_feature, press_feature)
#         if epoch < warmup_epochs:
#             loss = acc_loss + press_loss/3
#         else:
#             loss = acc_loss + 3*press_loss + 0.01*contract_loss
#         losses.append(loss.item())
#         acc_losses.append(acc_loss.item())
#         pres_losses.append(press_loss.item())
#         nt_losses.append(contract_loss.item())
#
#         acc_feature = acc_feature.detach().cpu().numpy()
#         press_feature = press_feature.detach().cpu().numpy()
#         represent_list.append(np.concatenate([acc_feature, press_feature], axis=1))
#         acc_out = acc_out.detach().cpu().numpy()
#         press_out = press_out.detach().cpu().numpy()
#         pred_list.append(np.concatenate([acc_out, press_out], axis=1))
#         sample_list.append(sample.detach().cpu().numpy())
#         label_list.append(label)
#
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#
#     if epoch%100 == 0:
#         for param_group in optimizer.param_groups:
#             print("Learning Rate:", param_group['lr'])
#         plot_func(represent_list, sample_list, label_list, pred_list, epoch)
#
#
#     avg_losses.append(np.average(losses))
#     acc_losses.append(np.average(acc_losses))
#     pres_losses.append(np.average(pres_losses))
#     nt_losses.append(np.average(nt_losses))
#
#     print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {np.average(losses):.4f}")
#
# print('Saving model at: ' + 'AE_NT_epoch%s' % str(epoch) \
#       + '_datalen%s_' % str(len_sw) + 'accpress.pth')
# torch.save(model.state_dict(), 'AE_NT_epoch%s' % str(epoch) + '_datalen%s_' % str(len_sw) + 'accpress.pth')
#
# # 示例字典
# data = {
#     'avg_losses': avg_losses,
#     'acc_losses': acc_losses,
#     'pres_losses': pres_losses,
#     'nt_losses': nt_losses
# }
#
# # 保存字典到 pickle 文件
# with open('AE_NT_loss.pkl', 'wb') as pickle_file:
#     pickle.dump(data, pickle_file)
#
# # # 读取 pickle 文件
# # with open('AE_NT_loss.pkl', 'rb') as pickle_file:
# #     loaded_data = pickle.load(pickle_file)
# #     print(loaded_data)
# print('Training finished')


# # training 2： 将两个autoencoder分开训练
# acc_model = Autoencoder3d4()
# press_model = Autoencoder1d()
# acc_model = acc_model.to(device)
# press_model = press_model.to(device)
#
# learning_rate = 0.01
# criterion = MSEloss()
# optimizer = optim.Adam(
#     list(acc_model.parameters()) +
#     list(press_model.parameters()),
#     lr=learning_rate, weight_decay=1e-3
# )
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.01)
#
# num_epochs = 1500
# warmup_epochs = 110
# avg_losses, avg_acc_loss, avg_pres_loss, avg_nt_loss = [], [], [], []
# for epoch in tqdm(range(num_epochs), total=num_epochs):
#     losses, acc_losses, pres_losses, nt_losses = [], [], [], []
#     sample_list, represent_list, pred_list, label_list = [], [], [], []
#     acc_model.train()
#     press_model.train()
#     for i, (sample, label) in enumerate(train_loader):
#         sample = sample.to(device=device, dtype=torch.float)
#         # acc_feature, acc_out, press_feature, press_out = model(sample[:,:3,:], sample[:,-1:,:])
#
#         acc_feature, acc_out = acc_model(sample[:,:3,:])
#         press_feature, press_out = press_model(sample[:,-1:,:])
#
#         acc_loss = criterion(acc_out, sample[:,:3,:])
#         press_loss = criterion(press_out, sample[:,-1:,:])
#         # contract_loss = ntxent_loss(acc_feature, press_feature)
#         loss = acc_loss + 3*press_loss
#
#         losses.append(loss.item())
#         acc_losses.append(acc_loss.item())
#         pres_losses.append(press_loss.item())
#         # nt_losses.append(contract_loss.item())
#
#         acc_feature = acc_feature.detach().cpu().numpy()
#         press_feature = press_feature.detach().cpu().numpy()
#         represent_list.append(np.concatenate([acc_feature, press_feature], axis=1))
#         acc_out = acc_out.detach().cpu().numpy()
#         press_out = press_out.detach().cpu().numpy()
#         pred_list.append(np.concatenate([acc_out, press_out], axis=1))
#         sample_list.append(sample.detach().cpu().numpy())
#         label_list.append(label)
#
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#
#     if epoch%100 == 0:
#         for param_group in optimizer.param_groups:
#             print("Learning Rate:", param_group['lr'])
#         plot_func(represent_list, sample_list, label_list, pred_list, epoch, 'AE2')
#
#     avg_losses.append(np.average(losses))
#     acc_losses.append(np.average(acc_losses))
#     pres_losses.append(np.average(pres_losses))
#     # nt_losses.append(np.average(nt_losses))
#
#     print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {np.average(losses):.4f}")
#
# print('Saving model at: ' + 'AE2_epoch%s' % str(epoch) \
#       + '_datalen%s_' % str(len_sw) + 'accpress.pth')
# torch.save(acc_model.state_dict(), 'AE2_epoch%s' % str(epoch) + '_datalen%s_' % str(len_sw) + 'acc1press.pth')
# torch.save(press_model.state_dict(), 'AE2_epoch%s' % str(epoch) + '_datalen%s_' % str(len_sw) + 'accpress1.pth')
#
# data = {
#     'avg_losses': avg_losses,
#     'acc_losses': acc_losses,
#     'pres_losses': pres_losses,
#     # 'nt_losses': nt_losses
# }
#
# # 保存字典到 pickle 文件
# with open('AE2_loss.pkl', 'wb') as pickle_file:
#     pickle.dump(data, pickle_file)


# # training 3： 将两个autoencoder分开训练，encoder output不接projector后用对比学习。
# ## warmup110后加ntloss后效果不错，但是没有接projector，lr是0.001。600epoch的实验测不用ntloss结果。
# acc_model = Autoencoder3d4()
# press_model = Autoencoder1d()
# acc_model = acc_model.to(device)
# press_model = press_model.to(device)
#
# learning_rate = 0.001
# criterion = MSEloss()
# optimizer = optim.Adam(
#     list(acc_model.parameters()) +
#     list(press_model.parameters()),
#     lr=learning_rate, weight_decay=1e-3
# )
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.01)
#
# ntxent_loss = NTXentloss()
# ntxent_loss = ntxent_loss.to(device)
#
# num_epochs = 1000
# warmup_epochs = 10
# name = 'AE2_nt'
# avg_losses, avg_acc_loss, avg_pres_loss, avg_nt_loss = [], [], [], []
# for epoch in tqdm(range(num_epochs), total=num_epochs):
#     losses, acc_losses, pres_losses, nt_losses = [], [], [], []
#     sample_list, represent_list, pred_list, label_list = [], [], [], []
#     acc_model.train()
#     press_model.train()
#     for i, (sample, label) in enumerate(train_loader):
#         sample = sample.to(device=device, dtype=torch.float)
#
#         acc_feature, acc_out = acc_model(sample[:,:3,:])
#         press_feature, press_out = press_model(sample[:,-1:,:])
#
#         acc_loss = criterion(acc_out, sample[:,:3,:])
#         press_loss = criterion(press_out, sample[:,-1:,:])
#         contract_loss = ntxent_loss(acc_feature, press_feature)
#         if epoch < warmup_epochs:
#             loss = acc_loss + 3*press_loss
#         else:
#             loss = acc_loss + 3*press_loss + 1*contract_loss
#
#         losses.append(loss.item())
#         acc_losses.append(acc_loss.item())
#         pres_losses.append(press_loss.item())
#         nt_losses.append(contract_loss.item())
#
#         acc_feature = acc_feature.detach().cpu().numpy()
#         press_feature = press_feature.detach().cpu().numpy()
#         represent_list.append(np.concatenate([acc_feature, press_feature], axis=1))
#         acc_out = acc_out.detach().cpu().numpy()
#         press_out = press_out.detach().cpu().numpy()
#         pred_list.append(np.concatenate([acc_out, press_out], axis=1))
#         sample_list.append(sample.detach().cpu().numpy())
#         label_list.append(label)
#
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#
#     if epoch%100 == 0:
#         for param_group in optimizer.param_groups:
#             print("Learning Rate:", param_group['lr'])
#         plot_func(represent_list, sample_list, label_list, pred_list, epoch, name)
#
#     avg_losses.append(np.average(losses))
#     acc_losses.append(np.average(acc_losses))
#     pres_losses.append(np.average(pres_losses))
#     nt_losses.append(np.average(nt_losses))
#
#     print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {np.average(losses):.4f}")
#
# print('Saving model at: ' + '%s_epoch%s' % (name, str(epoch)) \
#       + '_datalen%s_' % str(len_sw) + 'accpress.pth')
# torch.save(acc_model.state_dict(), '%s_epoch%s' % (name, str(epoch)) + '_datalen%s_' % str(len_sw) + 'acc1press.pth')
# torch.save(press_model.state_dict(), '%s_epoch%s' % (name, str(epoch)) + '_datalen%s_' % str(len_sw) + 'accpress1.pth')
#
# data = {
#     'avg_losses': avg_losses,
#     'acc_losses': acc_losses,
#     'pres_losses': pres_losses,
#     'nt_losses': nt_losses
# }
#
# # 保存字典到 pickle 文件
# with open('%s_loss.pkl'%name, 'wb') as pickle_file:
#     pickle.dump(data, pickle_file)


# # training 4： 使用同一个autoencoder。
# model = Autoencoder4d()
# model = model.to(device)
#
# learning_rate = 0.001
# criterion = MSEloss()
# optimizer = optim.Adam(
#     model.parameters(),
#     lr=learning_rate, weight_decay=1e-3
# )
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.01)
# num_epochs = 1000
# warmup_epochs = 10
# name = 'AE4d'
# avg_losses, avg_acc_loss, avg_pres_loss, avg_nt_loss = [], [], [], []
# for epoch in tqdm(range(num_epochs), total=num_epochs):
#     losses, acc_losses, pres_losses, nt_losses = [], [], [], []
#     sample_list, represent_list, pred_list, label_list = [], [], [], []
#     model.train()
#     for i, (sample, label) in enumerate(train_loader):
#         sample = sample.to(device=device, dtype=torch.float)
#         feature, out = model(sample)
#         loss = criterion(out, sample)
#         losses.append(loss.item())
#
#         feature = feature.detach().cpu().numpy()
#         represent_list.append(feature)
#         out = out.detach().cpu().numpy()
#         pred_list.append(out)
#         sample_list.append(sample.detach().cpu().numpy())
#         label_list.append(label)
#
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#
#     if epoch%100 == 0:
#         for param_group in optimizer.param_groups:
#             print("Learning Rate:", param_group['lr'])
#         plot_func(represent_list, sample_list, label_list, pred_list, epoch, name)
#
#     avg_losses.append(np.average(losses))
#     acc_losses.append(np.average(acc_losses))
#     pres_losses.append(np.average(pres_losses))
#     # nt_losses.append(np.average(nt_losses))
#
#     print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {np.average(losses):.4f}")
#
# print('Saving model at: ' + '%s_epoch%s' % (name, str(epoch)) \
#       + '_datalen%s_' % str(len_sw) + 'accpress.pth')
# torch.save(model.state_dict(), '%s_epoch%s' % (name, str(epoch)) + '_datalen%s_' % str(len_sw) + 'accpress.pth')
#
# data = {
#     'avg_losses': avg_losses,
#     'acc_losses': acc_losses,
#     'pres_losses': pres_losses,
#     # 'nt_losses': nt_losses
# }
#
# # 保存字典到 pickle 文件
# with open('%s_loss.pkl'%name, 'wb') as pickle_file:
#     pickle.dump(data, pickle_file)


# training 5： 单独训练AE后使用weight获得encoder output，然后concatenate
# 效果最好。
learning_rate = 0.001
press_model = Autoencoder1d()
press_model = press_model.to(device)
optimizer = optim.Adam(press_model.parameters(),
                       lr=learning_rate,
                       weight_decay=1e-3)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                       'min',
                                                       patience=5,
                                                       factor=0.01)

full_model_path = r'D:\code\DeepView\deepview\calculate_results\data\umineko\AE_reconstruct_epoch499_datalen50_pressure.pth'
if torch.cuda.is_available():
    press_model.load_state_dict(torch.load(full_model_path,
                                           weights_only=False))
else:
    press_model.load_state_dict(torch.load(full_model_path,
                                           weights_only=False,
                                           map_location=torch.device('cpu')))

train_set = data_loader_umineko(data_b[:,-1: :].astype(float), label_b.astype(int), device=device)
train_loader_pres = DataLoader(train_set, batch_size=batch_size, shuffle=False, drop_last=False)

press_repres_list, press_sample_list, press_pred_list, label_list = \
    AE_eval_time_series(train_loader_pres, press_model, device)
# plot_reconstruction_result('pressure',
#                            press_repres_list, press_sample_list,
#                            press_pred_list, label_list,
#                            'train_epoch_%s' % str(700))


acc_model = Autoencoder3d4()
acc_model = acc_model.to(device)
full_model_path = r'D:\code\DeepView\deepview\calculate_results\data\umineko\AE_reconstruct_epoch500_datalen50_accel.pth'
if torch.cuda.is_available():
    acc_model.load_state_dict(torch.load(full_model_path,
                                         weights_only=False))
else:
    acc_model.load_state_dict(torch.load(full_model_path,
                                         weights_only=False,
                                         map_location=torch.device('cpu')))

train_set = data_loader_umineko(data_b[:,:3 :].astype(float), label_b.astype(int), device=device)
train_loader_acc = DataLoader(train_set, batch_size=batch_size, shuffle=False, drop_last=False)

acc_repres_list, acc_sample_list, acc_pred_list, label_list = \
    AE_eval_time_series(train_loader_acc, acc_model, device)
# plot_reconstruction_result('accel',
#                            acc_repres_list, acc_sample_list,
#                            acc_pred_list, label_list,
#                            'train_epoch_%s' % str(500))


repres_list, sample_list, pred_list = [], [], []
for i in range(len(label_list)):
    repres_list.append(np.concatenate((acc_repres_list[i],
                                     press_repres_list[i]), axis=1))
    sample_list.append(np.concatenate((acc_sample_list[i],
                                     press_sample_list[i]), axis=1))
    pred_list.append(np.concatenate((acc_pred_list[i],
                                   press_pred_list[i]), axis=1))

plot_func(repres_list, sample_list, label_list, pred_list,
          '0', 'concatenate2')
print('')

### 使用dbscan聚类，然后判断聚类结果和label的一致性（只要有聚类和label重合或包含label即可）
# 两个encoder重新训练

CLIP_model = CrossModelAutoencoderContrastiveModel()

# 从自动编码器中获取编码器权重
acc_state_dict = acc_model.feature_extractor.state_dict()
pre_state_dict = press_model.feature_extractor.state_dict()

# 将编码器权重加载到分类器的特征提取器中
CLIP_model.acc_feature_extractor.load_state_dict(acc_state_dict)
CLIP_model.press_feature_extractor.load_state_dict(pre_state_dict)

CLIP_model = CLIP_model.to(device)
learning_rate = 0.0001
optimizer = optim.Adam(CLIP_model.parameters(), lr=learning_rate, weight_decay=1e-3)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.01)
#
num_epochs = 70
warmup_epochs = 1
name = 'clip2'
avg_losses, avg_acc_loss, avg_pres_loss, avg_nt_loss = [], [], [], []
for epoch in tqdm(range(num_epochs), total=num_epochs):
    losses, acc_losses, pres_losses, nt_losses = [], [], [], []
    sample_list, represent_list, pred_list, label_list = [], [], [], []
    CLIP_model.train()
    for i, (sample, label) in enumerate(train_loader):
        sample = sample.to(device=device, dtype=torch.float)
        acc_feature, acc_projout, press_feature, press_projout = (
            CLIP_model(sample[:,:3,:], sample[:,-1:,:]))

        loss = CLIP_model.calculate_loss(acc_projout, press_projout)
        losses.append(loss.item())

        acc_feature = acc_feature.detach().cpu().numpy()
        press_feature = press_feature.detach().cpu().numpy()
        represent_list.append(np.concatenate([acc_feature, press_feature], axis=1))
        sample_list.append(sample.detach().cpu().numpy())
        label_list.append(label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if epoch%10 == 0:
        for param_group in optimizer.param_groups:
            print("Learning Rate:", param_group['lr'])
        plot_func(represent_list, sample_list,
                  label_list, pred_list, epoch, name)

    avg_losses.append(np.average(losses))

    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {np.average(losses):.4f}")

print('Saving model at: ' + '%s_epoch%s' % (name, str(epoch)) \
      + '_datalen%s_' % str(len_sw) + 'accpress.pth')
torch.save(CLIP_model.state_dict(),
           '%s_epoch%s' % (name, str(epoch)) + '_datalen%s_' % str(len_sw) + 'accpress.pth')
    #
# 示例字典
data = {
    'avg_losses': avg_losses,
    'acc_losses': acc_losses,
    'pres_losses': pres_losses,
    'nt_losses': nt_losses
}

# 保存字典到 pickle 文件
with open('AE_NT_loss.pkl', 'wb') as pickle_file:
    pickle.dump(data, pickle_file)