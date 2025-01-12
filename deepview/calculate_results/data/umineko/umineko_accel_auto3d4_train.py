import pandas as pd
import numpy as np
import random
import torch
from tqdm import tqdm
from torch import optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from umap import UMAP
import plotly.express as px
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
    # torch,
    AE_eval_time_series,
    AE_train_time_series_resnet,
    # np,
    # tqdm
    Autoencoder3d4, # debug purpose
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

device = 'cuda'
len_sw = 50
# sensor_type = 'pressure'
sensor_type = 'accel'

activity_label = True
if activity_label:
    selected_columns = ['acc_x', 'acc_y', 'acc_z', 'label_id', 'back']  # without timestamps
    # selected_columns = ['pressure', 'label_id', 'label_id']  # without timestamps
  # without timestamps，activity labels
else:
    # selected_df['animal_tag_id'] = pd.factorize(selected_df['animal_tag'])[0]
    selected_columns = ['acc_x', 'acc_y', 'acc_z', 'file_id', 'label_id']  # without timestamps
    # selected_columns = ['pressure', 'file_id', 'label_id']  # without timestamps
  # 将animal tag作为label，判断鸟的embedding是否分开

# # tmp = selected_df[['GPS_velocity', 'GPS_bearing', 'label_id', 'acc_x']]
# selected_df['pressure'] = selected_df['pressure'] - 1013.25  # standard pressure
# tmp = selected_df[selected_columns]
# df_combined_fill = tmp.fillna(method='ffill').fillna(method='bfill')  # pressure has nan values
# selected_np = df_combined_fill.values

# data = selected_np[:, 0]
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
# selected_np[:, 0] = data_replaced
# # 绘图
# plt.figure(figsize=(10, 6))
# # plt.plot(data, 'bo-', label='Original Data')  # 原始数据用蓝色圆圈表示
# # plt.plot(np.where(outlier_mask)[0], data[outlier_mask], 'ro', label='Outliers')  # 异常值用红色圆圈表示
# plt.plot(data_replaced, 'go-', label='Data with Replacements')  # 替换后的数据用绿色圆圈表示
#
# plt.title('Data with Outliers Replaced by Nearest Neighbors')
# plt.xlabel('Index')
# plt.ylabel('Value')
# plt.legend()
# plt.grid(True)
# plt.savefig('a.png')
# print('')
# selected_np = selected_df[selected_columns].values

############################################
selected_np = selected_df[selected_columns].values
data_np = selected_np[:, :-2]
tmp_b_stand = gaussian_std(data_np)
selected_np[:, :-2] = tmp_b_stand
##########################################

tmp_b = sliding_window(selected_np[:, :-1], len_sw, len_sw)
# concatenate list
data_b = np.transpose(tmp_b[:, :, :-1], (0, 2, 1))  # [B, Len, dim-1] -> [B, dim-1, Len]
label_b = tmp_b[:, :, -1]  # [B, Len]

batch_size = 2048
train_set_r = data_loader_umineko(data_b.astype(float), label_b.astype(int), device=device)
train_loader = DataLoader(train_set_r, batch_size=batch_size,
                          shuffle=False, drop_last=False)

# model = Autoencoder1d()
model = Autoencoder3d4()
model = model.to(device)
criterion = MSEloss()

criterion = criterion.to(device)

learning_rate = 0.01
# training
start_epoch = 0
num_epochs = 1500

optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-3)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.01)
# learning rate update per epoch
#
# #--------------------load weights-------------------------------
# full_model_path = r'D:\code\DeepView\deepview\calculate_results\data\umineko\AE_reconstruct_epoch999_datalen50_accel.pth'
# if torch.cuda.is_available():
#     model.load_state_dict(torch.load(full_model_path, weights_only=False))
# else:
#     model.load_state_dict(torch.load(full_model_path, weights_only=False, map_location=torch.device('cpu')))
#
# representation_list, sample_list, pred_list, label_list = \
#     AE_eval_time_series(train_loader, model, device)
# repre_concat = np.concatenate(representation_list)
# repre_reshape = repre_concat.reshape(repre_concat.shape[0], -1)
#
# umap_3d = UMAP(n_components=3)
# proj_3d_gyro = umap_3d.fit_transform(repre_reshape)
#
# label_concat = np.concatenate(label_list)
# label_concat_vote = majority_value(label_concat)
# labeldict_findstr = {-2: 'unknown',
#                      0: 'stationary',
#                      1: 'bathing',
#                      7: 'forgaing_insect',
#                      2: 'flying',
#                      3: 'flight_cruising',
#                      4: 'flight_take_off',
#                      5: 'foraging',
#                      6: 'foraging_fish_poss',
#                      8: 'foraging_non-fish',
#                      9: 'foraging_steal',
#                      10: 'foraging_dive',
#                      11: 'surface_seizing',
#                      12: 'body_shaking',
#                      13: 'ground_active'}
# label_concat_vote_str = [labeldict_findstr[i] for i in label_concat_vote]
#
# fig_3d = px.scatter_3d(
#     proj_3d_gyro, x=0, y=1, z=2,
#     color=label_concat_vote_str,
#     labels={'color': 'activity'},
#     color_discrete_map={'unknown': 'lightgrey'},
# )
# # Reduce marker size for all points
# for trace in fig_3d.data:
#     trace.marker.size = 6  # Adjust the size to your preference (e.g., 6)
#
# # Update transparency for traces where activity is '-2.0'
# fig_3d.for_each_trace(lambda trace: trace.update(marker=dict(opacity=0.5)) if trace.name == '-2.0' else ())
# fig_3d.update_traces(marker=dict(line=dict(width=0)))  # remove boundary of point
# fig_3d.write_html("3dt-623.html")
#
# #############################################################
# umap_2d = UMAP(n_components=2)
# proj_2d_gyro = umap_2d.fit_transform(repre_reshape)
# fig_2d = px.scatter(
#     proj_2d_gyro, x=0, y=1,
#     color=label_concat_vote_str,
#     labels={'color': 'activity'},
#     color_discrete_map={'unknown': 'lightgrey'},
# )
# # Reduce marker size for all points
# for trace in fig_2d.data:
#     trace.marker.size = 6
#
# # Update transparency for traces where activity is '-2.0'
# fig_2d.for_each_trace(lambda trace: trace.update(marker=dict(opacity=0.5)) if trace.name == '-2.0' else ())
# fig_2d.update_traces(marker=dict(line=dict(width=0)))  # remove boundary of point
# fig_2d.write_html("2dt-623.html")
#
# #--------------------load weights-------------------------------
#

training_loss = []
for epoch in tqdm(range(start_epoch, num_epochs)):

    losses = AE_train_time_series_resnet(train_loader, model, criterion, optimizer, epoch, scheduler, device)
    training_loss.append(np.average(losses))
    if (epoch % 100 == 0) or (epoch == num_epochs - 1):
        # Print the learning rate
        for param_group in optimizer.param_groups:
            print("Learning Rate:", param_group['lr'])
        #     print('loss of the ' + str(epoch) + '-th training epoch is :' + losses.__str__())
        # reconstruction result
        representation_list, sample_list, pred_list, label_list = \
            AE_eval_time_series(train_loader, model, device)
        plot_reconstruction_result(sensor_type, representation_list, sample_list, pred_list, label_list, 'train_epoch_%s' % str(epoch))
    if epoch == 500:
        print('Saving model at: ' + 'AE_reconstruct_epoch%s' % str(epoch) \
              + '_datalen%s_' % str(len_sw) + sensor_type + '.pth')
        torch.save(model.state_dict(),
                   'AE_reconstruct_epoch%s' % str(epoch) + '_datalen%s_' % str(len_sw) + sensor_type + '.pth')

print('Saving model at: ' + 'AE_reconstruct_epoch%s' % str(epoch) \
      + '_datalen%s_' % str(len_sw) + sensor_type + '.pth')
torch.save(model.state_dict(), 'AE_reconstruct_epoch%s' % str(epoch) + '_datalen%s_' % str(len_sw) + sensor_type + '.pth')
# T_max=50, lr=0.0001, epoch=623