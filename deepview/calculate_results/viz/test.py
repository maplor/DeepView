import os

import matplotlib.pyplot as plt
# import numpy as np
# import os
# import matplotlib.pyplot as plt
# import pickle
# import pandas as pd
# from torch.utils.data import Dataset, DataLoader
# import torch
# import torch.nn as nn
#
# from scipy.interpolate import interp1d
# import math
# from datetime import datetime
# from tqdm import tqdm
# from sklearn.manifold import TSNE
#
# import pickle
import torch.optim as optim
# from utils import *
# from hubconf import load_weights
# from sslearning.models.accNet import Resnet
from umap import UMAP
import plotly.express as px
# from importlib.metadata import version, PackageNotFoundError

import pickle
import numpy as np
from tqdm import tqdm
import torch

from deepview.calculate_results.data.umineko_data import (
    read_umineko_data,
read_umineko_path,
extract_data_from_year_back,
label_dict,
get_accel_batch_data,
)

from deepview.calculate_results.models.utils import (
    Resnet,
load_weights,
freeze_feature_extractor,
plot_reconstruction_result,
MSEloss,
# torch,
AE_eval_time_series,
AE_train_time_series_resnet,
AE_train_time_series,
majority_value,
# np,
adjust_learning_rate,
# tqdm
Autoencoder3d,
Autoencoder1d,
Autoencoder2d,
Autoencoder3d_transf,
sliding_window,
data_loader_umineko
)
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader

def standardization(input_array, mean, std, bias=0.0):
    return ((input_array - mean) / np.maximum(std, 10 ** -5)) + bias

back_label_path = r'D:\logbot-data\BioTaggerData\masterLabelsByOtsuka\animal_id.csv'
umi_root_path = r'D:\logbot-data\BioTaggerData\export\raw\umineko\v.1.0.0'
file_paths = read_umineko_path(umi_root_path)

year = '2018'
data_path = r'D:\code\DeepView\deepview\calculate_results\data\umineko_%s.npy' % year
if os.path.exists(data_path):
    tmp = np.load(data_path, allow_pickle='TRUE').item()
    df_list = tmp['raw_data']
    df_clean_list = tmp['labeled_data']

selected_df, selected_clean_df = (
    extract_data_from_year_back(df_list, df_clean_list, 1))

selected_df['label_id'] = selected_df['label'].map(label_dict)
# selected_clean_df['label_id'] = selected_clean_df['label'].map(label_dict)

# 将df_list没有标签的位置赋值-2，标记灰色，其他label正常标记。观察是否有label的标记能在不同灰色cluster中
# Replace NaN with -2 in column A
selected_df['label_id'] = selected_df['label_id'].fillna(-2)

device = 'cuda'
len_sw = 100
sensor_type = 'accel'
selected_columns = ['acc_x', 'acc_y', 'acc_z',
                        'label_id']  # without timestamps

selected_np = selected_df[selected_columns].values

# datap = r'D:\code\DeepView\deepview\calculate_results\viz\pamap2\pamap2.npy'
# selected_np = np.load(datap)
data_np = selected_np[:,:-1]
print(data_np.shape)
mean = np.mean(data_np, axis=0)
std = np.std(data_np, axis=0)
print("mean:",mean)
print("std:",std)
tmp_b_stand = standardization(data_np, mean, std)

selected_np[:,:-1] = tmp_b_stand
# print(selected_np.shape)

tmp_b = sliding_window(selected_np, len_sw, len_sw)
# print(tmp_b.shape)

# concatenate list
data_b = np.transpose(tmp_b[:, :, :-1], (0, 2, 1))  # [B, Len, dim-1] -> [B, dim-1, Len]
label_b = tmp_b[:, :, -1]  # [B, Len]

batch_size = 512
train_set_r = data_loader_umineko(data_b, label_b, device=device)
train_loader = DataLoader(train_set_r, batch_size=batch_size,
                          shuffle=False, drop_last=False)

model = Autoencoder3d()
# model = Resnet(
#         output_size=15,
#         is_reconst=True,
#         len_sw= len_sw,
#         # is_simclr=True,
#         # is_eva=True,
#         resnet_version=1,
#                )
model = model.to(device)
# checkpoint/  )
# model = freeze_feature_extractor(model)

criterion = MSEloss()
criterion = criterion.to(device)

learning_rate = 0.001
optimizer = optim.Adam(
    model.parameters(), lr=learning_rate, amsgrad=True
)
optimizer = optim.Adam(
            model.parameters(), lr=learning_rate, amsgrad=True
        )
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

# lambda1 = lambda epoch: 1.0**epoch
# scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
# Initialize the ReduceLROnPlateau scheduler


 # training
start_epoch = 0
num_epochs = 1000
# lr = 0.0001
training_loss = []
for epoch in tqdm(range(start_epoch, num_epochs)):

    # learning_rate = adjust_learning_rate(
    #     learning_rate, optimizer, epoch, p_scheduler='cosine', p_epochs=num_epochs)
    # print('learning rate is %s'%str(learning_rate))

    losses = AE_train_time_series_resnet(train_loader, model, criterion, optimizer, epoch, scheduler, device)
    training_loss.append(losses)
    # if (epoch % 10 == 0) or (epoch == num_epochs - 1):
    #     print('loss of the ' + str(epoch) + '-th training epoch is :' + losses.__str__())

    if epoch % 100 == 0:
        representation_list, sample_list, pred_list, label_list = \
            AE_eval_time_series(train_loader, model, device)

        plot_reconstruction_result(representation_list, sample_list, pred_list, label_list)
        print('')

plt.plot(losses)
plt.show()
plt.close()
# #
# # print('Saving model at: ' + 'AE_reconstruct_epoch%s' % str(epoch) \
# #       + '_datalen%s_' % str(len_sw) +sensor_type+ '.pth')
# # torch.save(model.state_dict(), 'AE_reconstruct_epoch%s' % str(epoch) + \
# #            '_datalen%s_' % str(len_sw) +sensor_type+ '.pth')
# # reconstruction result
# representation_list, sample_list, pred_list, label_list = \
#             AE_eval_time_series(train_loader, model, device)
#
# plot_reconstruction_result(representation_list, sample_list, pred_list, label_list)
# print('')