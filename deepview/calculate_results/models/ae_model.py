import torch
import torch.nn as nn

class Act2Vec(nn.Module):
    """
    Implementation of Act2Vec model as described in the paper <https://arxiv.org/abs/1907.05597>.
    """
    def __init__(self, units, input_dim):
        super(Act2Vec, self).__init__()
        self.layer1 = nn.GRU(input_size=input_dim[-1]*input_dim[1], hidden_size=units, batch_first=True, bidirectional=True)
        self.units = units * 2  # Adjusted for bidirectional output
        self.repeat_length = input_dim[1]  # Number of times to repeat the vector
        self.layer3 = nn.GRU(input_size=self.units, hidden_size=units, batch_first=True)
        self.layer4 = nn.Linear(units, input_dim[2])

    def forward(self, inputs):
        inputs = inputs.reshape(inputs.shape[0], -1)
        encoder, state = self.layer1(inputs)
        x = self._repeat_vector(encoder, self.repeat_length)
        x, _ = self.layer3(x)
        x = self.layer4(x)
        return encoder, x

    def encoder(self, inputs):
        x, _ = self.layer1(inputs)
        return x

    def _repeat_vector(self, x, repeat_length):
        """
        Custom method to repeat the vector along the time dimension.
        """
        # Expand the dimensions and repeat
        batch_size = x.size(0)
        x = x.unsqueeze(1)  # Add a new dimension for time
        repeated_x = x.repeat(1, repeat_length, 1)  # Repeat along the time dimension
        return repeated_x


### %------------

from deepview.clustering_pytorch.datasets.factory import sliding_window
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
def AE_eval_time_series(train_loader, model, device):
    model.eval()

    representation_list = []
    sample_list, timestamp_list, label_list, pred_list, timestr_list, flag_list = [], [], [], [], [], []
    for i, (sample, label) in enumerate(train_loader):
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

def reduce_dimension_with_tsne(array, method='tsne'):
    # tsne or pca
    tsne = TSNE(n_components=2)  # 创建TSNE对象，降维到2维
    reduced_array = tsne.fit_transform(array)  # 对数组进行降维
    return reduced_array


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

# # prepare train loader of accelerometer
# with open('data.pkl', 'rb') as f:
#     df_ready = pickle.load(f)
# sensor_type = 'accel'
# batch_size = 512
# device = 'cuda'
# len_sw = 180
# selected_columns = ['acc_x', 'acc_y', 'acc_z',
#                    'label_id']  # without timestamps
# tmp_b = sliding_window(df_ready[selected_columns], len_sw)
#
# # concatenate list
# data_b = tmp_b[:,:,:-1]  # [B, Len, dim-1]
# label_b = tmp_b[:,:,-1]  # [B, Len]
#
# train_set_r = data_loader_umineko(data_b, label_b, device=device)
# train_loader = DataLoader(train_set_r, batch_size=batch_size,
#                            shuffle=False, drop_last=False)
#
# ### %------------
#
# out_channels = 32
# device = 'cuda'
# model = Act2Vec(out_channels, input_dim=(4503, 180, 3))
# model = model.to(device)
#
# optimizer = torch.optim.Adam(model.parameters(),
#                                      weight_decay=0.000001,
#                                      lr=0.0001)
# criterion = MSEloss()
# criterion = criterion.to(device)
# # training
# start_epoch = 0
# num_epochs = 40000
# lr = 0.0001
# # i = 1  # 应该一只鸟一个trainloader，暂时全拼接到一起
# # print('Starting %s-th bird data' % str(i))
# for epoch in tqdm(range(start_epoch, num_epochs)):
#
#     model.train()
#     losses = []
#     for i, (sample, label) in enumerate(train_loader):
#         sample = sample.to(dtype=torch.float)
#
#         x_encoded, output = model(sample)  # x_encoded.shape=batch512,outchannel128,len13
#         loss = criterion(sample, output)
#         losses.append(loss.item())
#
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#
#     if (epoch % 100 == 0) or (epoch == num_epochs-1):
#         print('loss of the ' + str(epoch) + '-th training epoch is :' + str(np.average(losses)))
#         # print('Saving model at: ' + 'AE_GRU_epoch%s' % str(epoch) \
#         #       + '_datalen%s_' % str(len_sw) + sensor_type + '.pth')
#         # torch.save(model.state_dict(), 'AE_GRU_epoch%s' % str(epoch) + \
#         #            '_datalen%s_' % str(len_sw) + sensor_type + '.pth')
#
#
# print('Saving model at: ' + 'AE_GRU_epoch%s' % str(epoch)\
#       + '_datalen%s_' % str(len_sw) + sensor_type + '.pth')
# torch.save(model.state_dict(), 'AE_GRU_epoch%s' % str(epoch) +\
#            '_datalen%s_' % str(len_sw) + sensor_type + '.pth')
#
#
# representation_list, sample_list, pred_list, label_list = \
#             AE_eval_time_series(train_loader, model, device)
#
# # tsne latent representation to shape=(2, len) PCA降维到形状为 (2, len)
# repre_concat = np.concatenate(representation_list)
# repre_reshape = repre_concat.reshape(repre_concat.shape[0], -1)
# repre_tsne = reduce_dimension_with_tsne(repre_reshape)
#
# sample_concat = np.concatenate(sample_list)
# pred_concat = np.concatenate(pred_list)
# sample_reshape = sample_concat.reshape(-1, sample_concat.shape[-1])
# pred_reshape = pred_concat.reshape(-1, pred_concat.shape[-1])
#
# plt.figure(1)
# plt.plot(sample_reshape[:100, 0], 'r')
# plt.plot(pred_reshape[:100, 0], 'b-.')
# plt.savefig('a.png')
# plt.close()