import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from tqdm import tqdm
# from dataloader_f import *
# import json
import torch
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

class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)

        # 如果输入和输出通道数不同，需要1x1卷积进行映射
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)  # 残差连接
        out = F.relu(out)
        return out

class ResNet1D(nn.Module):
    def __init__(self, block, layers):
        super(ResNet1D, self).__init__()
        self.in_channels = 16

        self.conv1 = nn.Conv1d(in_channels=3, out_channels=16, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm1d(16)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 16, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 128, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool1d(1)

        # self.fc = nn.Linear(128, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride):
        layers = []
        layers.append(block(self.in_channels, out_channels, stride))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):

        # 输入形状 (batch_size, 3, 600)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)  # (batch_size, 16, 150)

        x = self.layer1(x)  # (batch_size, 16, 150)
        x = self.layer2(x)  # (batch_size, 32, 75)
        x = self.layer3(x)  # (batch_size, 64, 38)
        x = self.layer4(x)  # (batch_size, 128, 19)

        x = self.avgpool(x)  # (batch_size, 128, 1)
        x = torch.flatten(x, 1)  # (batch_size, 128)

        return x
def ResNet18_1D_feature():
    return ResNet1D(ResidualBlock1D, [2, 2, 2, 2])
def ResNet34_1D_feature():
    return ResNet1D(ResidualBlock1D, [3, 4, 6, 3])


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        # self.lstm = nn.LSTM(input_size=3, hidden_size=8, num_layers=3, batch_first=True)

        self.resnet_traj = ResNet18_1D_feature()

        self.output_mlp = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 64)
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for layer in self.output_mlp:
            if isinstance(layer, nn.Linear):
                init.kaiming_normal_(layer.weight, nonlinearity='relu')
                if layer.bias is not None:
                    init.zeros_(layer.bias)

    def forward(self, traj_img):
        # traj_lstm, _ = self.lstm(traj)
        # traj = torch.cat([traj, traj_lstm], dim=-1).view(-1, 600 * (3 + 8))
        # traj = traj.view(-1, 180 * (3))
        x = self.resnet_traj(traj_img)

        # x = torch.cat([traj, traj_feat], dim=1)

        latent = self.output_mlp(x)
        return latent


class Autoencoder(nn.Module):
    def __init__(self, len_sw):
        super(Autoencoder, self).__init__()

        self.len_sw = len_sw
        # 编码器
        self.encoder = Encoder()

        # 解码器
        self.decoder = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, self.len_sw * 3),
            nn.Sigmoid()
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for layer in self.decoder:
            if isinstance(layer, nn.Linear):
                init.kaiming_normal_(layer.weight, nonlinearity='relu')
                if layer.bias is not None:
                    init.zeros_(layer.bias)

    def forward(self, traj_img):
        traj_img = traj_img.permute(0, 2, 1)
        latent = self.encoder(traj_img)
        x = self.decoder(latent)
        x = x.reshape(x.shape[0], self.len_sw, -1)
        return latent, x

    def get_latent_vector(self, x):
        return self.encoder(x)

# prepare train loader of accelerometer
with open('../data.pkl', 'rb') as f:
    df_ready = pickle.load(f)
sensor_type = 'accel'
batch_size = 512
from deepview.utils.device import get_device
device = get_device()
len_sw = 180
selected_columns = ['acc_x', 'acc_y', 'acc_z',
                   'label_id']  # without timestamps
tmp_b = sliding_window(df_ready[selected_columns], len_sw)

# concatenate list
data_b = tmp_b[:,:,:-1]  # [B, Len, dim-1]
label_b = tmp_b[:,:,-1]  # [B, Len]

train_set_r = data_loader_umineko(data_b, label_b, device=device)
train_loader = DataLoader(train_set_r, batch_size=batch_size,
                           shuffle=False, drop_last=False)

### %------------

out_channels = 32
from deepview.utils.device import get_device
device = get_device()
model = Autoencoder(len_sw)
model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(),
                                     weight_decay=0.000001,
                                     lr=0.0001)
criterion = MSEloss()
criterion = criterion.to(device)
# training
start_epoch = 0
num_epochs = 10000
lr = 0.0001
# i = 1  # 应该一只鸟一个trainloader，暂时全拼接到一起
# print('Starting %s-th bird data' % str(i))
for epoch in tqdm(range(start_epoch, num_epochs)):

    model.train()
    losses = []
    for i, (sample, label) in enumerate(train_loader):
        sample = sample.to(dtype=torch.float)

        x_encoded, output = model(sample)  # x_encoded.shape=batch512,outchannel128,len13
        loss = criterion(sample, output)
        losses.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch % 1000 == 0) or (epoch == num_epochs-1):
        print('loss of the ' + str(epoch) + '-th training epoch is :' + str(np.average(losses)))
        print('Saving model at: ' + 'resnet_epoch%s' % str(epoch)\
              + '_datalen%s_' % str(len_sw) + sensor_type + '.pth')
        torch.save(model.state_dict(), 'resnet_epoch%s' % str(epoch) +\
                   '_datalen%s_' % str(len_sw) + sensor_type + '.pth')

print('Saving model at: ' + 'resnet_epoch%s' % str(epoch)\
      + '_datalen%s_' % str(len_sw) + sensor_type + '.pth')
torch.save(model.state_dict(), 'resnet_epoch%s' % str(epoch) +\
           '_datalen%s_' % str(len_sw) + sensor_type + '.pth')


representation_list, sample_list, pred_list, label_list = \
            AE_eval_time_series(train_loader, model, device)

# tsne latent representation to shape=(2, len) PCA降维到形状为 (2, len)
repre_concat = np.concatenate(representation_list)
repre_reshape = repre_concat.reshape(repre_concat.shape[0], -1)
repre_tsne = reduce_dimension_with_tsne(repre_reshape)

sample_concat = np.concatenate(sample_list)
pred_concat = np.concatenate(pred_list)
sample_reshape = sample_concat.reshape(-1, sample_concat.shape[-1])
pred_reshape = pred_concat.reshape(-1, pred_concat.shape[-1])

plt.figure(1)
plt.plot(sample_reshape[:100, 0], 'r')
plt.plot(pred_reshape[:100, 0], 'b-.')
plt.savefig('resnet.png')
plt.close()



