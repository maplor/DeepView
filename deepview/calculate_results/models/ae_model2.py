import torch
import torch.nn as nn



class CNN_AE_encoder(nn.Module):
    def __init__(self, n_channels, out_channels=128):
        super(CNN_AE_encoder, self).__init__()

        self.n_channels = n_channels * 2

        self.linear = nn.Linear(n_channels, self.n_channels)
        kernel_size = 5
        self.e_conv1 = nn.Sequential(nn.Conv2d(self.n_channels, 32,
                                               (1, kernel_size), bias=False,
                                               padding=(0, kernel_size // 2)),
                                     nn.BatchNorm2d(32),
                                     nn.PReLU())  # Tanh is MoIL paper
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2, padding=0, return_indices=True)
        self.dropout = nn.Dropout(0.35)  # probability of samples to be zero

        self.e_conv2 = nn.Sequential(nn.Conv2d(32, 64,
                                               (1, kernel_size), bias=False,
                                               padding=(0, kernel_size // 2)),
                                     nn.BatchNorm2d(64),
                                     nn.PReLU())
        self.pool2 = nn.MaxPool1d(kernel_size=4, stride=2, padding=0, return_indices=True)

        self.e_conv3 = nn.Sequential(nn.Conv2d(64, out_channels,
                                               (1, kernel_size), bias=False,
                                               padding=(0, kernel_size // 2)),
                                     nn.BatchNorm2d(out_channels),
                                     nn.PReLU())
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2, padding=0, return_indices=True)

        self.out_samples = 25
        self.out_dim = out_channels
        self.units = 32

        self.layer1 = nn.GRU(input_size=out_channels*22,
                             hidden_size=self.units,
                             batch_first=True,
                             bidirectional=True)
        self.layer3 = nn.GRU(input_size=self.units*2,
                             hidden_size=self.units,
                             batch_first=True,
                             bidirectional=True)


        return

    def forward(self, x_input):  # x(batch,len180,dim6)
        x = self.linear(x_input)
        x = x.unsqueeze(2).permute(0, 3, 2, 1)  # outx(batch,dim,1,len)
        x1 = self.e_conv1(x)  # x1(batch,64,1,180)
        x1 = x1.squeeze(2)  # batch,32,180
        x, indice1 = self.pool1(x1)  # (batch,32,90)len减半,最后一维maxpool
        x = x.unsqueeze(2)
        x = self.dropout(x)
        # ---------
        x2 = self.e_conv2(x)  # batch,64,90
        x2 = x2.squeeze(2)
        x, indice2 = self.pool2(x2)
        x = x.unsqueeze(2)  # batch,64,45
        x = self.dropout(x)
        # ---------
        x3 = self.e_conv3(x)  # batch,128,45
        x3 = x3.squeeze(2)
        x3, indice3 = self.pool3(x3)  # xencoded(batch,128,15)
        # x_encoded # batch,128,15

        # lstm
        inputs = x3.reshape(x3.shape[0], -1) # b,32,22
        x4, _ = self.layer1(inputs)
        x_encoded, _ = self.layer3(x4)

        return x_encoded, [indice1, indice2, indice3]


class CNN_AE_decoder(nn.Module):
    def __init__(self, n_channels, out_channels=128):
        super(CNN_AE_decoder, self).__init__()

        self.n_channels = n_channels
        kernel_size = 5
        self.unpool1 = nn.MaxUnpool1d(kernel_size=2, stride=2, padding=0)
        self.d_conv1 = nn.Sequential(nn.ConvTranspose2d(out_channels, 64,
                                                        kernel_size=(1, kernel_size),
                                                        bias=False,
                                                        padding=(0, kernel_size // 2)),
                                     nn.BatchNorm2d(64),
                                     nn.PReLU())

        self.unpool2 = nn.MaxUnpool1d(kernel_size=4, stride=2, padding=0)
        self.d_conv2 = nn.Sequential(nn.ConvTranspose2d(64, 32,
                                                        kernel_size=(1, kernel_size),
                                                        stride=1, bias=False,
                                                        padding=(0, kernel_size // 2)),
                                     nn.BatchNorm2d(32),
                                     nn.PReLU())

        self.unpool3 = nn.MaxUnpool1d(kernel_size=2, stride=2, padding=0)
        self.d_conv3 = nn.Sequential(nn.ConvTranspose2d(32, n_channels,
                                                        kernel_size=(1, kernel_size),
                                                        stride=1, bias=False,
                                                        padding=(0, kernel_size // 2)),
                                     nn.BatchNorm2d(n_channels),
                                     nn.PReLU())

        self.linear = nn.Linear(n_channels, 3)
        if n_channels == 3:  # acc,gyro, where data length is 90
            self.reshapel = nn.Linear(89, 90)
        else:
            self.reshapel = nn.Linear(29, 30)

        self.units = 32
        out_channels = 128

        self.layer1 = nn.GRU(input_size=32,
                             hidden_size=16,
                             batch_first=True,
                             bidirectional=True)
        self.layer3 = nn.GRU(input_size=self.units*2,
                             hidden_size=self.units,
                             batch_first=True)
        return

    def forward(self, x, encode_indices):  # x_encoded(batch, 128, 25)

        # lstm
        x4, _ = self.layer3(x)
        x4 = x4.unsqueeze(1)
        repeated_x = x4.repeat(1, 22, 1)

        x_encoded, _ = self.layer1(repeated_x)
        x_encoded = x_encoded.permute(0, 2, 1)

        x = self.unpool1(x_encoded, encode_indices[-1])  # out(batch, 64, 47)
        x = x.unsqueeze(2)
        x = self.d_conv1(x)  # out(batch, 128, 45)
        x = x.squeeze(2)
        # x = self.lin1(x)
        # ---------
        x = self.unpool2(x, encode_indices[-2])  # out(batch, 64, 90)
        x = x.unsqueeze(2)
        x = self.d_conv2(x)  # out(batch, 32, 91)
        x = x.squeeze(2)
        # ---------
        x = self.unpool3(x, encode_indices[0])  # x_decoded(batch,32,180)
        x = x.unsqueeze(2)
        x_decoded = self.d_conv3(x)
        x_decoded = x_decoded.squeeze(2)  # batch, 6, 180 = AE input
        # x_decoded = self.reshapel(x_decoded)
        # x_decoded = self.linear(x_decoded)
        return x_decoded


class CNN_AE(nn.Module):
    def __init__(self, n_channels, out_channels=128):
        super(CNN_AE, self).__init__()

        # self.backbone = backbone
        self.n_channels = n_channels  # input data dimension

        self.lin2 = nn.Identity()
        # self.out_dim = 25 * out_channels


        self.encoder = CNN_AE_encoder(n_channels,
                                      out_channels=out_channels)
        self.decoder = CNN_AE_decoder(n_channels,
                                      out_channels=out_channels)

        return

    def forward(self, x):  # x(batch, len180, dim6)
        x_encoded, encode_indices = self.encoder(x)  # x_encoded(batch, 128, 25)
        # todo, encoder output 改成batch,dim
        decod_out = self.decoder(x_encoded, encode_indices)  # x_decoded(batch, 6, 179)

        x_decoded = decod_out.permute(0, 2, 1)
        # x_decoded(batch, 180, 6), x_encoded(batch, 128, 15)
        return x_encoded, x_decoded

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
# model = CNN_AE(3, out_channels)
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
# print('Saving model at: ' + 'cnngru_epoch%s' % str(epoch)\
#       + '_datalen%s_' % str(len_sw) + sensor_type + '.pth')
# torch.save(model.state_dict(), 'cnngru_epoch%s' % str(epoch) +\
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
# plt.savefig('cnngru.png')
# plt.close()