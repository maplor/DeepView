"""
Authors: Wouter Van Gansbeke, Simon Vandenhende
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ContrastiveModel(nn.Module):
    def __init__(self, backbone, head='mlp', features_dim=128):
        super(ContrastiveModel, self).__init__()
        self.backbone = backbone['backbone']
        self.backbone_dim = backbone['dim']
        self.head = head
        # self.encoder = backbone['backbone']

        if head == 'linear':
            self.contrastive_head = nn.Linear(self.backbone_dim, features_dim)

        elif head == 'mlp':
            self.contrastive_head = nn.Sequential(
                nn.Linear(self.backbone_dim, self.backbone_dim),
                nn.ReLU(),
                nn.Linear(self.backbone_dim, features_dim))

        else:
            raise ValueError('Invalid head {}'.format(head))

    def forward(self, x):  # x.shape=b*2,c,h,w
        # backboneout = self.encoder(x)  # out.shape=b*2,512
        backboneout = self.backbone(x)  # out.shape=b*2,512
        features = self.contrastive_head(backboneout[0].reshape(x.shape[0], -1))  # features.shape=b*2,b*2
        features = F.normalize(features, dim=1)
        return features


class ReconstructionFramework(nn.Module):
    def __init__(self, backbone):
        super(ReconstructionFramework, self).__init__()
        self.backbone = backbone

    def forward(self, x):
        x_encoded, x_decoded = self.backbone(x)
        return x_encoded, x_decoded


class ClusteringModel(nn.Module):
    def __init__(self, backbone, nclusters, nheads=1):
        super(ClusteringModel, self).__init__()
        self.backbone = backbone['backbone']
        self.backbone_dim = backbone['dim']
        self.nheads = nheads
        assert (isinstance(self.nheads, int))
        assert (self.nheads > 0)
        self.cluster_head = nn.ModuleList([nn.Linear(self.backbone_dim, nclusters) for _ in range(self.nheads)])

    def forward(self, x, forward_pass='default'):
        if forward_pass == 'default':
            features = self.backbone(x)
            out = [cluster_head(features) for cluster_head in self.cluster_head]

        elif forward_pass == 'backbone':
            out = self.backbone(x)

        elif forward_pass == 'head':
            out = [cluster_head(x) for cluster_head in self.cluster_head]

        elif forward_pass == 'return_all':
            features = self.backbone(x)
            out = {'features': features, 'output': [cluster_head(features) for cluster_head in self.cluster_head]}

        else:
            raise ValueError('Invalid forward pass {}'.format(forward_pass))

        return out


class CNN_AE_encoder(nn.Module):
    def __init__(self, n_channels, n_classes, out_channels=128, backbone=True):
        super(CNN_AE_encoder, self).__init__()

        self.n_channels = n_channels * 2
        # self.datalen = 180  #args.len_sw
        # self.n_classes = n_classes   # check if correct a

        self.linear = nn.Linear(n_channels, self.n_channels)
        kernel_size = 5
        self.e_conv1 = nn.Sequential(nn.Conv2d(self.n_channels, 32,
                                               (1, kernel_size), bias=False,
                                               padding=(0, kernel_size // 2)),
                                     nn.BatchNorm2d(32),
                                     nn.Tanh())  # Tanh is MoIL paper
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2, padding=0, return_indices=True)
        self.dropout = nn.Dropout(0.35)  # probability of samples to be zero

        self.e_conv2 = nn.Sequential(nn.Conv2d(32, 64,
                                               (1, kernel_size), bias=False,
                                               padding=(0, kernel_size // 2)),
                                     nn.BatchNorm2d(64),
                                     nn.Tanh())
        self.pool2 = nn.MaxPool1d(kernel_size=4, stride=2, padding=0, return_indices=True)

        self.e_conv3 = nn.Sequential(nn.Conv2d(64, out_channels,
                                               (1, kernel_size), bias=False,
                                               padding=(0, kernel_size // 2)),
                                     nn.BatchNorm2d(out_channels),
                                     nn.PReLU())
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2, padding=0, return_indices=True)

        self.out_samples = 25
        self.out_dim = out_channels

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
        x_encoded, indice3 = self.pool3(x3)  # xencoded(batch,128,15)
        # x_encoded # batch,128,15
        return x_encoded, [indice1, indice2, indice3]


class CNN_AE_decoder(nn.Module):
    def __init__(self, n_channels, n_classes, out_channels=128, backbone=True):
        super(CNN_AE_decoder, self).__init__()

        self.n_channels = n_channels
        kernel_size = 5
        self.unpool1 = nn.MaxUnpool1d(kernel_size=2, stride=2, padding=0)
        self.d_conv1 = nn.Sequential(nn.ConvTranspose2d(out_channels, 64,
                                                        kernel_size=(1, kernel_size),
                                                        bias=False,
                                                        padding=(0, kernel_size // 2)),
                                     nn.BatchNorm2d(64),
                                     nn.Tanh())

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
        return

    def forward(self, x_encoded, encode_indices):  # x_encoded(batch, 128, 25)
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

        n_classes = 5  # not used
        self.encoder = CNN_AE_encoder(n_channels, n_classes,
                                      out_channels=out_channels, backbone=True)
        self.decoder = CNN_AE_decoder(n_channels, n_classes,
                                      out_channels=out_channels, backbone=True)

        # # if backbone == False:
        # self.classifier = self.encoder.classifier
        # # self.out_dim

        return

    def forward(self, x):  # x(batch, len180, dim6)
        x_encoded, encode_indices = self.encoder(x)  # x_encoded(batch, 128, 25)
        # todo, encoder output 改成batch,dim
        decod_out = self.decoder(x_encoded, encode_indices)  # x_decoded(batch, 6, 179)

        x_decoded = decod_out.permute(0, 2, 1)
        # x_decoded(batch, 180, 6), x_encoded(batch, 128, 15)
        return x_encoded, x_decoded


# copy from cl-har
class CNN_AE_clhar(nn.Module):
    def __init__(self, n_channels, n_classes, out_channels=128, backbone=True):
        super(CNN_AE, self).__init__()

        self.backbone = backbone
        self.n_channels = n_channels

        self.e_conv1 = nn.Sequential(nn.Conv1d(n_channels, 32, kernel_size=8, stride=1, bias=False, padding=4),
                                     nn.BatchNorm1d(32),
                                     nn.ReLU())
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2, padding=1, return_indices=True)
        self.dropout = nn.Dropout(0.35)

        self.e_conv2 = nn.Sequential(nn.Conv1d(32, 64, kernel_size=8, stride=1, bias=False, padding=4),
                                     nn.BatchNorm1d(64),
                                     nn.ReLU())
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2, padding=1, return_indices=True)

        self.e_conv3 = nn.Sequential(nn.Conv1d(64, out_channels, kernel_size=8, stride=1, bias=False, padding=4),
                                     nn.BatchNorm1d(out_channels),
                                     nn.ReLU())
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2, padding=1, return_indices=True)

        self.unpool1 = nn.MaxUnpool1d(kernel_size=2, stride=2, padding=1)
        self.d_conv1 = nn.Sequential(
            nn.ConvTranspose1d(out_channels, 64, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(64),
            nn.ReLU())

        if n_channels == 9:  # ucihar
            self.lin1 = nn.Linear(33, 34)
        elif n_channels == 6:  # hhar
            self.lin1 = nn.Identity()
        elif n_channels == 3:  # shar
            self.lin1 = nn.Linear(39, 40)

        self.unpool2 = nn.MaxUnpool1d(kernel_size=2, stride=2, padding=1)
        self.d_conv2 = nn.Sequential(nn.ConvTranspose1d(64, 32, kernel_size=8, stride=1, bias=False, padding=4),
                                     nn.BatchNorm1d(32),
                                     nn.ReLU())

        self.unpool3 = nn.MaxUnpool1d(kernel_size=2, stride=2, padding=1)
        self.d_conv3 = nn.Sequential(nn.ConvTranspose1d(32, n_channels, kernel_size=8, stride=1, bias=False, padding=4),
                                     nn.BatchNorm1d(n_channels),
                                     nn.ReLU())

        if n_channels == 9:  # ucihar
            self.lin2 = nn.Linear(127, 128)
            self.out_dim = 18 * out_channels
        elif n_channels == 6:  # hhar
            self.lin2 = nn.Linear(99, 100)
            self.out_dim = 15 * out_channels
        elif n_channels == 3:  # shar
            self.out_dim = 21 * out_channels

        if backbone == False:
            self.classifier = nn.Linear(self.out_dim, n_classes)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x, indice1 = self.pool1(self.e_conv1(x))
        x = self.dropout(x)
        x, indice2 = self.pool2(self.e_conv2(x))
        x_encoded, indice3 = self.pool3(self.e_conv3(x))
        x = self.d_conv1(self.unpool1(x_encoded, indice3))
        x = self.lin1(x)
        x = self.d_conv2(self.unpool2(x, indice2))
        x = self.d_conv3(self.unpool1(x, indice1))
        if self.n_channels == 9:  # ucihar
            x_decoded = self.lin2(x)
        elif self.n_channels == 6:  # hhar
            x_decoded = self.lin2(x)
        elif self.n_channels == 3:  # shar
            x_decoded = x
        x_decoded = x_decoded.permute(0, 2, 1)
        x_encoded = x_encoded.reshape(x_encoded.shape[0], -1)

        if self.backbone:
            return x_decoded, x_encoded
        else:
            out = self.classifier(x_encoded)
            return out, x_decoded


# ----------------------------------------Backbone for SimCLR--------------------------------------

class FCN(nn.Module):
    def __init__(self, n_channels, out_channels=128, backbone=True):
        super(FCN, self).__init__()

        self.conv_block1 = nn.Sequential(nn.Conv1d(n_channels, 32, kernel_size=8, stride=1, bias=False, padding=4),
                                         nn.BatchNorm1d(32),
                                         nn.ReLU(),
                                         nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
                                         nn.Dropout(0.35))
        self.conv_block2 = nn.Sequential(nn.Conv1d(32, 64, kernel_size=8, stride=1, bias=False, padding=4),
                                         nn.BatchNorm1d(64),
                                         nn.ReLU(),
                                         nn.MaxPool1d(kernel_size=2, stride=2, padding=1))
        self.conv_block3 = nn.Sequential(nn.Conv1d(64, out_channels, kernel_size=8, stride=1, bias=False, padding=4),
                                         nn.BatchNorm1d(out_channels),
                                         nn.ReLU(),
                                         nn.MaxPool1d(kernel_size=2, stride=2, padding=1))

    def forward(self, x_in):
        # x_in: (batch, len, dim)
        x_in = x_in.permute(0, 2, 1)
        # x_in: (batch, dim(channel), len)
        x = self.conv_block1(x_in)
        x = self.conv_block2(x)
        x = self.conv_block3(x)

        # return: (batch, feature_dim)
        # feature_dim = out_channels * 25
        return x.reshape(x.shape[0], -1)


class DeepConvLSTM(nn.Module):
    def __init__(self, n_channels, conv_kernels=64, kernel_size=5, LSTM_units=128, backbone=True):
        super(DeepConvLSTM, self).__init__()

        self.conv1 = nn.Conv2d(1, conv_kernels, (kernel_size, 1))
        self.conv2 = nn.Conv2d(conv_kernels, conv_kernels, (kernel_size, 1))
        self.conv3 = nn.Conv2d(conv_kernels, conv_kernels, (kernel_size, 1))
        self.conv4 = nn.Conv2d(conv_kernels, conv_kernels, (kernel_size, 1))

        self.dropout = nn.Dropout(0.5)
        self.lstm = nn.LSTM(n_channels * conv_kernels, LSTM_units, num_layers=2)

        self.out_dim = LSTM_units

        self.activation = nn.ReLU()

    def forward(self, x):
        # x_in: (batch, len, dim)
        self.lstm.flatten_parameters()
        x = x.unsqueeze(1)
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = self.activation(self.conv3(x))
        x = self.activation(self.conv4(x))

        x = x.permute(2, 0, 3, 1)
        x = x.reshape(x.shape[0], x.shape[1], -1)

        x = self.dropout(x)

        x, h = self.lstm(x)
        x = x[-1, :, :]

        # return: (batch, LSTM_units)
        return x


class LSTM(nn.Module):
    def __init__(self, n_channels, LSTM_units=128, backbone=True):
        super(LSTM, self).__init__()

        self.backbone = backbone
        self.lstm = nn.LSTM(n_channels, LSTM_units, num_layers=2)
        self.out_dim = LSTM_units

    def forward(self, x):
        # x_in: (batch, len, dim)
        self.lstm.flatten_parameters()
        x = x.permute(1, 0, 2)
        x, (h, c) = self.lstm(x)
        x = x[-1, :, :]

        # return: (batch, LSTM_units)
        return x
