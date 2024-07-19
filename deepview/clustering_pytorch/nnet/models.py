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
        backboneout = self.backbone(x)  # out.shape=b*2,512
        features = self.contrastive_head(backboneout)  # features.shape=b*2,b*2
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


# autoencoder copy from CL-HAR
class CNN_AE_old(nn.Module):
    def __init__(self, n_channels, n_classes, out_channels=128, backbone=True):
        super(CNN_AE_old, self).__init__()

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


# class CNNAE_encoder(nn.Module):
#     def __init__(self, n_channels, out_channels):
#         super(CNNAE_encoder, self).__init__()
#
#         self.e_conv1 = nn.Sequential(nn.Conv1d(n_channels, 32, kernel_size=4,  #8
#                                                stride=1, bias=False, padding=4),
#                                      nn.BatchNorm1d(32),
#                                      nn.ReLU())
#         self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2, padding=1, return_indices=True)
#         self.dropout = nn.Dropout(0.35)
#         self.e_conv2 = nn.Sequential(nn.Conv1d(32, 64, kernel_size=8,
#                                                stride=1, bias=False, padding=4),
#                                      nn.BatchNorm1d(64),
#                                      nn.ReLU())
#         self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2, padding=1, return_indices=True)
#
#         self.e_conv3 = nn.Sequential(nn.Conv1d(64, out_channels, kernel_size=8,
#                                                stride=1, bias=False, padding=4),
#                                      nn.BatchNorm1d(out_channels),
#                                      nn.ReLU())
#         self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2, padding=1, return_indices=True)
#
#     def forward(self, x):
#         '''
#         input x.shape=batch, channel, length
#         the 1d-cnn layer uses stride=1, and generate more channels: m = nn.Conv1d(inchannel, outchannel, kernelsize, stride=1)
#         '''
#         # x = x.permute(0, 2, 1)
#         x, indice1 = self.pool1(self.e_conv1(x))  # xout.shape=batch512, outchannel32, len46
#         size1 = x.size()
#         x = self.dropout(x)
#         x, indice2 = self.pool2(self.e_conv2(x))  # xout.shape=batch512, outchannel64, len24
#         size2 = x.size()
#         x_encoded, indice3 = self.pool3(self.e_conv3(x))  # encoded.shape=batch512, outchannel128, len13
#         size3 = x_encoded.size()
#         return x_encoded, indice1, indice2, indice3, size1, size2, size3
#
#
# class CNNAE_decoder(nn.Module):
#     def __init__(self, n_channels, out_channels):
#         super(CNNAE_decoder, self).__init__()
#         self.unpool1 = nn.MaxUnpool1d(kernel_size=2, stride=2, padding=1)
#         self.d_conv1 = nn.Sequential(
#             nn.ConvTranspose1d(out_channels, 64, kernel_size=8, stride=1, bias=False, padding=4),
#             nn.BatchNorm1d(64),
#             nn.ReLU())
#
#         if n_channels == 9:  # ucihar
#             self.lin1 = nn.Linear(33, 34)
#         elif n_channels == 6:  # hhar
#             self.lin1 = nn.Identity()
#         elif n_channels == 3:  # shar
#             self.lin1 = nn.Linear(23, 24)
#             self.lin2 = nn.Linear(45, 46)
#             self.lin3 = nn.Linear(89, 90)
#
#         self.unpool2 = nn.MaxUnpool1d(kernel_size=2, stride=2, padding=1)
#         self.d_conv2 = nn.Sequential(nn.ConvTranspose1d(64, 32, kernel_size=8, stride=1, bias=False, padding=4),
#                                      nn.BatchNorm1d(32),
#                                      nn.ReLU())
#
#         self.unpool3 = nn.MaxUnpool1d(kernel_size=2, stride=2, padding=1)
#         self.d_conv3 = nn.Sequential(nn.ConvTranspose1d(32, n_channels, kernel_size=8, stride=1, bias=False, padding=4),
#                                      nn.BatchNorm1d(n_channels),
#                                      nn.ReLU())
#
#     def forward(self, x_encoded, indice1, indice2, indice3, size1, size2, size3):
#         '''
#         the input of decoder: # x_encoded.shape=batch512, outchannel128, len13
#         the second layer: # xout.shape=batch512, outchannel64, len24
#         the third layer: # xout.shape=batch512, outchannel32, len46
#         '''
#         x1 = self.unpool1(x_encoded, indice3, output_size=size3)
#         x2 = self.d_conv1(x1)
#         x = self.lin1(x2)
#         x = self.d_conv2(self.unpool2(x, indice2))
#         x = self.lin2(x)
#         x = self.d_conv3(self.unpool1(x, indice1))
#         x_decoded = self.lin3(x)
#         # if self.n_channels == 9:  # ucihar
#         #     x_decoded = self.lin2(x)
#         # elif self.n_channels == 6:  # hhar
#         #     x_decoded = self.lin2(x)
#         # elif self.n_channels == 3:  # shar
#         #     x_decoded = x
#         x_decoded = x_decoded.permute(0, 2, 1)
#         # x_encoded = x_encoded.reshape(x_encoded.shape[0], -1)
#         return x_decoded
#
#
# class CNN_AE(nn.Module):
#     def __init__(self, n_channels, out_channels):
#         super(CNN_AE, self).__init__()
#         self.encoder = CNNAE_encoder(n_channels, out_channels)
#         self.decoder = CNNAE_decoder(n_channels, out_channels)
#
#
#     def forward(self, x):
#         x_encoded, indice1, indice2, indice3, size1, size2, size3 = self.encoder(x)
#         out = self.decoder(x_encoded, indice1, indice2, indice3, size1, size2, size3)
#         return x_encoded, out


class CNN_AE_encoder(nn.Module):
    def __init__(self, n_channels, n_classes, out_channels=128, backbone=True):
        super(CNN_AE_encoder, self).__init__()

        self.n_channels = n_channels*2
        # self.datalen = 180  #args.len_sw
        # self.n_classes = n_classes   # check if correct a

        self.linear = nn.Linear(n_channels,  self.n_channels)
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
        decod_out = self.decoder(x_encoded, encode_indices)  # x_decoded(batch, 6, 179)

        # x_decoded = self.lin2(decod_out)

        x_decoded = decod_out.permute(0, 2, 1)
        # x_decoded(batch, 180, 6), x_encoded(batch, 128, 15)
        return x_encoded, x_decoded


class Autoencoder(nn.Module):
    def __init__(self, data_length, input_dim, encoded_dim=16):
        super(Autoencoder, self).__init__()

        # Encoder
        self.conv1 = nn.Conv1d(input_dim, 32, kernel_size=3, padding=1)  # (batch_size, 32, 180)
        self.pool1 = nn.MaxPool1d(2, stride=2)  # (batch_size, 32, 90)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)  # (batch_size, 64, 90)
        self.pool2 = nn.MaxPool1d(2, stride=2)  # (batch_size, 64, 45)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)  # (batch_size, 128, 45)
        self.pool3 = nn.MaxPool1d(3, stride=3)  # (batch_size, 128, 15)
        self.conv4 = nn.Conv1d(128, 256, kernel_size=3, padding=1)  # (batch_size, 256, 15)
        self.pool4 = nn.MaxPool1d(3, stride=3)  # (batch_size, 256, 5)
        self.fc1 = nn.Linear(256 * 5, 1024)  # (batch_size, 1024)
        self.fc2 = nn.Linear(1024, encoded_dim)  # (batch_size, 16)

        # Decoder
        self.fc3 = nn.Linear(encoded_dim, 1024)  # (batch_size, 1024)
        self.fc4 = nn.Linear(1024, 256 * 5)  # (batch_size, 256*5)
        self.deconv1 = nn.Conv1d(256, 128, kernel_size=3, padding=1)  # (batch_size, 128, 5)
        self.upsample1 = nn.Upsample(scale_factor=3)  # (batch_size, 128, 15)
        self.deconv2 = nn.Conv1d(128, 64, kernel_size=3, padding=1)  # (batch_size, 64, 15)
        self.upsample2 = nn.Upsample(scale_factor=3)  # (batch_size, 64, 45)
        self.deconv3 = nn.Conv1d(64, 32, kernel_size=3, padding=1)  # (batch_size, 32, 45)
        self.upsample3 = nn.Upsample(scale_factor=2)  # (batch_size, 32, 90)
        self.deconv4 = nn.Conv1d(32, input_dim, kernel_size=3, padding=1)  # (batch_size, 3, 90)
        self.upsample4 = nn.Upsample(scale_factor=2)  # (batch_size, 3, 180)

    def forward(self, x):
        # Encoder
        x = self.conv1(x.permute(0, 2, 1))  # Change to (batch_size, 3, 180) -> (batch_size, 32, 180)
        x = torch.relu(x)
        x = self.pool1(x)  # (batch_size, 32, 180) -> (batch_size, 32, 90)
        x = self.conv2(x)  # (batch_size, 32, 90) -> (batch_size, 64, 90)
        x = torch.relu(x)
        x = self.pool2(x)  # (batch_size, 64, 90) -> (batch_size, 64, 45)
        x = self.conv3(x)  # (batch_size, 64, 45) -> (batch_size, 128, 45)
        x = torch.relu(x)
        x = self.pool3(x)  # (batch_size, 128, 45) -> (batch_size, 128, 15)
        x = self.conv4(x)  # (batch_size, 128, 15) -> (batch_size, 256, 15)
        x = torch.relu(x)
        x = self.pool4(x)  # (batch_size, 256, 15) -> (batch_size, 256, 5)
        x = x.view(x.size(0), -1)  # Flatten (batch_size, 256*5)
        x = self.fc1(x)  # (batch_size, 256*5) -> (batch_size, 1024)
        x = torch.relu(x)
        x = self.fc2(x)  # (batch_size, 1024) -> (batch_size, 16)
        encoded = x

        # Decoder
        x = self.fc3(encoded)  # (batch_size, 16) -> (batch_size, 1024)
        x = torch.relu(x)
        x = self.fc4(x)  # (batch_size, 1024) -> (batch_size, 256*5)
        x = torch.relu(x)
        x = x.view(x.size(0), 256, 5)  # Reshape (batch_size, 256, 5)
        x = self.deconv1(x)  # (batch_size, 256, 5) -> (batch_size, 128, 5)
        x = torch.relu(x)
        x = self.upsample1(x)  # (batch_size, 128, 5) -> (batch_size, 128, 15)
        x = self.deconv2(x)  # (batch_size, 128, 15) -> (batch_size, 64, 15)
        x = torch.relu(x)
        x = self.upsample2(x)  # (batch_size, 64, 15) -> (batch_size, 64, 45)
        x = self.deconv3(x)  # (batch_size, 64, 45) -> (batch_size, 32, 45)
        x = torch.relu(x)
        x = self.upsample3(x)  # (batch_size, 32, 45) -> (batch_size, 32, 90)
        x = self.deconv4(x)  # (batch_size, 32, 90) -> (batch_size, 3, 90)
        x = torch.relu(x)
        x = self.upsample4(x)  # (batch_size, 3, 90) -> (batch_size, 3, 180)
        decoded = x.permute(0, 2, 1)  # Change back to (batch_size, 180, 3)

        return encoded, decoded


# class Autoencoder(nn.Module):
#     def __init__(self, data_length, input_dim, encoded_dim=16):
#         super(Autoencoder, self).__init__()
#
#         self.data_length = data_length
#         self.input_dim = input_dim
#         self.encoded_dim = encoded_dim
#
#         # Encoder
#         self.conv1 = nn.Conv1d(input_dim, 32, kernel_size=3, stride=2, padding=1)  # (batch_size, 32, data_length/2)
#         self.conv2 = nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1)  # (batch_size, 64, data_length/4)
#         self.conv3 = nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1)  # (batch_size, 128, data_length/8)
#         self.conv4 = nn.Conv1d(128, 256, kernel_size=3, stride=2, padding=1)  # (batch_size, 256, data_length/16)
#
#         # Calculate flattened size after convolutions
#         self.flatten_param = math.ceil(data_length / 16)
#         self.flattened_size = 256 * self.flatten_param
#
#         self.fc1 = nn.Linear(self.flattened_size, 1024)  # (batch_size, 1024)
#         self.fc2 = nn.Linear(1024, encoded_dim)  # (batch_size, encoded_dim)
#
#         # Decoder
#         self.fc3 = nn.Linear(encoded_dim, 1024)  # (batch_size, 1024)
#         self.fc4 = nn.Linear(1024, self.flattened_size)  # (batch_size, flattened_size)
#
#         self.deconv1 = nn.ConvTranspose1d(256, 128, kernel_size=3, stride=2, padding=1,
#                                           output_padding=1)  # (batch_size, 128, data_length/8)
#         self.deconv2 = nn.ConvTranspose1d(128, 64, kernel_size=3, stride=2, padding=1,
#                                           output_padding=1)  # (batch_size, 64, data_length/4)
#         self.deconv3 = nn.ConvTranspose1d(64, 32, kernel_size=3, stride=2, padding=1,
#                                           output_padding=1)  # (batch_size, 32, data_length/2)
#         self.deconv4 = nn.ConvTranspose1d(32, input_dim, kernel_size=3, stride=2, padding=1,
#                                           output_padding=1)  # (batch_size, input_dim, data_length)
#         self.conv5 = nn.Conv1d(input_dim, input_dim, kernel_size=3,
#                                padding=1)  # Final layer to adjust size if necessary
#         self.conv6 = nn.Conv1d(3, 3, kernel_size=3, stride=1, padding=1)  # (batch_size, 3, 180)
#
#     def forward(self, x):
#         # Encoder
#         x = self.conv1(x.permute(0, 2, 1))  # Change to (batch_size, input_dim, data_length)
#         x = torch.relu(x)
#         x = self.conv2(x)
#         x = torch.relu(x)
#         x = self.conv3(x)
#         x = torch.relu(x)
#         x = self.conv4(x)
#         x = torch.relu(x)
#         x = x.view(x.size(0), -1)  # Flatten
#         x = self.fc1(x)
#         x = torch.relu(x)
#         x = self.fc2(x)
#         encoded = x
#
#         # Decoder
#         x = self.fc3(encoded)
#         x = torch.relu(x)
#         x = self.fc4(x)
#         x = torch.relu(x)
#         x = x.view(x.size(0), 256, self.flatten_param)  # Reshape
#         x = self.deconv1(x)
#         x = torch.relu(x)
#         x = self.deconv2(x)
#         x = torch.relu(x)
#         x = self.deconv3(x)
#         x = torch.relu(x)
#         x = self.deconv4(x)
#         x = torch.relu(x)
#         x = self.conv5(x)
#         x = torch.relu(x)
#         x = self.conv6(x)  # (batch_size, 3, 192) -> (batch_size, 3, 180)
#         decoded = x.permute(0, 2, 1)  # Change back to (batch_size, 180, 3)
#         return encoded, decoded


# # Define data length and input dimension
# data_length = 180
# input_dim = 3
#
# # Create the model
# model = Autoencoder(data_length=data_length, input_dim=input_dim)
# print(model)
#
# # Example input
# example_input = torch.randn(128, data_length, input_dim)
# example_output = model(example_input)
# print("Output shape:", example_output.shape)



class AE_linear(nn.Module):
    def __init__(self, n_channels, len_sw, n_classes, outdim=128, backbone=True):
        super(AE_linear, self).__init__()

        self.backbone = backbone
        self.len_sw = len_sw

        self.e1 = nn.Linear(n_channels, 8)
        self.e2 = nn.Linear(8 * len_sw, 2 * len_sw)
        self.e3 = nn.Linear(2 * len_sw, outdim)

        self.d1 = nn.Linear(outdim, 2 * len_sw)
        self.d2 = nn.Linear(2 * len_sw, 8 * len_sw)
        self.d3 = nn.Linear(8, n_channels)

        self.out_dim = outdim

        if backbone == False:
            self.classifier = nn.Linear(outdim, n_classes)

    def forward(self, x):
        x_e1 = self.e1(x)
        x_e1 = x_e1.reshape(x_e1.shape[0], -1)
        x_e2 = self.e2(x_e1)
        x_encoded = self.e3(x_e2)

        x_d1 = self.d1(x_encoded)
        x_d2 = self.d2(x_d1)
        x_d2 = x_d2.reshape(x_d2.shape[0], self.len_sw, 8)
        x_decoded = self.d3(x_d2)

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