import torch
from torch import nn
import numpy as np
import torch.nn.functional as F

class SimpleNN_1s(nn.Module):
    def __init__(self, input_dim=128 * 6, embedding_dim=128, number_classes=5):
        super(SimpleNN_1s, self).__init__()
        self.acc_encoder = Autoencoder3d4().feature_extractor
        # self.pre_encoder = Autoencoder1d().feature_extractor

        self.linear = nn.Linear(input_dim*1, 32)
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
        # press = data[:, -1:, :]
        acc_fea, _, _ = self.acc_encoder(accel)
        # pre_fea, _, _ = self.pre_encoder(press)
        # concat_fea = torch.concat([acc_fea, pre_fea], dim=1)
        concat_fea = self.linear(acc_fea)
        # concat_fea = self.linear(acc_fea)
        if if_contrast:
            output = self.projector(concat_fea)  # output(batch, 128)
        else:
            output = self.classifier(concat_fea)
        return output, concat_fea

class SimpleNN_13s(nn.Module):
    def __init__(self, input_dim=128 * 6, number_classes=5):
        super(SimpleNN_13s, self).__init__()
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

class SimpleNN_13s_SHAP(nn.Module):
    def __init__(self, input_dim=128 * 6, number_classes=5):
        super(SimpleNN_13s_SHAP, self).__init__()
        self.encoder3d = Autoencoder3d4().feature_extractor
        self.encoder1d = Autoencoder1d().feature_extractor

        self.classifier = nn.Sequential(
            nn.Linear(input_dim * 2, 32),
            # nn.ELU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, number_classes),
            # nn.Softmax(dim=1),
        )

        self.projector = nn.Sequential(
            nn.Linear(input_dim * 2, 32),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 16)
        )

    def forward(self, data, if_contrast=True):
        accel = data[:, :3, :]
        press = data[:, -1:, :]
        acc_fea, _, _ = self.encoder3d(accel)
        pre_fea, _, _ = self.encoder1d(press)
        concat_fea = torch.concat([acc_fea, pre_fea], dim=1)
        if if_contrast:
            output = self.projector(concat_fea)  # output(batch, 128)
        else:
            output = self.classifier(concat_fea)
        return output, concat_fea

class SimpleNN_13s_SHAP_clean(nn.Module):
    def __init__(self, input_dim=128 * 6, number_classes=5):
        super(SimpleNN_13s_SHAP_clean, self).__init__()
        self.encoder3d = Autoencoder3d4().feature_extractor
        self.encoder1d = Autoencoder1d().feature_extractor
        self.classifier = nn.Sequential(
            nn.Linear(input_dim * 2, 32),
            # nn.ELU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, number_classes),
            # nn.Softmax(dim=1),
        )

    def forward(self, data, if_contrast=True):
        accel = data[:, :3, :]
        press = data[:, -1:, :]
        acc_fea, _, _ = self.encoder3d(accel)
        pre_fea, _, _ = self.encoder1d(press)
        concat_fea = torch.concat([acc_fea, pre_fea], dim=1)
        output = self.classifier(concat_fea)
        return output


class SimpleNN_33s(nn.Module):
    def __init__(self, input_dim=128 * 6, embedding_dim=128, number_classes=5):
        super(SimpleNN_33s, self).__init__()
        self.acc_encoder = Autoencoder3d4().feature_extractor
        self.gyr_encoder = Autoencoder3d4().feature_extractor

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
        press = data[:, 3:, :]
        acc_fea, _, _ = self.acc_encoder(accel)
        pre_fea, _, _ = self.gyr_encoder(press)
        concat_fea = torch.concat([acc_fea, pre_fea], dim=1)
        concat_fea = self.linear(concat_fea)
        # concat_fea = self.linear(acc_fea)
        if if_contrast:
            output = self.projector(concat_fea)  # output(batch, 128)
        else:
            output = self.classifier(concat_fea)
        return output, concat_fea


class SimpleNN_32s(nn.Module):
    def __init__(self, input_dim=128 * 6, embedding_dim=128, number_classes=5):
        super(SimpleNN_32s, self).__init__()
        self.acc_encoder = Autoencoder3d4().feature_extractor
        self.gyr_encoder = Autoencoder2d().feature_extractor

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
        press = data[:, 3:, :]
        acc_fea, _, _ = self.acc_encoder(accel)
        pre_fea, _, _ = self.gyr_encoder(press)
        concat_fea = torch.concat([acc_fea, pre_fea], dim=1)
        concat_fea = self.linear(concat_fea)
        # concat_fea = self.linear(acc_fea)
        if if_contrast:
            output = self.projector(concat_fea)  # output(batch, 128)
        else:
            output = self.classifier(concat_fea)
        return output, concat_fea

class SimpleNN_11s(nn.Module):
    def __init__(self, input_dim=128 * 6, embedding_dim=128, number_classes=5):
        super(SimpleNN_11s, self).__init__()
        self.acc_encoder = Autoencoder1d().feature_extractor
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
        accel = data[:, :1, :]
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


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0, temperature=200):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin  # margin: 边缘参数，用于控制负样本对之间的最小距离
        self.temperature = temperature  # temperature: 温度参数，用于控制 softmax 的平滑程度

    def nt_xent_loss(self, features, temperature=1, eps=1e-6):
        """
        计算 NT-Xent 损失。

        Args:
            features: 特征向量，形状为 (batch_size * 2, feature_dim)。
            temperature: 温度系数。
            eps: 数值稳定性的小量。

        Returns:
            损失值。
        """
        batch_size = features.shape[0] // 2

        # 计算所有样本对之间的相似度
        similarity_matrix = torch.matmul(features, features.T)

        # 去除对角线上的自身相似度
        mask = torch.eye(features.shape[0], dtype=torch.bool, device=features.device)
        similarity_matrix = similarity_matrix.masked_fill(mask, value=-1e9)

        # 获取正样本对的相似度
        sim_ij = torch.diag(similarity_matrix, batch_size)
        sim_ji = torch.diag(similarity_matrix, -batch_size)
        positives = torch.cat([sim_ij, sim_ji], dim=0)

        # 计算所有负样本的相似度
        negatives = similarity_matrix[~mask].reshape(2 * batch_size, -1)

        # 计算损失
        numerator = torch.exp(positives / temperature)
        denominator = torch.sum(torch.exp(negatives / temperature), dim=1)
        loss = -torch.mean(torch.log(numerator / (denominator + eps)))

        return loss

    def forward(self, feature):
        """
        计算对比损失。
        Returns:
            损失值。
        """
        loss_contrastive = self.nt_xent_loss(feature, temperature=self.temperature)

        return loss_contrastive

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


#-----------------------Resnet------------------------------------------

class ResBlock(nn.Module):
    r""" Basic bulding block in Resnets:

       bn-relu-conv-bn-relu-conv
      /                         \
    x --------------------------(+)->

    """

    def __init__(
        self, in_channels, out_channels, kernel_size=5, stride=1, padding=2
    ):

        super(ResBlock, self).__init__()

        self.bn1 = nn.BatchNorm1d(in_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            bias=False,
            padding_mode="circular",
        )
        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            bias=False,
            padding_mode="circular",
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        x = self.relu(self.bn1(x))
        x = self.conv1(x)
        x = self.relu(self.bn2(x))
        x = self.conv2(x)

        x = x + identity

        return x

class Reconstructor(nn.Module):
    def __init__(self, input_size=512, len_sw=300):
        super().__init__()
        self.len_sw = len_sw
        self.decoder = nn.Sequential(
            nn.Linear(input_size, 2048),
            nn.Mish(),
            nn.Linear(2048, 1024),
            nn.Mish(),
            nn.Linear(1024, self.len_sw * 3),
            # nn.PReLU()
            # nn.Softmax()
        )

    def forward(self, x):
        x = self.decoder(x)
        x = x.reshape(x.shape[0], -1, self.len_sw)  # batch,dim,len
        return x


class ReconstructorConv(nn.Module):
    def __init__(self, input_size=512, len_sw=300):
        super().__init__()
        self.len_sw = len_sw
        # self.decoder = nn.Sequential(
        #     nn.Linear(input_size, 2048),
        #     nn.Mish(),
        #     nn.Linear(2048, 1024),
        #     nn.Mish(),
        #     nn.Linear(1024, self.len_sw * 3),
        #     nn.Mish()
        # )

        self.Conv1 = nn.Conv1d(1, 16, 3, stride=1)
        self.bn1 = nn.BatchNorm1d(16)
        self.drop = nn.Dropout(0.5)
        self.ac1 = nn.Mish()
        self.Conv2 = nn.Conv1d(16, 3, 3, stride=2)
        self.bn2 = nn.BatchNorm1d(3)
        self.drop2 = nn.Dropout(0.5)
        self.ac2 = nn.Mish()
        self.linear1 = nn.Linear(510, self.len_sw)
        self.ac3 = nn.Mish()

    def forward(self, x):  # batch, 1024
        # x = self.decoder(x)
        # x = x.reshape(x.shape[0], -1, self.len_sw)  # batch,dim,len
        x = x.unsqueeze(1)
        x = self.Conv1(x)
        x = self.bn1(x)
        x = self.drop(x)
        x = self.ac1(x)

        x = self.Conv2(x)
        x = self.bn2(x)
        x = self.drop2(x)
        x = self.ac2(x)

        # x = x.reshape(x.shape[0], self.len_sw, -1)
        x = self.linear1(x)
        x = self.ac3(x)
        # x = x.reshape(x.shape[0], -1, self.len_sw)

        return x

class Downsample(nn.Module):
    r"""Downsampling layer that applies anti-aliasing filters.
    For example, order=0 corresponds to a box filter (or average downsampling
    -- this is the same as AvgPool in Pytorch), order=1 to a triangle filter
    (or linear downsampling), order=2 to cubic downsampling, and so on.
    See https://richzhang.github.io/antialiased-cnns/ for more details.
    """

    def __init__(self, channels=None, factor=2, order=1):
        super(Downsample, self).__init__()
        assert factor > 1, "Downsampling factor must be > 1"
        self.stride = factor
        self.channels = channels
        self.order = order

        # Figure out padding and check params make sense
        # The padding is given by order*(factor-1)/2
        # so order*(factor-1) must be divisible by 2
        total_padding = order * (factor - 1)
        assert total_padding % 2 == 0, (
            "Misspecified downsampling parameters."
            "Downsampling factor and order must be such "
            "that order*(factor-1) is divisible by 2"
        )
        self.padding = int(order * (factor - 1) / 2)

        box_kernel = np.ones(factor)
        kernel = np.ones(factor)
        for _ in range(order):
            kernel = np.convolve(kernel, box_kernel)
        kernel /= np.sum(kernel)
        kernel = torch.Tensor(kernel)
        self.register_buffer(
            "kernel", kernel[None, None, :].repeat((channels, 1, 1))
        )

    def forward(self, x):
        return F.conv1d(
            x,
            self.kernel,
            stride=self.stride,
            padding=self.padding,
            groups=x.shape[1],
        )

class Resnet(nn.Module):
    r"""The general form of the architecture can be described as follows:

    x->[Conv-[ResBlock]^m-BN-ReLU-Down]^n->y

    In other words:

            bn-relu-conv-bn-relu-conv                        bn-
           /                         \                      /
    x->conv --------------------------(+)-bn-relu-down-> conv ----

    """

    def __init__(
        self,
        output_size=1,
        n_channels=3,
            len_sw=300,
        is_eva=False,
        resnet_version=1,
        epoch_len=10,
        is_mtl=False,
        is_simclr=False,
            is_reconst=False
    ):
        super(Resnet, self).__init__()

        # Architecture definition. Each tuple defines
        # a basic Resnet layer Conv-[ResBlock]^m]-BN-ReLU-Down
        # isEva: change the classifier to two FC with ReLu
        # For example, (64, 5, 1, 5, 3, 1) means:
        # - 64 convolution filters
        # - kernel size of 5
        # - 1 residual block (ResBlock)
        # - ResBlock's kernel size of 5
        # - downsampling factor of 3
        # - downsampling filter order of 1
        # In the below, note that 3*3*5*5*4 = 900 (input size)
        if resnet_version == 1:
            if epoch_len == 5:
                cgf = [
                    (64, 5, 2, 5, 2, 2),
                    (128, 5, 2, 5, 2, 2),
                    (256, 5, 2, 5, 3, 1),
                    (256, 5, 2, 5, 3, 1),
                    (512, 5, 0, 5, 3, 1),
                ]
            elif epoch_len == 10:
                cgf = [
                    (64, 5, 2, 5, 2, 2),
                    (128, 5, 2, 5, 2, 2),
                    (256, 5, 2, 5, 5, 1),
                    (512, 5, 2, 5, 5, 1),
                    (1024, 5, 0, 5, 3, 1),
                ]
            else:
                cgf = [
                    (64, 5, 2, 5, 3, 1),
                    (128, 5, 2, 5, 3, 1),
                    (256, 5, 2, 5, 5, 1),
                    (512, 5, 2, 5, 5, 1),
                    (1024, 5, 0, 5, 4, 0),
                ]
        else:
            cgf = [
                (64, 5, 2, 5, 3, 1),
                (64, 5, 2, 5, 3, 1),
                (128, 5, 2, 5, 5, 1),
                (128, 5, 2, 5, 5, 1),
                (256, 5, 2, 5, 4, 0),
            ]  # smaller resnet
        in_channels = n_channels
        feature_extractor = nn.Sequential()
        for i, layer_params in enumerate(cgf):
            (
                out_channels,
                conv_kernel_size,
                n_resblocks,
                resblock_kernel_size,
                downfactor,
                downorder,
            ) = layer_params
            feature_extractor.add_module(
                f"layer{i+1}",
                Resnet.make_layer(
                    in_channels,
                    out_channels,
                    conv_kernel_size,
                    n_resblocks,
                    resblock_kernel_size,
                    downfactor,
                    downorder,
                ),
            )
            in_channels = out_channels

        self.feature_extractor = feature_extractor
        self.is_mtl = is_mtl

        # Classifier input size = last out_channels in previous layer
        if is_eva:
            self.classifier = EvaClassifier(
                input_size=out_channels, output_size=output_size
            )
        elif is_mtl:
            self.aot_h = Classifier(
                input_size=out_channels, output_size=output_size
            )
            self.scale_h = Classifier(
                input_size=out_channels, output_size=output_size
            )
            self.permute_h = Classifier(
                input_size=out_channels, output_size=output_size
            )
            self.time_w_h = Classifier(
                input_size=out_channels, output_size=output_size
            )
        elif is_simclr:
            self.classifier = ProjectionHead(
                input_size=out_channels, encoding_size=output_size
            )
        elif is_reconst:
            self.classifier = Reconstructor(
                input_size=out_channels,
                len_sw=len_sw,
            )

        weight_init(self)

    @staticmethod
    def make_layer(
        in_channels,
        out_channels,
        conv_kernel_size,
        n_resblocks,
        resblock_kernel_size,
        downfactor,
        downorder=1,
    ):
        r""" Basic layer in Resnets:

        x->[Conv-[ResBlock]^m-BN-ReLU-Down]->

        In other words:

                bn-relu-conv-bn-relu-conv
               /                         \
        x->conv --------------------------(+)-bn-relu-down->

        """

        # Check kernel sizes make sense (only odd numbers are supported)
        assert (
            conv_kernel_size % 2
        ), "Only odd number for conv_kernel_size supported"
        assert (
            resblock_kernel_size % 2
        ), "Only odd number for resblock_kernel_size supported"

        # Figure out correct paddings
        conv_padding = int((conv_kernel_size - 1) / 2)
        resblock_padding = int((resblock_kernel_size - 1) / 2)

        modules = [
            nn.Conv1d(
                in_channels,
                out_channels,
                conv_kernel_size,
                1,
                conv_padding,
                bias=False,
                padding_mode="circular",
            )
        ]

        for i in range(n_resblocks):
            modules.append(
                ResBlock(
                    out_channels,
                    out_channels,
                    resblock_kernel_size,
                    1,
                    resblock_padding,
                )
            )

        modules.append(nn.BatchNorm1d(out_channels))
        modules.append(nn.ReLU(True))
        modules.append(Downsample(out_channels, downfactor, downorder))

        return nn.Sequential(*modules)

    def forward(self, x):
        feats = self.feature_extractor(x)

        if self.is_mtl:
            aot_y = self.aot_h(feats.view(x.shape[0], -1))
            scale_y = self.scale_h(feats.view(x.shape[0], -1))
            permute_y = self.permute_h(feats.view(x.shape[0], -1))
            time_w_h = self.time_w_h(feats.view(x.shape[0], -1))
            return aot_y, scale_y, permute_y, time_w_h
        else:
            y = self.classifier(feats.view(x.shape[0], -1))
            return feats, y
        return y

#-----------------------Resnet------------------------------------------

#-----------------------autoencoder------------------------------------------
class Encoder3d(nn.Module):
    def __init__(self):
        super(Encoder3d, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=3, out_channels=64, kernel_size=6, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(kernel_size=2)

        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=6, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(kernel_size=2)

        self.conv3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=4, padding=1)
        self.bn3 = nn.BatchNorm1d(256)
        self.pool3 = nn.MaxPool1d(kernel_size=2)

        self.flattened_size = 256 * 4#12  # Adjust this based on input size and pooling
        self.fc = nn.Linear(self.flattened_size, 64)

    def forward(self, x):
        x = self.pool1(F.elu(self.bn1(self.conv1(x))))
        x = self.pool2(F.elu(self.bn2(self.conv2(x))))
        x = self.pool3(F.elu(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)  # Latent space
        return x

# Decoder structure
class Decoder3d(nn.Module):
    def __init__(self):
        super(Decoder3d, self).__init__()
        self.fc = nn.Linear(64, 256 * 4)  # Match encoder flattened size
        self.unflatten = nn.Unflatten(1, (256, 4))

        self.conv_trans1 = nn.ConvTranspose1d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm1d(128)

        self.conv_trans2 = nn.ConvTranspose1d(in_channels=128, out_channels=64, kernel_size=6, stride=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)

        self.conv_trans3 = nn.ConvTranspose1d(in_channels=64, out_channels=3, kernel_size=6, stride=2, padding=1)
        self.bn3 = nn.BatchNorm1d(3)
        self.linear = nn.Linear(52, 50)

    def forward(self, x):
        x = F.elu(self.fc(x))
        x = self.unflatten(x)
        x = F.elu(self.bn1(self.conv_trans1(x)))
        x = F.elu(self.bn2(self.conv_trans2(x)))
        x = F.elu(self.bn3(self.conv_trans3(x)))  # Sigmoid to normalize output between 0 and 1
        # x = torch.sigmoid(self.conv_trans3(x))  # Sigmoid to normalize output between 0 and 1
        x = self.linear(x)
        return x

class Autoencoder3d(nn.Module):
    def __init__(self, is_reconst=True, is_classify=False):
        super(Autoencoder3d, self).__init__()
        self.feature_extractor = Encoder3d()

        self.is_reconst = is_reconst
        self.is_classify = is_classify
        if self.is_reconst:
            self.decoder = Decoder3d()
        elif self.is_classify:
            # self.classify = EvaClassifier(output_size=15)
            self.classify = MLP(output_size=6)  # number of classes
        else:
            print('error: no module in Autoencoder3d.')

        weight_init(self)

    def forward(self, x):
        feature = self.feature_extractor(x)
        # out = self.decoder(feature)
        if self.is_reconst:
            out = self.decoder(feature)
        elif self.is_classify:
            out = self.classify(feature)
        else:
            out = self.decoder(feature)
            print('error: no module in Autoencoder3d.')
        return feature, out


# transfer the channel and length of the input data: Autoencoder3d transfer
class Encoder3d4(nn.Module):
    def __init__(self):
        super(Encoder3d4, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=3,
                               out_channels=64,
                               kernel_size=3,
                               padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(kernel_size=2,
                                  stride=2,
                                  return_indices=True)

        self.conv2 = nn.Conv1d(in_channels=64,
                               out_channels=128,
                               kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(kernel_size=2,
                                  stride=2,
                                  return_indices=True)

        self.conv3 = nn.Conv1d(in_channels=128,
                               out_channels=128,
                               kernel_size=3,
                               padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        self.pool3 = nn.MaxPool1d(kernel_size=2,
                                  stride=2,
                                  return_indices=True)

        self.flattened_size = 128 * 6  # 12  # Adjust this based on input size and pooling
        self.fc = nn.Linear(self.flattened_size, 64)

    def forward(self, x):
        # input: batch, channel, length
        # x = input.permute(0, 2, 1)
        size1 = x.size()
        x, idx1 = self.pool1(F.elu(self.bn1(self.conv1(x))))
        size2 = x.size()
        x, idx2 = self.pool2(F.elu(self.bn2(self.conv2(x))))
        size3 = x.size()
        x, idx3 = self.pool3(F.elu(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)  # Flatten: batch, 128*6
        # x = self.fc(x)  # Latent space
        return x, [idx1, idx2, idx3], [size1, size2, size3]


# Decoder structure with skip connections and strides=1 for finer upsampling
class Decoder3d4(nn.Module):
    def __init__(self):
        super(Decoder3d4, self).__init__()
        # self.fc = nn.Linear(64, 128 * 6)  # Match encoder flattened size
        self.unflatten = nn.Unflatten(1, (128, 6))

        self.conv_trans1 = nn.ConvTranspose1d(in_channels=128,
                                              out_channels=128,
                                              kernel_size=3,
                                              stride=1,
                                              padding=1)
        self.bn1 = nn.BatchNorm1d(128)
        self.unpool1 = nn.MaxUnpool1d(2, stride=2)

        self.conv_trans2 = nn.ConvTranspose1d(in_channels=128,
                                              out_channels=64,
                                              kernel_size=3,
                                              stride=1,
                                              padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.unpool2 = nn.MaxUnpool1d(2, stride=2)

        self.conv_trans3 = nn.ConvTranspose1d(in_channels=64,
                                              out_channels=3,
                                              kernel_size=3,
                                              stride=1,
                                              padding=1)
        self.bn3 = nn.BatchNorm1d(3)
        self.unpool3 = nn.MaxUnpool1d(2, stride=2)
        # self.linear = nn.Linear(47, 48)

    def forward(self, x, idxs, sizes):
        [idx1, idx2, idx3] = idxs
        [size1, size2, size3] = sizes
        # x = F.elu(self.fc(x))
        x = self.unflatten(x)
        x = self.unpool1(x, idx3,
                         output_size=size3)
        x = F.elu(self.bn1(self.conv_trans1(x)))
        x = self.unpool2(x, idx2,
                         output_size=size2)
        x = F.elu(self.bn2(self.conv_trans2(x)))
        x = self.unpool3(x, idx1,
                         output_size=size1)
        x = self.bn3(self.conv_trans3(x))

        # x = torch.sigmoid(self.conv_trans3(x))  # Sigmoid to normalize output between 0 and 1
        # x = self.linear(x)
        return x


class Autoencoder3d4(nn.Module):
    def __init__(self, is_reconst=True, is_classify=False):
        super(Autoencoder3d4, self).__init__()
        self.feature_extractor = Encoder3d4()

        self.is_reconst = is_reconst
        self.is_classify = is_classify
        if self.is_reconst:
            self.decoder = Decoder3d4()
        elif self.is_classify:
            # self.classify = EvaClassifier(output_size=12)
            self.classify = MLP(output_size=5)
        else:
            print('error: no module in Autoencoder3d.')

        # Weight initialization (if you use custom weight_init function)
        weight_init(self)

    def forward(self, x):
        feature, idxs, sizes = self.feature_extractor(x)  # Get features and skip connections
        if self.is_reconst:
            out = self.decoder(feature, idxs, sizes)  # Pass skip connections to the decoder
        elif self.is_classify:
            out = self.classify(feature)
        else:
            out = self.decoder(feature)
            print('error: no module in Autoencoder3d.')
        return feature, out


class Encoder4d(nn.Module):
    def __init__(self):
        super(Encoder4d, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=4,
                               out_channels=64,
                               kernel_size=3,
                               padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(kernel_size=2,
                                  stride=2,
                                  return_indices=True)

        self.conv2 = nn.Conv1d(in_channels=64,
                               out_channels=128,
                               kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(kernel_size=2,
                                  stride=2,
                                  return_indices=True)

        self.conv3 = nn.Conv1d(in_channels=128,
                               out_channels=128,
                               kernel_size=3,
                               padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        self.pool3 = nn.MaxPool1d(kernel_size=2,
                                  stride=2,
                                  return_indices=True)

        self.flattened_size = 128 * 6  # 12  # Adjust this based on input size and pooling
        self.fc = nn.Linear(self.flattened_size, 64)

    def forward(self, x):
        # input: batch, channel, length
        # x = input.permute(0, 2, 1)
        size1 = x.size()
        x, idx1 = self.pool1(F.elu(self.bn1(self.conv1(x))))
        size2 = x.size()
        x, idx2 = self.pool2(F.elu(self.bn2(self.conv2(x))))
        size3 = x.size()
        x, idx3 = self.pool3(F.elu(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)  # Flatten: batch, 128*6
        # x = self.fc(x)  # Latent space
        return x, [idx1, idx2, idx3], [size1, size2, size3]

class Decoder4d(nn.Module):
    def __init__(self):
        super(Decoder4d, self).__init__()
        # self.fc = nn.Linear(64, 128 * 6)  # Match encoder flattened size
        self.unflatten = nn.Unflatten(1, (128, 6))

        self.conv_trans1 = nn.ConvTranspose1d(in_channels=128,
                                              out_channels=128,
                                              kernel_size=3,
                                              stride=1,
                                              padding=1)
        self.bn1 = nn.BatchNorm1d(128)
        self.unpool1 = nn.MaxUnpool1d(2, stride=2)

        self.conv_trans2 = nn.ConvTranspose1d(in_channels=128,
                                              out_channels=64,
                                              kernel_size=3,
                                              stride=1,
                                              padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.unpool2 = nn.MaxUnpool1d(2, stride=2)

        self.conv_trans3 = nn.ConvTranspose1d(in_channels=64,
                                              out_channels=4,
                                              kernel_size=3,
                                              stride=1,
                                              padding=1)
        self.bn3 = nn.BatchNorm1d(4)
        self.unpool3 = nn.MaxUnpool1d(2, stride=2)
        # self.linear = nn.Linear(47, 48)

    def forward(self, x, idxs, sizes):
        [idx1, idx2, idx3] = idxs
        [size1, size2, size3] = sizes
        # x = F.elu(self.fc(x))
        x = self.unflatten(x)
        x = self.unpool1(x, idx3,
                         output_size=size3)
        x = F.elu(self.bn1(self.conv_trans1(x)))
        x = self.unpool2(x, idx2,
                         output_size=size2)
        x = F.elu(self.bn2(self.conv_trans2(x)))
        x = self.unpool3(x, idx1,
                         output_size=size1)
        x = self.bn3(self.conv_trans3(x))

        # x = torch.sigmoid(self.conv_trans3(x))  # Sigmoid to normalize output between 0 and 1
        # x = self.linear(x)
        return x
class Autoencoder4d(nn.Module):
    def __init__(self, is_reconst=True, is_classify=False):
        super(Autoencoder4d, self).__init__()
        self.feature_extractor = Encoder4d()

        self.is_reconst = is_reconst
        self.is_classify = is_classify
        if self.is_reconst:
            self.decoder = Decoder4d()
        elif self.is_classify:
            # self.classify = EvaClassifier(output_size=12)
            self.classify = MLP(output_size=5)
        else:
            print('error: no module in Autoencoder3d.')

        # Weight initialization (if you use custom weight_init function)
        weight_init(self)

    def forward(self, x):
        feature, idxs, sizes = self.feature_extractor(x)  # Get features and skip connections
        if self.is_reconst:
            out = self.decoder(feature, idxs, sizes)  # Pass skip connections to the decoder
        elif self.is_classify:
            out = self.classify(feature)
        else:
            out = self.decoder(feature)
            print('error: no module in Autoencoder3d.')
        return feature, out



class Encoder2d(nn.Module):
    def __init__(self):
        super(Encoder2d, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=2, out_channels=64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(kernel_size=2,
                                  stride=2,
                                  return_indices=True)

        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(kernel_size=2,
                                  stride=2,
                                  return_indices=True)

        self.conv3 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        self.pool3 = nn.MaxPool1d(kernel_size=2,
                                  stride=2,
                                  return_indices=True)

        # Calculate the flattened size after all convolutions and pooling
        self.flattened_size = 128 * 6  # Adjust this to match the output of the last pooling layer
        self.fc = nn.Linear(self.flattened_size, 64)

    def forward(self, x):
        # input: batch, channel, length
        size1 = x.size()
        x, idx1 = self.pool1(F.elu(self.bn1(self.conv1(x))))
        size2 = x.size()
        x, idx2 = self.pool2(F.elu(self.bn2(self.conv2(x))))
        size3 = x.size()
        x, idx3 = self.pool3(F.elu(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)  # Flatten: batch, 128*6
        # x = self.fc(x)  # Latent space
        return x, [idx1, idx2, idx3], [size1, size2, size3]


class Decoder2d(nn.Module):
    def __init__(self):
        super(Decoder2d, self).__init__()
        # self.fc = nn.Linear(64, 64 * 37)  # Adjust size if necessary
        self.unflatten = nn.Unflatten(1, (128, 6))

        self.conv_trans1 = nn.ConvTranspose1d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        # self.upsample1 = nn.Upsample(scale_factor=2)
        self.unpool1 = nn.MaxUnpool1d(2, stride=2)
        self.bn1 = nn.BatchNorm1d(128)

        self.conv_trans2 = nn.ConvTranspose1d(in_channels=128, out_channels=64, kernel_size=3, padding=1)
        # self.upsample2 = nn.Upsample(scale_factor=2)
        self.unpool2 = nn.MaxUnpool1d(2, stride=2)
        self.bn2 = nn.BatchNorm1d(64)

        self.conv_trans3 = nn.ConvTranspose1d(in_channels=64, out_channels=2, kernel_size=3, padding=1)
        # self.upsample3 = nn.Upsample(scale_factor=2)
        self.unpool3 = nn.MaxUnpool1d(2, stride=2)
        self.bn3 = nn.BatchNorm1d(1)

        # self.linear = nn.Linear(296, 300)
    def forward(self, x, idxs, sizes):
        [idx1, idx2, idx3] = idxs
        [size1, size2, size3] = sizes
        x = self.unflatten(x)
        x = self.unpool1(x, idx3,
                         output_size=size3)
        x = F.elu(self.bn1(self.conv_trans1(x)))
        x = self.unpool2(x, idx2,
                         output_size=size2)
        x = F.elu(self.bn2(self.conv_trans2(x)))
        x = self.unpool3(x, idx1,
                         output_size=size1)
        x = self.bn3(self.conv_trans3(x))
        return x


class Autoencoder2d(nn.Module):
    def __init__(self):
        super(Autoencoder2d, self).__init__()
        self.feature_extractor = Encoder2d()
        self.decoder = Decoder2d()

        weight_init(self)

    def forward(self, x):
        feature, idxs, sizes = self.feature_extractor(x)
        out = self.decoder(feature, idxs, sizes)
        return feature, out


class Encoder1d(nn.Module):
    def __init__(self):
        super(Encoder1d, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(kernel_size=2,
                                  stride=2,
                                  return_indices=True)

        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(kernel_size=2,
                                  stride=2,
                                  return_indices=True)

        self.conv3 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        self.pool3 = nn.MaxPool1d(kernel_size=2,
                                  stride=2,
                                  return_indices=True)

        # Calculate the flattened size after all convolutions and pooling
        self.flattened_size = 128 * 6  # Adjust this to match the output of the last pooling layer
        self.fc = nn.Linear(self.flattened_size, 64)

    def forward(self, x):
        # input: batch, channel, length
        size1 = x.size()
        x, idx1 = self.pool1(F.elu(self.bn1(self.conv1(x))))
        size2 = x.size()
        x, idx2 = self.pool2(F.elu(self.bn2(self.conv2(x))))
        size3 = x.size()
        x, idx3 = self.pool3(F.elu(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)  # Flatten: batch, 128*6
        # x = self.fc(x)  # Latent space
        return x, [idx1, idx2, idx3], [size1, size2, size3]


class Decoder1d(nn.Module):
    def __init__(self):
        super(Decoder1d, self).__init__()
        # self.fc = nn.Linear(64, 64 * 37)  # Adjust size if necessary
        self.unflatten = nn.Unflatten(1, (128, 6))

        self.conv_trans1 = nn.ConvTranspose1d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        # self.upsample1 = nn.Upsample(scale_factor=2)
        self.unpool1 = nn.MaxUnpool1d(2, stride=2)
        self.bn1 = nn.BatchNorm1d(128)

        self.conv_trans2 = nn.ConvTranspose1d(in_channels=128, out_channels=64, kernel_size=3, padding=1)
        # self.upsample2 = nn.Upsample(scale_factor=2)
        self.unpool2 = nn.MaxUnpool1d(2, stride=2)
        self.bn2 = nn.BatchNorm1d(64)

        self.conv_trans3 = nn.ConvTranspose1d(in_channels=64, out_channels=1, kernel_size=3, padding=1)
        # self.upsample3 = nn.Upsample(scale_factor=2)
        self.unpool3 = nn.MaxUnpool1d(2, stride=2)
        self.bn3 = nn.BatchNorm1d(1)

        # self.linear = nn.Linear(296, 300)
    def forward(self, x, idxs, sizes):
        [idx1, idx2, idx3] = idxs
        [size1, size2, size3] = sizes
        x = self.unflatten(x)
        x = self.unpool1(x, idx3,
                         output_size=size3)
        x = F.elu(self.bn1(self.conv_trans1(x)))
        x = self.unpool2(x, idx2,
                         output_size=size2)
        x = F.elu(self.bn2(self.conv_trans2(x)))
        x = self.unpool3(x, idx1,
                         output_size=size1)
        x = self.bn3(self.conv_trans3(x))
        return x


class Autoencoder1d(nn.Module):
    def __init__(self):
        super(Autoencoder1d, self).__init__()
        self.feature_extractor = Encoder1d()
        self.decoder = Decoder1d()

        weight_init(self)

    def forward(self, x):
        feature, idxs, sizes = self.feature_extractor(x)
        out = self.decoder(feature, idxs, sizes)
        return feature, out
#-----------------------autoencoder------------------------------------------

# Model weight initialization function
def weight_init(m, mode="fan_out", nonlinearity="relu"):
    if isinstance(m, nn.Conv1d) or isinstance(m, nn.ConvTranspose1d):
        nn.init.kaiming_normal_(m.weight, mode=mode, nonlinearity=nonlinearity)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode=mode, nonlinearity=nonlinearity)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

class EvaClassifier(nn.Module):
    def __init__(self, input_size=64, nn_size=512, output_size=2):
        super(EvaClassifier, self).__init__()
        self.linear1 = torch.nn.Linear(input_size, nn_size)
        self.linear2 = torch.nn.Linear(nn_size, output_size)

    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = F.softmax(x, dim=-1)
        return x

class Classifier(nn.Module):
    def __init__(self, input_size=1024, output_size=2):
        super(Classifier, self).__init__()
        self.linear1 = torch.nn.Linear(input_size, output_size)

    def forward(self, x):
        y_pred = self.linear1(x)
        return y_pred


class MLP(nn.Module):
    def __init__(self, input_size=64, hidden_size=32, output_size=5):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        out = self.softmax(x)
        return out


class ProjectionHead(nn.Module):
    def __init__(self, input_size=1024, nn_size=256, encoding_size=100):
        super(ProjectionHead, self).__init__()
        self.linear1 = torch.nn.Linear(input_size, nn_size)
        self.linear2 = torch.nn.Linear(nn_size, encoding_size)

    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        return x


# cross model contrastive learning models
class ProjectionHead(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        """
        Args:
            input_dim: 输入特征的维度 (e.g., 编码器输出的维度)
            hidden_dim: 隐藏层的维度 (通常设置较大，如 2048)
            output_dim: 输出维度 (e.g., 对比学习空间的维度，如 128)
        """
        super(ProjectionHead, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        """
        Forward pass for the projection head.
        Args:
            x: 输入特征 (通常是编码器输出的特征向量)
        Returns:
            z: 经过投影并 L2 归一化的特征向量
        """
        x = F.relu(self.fc1(x))  # 全连接层 + ReLU
        x = self.fc2(x)  # 第二个全连接层
        z = F.normalize(x, dim=1)  # L2 归一化
        return z
class CrossModelAutoencoderContrastiveModel(nn.Module):
    def __init__(self, input_size=1024, hidden_size=512, output_size=128):
        super(CrossModelAutoencoderContrastiveModel, self).__init__()
        self.acc_feature_extractor = Encoder3d4()
        self.acc_decoder = Decoder3d4()
        self.press_feature_extractor = Encoder1d()
        self.press_decoder = Decoder1d()

        self.accel_projection = ProjectionHead(768, 512, 215)
        self.press_projection = ProjectionHead(768, 512, 215)

        self.temperature = 0.3


    def forward(self, acc, press):
        acc_feature, idxs, sizes = self.acc_feature_extractor(acc)  # Get features and skip connections
        press_feature, idxs, sizes = self.press_feature_extractor(press)

        acc_features = self.accel_projection(acc_feature)
        press_features = self.press_projection(press_feature)

        # 归一化嵌入
        acc_features = F.normalize(acc_features, p=2, dim=1)
        press_features = F.normalize(press_features, p=2, dim=1)

        # feature extractor visualization, projector for regression
        return acc_feature, acc_features, press_feature, press_features

    def calculate_loss(self, acc_features, press_features):

        # 计算相似度矩阵 (余弦相似度)
        logits_per_image = torch.matmul(acc_features, press_features.t()) / self.temperature
        logits_per_text = logits_per_image.t()

        # 创建标签：每个图像与其对应文本的索引是匹配的
        batch_size = acc_features.size(0)
        targets = torch.arange(batch_size, device=acc_features.device)

        # 计算交叉熵损失
        loss_image_to_text = F.cross_entropy(logits_per_image, targets)
        loss_text_to_image = F.cross_entropy(logits_per_text, targets)

        # 返回平均损失
        loss = (loss_image_to_text + loss_text_to_image) / 2
        return loss


# 对比学习损失 (NT-Xent Loss)
class NTXentloss(nn.Module):
    def __init__(self):
        super(NTXentloss, self).__init__()
    def forward(self, features_1, features_2, temperature=0.5):
        """
        Compute NT-Xent contrastive loss for two sets of features.
        :param features_1: Tensor of shape (batch_size, hidden_dim) from modality 1
        :param features_2: Tensor of shape (batch_size, hidden_dim) from modality 2
        :param temperature: Temperature scaling factor
        :return: Contrastive loss scalar
        """
        batch_size = features_1.size(0)

        # Normalize features
        features_1 = F.normalize(features_1, dim=1)
        features_2 = F.normalize(features_2, dim=1)

        # Concatenate features to form a joint batch
        features = torch.cat([features_1, features_2], dim=0)  # (2 * batch_size, hidden_dim)

        # Compute similarity matrix (2N x 2N)
        similarity_matrix = torch.matmul(features, features.T)  # (2 * batch_size, 2 * batch_size)

        # Remove self-similarity by masking the diagonal
        mask = torch.eye(2 * batch_size, device=features.device).bool()
        similarity_matrix = similarity_matrix.masked_fill(mask, float('-inf'))

        # Scale by temperature
        logits = similarity_matrix / temperature

        # Create labels: positives are diagonal in cross-modality (e.g., [0, batch_size], [1, batch_size + 1], ...)
        labels = torch.cat([torch.arange(batch_size, 2 * batch_size),
                           torch.arange(0, batch_size)]).to(features.device)

        # Compute NT-Xent loss using cross entropy
        loss = F.cross_entropy(logits, labels)
        return loss
