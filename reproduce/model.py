"""
Model module
============
Reproduces the "模型与 Supervised Contrastive Learning" block of the notebook.

SimpleNN
--------
Two parallel 1-D conv autoencoder encoders:
  * acc_encoder : Autoencoder3d4 feature extractor over the 3 accel channels.
  * pre_encoder : Autoencoder1d  feature extractor over the 1 pressure channel.
Each encoder flattens to 128*6 = 768 features; concatenated -> 1536, then a
linear layer projects to a 32-d shared embedding.  From there:
  * projector  -> 16-d head used for supervised contrastive learning.
  * classifier -> softmax over `number_classes` for the supervised head.

`forward(data, if_contrast=True)` returns (head_output, shared_32d_feature).

SupContrastiveLoss
------------------
Supervised contrastive loss (temperature 0.3) over the projector embeddings.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from model_func import Autoencoder3d4, Autoencoder1d


class SimpleNN(nn.Module):
    def __init__(self, input_dim=128 * 6, embedding_dim=128, number_classes=5):
        super(SimpleNN, self).__init__()
        self.acc_encoder = Autoencoder3d4().feature_extractor
        self.pre_encoder = Autoencoder1d().feature_extractor

        self.linear = nn.Linear(input_dim * 2, 32)
        self.projector = nn.Sequential(
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
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
        if if_contrast:
            output = self.projector(concat_fea)
        else:
            output = self.classifier(concat_fea)
        return output, concat_fea


class SupContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.3):
        super(SupContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        features = F.normalize(features, dim=1)
        similarity_matrix = torch.matmul(features, features.T) / self.temperature

        # numerical stability
        similarity_matrix = similarity_matrix - torch.max(
            similarity_matrix, dim=1, keepdim=True
        )[0]

        labels1 = labels.unsqueeze(1)
        mask = torch.eye(similarity_matrix.size(0), dtype=torch.bool,
                         device=features.device)
        positive_mask = (labels1 == labels1.T) & ~mask

        exp_sim = torch.exp(similarity_matrix)
        numerator = exp_sim * positive_mask
        denominator = exp_sim * ~mask

        numerator_sum = numerator.sum(dim=1) + 1e-8
        denominator_sum = denominator.sum(dim=1) + 1e-8

        valid_mask = numerator_sum > 0
        loss = -torch.log(numerator_sum[valid_mask] / denominator_sum[valid_mask])
        loss = loss.mean()
        return loss
