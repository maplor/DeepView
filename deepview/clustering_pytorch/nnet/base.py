#
# DeepLabCut Toolbox (deeplabcut.org)
# © A. & M.W. Mathis Labs
# https://github.com/DeepLabCut/DeepLabCut
#
# Please see AUTHORS for contributors.
# https://github.com/DeepLabCut/DeepLabCut/blob/master/AUTHORS
#
# Licensed under GNU Lesser General Public License v3.0
#
import abc
import tensorflow as tf
from deepview.clustering_pytorch.datasets import Batch
# from deeplabcut.pose_estimation_tensorflow.core1 import predict_multianimal
# from .layers import prediction_layer
# from .utils import make_2d_gaussian_kernel
import torch.nn as nn
from common_config import adjust_learning_rate
from train_utils import AE_train_time_series, AE_eval_time_series
from prepare_data.time_series_preparation import *


class BaseNet(metaclass=abc.ABCMeta):
    def __init__(self, cfg):
        self.cfg = cfg

    @abc.abstractmethod
    def extract_features(self, inputs):
        ...

    @abc.abstractmethod
    def get_net(self, inputs):
        ...

    def train(self, batch):
        heads = self.get_net(batch[Batch.inputs])
        from models import CNN_AE
        device = 'cpu'
        # Prepare time-series Dataset
        train_dataloaders = prepare_data(args)
        backbone = CNN_AE(n_channels=3, out_channels=128)
        model = ReconstructionFramework(backbone)
        loss_value = 0
        for epoch in range(1, p['epochs']):
            # Adjust lr
            lr = adjust_learning_rate(p, optimizer, epoch)
            print('Adjusted learning rate to {:.5f}'.format(lr))
            loss_value = AE_train_time_series(train_dataloaders[0], model, criterion, optimizer, epoch, device)
            print('loss of the ' + str(epoch) + '-th training epoch is :' + loss_value.__str__())

        # batch[Batch.part_score_targets], heads[pred_layer]

        loss = {"part_loss": loss_value}
        total_loss = loss["part_loss"]
        loss["total_loss"] = total_loss
        return loss

    def test(self, inputs):
        heads = self.get_net(inputs)
        return self.add_inference_layers(heads)

    def prediction_layers(
        self,
        features,
        scope="pose",
        reuse=None,
    ):
        out = {}
        n_joints = self.cfg["num_joints"]
        with tf.compat.v1.variable_scope(scope, reuse=reuse):
            out["part_pred"] = prediction_layer(
                self.cfg,
                features,
                "part_pred",
                n_joints + self.cfg.get("num_idchannel", 0),
            )
            if self.cfg["location_refinement"]:
                out["locref"] = prediction_layer(
                    self.cfg,
                    features,
                    "locref_pred",
                    n_joints * 2,
                )
            if (
                self.cfg["pairwise_predict"]
                and "multi-animal" not in self.cfg["dataset_type"]
            ):
                out["pairwise_pred"] = prediction_layer(
                    self.cfg,
                    features,
                    "pairwise_pred",
                    n_joints * (n_joints - 1) * 2,
                )
            if (
                self.cfg["partaffinityfield_predict"]
                and "multi-animal" in self.cfg["dataset_type"]
            ):
                out["pairwise_pred"] = prediction_layer(
                    self.cfg,
                    features,
                    "pairwise_pred",
                    self.cfg["num_limbs"] * 2,
                )
        out["features"] = features
        return out


class ReconstructionFramework(nn.Module):
    def __init__(self, backbone):
        super(ReconstructionFramework, self).__init__()
        self.backbone = backbone

    def forward(self, x):
        x_encoded, x_decoded = self.backbone(x)
        return x_encoded, x_decoded


# from models import CNN_AE
# backbone = CNN_AE(n_channels=3, out_channels=128)
# model = ReconstructionFramework(backbone)

# Criterion
# print(colored('Retrieve criterion', 'blue'))
criterion = get_criterion(p)
# print('Criterion is {}'.format(criterion.__class__.__name__))
criterion = criterion.to(device)

# Optimizer and scheduler
# print(colored('Retrieve optimizer', 'blue'))
optimizer = get_optimizer(p, model)
