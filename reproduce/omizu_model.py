"""
Omizunagidori model module
===========================
Omizunagidori counterpart of `bear_model.py` / `turtle_model.py`.  The
omizunagidori dataset has a single 3-channel accelerometer stream (no pressure
channel) and a window length of 50, so (identical encoder geometry to bear):

  * one `Encoder3d4` feature extractor over the 3 accel channels.
    After three /2 max-pools, 50 -> 25 -> 12 -> 6, giving 128*6 = 768 features.
  * linear 768 -> 32 shared embedding.
  * projector  -> 16-d head for supervised contrastive learning.
  * classifier -> 7 omizunagidori behaviour classes (raw logits).

`forward(data, if_contrast=True)` returns (head_output, shared_32d_feature),
matching the umineko `SimpleNN` / bear `BearNN` interface so the generic
train_eval helpers (train_model / evaluate_* / uncertainty_sampling /
AE_eval_time_series) work unchanged.
"""

import torch
import torch.nn as nn

from model_func import Encoder3d4
# reuse the identical supervised-contrastive loss
from model import SupContrastiveLoss  # noqa: F401  (re-exported for the notebook)

# 128 channels * (win / 2 / 2 / 2).  win=50 -> 128*6=768;  win=200 -> 128*25=3200.
# Pass input_dim explicitly to OmizuNN when using a non-50 window.
FEATURE_DIM = 128 * 6


class OmizuNN(nn.Module):
    def __init__(self, input_dim=FEATURE_DIM, number_classes=7):
        super(OmizuNN, self).__init__()
        self.acc_encoder = Encoder3d4()

        self.linear = nn.Linear(input_dim, 32)
        self.projector = nn.Sequential(
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
        )
        # NOTE: classifier emits raw logits (no Softmax).  nn.CrossEntropyLoss
        # already applies log-softmax internally; a Softmax layer here would
        # double-softmax, flatten gradients and starve the minority classes.
        # Apply torch.softmax explicitly wherever probabilities are needed
        # (uncertainty sampling, confidence thresholding).
        self.classifier = nn.Sequential(
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, number_classes),
        )

    def forward(self, data, if_contrast=True):
        accel = data[:, :3, :]
        acc_fea, _, _ = self.acc_encoder(accel)
        concat_fea = self.linear(acc_fea)
        if if_contrast:
            output = self.projector(concat_fea)
        else:
            output = self.classifier(concat_fea)
        return output, concat_fea


def freeze_encoders(model):
    """Omizu freeze: train classifier only (single-encoder variant of the
    umineko `freeze_encoders`, which referenced a second pressure encoder)."""
    for param in model.acc_encoder.parameters():
        param.requires_grad = False
    for param in model.linear.parameters():
        param.requires_grad = False
    for param in model.projector.parameters():
        param.requires_grad = False
    for param in model.classifier.parameters():
        param.requires_grad = True
