# This Python file uses the following encoding: utf-8
import numpy as np
import torch
import torch.nn as nn
# from src import utils
# from deepview.supv_learning_pytorch.utils.utils import *
from ..utils import utils

class LSTM(nn.Module):
    """
    Long Short-Term Memory
    """

    def __init__(self, cfg):
        super().__init__()
        
        self.in_ch = cfg.dataset.in_ch
        self.window_size = cfg.dataset.window_size
        self.num_classes = cfg.dataset.n_classes
        self.num_lstm_layers = cfg.model.num_lstm_layers
        self.num_lstm_hidden_units = cfg.model.num_lstm_hidden_units
        self.bidirectional_lstm = cfg.model.bidirectional_lstm
        self.out_layer_type = cfg.model.out_layer_type
        self.out_dropout_rate = cfg.model.out_dropout_rate  
        if 'cuda:' in str(cfg.train.cuda):
            self.cuda_device = cfg.train.cuda
        else:
            self.cuda_device = 'cuda:' + str(cfg.train.cuda)

        # -- [2] LSTM Encoder --
        lstm_blocks = []
        for i in range(self.num_lstm_layers):  
            if self.bidirectional_lstm == True:
                if i == 0:
                    # input_lstm_units = self.num_conv_filters
                    input_lstm_units = self.in_ch
                else:
                    input_lstm_units = self.num_lstm_hidden_units * 2
                output_lstm_units = self.num_lstm_hidden_units * 2
            else:
                if i == 0:
                    # input_lstm_units = self.num_conv_filters
                    input_lstm_units = self.in_ch
                else:
                    input_lstm_units = self.num_lstm_hidden_units
                output_lstm_units = self.num_lstm_hidden_units
            lstm_blocks.append(
                nn.Sequential(
                    nn.LSTM(input_lstm_units, 
                            self.num_lstm_hidden_units, 
                            batch_first=True, 
                            bidirectional=self.bidirectional_lstm)
                )
            )
        self.lstm_blocks = nn.ModuleList(lstm_blocks)

        dropout_blocks = []
        for i in range (self.num_lstm_layers):
            dropout_blocks.append(
                nn.Sequential(
                    nn.Dropout(p=0.5)
                )
            )
        self.dropout_blocks = nn.ModuleList(dropout_blocks)
        
        if self.bidirectional_lstm == True:
            self.output_lstm_units = self.num_lstm_hidden_units*2
        else:
            self.output_lstm_units = self.num_lstm_hidden_units
            
        # -- [3] Output --
        self.dim_after_flatten = self.output_lstm_units*self.window_size
        if self.out_layer_type == "flatten-linear":
            self.out = nn.Sequential(
                nn.Flatten(start_dim=1, end_dim=-1), # (B, CH, T, 1) -> (B, CH*T)
                nn.Dropout(p=0.5),
                nn.Linear(self.dim_after_flatten,
                          128),
                nn.Dropout(p=0.2),
                nn.Linear(128,
                          self.num_classes)
            )
        elif self.out_layer_type == "flatten-linear-2":
            self.out = nn.Sequential(
                nn.Flatten(start_dim=1, end_dim=-1), # (B, CH, T, 1) -> (B, CH*T)
                nn.Dropout(p=self.out_dropout_rate),
                nn.Linear(self.dim_after_flatten,
                          self.num_classes) # (B, CH*T) -> (B, C)
            )
        else:
            # https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
            self.out = nn.Conv2d(self.output_lstm_units, # in_channels
                                 self.num_classes, # out_channels
                                 1, # square kernels -> kernel_size=(1, 1) 1*1 convolution layer
                                 stride=1, 
                                 padding=0)
        

    def forward(self, 
                x: torch.Tensor, 
                y: torch.Tensor, 
                mixup=False, 
                mixup_alpha=0.2):
        """
        Args:
            x (torch.Tensor): shape = (B, CH, T, 1)
        Returns:
            torch.Tensor: shape = (B, N_CLASSES, T, 1)
        """

        # -- [2] LSTM --
        # Reshape: (B, CH, T, 1) -> (B, T, CH)
        # nn.LSTM(): batch_first=True
        x = x.squeeze(3).transpose(1, 2)
        for i in range(self.num_lstm_layers):
            x, _ = self.lstm_blocks[i](x)
            x = self.dropout_blocks[i](x)
        # Reshape: (B, T, CH) -> (B, CH, T, 1)
        x = x.transpose(1, 2).unsqueeze(3)

        # Manifold Mixup
        if mixup == True:
            # print(f"mixup: {mixup}")
            # print(f"mixup_alpha = {mixup_alpha}")
            y_onehot = utils.to_one_hot(y, self.num_classes)
            # print(f"y_onehot.shape: {y_onehot.shape}")

            # mixup using for loop
            # x, y_onehot = utils.mixup(x, y_onehot, mixup_alpha=mixup_alpha)
            
            # mixup without for loop (should be faster)
            batch_size = x.size()[0]
            lam = np.random.beta(mixup_alpha, mixup_alpha, batch_size)
            lam = torch.from_numpy(np.array([lam]).astype('float32')).to(f"{self.cuda_device}")
            x, y_onehot = utils.mixup_process(x, y_onehot, lam)
            
            x = x.to(f"{self.cuda_device}")
            y_onehot = y_onehot.to(f"{self.cuda_device}")

        feats = x
        
        # -- [3] Output Layer --
        x = self.out(x)
        
        if mixup == True:
            if "flatten-linear" in self.out_layer_type:
                # (B, C)
                y_onehot = y_onehot
            else:
                # (B, T, 1, C) -> (B, C, T, 1)
                # print(f"{y_onehot.shape}") # torch.Size([128, 50, 1, 6])
                y_onehot = y_onehot.transpose(1, 3).transpose(2, 3)
                # print(f"{y_onehot.shape}") # torch.Size([128, 6, 50, 1])
            return x, y_onehot, feats
        else:
            return x, None, feats