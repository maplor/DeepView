# This Python file uses the following encoding: utf-8
import numpy as np
import torch
import torch.nn as nn
from ..utils import utils


class DeepConvLSTMSelfAttn(nn.Module): # DCLSA
    """
    Imprementation of a DeepConvLSTM with Self-Attention used in 
    'Deep ConvLSTM with self-attention for human activity decoding
    using wearable sensors' (Sensors 2020).

    Note:
        https://ieeexplore.ieee.org/document/9296308 (Sensors 2020)
        
    """

    def __init__(self, cfg):
        super().__init__()
        
        self.in_ch = cfg.dataset.in_ch
        self.window_size = cfg.dataset.window_size
        self.num_classes = cfg.dataset.n_classes
        self.kernel_size = cfg.model.kernel_size
        self.num_conv_layers = cfg.model.num_conv_layers
        self.num_conv_filters = cfg.model.num_conv_filters
        self.dropout_after_conv = cfg.model.dropout_after_conv
        self.num_lstm_hidden_units = cfg.model.num_lstm_hidden_units
        self.num_lstm_layers = cfg.model.num_lstm_layers
        self.bidirectional_lstm = cfg.model.bidirectional_lstm
        self.num_attn_heads = cfg.model.num_attn_heads
        self.num_attn_layers = cfg.model.num_attn_layers
        self.out_layer_type = cfg.model.out_layer_type
        self.out_dropout_rate = cfg.model.out_dropout_rate  
        if 'cuda:' in str(cfg.train.cuda):
            self.cuda_device = cfg.train.cuda
        else:
            self.cuda_device = 'cuda:' + str(cfg.train.cuda)

        # Added for experiment 06 model params (hyperparameter tuning)
        # if cfg.metadata.approach == "experiment-06-model-params" and cfg.metadata.task == "model-params":
        #     self.conv_dropout = cfg.model.conv_dropout # bool
        #     self.num_lstm_hidden_units = self.num_conv_filters # 64 or 128
        # else:
        #     self.conv_dropout = False
        
        # NOTE: The first block is input layer.
        
        # -- [1] Embedding Layer --
        # Convolutions
        conv_blocks = []
        for i in range(self.num_conv_layers):
            in_ch_ = self.in_ch if i == 0 else self.num_conv_filters
            conv_blocks.append(
                nn.Sequential(
                    nn.Conv2d(in_ch_, 
                              self.num_conv_filters, 
                              kernel_size=(self.kernel_size, 1), 
                              # stride=1, # stride=1 as default
                              # dilation=1, # dilation=1 as default
                              padding=(self.kernel_size // 2, 0)), # 5 // 2 → 2
                    nn.BatchNorm2d(self.num_conv_filters),
                    nn.ReLU(),
                )
            )
        self.conv_blocks = nn.ModuleList(conv_blocks)

        # Added for experiment 06 model params (hyperparameter tuning)
        if self.dropout_after_conv == True:
            self.conv_dropout_layer = nn.Dropout(p=0.5)
        
        # -- [2] LSTM Encoder --
        lstm_blocks = []
        for i in range(self.num_lstm_layers):  
            if self.bidirectional_lstm == True:
                if i == 0:
                    input_lstm_units = self.num_conv_filters
                else:
                    input_lstm_units = self.num_lstm_hidden_units * 2
                self.output_lstm_units = self.num_lstm_hidden_units * 2
            else:
                if i == 0:
                    input_lstm_units = self.num_conv_filters
                else:
                    input_lstm_units = self.num_lstm_hidden_units
                self.output_lstm_units = self.num_lstm_hidden_units
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

        # -- [3] Self-Attention --
        attn_blocks = []
        for i in range(self.num_attn_layers):
            attn_blocks.append(
                nn.MultiheadAttention(self.output_lstm_units, 
                                      self.num_attn_heads, 
                                      batch_first=True,)
            )
        self.attn_blocks = nn.ModuleList(attn_blocks)

        # -- [4] Output Layer --
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
            self.out = nn.Conv2d(self.output_lstm_units, 
                                 self.num_classes, 
                                 1, 
                                 stride=1, 
                                 padding=0)

    def forward(self, 
                x: torch.Tensor, 
                y: torch.Tensor, 
                mixup=False, 
                mixup_alpha=0.2) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): shape = (B, CH, T, 1)
        Returns:
            torch.Tensor: shape = (B, N_CLASSES, T, 1)
        """
        # -- [1] Embedding Layer --
        # Convolutions
        for i in range(self.num_conv_layers):
            x = self.conv_blocks[i](x)

        # Added for experiment 06 model params (hyperparameter tuning)
        if self.dropout_after_conv == True:
            x = self.conv_dropout_layer(x)

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
        
        # -- [2] LSTM Encoder --
        # Reshape: (B, CH, T, 1) -> (B, T, CH)
        # nn.LSTM(): batch_first=True
        x = x.squeeze(3).transpose(1, 2)
        for i in range(self.num_lstm_layers):
            x, _ = self.lstm_blocks[i](x)
            x = self.dropout_blocks[i](x)

        # -- [3] Self-Attention --
        for i in range(self.num_attn_layers):
            x, w = self.attn_blocks[i](x.clone(), x.clone(), x.clone())
        # Reshape: (B, T, CH) -> (B, CH, T, 1)
        x = x.transpose(1, 2).unsqueeze(3)
        
        # -- [4] Output Layer --
        
        feats = x
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