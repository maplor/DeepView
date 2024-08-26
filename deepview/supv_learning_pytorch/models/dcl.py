
import numpy as np
import torch
import torch.nn as nn
# from deepview.supv_learning_pytorch.utils.utils import *
from ..utils import utils

class DeepConvLSTM(nn.Module): # DCL
    """
    Imprementation of DeepConvLSTM [Sensors 2016].
    
    Note:
        https://www.mdpi.com/1424-8220/16/1/115 (Sensors, 2016)
    
    """

    def __init__(self, cfg):
        super().__init__()
        
        self.in_ch = cfg.dataset.in_ch
        self.window_size = cfg.dataset.window_size
        self.num_classes = cfg.dataset.n_classes
        self.kernel_size = cfg.model.kernel_size                    
        self.num_conv_layers = cfg.model.num_conv_layers
        self.num_conv_filters = cfg.model.num_conv_filters
        self.num_lstm_layers = cfg.model.num_lstm_layers
        self.num_lstm_hidden_units = cfg.model.num_lstm_hidden_units
        self.bidirectional_lstm = cfg.model.bidirectional_lstm
        self.out_layer_type = cfg.model.out_layer_type
        self.out_dropout_rate = cfg.model.out_dropout_rate  
        if 'cuda:' in str(cfg.train.cuda):
            self.cuda_device = cfg.train.cuda
        else:
            self.cuda_device = 'cuda:' + str(cfg.train.cuda)

        # -- [1] CNN (Convolution layers) --
        blocks = []
        for i in range(self.num_conv_layers):
            in_ch_ = self.in_ch if i == 0 else self.num_conv_filters
            blocks.append(
                nn.Sequential(
                    nn.Conv2d(in_ch_, 
                              self.num_conv_filters, 
                              # stride=1, # stride=1 as default
                              # dilation=1, # dilation=1 as default
                              kernel_size=(self.kernel_size, 1), 
                              padding=(self.kernel_size // 2, 0)), # 5 // 2 → 2
                    nn.BatchNorm2d(self.num_conv_filters),
                    nn.ReLU(),
                )
            )
        self.conv_blocks = nn.ModuleList(blocks)

        # -- [2] LSTM Encoder --
        lstm_blocks = []
        for i in range(self.num_lstm_layers):  
            if self.bidirectional_lstm == True:
                if i == 0:
                    input_lstm_units = self.num_conv_filters
                else:
                    input_lstm_units = self.num_lstm_hidden_units * 2
                output_lstm_units = self.num_lstm_hidden_units * 2
            else:
                if i == 0:
                    input_lstm_units = self.num_conv_filters
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
        # -- [1] Convolution Layer --
        for block in self.conv_blocks:
            x = block(x)                 

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
        
        # -- [2] LSTM --
        # Reshape: (B, CH, T, 1) -> (B, T, CH)
        # nn.LSTM(): batch_first=True
        x = x.squeeze(3).transpose(1, 2)
        for i in range(self.num_lstm_layers):
            x, _ = self.lstm_blocks[i](x)
            x = self.dropout_blocks[i](x)

        # Reshape: (B, T, CH) -> (B, CH, T, 1)
        x = x.transpose(1, 2).unsqueeze(3) 
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

class DeepConvLstmV3(nn.Module):
    def __init__(self, in_ch: int = 3, num_classes: int = 11):
        super().__init__()
        # if num_classes is None:
        #     num_classes = len(OPENPACK_OPERATIONS)

        # -- [1] CNN --
        # *** Edit Here ***
        num_conv_layers = 4  # convolutional layers (Default: 4)
        num_conv_filter = 64  # convolutional filters (Default: 64)
        ks = 5  # kernel size,
        # ******************

        blocks = []
        for i in range(num_conv_layers):
            in_ch_ = in_ch if i == 0 else 64
            blocks.append(
                nn.Sequential(
                    nn.Conv2d(in_ch_, 64, kernel_size=(5, 1), padding=(2, 0)),
                    nn.BatchNorm2d(64),
                    nn.ReLU(),
                )
            )
        self.conv_blocks = nn.ModuleList(blocks)

        # -- [2] LSTM --
        # *** Edit Here ***
        hidden_units = 128  # number of hidden units for Bi-LSTM
        # ******************

        # NOTE: enable ``bidirectional``
        self.lstm6 = nn.LSTM(num_conv_filter, hidden_units, batch_first=True, bidirectional=True)
        self.lstm7 = nn.LSTM(hidden_units*2, hidden_units, batch_first=True,  bidirectional=True)
        self.dropout6 = nn.Dropout(p=0.3)
        self.dropout7 = nn.Dropout(p=0.3)

        # -- [3] Output --
        self.out8 = nn.Conv2d(
            hidden_units * 2,
            num_classes,
            1,
            stride=1,
            padding=0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): shape = (B, CH, T, 1)
        Returns:
            torch.Tensor: shape = (B, N_CLASSES, T, 1)
        """
        # input shape from (B, T, CH) to (B, CH, T, 1)
        x = x.unsqueeze(3)
        x = x.transpose(2, 1)

        # -- [1] Conv --
        for block in self.conv_blocks:
            x = block(x)

        # -- [2] LSTM --
        # Reshape: (B, CH, 1, T) -> (B, T, CH)
        x = x.squeeze(3).transpose(1, 2)

        x, _ = self.lstm6(x)
        x = self.dropout6(x)
        x, _ = self.lstm7(x)
        x = self.dropout7(x)

        # Reshape: (B, T, CH) -> (B, CH, T, 1)
        x = x.transpose(1, 2).unsqueeze(3)

        # -- [3] Output --
        x = self.out8(x)
        x_out = torch.transpose(x.squeeze(-1), 2, 1)  # output.shape=(batch,datalen,cls)

        return x_out