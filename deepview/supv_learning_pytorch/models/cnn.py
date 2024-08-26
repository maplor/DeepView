

import numpy as np
import torch
import torch.nn as nn
from ..utils import utils

class CNN(nn.Module):
    """
    Convolutional Neural Network
    """
    
    def __init__(self, cfg):
        super().__init__()
        
        self.in_ch = cfg.dataset.in_ch
        self.window_size = cfg.dataset.window_size
        self.num_classes = cfg.dataset.n_classes
        self.kernel_size = cfg.model.kernel_size          
        self.num_conv_layers = cfg.model.num_conv_layers
        self.num_conv_filters = cfg.model.num_conv_filters
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
                              kernel_size=(self.kernel_size, 1), 
                              # stride=1, # stride=1 as default
                              # dilation=1, # dilation=1 as default
                              padding=(self.kernel_size//2, 0)), # 5 // 2 → 2
                    nn.BatchNorm2d(self.num_conv_filters),
                    nn.ReLU(),
                )
            )
        self.conv_blocks = nn.ModuleList(blocks)

        # -- [2] Output layer  --
        self.dim_after_flatten = self.num_conv_filters*self.window_size
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
            self.out = nn.Conv2d(self.num_conv_filters, # in_channels
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
        # -- [1] CNN (Convolution layers) --
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
        
        # Reshape: (B, CH, T, 1) -> (B, T, CH)
        # x = x.squeeze(3).transpose(1, 2)
        # Reshape: (B, T, CH) -> (B, CH, T, 1)
        # x = x.transpose(1, 2).unsqueeze(3)

        feats = x
        
        # -- [2] Output layer --
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
