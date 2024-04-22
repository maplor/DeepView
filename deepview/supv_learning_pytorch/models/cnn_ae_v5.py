import numpy as np
import torch
import torch.nn as nn
from ..utils import utils

# The code is based on https://github.com/Tian0426/CL-HAR, but modified for this study.

class CNN_AE5(nn.Module): # CNN-AE or AE5 
    def __init__(self, cfg):
        super(CNN_AE5, self).__init__()
        
        self.in_ch = cfg.dataset.in_ch
        self.window_size = cfg.dataset.window_size
        self.num_classes = cfg.dataset.n_classes
        self.pretrain = cfg.model.pretrain
        self.out_layer_type = cfg.model.out_layer_type
        self.out_dropout_rate = cfg.model.out_dropout_rate
        if 'cuda:' in str(cfg.train.cuda):
            self.cuda_device = cfg.train.cuda
        else:
            self.cuda_device = 'cuda:' + str(cfg.train.cuda)
            
        self.e_conv1 = nn.Sequential(
            nn.Conv2d(
                self.in_ch,
                128,
                kernel_size=(5,1),
                stride=1,
                padding=(2,0),
                bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU())
        self.pool1 = nn.MaxPool2d(
            kernel_size=(2,1),
            stride=2,
            padding=(1,0),
            return_indices=True)
        self.dropout = nn.Dropout(0.2)

        self.e_conv2 = nn.Sequential(
            nn.Conv2d(
                128,
                128,
                kernel_size=(5,1),
                stride=1,
                padding=(2,0),
                bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU())
        self.pool2 = nn.MaxPool2d(
            kernel_size=(2,1),
            stride=2,
            padding=(1,0),
            return_indices=True)

        self.e_conv3 = nn.Sequential(
            nn.Conv2d(
                128,
                128,
                kernel_size=(5,1),
                stride=1,                
                padding=(2,0),
                bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU())
        self.pool3 = nn.MaxPool2d(
            kernel_size=(2,1),
            stride=2,
            padding=(1,0),
            return_indices=True)

        self.unpool1 = nn.MaxUnpool2d(kernel_size=2, stride=2, padding=1)
        self.d_conv1 = nn.Sequential(
            nn.ConvTranspose2d(
                128,
                128,
                # kernel_size=(1,5),
                # stride=1,
                # padding=(0,2),
                kernel_size=(5,1),
                stride=1,
                padding=(2,0),
                bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU())

        self.unpool2 = nn.MaxUnpool2d(kernel_size=2, stride=2, padding=1)
        self.d_conv2 = nn.Sequential(
            nn.ConvTranspose2d(
                128,
                128,
                # kernel_size=(1,5),
                # stride=1,
                # padding=(0,2),
                kernel_size=(5,1),
                stride=1,
                padding=(2,0),
                bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU())

        self.unpool3 = nn.MaxUnpool2d(kernel_size=2, stride=2, padding=1)
        self.d_conv3 = nn.Sequential(
            nn.ConvTranspose2d(
                128,
                self.in_ch,
                # kernel_size=(1,5),
                # stride=1,
                # padding=(0,2),
                kernel_size=(5,1),
                stride=1,
                padding=(2,0),
                bias=False),
            nn.BatchNorm2d(self.in_ch)
            #, nn.ReLU(),
            )
        
        # Linear Layer
        self.linear1 = nn.Linear(8, self.window_size)
        self.linear2 = nn.Linear(self.window_size, self.window_size)

        # -- [4] Output Layer --
        self.dim_after_flatten = 128*self.window_size
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
            self.out = nn.Conv2d(128, 
                                 self.num_classes, 
                                 1,
                                 stride=1, 
                                 padding=0)


    def forward(self, x, y=None, mixup=False, mixup_alpha=0.2):
        # -- [1] Encoder --
        # 0: torch.Size([128, 3, 50, 1]) # (B, CH, T, 1)
        # 1: torch.Size([128, 128, 26, 1])
        # 2: torch.Size([128, 128, 14, 1])
        # 3: torch.Size([128, 128, 8, 1])
        x = self.e_conv1(x)
        x, indice1 = self.pool1(x)
        x = self.dropout(x)
        x = self.e_conv2(x)
        x, indice2 = self.pool2(x)
        x = self.e_conv3(x)
        x, indice3 = self.pool3(x)
        x_encoded = x
        x_encoded = self.linear1(x_encoded.squeeze(3)) # torch.Size([128, 128, 50]) 
        x_encoded = self.dropout(x_encoded) 
        x_encoded = x_encoded.unsqueeze(3) # torch.Size([128, 128, 50, 1]) 
        
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
            x_encoded, y_onehot = utils.mixup_process(x_encoded, y_onehot, lam)
            
            x_encoded = x_encoded.to(f"{self.cuda_device}")
            y_onehot = y_onehot.to(f"{self.cuda_device}")

        # -- [2] Decoder --
        # 0: torch.Size([128, 128, 8, 1]
        # 1: torch.Size([128, 128, 14, 1]
        # 2: torch.Size([128, 128, 26, 1]
        # 3: torch.Size([128, 3, 50, 1]
        x = self.unpool1(x, indice3, output_size=torch.Size([128, 128, 14, 1]))
        x = self.d_conv1(x)
        x = self.unpool2(x, indice2, output_size=torch.Size([128, 128, 26, 1]))
        x = self.d_conv2(x)
        x = self.unpool3(x, indice1, output_size=torch.Size([128, 128, 50, 1]))
        x = self.d_conv3(x) # torch.Size([128, 3, 50, 1]) without ReLU
        x = self.dropout(x.squeeze(3)) # torch.Size([128, 3, 50])
        x_decoded = self.linear2(x).unsqueeze(3) # torch.Size([128, 3, 50, 1])

        # -- [3] Output --
        feats = x_encoded
        out = self.out(x_encoded) # torch.Size([128, 128, 50, 1])  -> torch.Size([128, 6, 50, 1]) 
        
        if self.pretrain == True:
            return out, x_decoded, x_encoded
        else:
            if mixup == True:
                if "flatten-linear" in self.out_layer_type:
                    # (B, C)
                    y_onehot = y_onehot
                else:
                    # (B, T, 1, C) -> (B, C, T, 1)
                    # print(f"{y_onehot.shape}") # torch.Size([128, 50, 1, 6])
                    y_onehot = y_onehot.transpose(1, 3).transpose(2, 3)
                    # print(f"{y_onehot.shape}") # torch.Size([128, 6, 50, 1])
                return out, y_onehot, feats
            else:
                return out, None, feats