
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List
from torch.autograd import Variable
from einops import rearrange, repeat
from ..utils import utils

# The code is based on https://github.com/Tian0426/CL-HAR, but modified for this study.

class Transformer(nn.Module):
    def __init__(self, cfg):
        super(Transformer, self).__init__()
        
        self.in_ch = cfg.dataset.in_ch
        self.window_size = cfg.dataset.window_size
        self.num_classes = cfg.dataset.n_classes
        self.dim = cfg.model.dim
        self.depth = cfg.model.depth
        self.heads = cfg.model.heads
        self.mlp_dim = cfg.model.mlp_dim
        self.dropout = cfg.model.dropout
        self.out_layer_type = cfg.model.out_layer_type
        self.out_dropout_rate = cfg.model.out_dropout_rate  
        if 'cuda:' in str(cfg.train.cuda):
            self.cuda_device = cfg.train.cuda
        else:
            self.cuda_device = 'cuda:' + str(cfg.train.cuda)

        self.transformer = Seq_Transformer(
            n_channel=self.in_ch,
            window_size=self.window_size,
            n_classes=self.num_classes,
            dim=self.dim,
            depth=self.depth,
            heads=self.heads,
            mlp_dim=self.mlp_dim,
            dropout=self.dropout)
        
        # self.classifier = nn.Linear(self.dim, self.num_classes) not used -> comment out
        
        # -- [4] Output Layer --
        self.dim_after_flatten = self.dim*self.window_size
        self.out = nn.Conv2d(self.dim, 
                             self.num_classes, 
                             1, 
                             stride=1, 
                             padding=0)
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
            self.out = nn.Conv2d(self.dim, # in_channels
                                 self.num_classes, # out_channels
                                 1, # square kernels -> kernel_size=(1, 1) 1*1 convolution layer
                                 stride=1, 
                                 padding=0)

    def forward(self, x, y=None, mixup=False, mixup_alpha=0.2):
        # Reshape: (B, CH, T, 1) -> (B, T, CH)
        x = x.squeeze(3).transpose(1, 2) 
        c_t, x = self.transformer(x)
        # feats = x # torch.Size([128, 50, 128])
        x = x.transpose(1, 2).unsqueeze(3) # torch.Size([128, 128, 50, 1])
        
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
        
        feats = x # torch.Size([128, 128, 50, 1])
        
        x = self.out(x) # torch.Size([128, 6, 50, 1]) or torch.Size([128, 6])
        
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

        
class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2)
                             * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dropout=0.):
        super().__init__()
        self.heads = heads
        self.scale = dim ** -0.5
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask=None):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)
        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value=True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, float('-inf'))
            del mask
        self.attn = dots.softmax(dim=-1)
        out = torch.einsum('bhij,bhjd->bhid', self.attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out


class Base_Transformer(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads=heads, dropout=dropout))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)))
            ]))

    def forward(self, x, mask=None):
        for attn, ff in self.layers:
            x = attn(x, mask=mask)
            x = ff(x)
        return x


class Seq_Transformer(nn.Module):
    def __init__(
            self,
            n_channel,
            window_size,
            n_classes,
            dim=128,
            depth=4,
            heads=4,
            mlp_dim=64,
            dropout=0.1):
        super().__init__()
        self.patch_to_embedding = nn.Linear(n_channel, dim)
        self.c_token = nn.Parameter(torch.randn(1, 1, dim))
        self.position = PositionalEncoding(d_model=dim, max_len=window_size)
        self.transformer = Base_Transformer(dim, depth, heads, mlp_dim, dropout)
        self.to_c_token = nn.Identity()

    def forward(self, forward_seq):
        x = self.patch_to_embedding(forward_seq)
        x = self.position(x)
        b, n, _ = x.shape
        c_tokens = repeat(self.c_token, '() n d -> b n d', b=b)
        x = torch.cat((c_tokens, x), dim=1)
        x = self.transformer(x)
        x_ = x[:, 1:]
        c_t = self.to_c_token(x[:, 0])
        return c_t, x_