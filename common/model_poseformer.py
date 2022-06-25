## Our PoseFormer model was revised from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py

import math
import logging
from functools import partial
from collections import OrderedDict
# from einops import rearrange, repeat
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.helpers import load_pretrained
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model

from common.mhformer import Model



class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features  # out_features：32
        hidden_features = hidden_features or in_features  # hidden_features：64
        self.fc1 = nn.Linear(in_features, hidden_features) # 32,64
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)  # 64,32
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class DynamicPosBias(nn.Module):
    def __init__(self, dim, num_heads, residual):
        super().__init__()
        self.residual = residual
        self.num_heads = num_heads
        self.pos_dim = dim // 4
        self.pos_proj = nn.Linear(2, self.pos_dim)
        self.pos1 = nn.Sequential(
            nn.LayerNorm(self.pos_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.pos_dim, self.pos_dim),
        )
        self.pos2 = nn.Sequential(
            nn.LayerNorm(self.pos_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.pos_dim, self.pos_dim)
        )
        self.pos3 = nn.Sequential(
            nn.LayerNorm(self.pos_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.pos_dim, self.num_heads)
        )
    def forward(self, biases):
        if self.residual:
            pos = self.pos_proj(biases) # 2Wh-1 * 2Ww-1, heads
            pos = pos + self.pos1(pos)
            pos = pos + self.pos2(pos)
            pos = self.pos3(pos)
        else:
            pos = self.pos3(self.pos2(self.pos1(self.pos_proj(biases))))
        return pos

    def flops(self, N):
        flops = N * 2 * self.pos_dim
        flops += N * self.pos_dim * self.pos_dim
        flops += N * self.pos_dim * self.pos_dim
        flops += N * self.pos_dim * self.num_heads
        return flops

#   跨步卷积
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Conv1d(d_model, d_ff, kernel_size=1, stride=1)
        self.w_2 = nn.Conv1d(d_ff, d_model, kernel_size=3, stride=3, padding = 1)

        self.gelu = nn.ReLU()

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.w_2(self.dropout(self.gelu(self.w_1(x))))
        x = x.permute(0, 2, 1)

        return x

class Attention(nn.Module):
    def  __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., block_size=128):
        super().__init__()
        self.num_heads = num_heads
        self.block_size = block_size
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)  # 32,96
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.time_shift = nn.ZeroPad2d((0, 0, 1, 0))
        self.time_weighting = nn.Parameter(torch.ones(self.num_heads, self.block_size, self.block_size))

    def forward(self, x):
        B, N, C = x.shape  # 432,17,32
        x = torch.cat([self.time_shift(x)[:, :N, :C // 2], x[:, :N, C // 2:]], dim=2)
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)  #reshape(432, 17, 3, 8, 4)
        # 3,432,8,17,4
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)
        # 432,8,17,4
        attn = (q @ k.transpose(-2, -1)) * self.scale  # 432,8,17,17
        attn = attn.softmax(dim=-1)
        attn = attn * self.time_weighting[:, :N, :N]
        attn = self.attn_drop(attn)   # 432,8,17,17

        x = (attn @ v).transpose(1, 2).reshape(B, N, C) # 432,17,32
        x = self.proj(x)
        x = self.proj_drop(x)
        return x   # 432,17,32

class Block(nn.Module):

    # def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=1, drop=0., attn_drop=0.,
    #              drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,group_size=int(pow(17,0.5))):
    def __init__(self, dim, num_heads,kernel_size,mlp_ratio=2., dim_qk = None, dim_v = None, stride = 1,dilation = 1,qkv_bias=False, qk_scale=1, drop=0., attn_drop=0.,
                     drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,group_width=8, groups=1,lam=1, gamma=1,** kwargs):
        # qk_scale = None
        super().__init__()
        self.norm1 = norm_layer(dim)
        # self.group_size = group_size
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop,block_size=128)
        # self.attn = Attention1(
        #     dim, group_size=to_2tuple(self.group_size), num_heads=num_heads,
        #     qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop,
        #     position_bias=True)
        # self.attn = ELSA(dim, num_heads, dim_qk=dim_qk, dim_v=dim_v, kernel_size=kernel_size,
        #                  stride=stride, dilation=dilation,
        #                  qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop,
        #                  group_width=group_width, groups=groups, lam=lam, gamma=gamma, **kwargs)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)  # mlp_hidden_dim：64
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)  # 32,64，GELU，0


    def forward(self, x):
        # self.stride_num = stride_num
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class Block1(nn.Module):

    def __init__(self, dim, num_heads,mlp_ratio=2.,qkv_bias=False, qk_scale=1, drop=0., attn_drop=0.,
                     drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,group_width=8, groups=1,lam=1, gamma=1,** kwargs):
        # qk_scale = None
        super().__init__()
        self.norm1 = norm_layer(dim)
        # self.group_size = group_size
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop,block_size=128)
        # self.attn = Attention1(
        #     dim, group_size=to_2tuple(self.group_size), num_heads=num_heads,
        #     qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop,
        #     position_bias=True)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)  # mlp_hidden_dim：64
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)  # 32,64，GELU，0


    def forward(self, x):
        # self.stride_num = stride_num
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class PoseTransformer(nn.Module):
    def __init__(self, num_frame=81, num_joints=17, in_chans=2, embed_dim_ratio=32, depth=4,
                 num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.2, norm_layer=None, kernel_size=5, d_model=768,
                 d_ff=1536, dropout=0.1,embed_dim = 768):
    # def __init__(self, num_frame=81, num_joints=17, in_chans=2, embed_dim_ratio=32, depth=4,
    #              num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None,
    #              drop_rate=0., attn_drop_rate=0., drop_path_rate=0.2,norm_layer=None,kernel_size=5,d_model=768, d_ff=1536, dropout=0.1):

        """    ##########hybrid_backbone=None, representation_size=None,
        Args:
            num_frame (int, tuple): input frame number
            num_joints (int, tuple): joints number
            in_chans (int): number of input channels, 2D joints have 2 channels: (x,y)
            embed_dim_ratio (int): embedding dimension ratio
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            norm_layer: (nn.Module): normalization layer
        """
        super().__init__()

        # norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        # embed_dim = embed_dim_ratio * num_joints   #### temporal embed_dim is num_joints * spatial embedding dim ratio
        embed_dim = 768
        out_dim = num_joints * 3     #### output dimension is num_joints * 3

        ### spatial patch embedding
        # self.Spatial_patch_to_embedding = nn.Linear(in_chans, embed_dim_ratio) # 2，32
        # self.Spatial_pos_embed = nn.Parameter(torch.zeros(1, num_joints, embed_dim_ratio))  # 1， 17， 32
        #
        # geometry position embedding

        # self.Temporal_pos_embed = nn.Parameter(torch.zeros(1, num_frame, embed_dim))  # 1, 9， 32*17



        # dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        #
        # self.Spatial_blocks = nn.ModuleList([
        #     Block(
        #         dim=embed_dim_ratio, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
        #         drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,kernel_size = kernel_size)
        #     # group_size=group_size
        #     for i in range(depth)])
        #
        # self.blocks = nn.ModuleList([
        #     Block(
        #         dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
        #         drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,kernel_size = kernel_size)
        #     for i in range(depth)])

        # self.Spatial_norm = norm_layer(embed_dim_ratio)


        ####### A easy way to implement weighted mean
        self.weighted_mean = torch.nn.Conv1d(in_channels=num_frame, out_channels=1, kernel_size=1)


        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim , out_dim),
        )
        self.model = Model()
        self.num_joints = num_joints
        self.pos_drop = nn.Dropout(p=drop_rate)
        # self.Temporal_norm = norm_layer(embed_dim)
        self.stride_conv = PositionwiseFeedForward(d_model, d_ff, dropout)


    # def Spatial_forward_features(self, x):
    #     b, _, f, p = x.shape  ##### b is batch size, f is number of frames, p is number of joints
    #     x = rearrange(x, 'b c f p  -> (b f) p  c', )
    #
    #     x = self.Spatial_patch_to_embedding(x)
    #     x += self.Spatial_pos_embed
    #     x = self.pos_drop(x)
    #
    #     for blk in self.Spatial_blocks:
    #         x = blk(x)
    #
    #     x = self.Spatial_norm(x)
    #     x = rearrange(x, '(b f) w c -> b f (w c)', f=f)
    #     return x

    def forward_features(self, x, N=4):
        b = x.shape[0]  # 16,27,544 -> 16, 81, 1536
        # x += self.Temporal_pos_embed   # x += 16,81,1536
        x = self.pos_drop(x)   # x 16,27,544   16, 81,1536
    #     # for blk in self.blocks:
    #     #     x = blk(x)
    #     #
    #     x = self.Temporal_norm(x) # 16,81,1536
    #
        for i in range(N):
            x = self.stride_conv(x)
    #     ##### x size [b, f, emb_dim], then take weighted mean on frame dimension, we only predict 3D pose of the center frame
    #     # x = self.weighted_mean(x) # 16,27,544 -> 16,1,1536
        x = x.view(b, 1, -1)
        return x

    def forward(self, x):
        b = x.shape[0]
        # x = x.permute(0, 3, 1, 2)  # 16, 81, 17, 2
        # b, _, _, p = x.shape  # 16, 2, 81, 17
        ### now x is [batch_size, 2 channels, receptive frames, joint_num], following image data
        # x = self.Spatial_forward_features(x)  # 16,27,544   16, 81, 544
        x = self.model(x)
        # x_ST = x
        x = self.forward_features(x)   # 16,1,1536
        x = self.head(x)   # 16, 1, 51

        x = x.view(b, 1, self.num_joints, -1)

        return x

