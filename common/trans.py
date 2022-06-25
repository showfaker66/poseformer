import math
import logging
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath
from .base import elu_feature_map

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=9, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., length=27,block_size=128):
        super().__init__()
        self.feature_map = (
            elu_feature_map(num_heads)
        )

        self.block_size = block_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        # self.time_shift = nn.ZeroPad2d((0, 0, 1, 0))
        # self.time_weighting = nn.Parameter(torch.ones(self.num_heads, self.block_size, self.block_size))

    # def forward(self, x):
    #     B, N, C = x.shape   # 16, 34, 81
    #     # x = torch.cat([self.time_shift(x)[:, :N, :C // 2], x[:, :N, C // 2:]], dim=2)
    #     qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)   # 3, 16, 9, 34, 9
    #     q, k, v = qkv[0], qkv[1], qkv[2]   # 16, 9, 34, 9
    #
    #     self.feature_map.new_feature_map(q.device)
    #     Q = self.feature_map.forward_queries(q)
    #     K = self.feature_map.forward_keys(k)
    #     # K = K * key_lengths.float_matrix[:, :, None, None]
    #     KV = torch.einsum("nshd,nshm->nhmd", K, v)
    #     Z = 1 / (torch.einsum("nlhd,nhd->nlh", Q, K.sum(dim=1)) + self.eps)
    #     x = torch.einsum("nlhd,nhmd,nlh->nlhm", Q, KV, Z)
    #
    #     x = self.proj(x)
    #     x = self.proj_drop(x)
    #     return x

    def forward(self, x):
        B, N, C = x.shape  # 16, 34, 81
        # x = torch.cat([self.time_shift(x)[:, :N, :C // 2], x[:, :N, C // 2:]], dim=2)
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)  # 3, 16, 9, 34, 9
        q, k, v = qkv[0], qkv[1], qkv[2]  # 16, 9, 34, 9

        attn = (q @ k.transpose(-2, -1)) * self.scale  # 16, 9, 34, 34
        attn = attn.softmax(dim=-1)
        # attn = attn * self.time_weighting[:, :N, :N]

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)  # 16, 9, 34, 34 -> 16, 34, 81
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_hidden_dim, qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, length=27):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, \
            qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, length=length,block_size=128)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class PoolFormerBlock(nn.Module):
    def  __init__(self, dim, pool_size=3, mlp_ratio=2.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 drop=0., drop_path=0.,
                 use_layer_scale=True, layer_scale_init_value=1e-5):
        super().__init__()

        self.norm1 = norm_layer(dim)
        self.token_mixer = Pooling(pool_size=pool_size)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)

        # The following two techniques are useful to train deep PoolFormers.
        self.drop_path = DropPath(drop_path) if drop_path > 0. \
            else nn.Identity()
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale_1 = nn.Parameter(
                layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            self.layer_scale_2 = nn.Parameter(
                layer_scale_init_value * torch.ones((dim)), requires_grad=True)

    def forward(self, x) :
        if self.use_layer_scale:
            x = x + self.drop_path(
                self.layer_scale_1.unsqueeze(0).unsqueeze(0)    # self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) -> 81, 1, 1
                * self.token_mixer(self.norm1(x)))                # self.token_mixer(self.norm1(x)) -> 16, 34, 81
            x = x + self.drop_path(
                self.layer_scale_2.unsqueeze(0).unsqueeze(0)
                * self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.token_mixer(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class Pooling(nn.Module):
    """
    Implementation of pooling for PoolFormer
    --pool_size: pooling size
    """
    def __init__(self, pool_size=3):
        super().__init__()
        self.pool = nn.AvgPool2d(
            pool_size, stride=1, padding=pool_size//2, count_include_pad=False)

    def forward(self, x):
        return self.pool(x) - x

class Transformer(nn.Module):
    def __init__(self, depth=4, embed_dim=81, mlp_hidden_dim=162, h=9, drop_rate=0.1, length=34):
        super().__init__()
        drop_path_rate = 0.2
        attn_drop_rate = 0.
        qkv_bias = True
        qk_scale = None

        norm_layer = partial(nn.LayerNorm, eps=1e-6)

        self.Temporal_pos_embed = nn.Parameter(torch.zeros(1, length, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  

        # self.blocks = nn.ModuleList([
        #     Block(
        #         dim=embed_dim, num_heads=h, mlp_hidden_dim=mlp_hidden_dim, qkv_bias=qkv_bias, qk_scale=qk_scale,
        #         drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, length=length)
        #     for i in range(depth)])
        self.blocks = nn.ModuleList([
            PoolFormerBlock(
                embed_dim, pool_size=3, mlp_ratio=2.,
                act_layer=nn.GELU, norm_layer=norm_layer,
                drop=drop_rate, drop_path=dpr[i],
                use_layer_scale=True,
                layer_scale_init_value=1e-5)
            for i in range(depth)])

        self.Temporal_norm = norm_layer(embed_dim)

    def forward(self, x):
        x += self.Temporal_pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.Temporal_norm(x)

        return x

