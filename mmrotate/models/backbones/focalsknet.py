import math
import warnings
from functools import partial

import torch
import torch.nn as nn
from mmcv.cnn import build_norm_layer
from mmcv.cnn.utils.weight_init import (constant_init, normal_init, trunc_normal_init)
from mmcv.runner import BaseModule
from timm.models.layers import DropPath, to_2tuple
from torch.nn.modules.utils import _pair as to_2tuple
from fvcore.nn import flop_count_table, flop_count_str, FlopCountAnalysis
from ..builder import ROTATED_BACKBONES

# from ..utils import DCNv3

# this page for Mix of LSKNet & FocalNet

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class LSK_input_gate(nn.Module):
    # input gate
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)  # 5x5 in,out size same.
        self.conv_spatial = nn.Conv2d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
        self.proj = nn.Linear(dim, dim)  # 32,32
        self.light_conv = nn.Conv2d(dim, 2, 1, bias=True)

    def forward(self, x):
        attn1 = self.conv0(x)  # (5,1) RF = 5
        attn2 = self.conv_spatial(attn1)  # (7,3) RF_last = 18 + 5

        gate = self.light_conv(x)
        attn = attn1 * gate[:, 0, :, :].unsqueeze(1) + attn2 * gate[:, 1, :, :].unsqueeze(1)
        attn = linear_p(attn, self.proj)
        return x * attn


class LSK_input_gate_pooling_q(nn.Module):
    # input gate + que
    def __init__(self, dim):
        super().__init__()
        self.act = nn.GELU()
        self.conv0 = nn.Sequential(nn.Conv2d(dim, dim, 5, padding=2, groups=dim), self.act)
        self.conv1 = nn.Sequential(nn.Conv2d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3), self.act)
        self.proj = nn.Linear(dim, dim)
        self.focal_level = 2
        self.f = nn.Linear(dim, 2 * dim + (self.focal_level + 1), bias=True)

    def forward(self, x):
        # but flops is too high (lsknet block = 269, this = 363)
        _, C, _, _ = x.shape
        x1 = linear_p(x, self.f)
        q, v, gates = torch.split(x1, (C, C, self.focal_level + 1), 1)

        attn1 = self.conv0(v)  # (5,1) RF = 5
        attn2 = self.conv1(attn1)  # (7,3) RF_last = 18 + 5
        pooling = self.act(attn2.mean(2, keepdim=True).mean(3, keepdim=True))

        attn = (attn1 * gates[:, 0, :, :].unsqueeze(1) + attn2 * gates[:, 1, :, :].unsqueeze(1) +
                pooling * gates[:, 2, :, :].unsqueeze(1))
        attn = linear_p(attn, self.proj)
        return q * attn




class LSK_input_gate_dcnv3(nn.Module):
    # input gate
    def __init__(self, dim):
        super().__init__()
        self.conv0 = DCNv3_(dim, 3, pad=1, group=dim // 32)  # 5x5 in,out size same.
        self.conv1 = DCNv3_(dim, 3, pad=1, group=dim // 32)
        self.light_conv = nn.Conv2d(dim, 2, 1, bias=True)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        x1 = x.permute(0, 2, 3, 1).contiguous()
        attn1 = self.conv0(x1)
        attn2 = self.conv1(attn1)  # (3,1)
        attn = self.proj(attn2)
        attn = attn.permute(0, 3, 1, 2).contiguous()
        return x * attn


class LSKMix_pooling(nn.Module):
    # pooling layer + no sigmoid
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)  # 5x5 in,out size same.
        self.conv_spatial = nn.Conv2d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
        self.conv1 = nn.Conv2d(dim, dim // 2, 1)
        self.conv2 = nn.Conv2d(dim, dim // 2, 1)
        self.conv_squeeze = nn.Conv2d(3, 3, 7, padding=3)
        self.conv = nn.Conv2d(dim // 2, dim, 1)
        self.light_conv = nn.Conv2d(dim, 1, 1, bias=True)
        self.act = nn.GELU()

    def forward(self, x):
        attn1 = self.conv0(x)  # (5,1) RF = 5
        attn2 = self.conv_spatial(attn1)  # (7,3) RF_last = 18 + 5

        attn1 = self.conv1(attn1)  # first layer
        attn2 = self.conv2(attn2)  # second layer

        avgpooling = self.act(attn2.mean(2, keepdim=True).mean(3, keepdim=True))
        pooling = avgpooling
        gate = self.light_conv(x)
        attn = torch.cat([attn1, attn2], dim=1)
        avg_attn = torch.mean(attn, dim=1, keepdim=True)
        max_attn, _ = torch.max(attn, dim=1, keepdim=True)
        agg = torch.cat([avg_attn, max_attn, gate], dim=1)
        sig = self.conv_squeeze(agg).sigmoid()
        attn = attn1 * sig[:, 0, :, :].unsqueeze(1) + attn2 * sig[:, 1, :, :].unsqueeze(1) + pooling * sig[:, 2, :,
                                                                                                       :].unsqueeze(1)
        attn = self.conv(attn)
        return x * attn


class LSKMix_input_gate_pooling(nn.Module):
    # input gate & pooling layer
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Sequential(nn.Conv2d(dim, dim, 5, padding=2, groups=dim), nn.GELU())
        self.conv1 = nn.Sequential(nn.Conv2d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3), nn.GELU())
        self.proj = nn.Linear(dim, dim)
        self.focal_level = 2
        self.f = nn.Linear(dim, self.focal_level + 1, bias=True)
        self.act = nn.GELU()

    def forward(self, x):
        attn1 = self.conv0(x)  # (5,1) RF = 5
        attn2 = self.conv_spatial(attn1)  # (7,3) RF_last = 18 + 5
        pooling = self.act(attn2.mean(2, keepdim=True).mean(3, keepdim=True))
        gate = linear_p(x, self.f)
        attn = attn1 * gate[:, 0, :, :].unsqueeze(1) + attn2 * gate[:, 1, :, :].unsqueeze(1) + pooling * gate[:, 2, :,
                                                                                                         :].unsqueeze(1)
        attn = linear_p(attn, self.proj)
        return x * attn


class LSKMix_input_gate_pooling_3layer(nn.Module):
    # input gate & pooling layer & softmax
    def __init__(self, dim, test_type=None):
        super().__init__()
        # padding = (dilation * (kernel_size - 1)) / 2
        self.conv0 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)  # 5x5 in,out size same.
        self.conv_spatial = nn.Conv2d(dim, dim, 5, padding=4, groups=dim, dilation=2)
        self.conv_spatial2 = nn.Conv2d(dim, dim, 5, padding=6, groups=dim, dilation=3)
        self.conv1 = nn.Conv2d(dim, dim // 2, 1)
        self.conv2 = nn.Conv2d(dim, dim // 2, 1)
        self.conv_squeeze = nn.Conv2d(2, 2, 7, padding=3)
        self.conv = nn.Conv2d(dim, dim, 1)
        self.light_conv = nn.Conv2d(dim, 4, 1, bias=True)
        self.act = nn.GELU()
        self.softmax = nn.Softmax(2)
        if test_type == '3_layer_softmax':
            self.softmax_ = True

    def forward(self, x):
        # N, C, H, W = x.shape

        attn1 = self.conv0(x)  # (5,1) RF = 5
        gate = self.light_conv(x)
        if self.softmax_:
            gate = self.softmax(gate.view(*gate.size()[:2], -1)).view_as(gate)

        attn2 = self.conv_spatial(attn1)  # (7,3) RF_last = 18 + 5
        attn3 = self.conv_spatial2(attn2)
        pooling = self.act(attn3.mean(2, keepdim=True).mean(3, keepdim=True))

        attn = (attn1 * gate[:, 0, :, :].unsqueeze(1) +
                attn2 * gate[:, 1, :, :].unsqueeze(1) +
                attn3 * gate[:, 2, :, :].unsqueeze(1) +
                pooling * gate[:, 3, :, :].unsqueeze(1))
        attn = self.conv(attn)
        return x * attn


class LSKblock_q(nn.Module):
    # input gate + que
    def __init__(self, dim):
        super().__init__()
        self.act = nn.GELU()
        self.conv0 = nn.Sequential(nn.Conv2d(dim, dim, 5, padding=2, groups=dim), self.act)
        self.conv1 = nn.Sequential(nn.Conv2d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3), self.act)
        self.conv = nn.Conv2d(dim // 2, dim, 1)
        self.conv_gate1 = nn.Conv2d(dim, dim // 2, 1)
        self.conv_gate2 = nn.Conv2d(dim, dim // 2, 1)
        self.conv_squeeze = nn.Conv2d(2, 2, 7, padding=3)
        self.f = nn.Linear(dim, 2 * dim, bias=True)

    def forward(self, x):
        # but flops is too high (lsknet block = 269, this = ?)
        _, C, _, _ = x.shape
        x1 = linear_p(x, self.f)
        q, v = torch.split(x1, (C, C), 1)

        attn1 = self.conv0(v)  # (5,1) RF = 5
        attn2 = self.conv1(attn1)  # (7,3) RF_last = 18 + 5

        attn1 = self.conv_gate1(attn1)  # first layer
        attn2 = self.conv_gate2(attn2)
        attn = torch.cat([attn1, attn2], dim=1)
        avg_attn = torch.mean(attn, dim=1, keepdim=True)
        max_attn, _ = torch.max(attn, dim=1, keepdim=True)
        agg = torch.cat([avg_attn, max_attn], dim=1)
        sig = self.conv_squeeze(agg).sigmoid()

        attn = attn1 * sig[:, 0, :, :].unsqueeze(1) + attn2 * sig[:, 1, :, :].unsqueeze(1)
        attn = self.conv(attn)
        return q * attn


class LSKblock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)  # 5x5 in,out size same.
        self.conv_spatial = nn.Conv2d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
        self.conv1 = nn.Conv2d(dim, dim // 2, 1)
        self.conv2 = nn.Conv2d(dim, dim // 2, 1)
        self.conv_squeeze = nn.Conv2d(2, 2, 7, padding=3)
        self.conv = nn.Conv2d(dim // 2, dim, 1)

    def forward(self, x):
        attn1 = self.conv0(x)  # (5,1) RF = 5
        attn2 = self.conv_spatial(attn1)  # (7,3) RF_last = 18 + 5

        attn1 = self.conv1(attn1)  # first layer
        attn2 = self.conv2(attn2)  # second layer

        attn = torch.cat([attn1, attn2], dim=1)
        avg_attn = torch.mean(attn, dim=1, keepdim=True)
        max_attn, _ = torch.max(attn, dim=1, keepdim=True)
        agg = torch.cat([avg_attn, max_attn], dim=1)

        sig = self.conv_squeeze(agg).sigmoid()
        attn = attn1 * sig[:, 0, :, :].unsqueeze(1) + attn2 * sig[:, 1, :, :].unsqueeze(1)
        attn = self.conv(attn)
        return x * attn


class FocalNet_Conv_nothing(nn.Module):  # 612
    def __init__(self, dim, test=None):
        super().__init__()
        self.dim, self.test = dim, test
        self.focal_level, self.focal_window, self.focal_factor = 2, 5, 2
        self.f = nn.Linear(dim, self.focal_level + 1, bias=True)
        self.h = nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0, groups=1, bias=True)
        self.focal_layers = nn.ModuleList()
        self.act = nn.GELU()
        for k in range(self.focal_level):
            kernel_size = self.focal_factor * k + self.focal_window
            self.focal_layers.append(
                nn.Sequential(
                    nn.Conv2d(dim, dim, kernel_size=kernel_size, stride=1, groups=dim, padding=(kernel_size // 2), bias=False),
                )
            )
    def forward(self, x):
        _, C, _, _ = x.shape
        gates = linear_p(x, self.f)
        v = x.clone()
        ctx_all = 0
        for l in range(self.focal_level):
            v = self.focal_layers[l](v)
            ctx_all = ctx_all + v * gates[:, l:l + 1]
        if self.test == 1:
            ctx_global = v.mean(2, keepdim=True).mean(3, keepdim=True)
            ctx_all = ctx_all + ctx_global * gates[:, self.focal_level:]

        x_out = x * self.h(ctx_all)
        return x_out
class FocalNet_Conv(nn.Module):  # 612
    def __init__(self, dim, test=None):
        super().__init__()
        self.dim = dim
        self.first = 1
        # specific args for focalv3
        if test == None:
            self.focal_level, self.focal_window, self.focal_factor = 3, 5, 2
        elif test == 2:
            self.focal_level, self.focal_window, self.focal_factor = 2, 5, 2
        elif test == 3:
            self.focal_level, self.focal_window, self.focal_factor = 2, 5, 2
        elif test == 4:
            self.focal_level, self.focal_window, self.focal_factor = 2, 9, 2
        elif test == 5:
            self.focal_level, self.focal_window, self.focal_factor = 2, 5, 2
            self.first = 0
        elif test == 6:
            self.focal_level, self.focal_window, self.focal_factor = 2, 5, 2
            self.first = 0
        elif test == 7:
            self.focal_level, self.focal_window, self.focal_factor = 2, 5, 2
            self.first = 0
        else:
            self.focal_level, self.focal_window, self.focal_factor = 2, 7, 2

        self.f = nn.Linear(dim, 2 * dim + (self.focal_level + 1), bias=True)
        self.h = nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0, groups=1, bias=True)

        self.act = nn.GELU()
        self.proj = nn.Linear(dim, dim)
        self.focal_layers = nn.ModuleList()

        self.norm1 = nn.LayerNorm(dim)

        for k in range(self.focal_level):
            kernel_size = self.focal_factor * k + self.focal_window if not test == 4 else -1 * self.focal_factor * k + self.focal_window
            dilation = self.focal_factor * k + 1 if test == 2 else 1
            if self.first or test == 7:
                self.focal_layers.append(
                    nn.Sequential(
                        nn.Conv2d(dim, dim, kernel_size=kernel_size, stride=1, dilation=dilation, groups=dim,
                                  padding=(kernel_size // 2)*dilation, bias=False),
                        nn.GELU(),
                    )
                )
            else:
                self.focal_layers.append(
                    nn.Sequential(
                        nn.Conv2d(dim, dim, kernel_size=kernel_size, stride=1, dilation=dilation, groups=dim,
                                  padding=(kernel_size // 2)*dilation, bias=False),
                    )
                )
        if test == 6:
            self.first = 1
    def forward(self, x):
        _, C, _, _ = x.shape
        x = linear_p(x, self.norm1) if self.first else x
        x1 = linear_p(x, self.f)
        q, v, gates = torch.split(x1, (C, C, self.focal_level + 1), 1)
        ctx_all = 0
        for l in range(self.focal_level):
            v = self.focal_layers[l](v)
            ctx_all = ctx_all + v * gates[:, l:l + 1]
        ctx_global = self.act(v.mean(2, keepdim=True).mean(3, keepdim=True))
        ctx_all = ctx_all + ctx_global * gates[:, self.focal_level:]

        x_out = q * self.h(ctx_all)
        return x_out


class Attention(nn.Module):
    def __init__(self, d_model, test_type=None):
        super().__init__()
        self.first = 1
        self.proj_1 = nn.Conv2d(d_model, d_model, 1)
        self.activation = nn.GELU()
        if test_type == None:
            self.spatial_gating_unit = LSKblock(d_model)
        elif test_type == 'input_gate':  # pooling layer is also featuremap
            self.spatial_gating_unit = LSK_input_gate(d_model)
        elif test_type == 'input_gate_pooling':
            self.spatial_gating_unit = LSKMix_input_gate_pooling(d_model)
        elif test_type == 'pooling':
            self.spatial_gating_unit = LSKMix_pooling(d_model)
        elif test_type == '3_layer_softmax':
            self.spatial_gating_unit = LSKMix_input_gate_pooling_3layer(d_model, test_type)
        elif test_type == '3_layer':
            self.spatial_gating_unit = LSKMix_input_gate_pooling_3layer(d_model)
        elif test_type == 'dcnv3':  # 83.9
            self.spatial_gating_unit = LSK_input_gate_dcnv3(d_model)
        elif test_type == 'input_gate_pooling_q':
            self.spatial_gating_unit = LSK_input_gate_pooling_q(d_model)
        elif test_type == 'FocalNet_Conv':
            self.spatial_gating_unit = FocalNet_Conv(d_model)
        elif test_type == 'FocalNet_Conv2':
            self.spatial_gating_unit = FocalNet_Conv(d_model, 1)
        elif test_type == 'FocalNet_dilation':
            self.spatial_gating_unit = FocalNet_Conv(d_model, 2)
        elif test_type == 'FocalNet_no_dilation':
            self.spatial_gating_unit = FocalNet_Conv(d_model, 3)
        elif test_type == 'FocalNet_reverse':
            self.spatial_gating_unit = FocalNet_Conv(d_model, 4)
        elif test_type == 'no_element':
            self.spatial_gating_unit = FocalNet_Conv(d_model, 5)
        elif test_type == 'nothing':
            self.spatial_gating_unit = FocalNet_Conv_nothing(d_model)
        elif test_type == 'nothing_even_pooling':
            self.spatial_gating_unit = FocalNet_Conv_nothing(d_model, 1)
        elif test_type == 'q_block':
            self.spatial_gating_unit = LSKblock_q(d_model)
        elif test_type == 'no_GELU':
            self.spatial_gating_unit = FocalNet_Conv(d_model, 6)
        elif test_type == 'no_LN':
            self.spatial_gating_unit = FocalNet_Conv(d_model, 7)
        self.proj_2 = nn.Conv2d(d_model, d_model, 1)  #

    def forward(self, x):
        if self.first:
            self.first = 0
            flops = FlopCountAnalysis(self.spatial_gating_unit, x)
        shorcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        x = x + shorcut
        return x


class Block(nn.Module):
    def __init__(self, dim, mlp_ratio=4., drop=0., drop_path=0., act_layer=nn.GELU, stage=0, norm_cfg=None,
                 test_type=None):
        super().__init__()
        if norm_cfg:
            self.norm1 = build_norm_layer(norm_cfg, dim)[1]
            self.norm2 = build_norm_layer(norm_cfg, dim)[1]
        else:
            self.norm1 = nn.BatchNorm2d(dim)
            self.norm2 = nn.BatchNorm2d(dim)

        self.attn = Attention(dim, test_type)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        layer_scale_init_value = 1e-2
        self.layer_scale_1 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)

    def forward(self, x):
        x = x + self.drop_path(self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * self.attn(self.norm1(x)))
        x = x + self.drop_path(self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * self.mlp(self.norm2(x)))
        return x


class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768, norm_cfg=None, test_type=None):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        if norm_cfg:
            self.norm = build_norm_layer(norm_cfg, embed_dim)[1]
        else:
            self.norm = nn.BatchNorm2d(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = self.norm(x)
        return x, H, W


@ROTATED_BACKBONES.register_module()
class FocaL_SKNet(BaseModule):
    def __init__(self, img_size=224, in_chans=3, embed_dims=[64, 128, 256, 512],
                 mlp_ratios=[8, 8, 4, 4], drop_rate=0., drop_path_rate=0., norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 depths=[3, 4, 6, 3], num_stages=4,
                 pretrained=None,
                 init_cfg=None,
                 norm_cfg=None, test_type=None):
        super().__init__(init_cfg=init_cfg)

        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be set at the same time'
        if isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        elif pretrained is not None:
            raise TypeError('pretrained must be a str or None')
        self.depths = depths
        self.num_stages = num_stages

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0

        for i in range(num_stages):
            patch_embed = OverlapPatchEmbed(img_size=img_size if i == 0 else img_size // (2 ** (i + 1)), patch_size=7 if i == 0 else 3, stride=4 if i == 0 else 2,
                                            in_chans=in_chans if i == 0 else embed_dims[i - 1], embed_dim=embed_dims[i], norm_cfg=norm_cfg, test_type=test_type)

            block = nn.ModuleList([Block(dim=embed_dims[i], mlp_ratio=mlp_ratios[i], drop=drop_rate,
                                         drop_path=dpr[cur + j], norm_cfg=norm_cfg, test_type=test_type, stage=i)
                                   for j in range(depths[i])])
            norm = norm_layer(embed_dims[i])
            cur += depths[i]

            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"block{i + 1}", block)
            setattr(self, f"norm{i + 1}", norm)

    def init_weights(self):
        print('init cfg', self.init_cfg)
        if self.init_cfg is None:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    trunc_normal_init(m, std=.02, bias=0.)
                elif isinstance(m, nn.LayerNorm):
                    constant_init(m, val=1.0, bias=0.)
                elif isinstance(m, nn.Conv2d):
                    fan_out = m.kernel_size[0] * m.kernel_size[
                        1] * m.out_channels
                    fan_out //= m.groups
                    normal_init(
                        m, mean=0, std=math.sqrt(2.0 / fan_out), bias=0)
        else:
            super(FocaL_SKNet, self).init_weights()

    def freeze_patch_emb(self):
        self.patch_embed1.requires_grad = False

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed1', 'pos_embed2', 'pos_embed3', 'pos_embed4', 'cls_token'}  # has pos_embed may be better

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        B = x.shape[0]
        outs = []
        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            block = getattr(self, f"block{i + 1}")
            norm = getattr(self, f"norm{i + 1}")
            x, H, W = patch_embed(x)  #
            # if not H == 256:
            #     a = x
            for blk in block:
                x = blk(x)
            x = x.flatten(2).transpose(1, 2)
            x = norm(x)
            x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            outs.append(x)
        return outs

    def forward(self, x):
        x = self.forward_features(x)  # 1,3,1024,1024
        # x = self.head(x)
        return x


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x):
        x = self.dwconv(x)
        return x


def _conv_filter(state_dict, patch_size=16):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k:
            v = v.reshape((v.shape[0], 3, patch_size, patch_size))
        out_dict[k] = v

    return out_dict


def linear_p(x, f):
    x = x.permute(0, 2, 3, 1).contiguous()
    x = f(x)
    x = x.permute(0, 3, 1, 2).contiguous()
    return x
