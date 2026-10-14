"""STM blocks, the shared STMFE backbone, and scale-specific STMDE modules."""

import torch
import torch.nn as nn
from einops import rearrange
from models._compat import trunc_normal_, DropPath
import torch.nn.functional as F
from models.stssm import STSSM
from models.tree_scan_utils.segment_tree import SegmentTree
from models.block.base import CBR, ConvLNGELU

class ToChannelsFirst(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.permute(0, 3, 1, 2)

class ToChannelsLast(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.permute(0, 2, 3, 1)


def build_norm_layer(dim,
                     norm_layer,
                     in_format='channels_last',
                     out_format='channels_last',
                     eps=1e-6):
    layers = []
    if norm_layer == 'BN':
        if in_format == 'channels_last':
            layers.append(ToChannelsFirst())
        layers.append(nn.BatchNorm2d(dim))
        if out_format == 'channels_last':
            layers.append(ToChannelsLast())
    elif norm_layer == 'LN':
        if in_format == 'channels_first':
            layers.append(ToChannelsLast())
        layers.append(nn.LayerNorm(dim, eps=eps))
        if out_format == 'channels_first':
            layers.append(ToChannelsFirst())
    else:
        raise NotImplementedError(
            f'build_norm_layer does not support {norm_layer}')
    return nn.Sequential(*layers)


def build_act_layer(act_layer):
    if act_layer == 'ReLU':
        return nn.ReLU(inplace=True)
    elif act_layer == 'SiLU':
        return nn.SiLU(inplace=True)
    elif act_layer == 'GELU':
        return nn.GELU()

    raise NotImplementedError(f'build_act_layer does not support {act_layer}')


class Stem(nn.Module):
    r""" Stem layer
    Args:
        in_chans (int): number of input channels
        out_chans (int): number of output channels
        act_layer (str): activation layer
        norm_layer (str): normalization layer
    """

    def __init__(self,
                 in_chans=3,
                 out_chans=96,
                 act_layer='GELU',
                 norm_layer='BN'):
        super().__init__()
        self.conv1 = nn.Conv2d(in_chans,
                               out_chans // 2,
                               kernel_size=3,
                               stride=2,
                               padding=1)
        self.norm1 = build_norm_layer(out_chans // 2, norm_layer,
                                      'channels_first', 'channels_first')
        self.act = build_act_layer(act_layer)
        self.conv2 = nn.Conv2d(out_chans // 2,
                               out_chans,
                               kernel_size=3,
                               stride=2,
                               padding=1)
        self.norm2 = build_norm_layer(out_chans, norm_layer, 'channels_first',
                                      'channels_last')

    def forward(self, x, return_intermediate=False):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act(x)
        if return_intermediate:
            inter = x
        x = self.conv2(x)
        x = self.norm2(x)
        if return_intermediate:
            return x, inter
        return x


class Downsample(nn.Module):


    def __init__(self, channels, norm_layer='LN'):
        super().__init__()
        self.conv = nn.Conv2d(channels,
                              2 * channels,
                              kernel_size=3,
                              stride=2,
                              padding=1,
                              bias=False)
        self.norm = build_norm_layer(2 * channels, norm_layer,
                                     'channels_first', 'channels_last')

    def forward(self, x):
        x = self.conv(x.permute(0, 3, 1, 2))
        x = self.norm(x)
        return x


class FeedForward(nn.Module):

    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer='GELU',
                 drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = build_act_layer(act_layer)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class STMBlock(nn.Module):


    def __init__(self,
                 channels,
                 mlp_ratio=4.,
                 drop=0.,
                 norm_layer='LN',
                 drop_path=0.,
                 act_layer='GELU',
                 post_norm=False,
                 layer_scale=None,
                 with_cp=False,
                 k=-1,
                 x=-1,
                 ):
        super().__init__()
        self.channels = channels
        self.mlp_ratio = mlp_ratio
        self.with_cp = with_cp

        self.norm1 = build_norm_layer(channels, 'LN')
        self.post_norm = post_norm
        self.stssm = STSSM(
                d_model=channels, 
                d_state=1, 
                ssm_ratio=2,
                ssm_rank_ratio=2,
                dt_rank='auto',
                act_layer=nn.SiLU,
                # ==========================
                d_conv=3,
                conv_bias=False,
                # ==========================
                dropout=0.0,
                # ==========================segment tree
                k=k,
                x=x,
            )
        self.drop_path = DropPath(drop_path) if drop_path > 0. \
            else nn.Identity()
        self.norm2 = build_norm_layer(channels, 'LN')
        self.mlp = FeedForward(in_features=channels,
                            hidden_features=int(channels * mlp_ratio),
                            act_layer=act_layer,
                            drop=drop)
        self.layer_scale = layer_scale is not None
        if self.layer_scale:
            self.gamma1 = nn.Parameter(layer_scale * torch.ones(channels),
                                       requires_grad=True)
            self.gamma2 = nn.Parameter(layer_scale * torch.ones(channels),
                                       requires_grad=True)
    def forward(self, x):

        def _inner_forward(x):
            if not self.layer_scale:
                if self.post_norm:
                    x = x + self.drop_path(self.norm1(self.stssm(x)))
                    x = x + self.drop_path(self.norm2(self.mlp(x)))
                else:
                    x = x + self.drop_path(self.stssm(self.norm1(x)))
                    x = x + self.drop_path(self.mlp(self.norm2(x)))
                return x
            if self.post_norm:
                x = x + self.drop_path(self.gamma1 * self.norm1(self.stssm(x)))
                x = x + self.drop_path(self.gamma2 * self.norm2(self.mlp(x)))
            else:
                x = x + self.drop_path(self.gamma1 * self.stssm(self.norm1(x)))
                x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
            return x

        if self.with_cp and x.requires_grad:
            x = checkpoint.checkpoint(_inner_forward, x)
        else:
            x = _inner_forward(x)
        return x


class STMStage(nn.Module):

    def __init__(self,
                 channels,
                 depth,
                 downsample=True,
                 mlp_ratio=4.,
                 drop=0.,
                 drop_path=0.,
                 act_layer='GELU',
                 norm_layer='LN',
                 post_norm=False,
                 layer_scale=None,
                 with_cp=False,
                 k=-1,
                 x=-1,
                 ):
        super().__init__()
        self.channels = channels
        self.depth = depth
        self.post_norm = post_norm

        self.blocks = nn.ModuleList([
            STMBlock(
                channels=channels,
                mlp_ratio=mlp_ratio,
                drop=drop,
                drop_path=drop_path[i] if isinstance(
                    drop_path, list) else drop_path,
                act_layer=act_layer,
                norm_layer=norm_layer,
                post_norm=post_norm,
                layer_scale=layer_scale,
                with_cp=with_cp,
                k=k,
                x=x,
        ) for i in range(depth)
        ])
        self.norm = build_norm_layer(channels, 'LN')
        self.downsample = Downsample(
            channels=channels, norm_layer=norm_layer) if downsample else None

    def forward(self, x, return_wo_downsample=False):
        for i, blk in enumerate(self.blocks):
            x = blk(x)
        if not self.post_norm or self.center_feature_scale:
            x = self.norm(x)
        if return_wo_downsample:
            x_ = x
        if self.downsample is not None:
            x = self.downsample(x)

        if return_wo_downsample:
            return x, x_
        return x

class STMBackbone(nn.Module):

    def __init__(self, channels=80, depths=[2, 2, 9, 2],
                 mlp_ratio=4., drop_rate=0., drop_path_rate=0.2, drop_path_type='linear'
                 , act_layer='GELU', norm_layer='LN', layer_scale=None, post_norm=False,
                 with_cp=False, cls_scale=1.5, stage_k=[-1,-1,-1,-1], stage_x=[-1,-1,-1,-1], **kwargs):
        super().__init__()
        self.num_stages = len(depths)
        self.depths = depths
        self.channels = channels
        self.post_norm = post_norm
        self.mlp_ratio = mlp_ratio

        in_chans = 3
        self.stem = Stem(in_chans=in_chans,
                                     out_chans=channels,
                                     act_layer=act_layer,
                                     norm_layer=norm_layer)
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))
        ]
        if drop_path_type == 'uniform':
            for i in range(len(dpr)):
                dpr[i] = drop_path_rate

        self.stages = nn.ModuleList()
        for i in range(self.num_stages):
            k, x = stage_k[i], stage_x[i]
            print(f"STMFE stage {i + 1} : k={k}, x={x}")
            stage = STMStage(
                channels=int(channels * 2**i),
                depth=depths[i],
                mlp_ratio=self.mlp_ratio,
                drop=drop_rate,
                drop_path=dpr[sum(depths[:i]):sum(depths[:i + 1])],
                act_layer=act_layer,
                norm_layer=norm_layer,
                post_norm=post_norm,
                downsample=(i < self.num_stages - 1),
                layer_scale=layer_scale,
                with_cp=with_cp,
                k=k,
                x=x,
            )
            self.stages.append(stage)
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

        
    def forward(self, x):
        x = self.stem(x)
        x = self.pos_drop(x)
        out=[]
        for stage in self.stages:
            x, x_ = stage(x, return_wo_downsample=True)
            out.append(x_)
        
        out = [ element.permute(0, 3, 1, 2) for element in out]
        return out[0],out[1],out[2],out[3]
    def forward_features(self, x):
        stem, stem_inter = self.stem(x, return_intermediate=True)
        x = self.pos_drop(stem)
        out = []
        for stage in self.stages:
            x, x_ = stage(x, return_wo_downsample=True)
            out.append(x_)
        out = [element.permute(0, 3, 1, 2) for element in out]
        return stem_inter, out[0], out[1], out[2], out[3]


class STMDE(nn.Module):
    

    def __init__(self, in_channels, out_channels, drop_path_rate=0, k=-1, x=-1):
        super().__init__()
        self.input_projection = ConvLNGELU(in_channels, out_channels)
        self.output_projection = CBR(out_channels, out_channels)
        self.stm_block = STMBlock(
            channels=out_channels,
            mlp_ratio=4,
            drop=0.,
            norm_layer='LN',
            drop_path=drop_path_rate,
            act_layer='GELU',
            post_norm=False,
            layer_scale=None,
            with_cp=False,
            k=k,
            x=x,
        )

    def forward(self, F_pre, F_post):
        feature = torch.cat([F_pre, F_post], dim=1)
        feature = self.input_projection(feature)
        feature = feature + self.stm_block(feature)
        feature = feature.permute(0, 3, 1, 2)
        return self.output_projection(feature)
