import torch
import torch.nn as nn
import torch.nn.functional as F

from models.stm import STMBackbone


def _standard_dims(channels: int) -> dict:
    return {
        "F0": channels // 2,
        "F1": channels * 1,
        "F2": channels * 2,
        "F3": channels * 4,
        "F4": channels * 8,
    }


class STMFE(nn.Module):

    def __init__(self, channels=128, k=(10, 10, 10, 10), x=(200, 50, 10, 3)):
        super().__init__()
        self.channels = channels
        self.backbone = STMBackbone(channels=channels, stage_k=list(k), stage_x=list(x))
        self.feat_dims = _standard_dims(channels)

    def forward_features(self, x):
        F0, F1, F2, F3, F4 = self.backbone.forward_features(x)
        return {"F0": F0, "F1": F1, "F2": F2, "F3": F3, "F4": F4}


class SS2DFeatureExtractor(nn.Module):

    def __init__(self, channels=128, depths=(2, 2, 9, 2), dims=(96, 192, 384, 768)):
        # lazy import: the VMamba backbone is only needed for the SS2D encoder variant
        from models.vmamba import Backbone_VSSM
        super().__init__()
        self.channels = channels
        self.dims = list(dims)
        self.backbone = Backbone_VSSM(
            depths=list(depths), dims=self.dims, out_indices=(0, 1, 2, 3), norm_layer="ln"
        )
        self.conv1 = nn.Conv2d(3, channels // 2, kernel_size=3, stride=2, padding=1)
        self.norm1 = nn.BatchNorm2d(channels // 2)
        self.act = nn.GELU()
        target = [channels, channels * 2, channels * 4, channels * 8]
        self.proj = nn.ModuleList([nn.Conv2d(d, t, 1) for d, t in zip(self.dims, target)])
        self.feat_dims = _standard_dims(channels)

    def forward_features(self, x):
        F0 = self.act(self.norm1(self.conv1(x)))
        outs = self.backbone(x)  # 4 stages, channels_first
        F1, F2, F3, F4 = [self.proj[i](o) for i, o in enumerate(outs)]
        return {"F0": F0, "F1": F1, "F2": F2, "F3": F3, "F4": F4}


def build_stmfe(opt):
    enc = getattr(opt, "feature_extractor", "stmfe")
    channels = int(getattr(opt, "base_channels", 128))
    k = list(getattr(opt, "k", [10, 10, 10, 10]))
    x = list(getattr(opt, "x", [200, 50, 10, 3]))
    if enc == "ss2d":
        return SS2DFeatureExtractor(channels=channels)
    return STMFE(channels=channels, k=k, x=x)
