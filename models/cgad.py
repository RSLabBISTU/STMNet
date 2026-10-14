from __future__ import annotations

from typing import Dict, Iterable, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.stssm import STSSM

class ConvLNReLU(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, dropout: float = 0.0):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False)
        self.ln = nn.LayerNorm(out_channels)
        self.act = nn.ReLU(inplace=True)
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = x.permute(0, 2, 3, 1).contiguous()
        x = self.ln(x)
        x = self.act(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        return self.drop(x)


class TopDown(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.g2 = nn.Sequential(
            ConvLNReLU(dim * 2, dim, kernel_size=3),
            nn.Conv2d(dim, dim, kernel_size=1),
            nn.Sigmoid(),
        )
        self.g1 = nn.Sequential(
            ConvLNReLU(dim * 2, dim, kernel_size=3),
            nn.Conv2d(dim, dim, kernel_size=1),
            nn.Sigmoid(),
        )
        self.refine = ConvLNReLU(dim, dim, kernel_size=3)

    def forward(self, F_high: torch.Tensor, F_low: torch.Tensor) -> torch.Tensor:
        F_high_up = F.interpolate(F_high, size=F_low.shape[-2:], mode="bilinear", align_corners=False)
        g2 = self.g2(torch.cat([F_high_up, F_low], dim=1))
        g1 = self.g1(torch.cat([F_high_up, F_low], dim=1))
        return self.refine(g2 * F_low + g1 * F_high_up)


class BottomUp(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.g1 = nn.Sequential(
            ConvLNReLU(dim * 2, dim, kernel_size=3),
            nn.Conv2d(dim, dim, kernel_size=1),
            nn.Sigmoid(),
        )
        self.g2 = nn.Sequential(
            ConvLNReLU(dim * 2, dim, kernel_size=3),
            nn.Conv2d(dim, dim, kernel_size=1),
            nn.Sigmoid(),
        )
        self.refine = ConvLNReLU(dim, dim, kernel_size=3)

    def forward(self, F_low: torch.Tensor, F_high: torch.Tensor) -> torch.Tensor:
        F_low_resized = F.interpolate(F_low, size=F_high.shape[-2:], mode="bilinear", align_corners=False)
        g1 = self.g1(torch.cat([F_high, F_low_resized], dim=1))
        g2 = self.g2(torch.cat([F_high, F_low_resized], dim=1))
        return self.refine(g1 * F_high + g2 * F_low_resized)


class MultiScaleFusion(nn.Module):
    def __init__(self, dim: int, levels: int = 3):
        super().__init__()
        self.fuse = ConvLNReLU(dim * levels, dim, kernel_size=1)

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        target_size = features[0].shape[-2:]
        aligned = [
            feature
            if feature.shape[-2:] == target_size
            else F.interpolate(feature, size=target_size, mode="bilinear", align_corners=False)
            for feature in features
        ]
        return self.fuse(torch.cat(aligned, dim=1))


def _import_mamba():
    try:
        from mamba_ssm import Mamba
    except ImportError as exc:
        raise ImportError(
            "Official Mamba is required. Install it: pip install 'mamba-ssm[causal-conv1d]' --no-build-isolation"
        ) from exc
    return Mamba


class ResidualMambaBlock(nn.Module):
    def __init__(self, mamba: nn.Module, dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.mamba = mamba

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.mamba(self.norm(x))


class FourDirectionalScan(nn.Module):
    def __init__(
        self,
        dim: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        num_layers: int = 1,
        directions: Iterable[str] = ("lr", "rl", "tb", "bt"),
    ):
        super().__init__()
        self.dim = int(dim)
        self.directions = tuple(directions)
        Mamba = _import_mamba()
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    *[
                        ResidualMambaBlock(
                            Mamba(d_model=self.dim, d_state=d_state, d_conv=d_conv, expand=expand),
                            dim=self.dim,
                        )
                        for _ in range(num_layers)
                    ]
                )
                for _ in self.directions
            ]
        )
        self.fuse = ConvLNReLU(self.dim * len(self.directions), self.dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs: List[torch.Tensor] = []
        for direction, block in zip(self.directions, self.blocks):
            seq, shape = self._to_sequence(x, direction)
            seq = block(seq)
            outputs.append(self._from_sequence(seq, shape, direction))
        return self.fuse(torch.cat(outputs, dim=1))

    def _to_sequence(self, x: torch.Tensor, direction: str):
        b, c, h, w = x.shape
        if direction == "lr":
            seq = x.permute(0, 2, 3, 1).reshape(b, h * w, c)
        elif direction == "rl":
            seq = torch.flip(x, dims=[3]).permute(0, 2, 3, 1).reshape(b, h * w, c)
        elif direction == "tb":
            seq = x.permute(0, 3, 2, 1).reshape(b, h * w, c)
        elif direction == "bt":
            seq = torch.flip(x, dims=[2]).permute(0, 3, 2, 1).reshape(b, h * w, c)
        else:
            raise ValueError(f"Unsupported scan direction: {direction}")
        return seq.contiguous(), (b, c, h, w)

    def _from_sequence(self, seq: torch.Tensor, shape, direction: str) -> torch.Tensor:
        b, c, h, w = shape
        if direction == "lr":
            x = seq.reshape(b, h, w, c).permute(0, 3, 1, 2)
        elif direction == "rl":
            x = seq.reshape(b, h, w, c).permute(0, 3, 1, 2)
            x = torch.flip(x, dims=[3])
        elif direction == "tb":
            x = seq.reshape(b, w, h, c).permute(0, 3, 2, 1)
        elif direction == "bt":
            x = seq.reshape(b, w, h, c).permute(0, 3, 2, 1)
            x = torch.flip(x, dims=[2])
        else:
            raise ValueError(f"Unsupported scan direction: {direction}")
        return x.contiguous()


class SegmentTreeScan(nn.Module):
    def __init__(
        self,
        dim: int,
        k: int = 10,
        x: int = 200,
        d_state: int = 1,
        ssm_ratio: float = 2.0,
        ssm_rank_ratio: float = 2.0,
        d_conv: int = 3,
    ):
        super().__init__()
        self.stssm = STSSM(
            d_model=dim,
            d_state=d_state,
            ssm_ratio=ssm_ratio,
            ssm_rank_ratio=ssm_rank_ratio,
            dt_rank="auto",
            act_layer=nn.SiLU,
            d_conv=d_conv,
            conv_bias=False,
            dropout=0.0,
            k=k,
            x=x,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 1).contiguous()
        x = self.stssm(x)
        return x.permute(0, 3, 1, 2).contiguous()


def build_global_scan(num_dim: int, cfg: Dict):
    global_scan_type = cfg.get("global_scan_type", "four_directional")
    if global_scan_type == "segment_tree":
        return SegmentTreeScan(
            dim=num_dim,
            k=int(cfg.get("scan_k", 10)),
            x=int(cfg.get("scan_x", 200)),
            d_state=int(cfg.get("scan_state_dim", 1)),
        )
    return FourDirectionalScan(
        dim=num_dim,
        d_state=int(cfg.get("scan_state_dim", 16)),
        d_conv=int(cfg.get("scan_conv_width", 4)),
        expand=int(cfg.get("scan_expand", 2)),
        num_layers=int(cfg.get("scan_num_layers", 1)),
    )


class CGAD(nn.Module):
    difference_order = ("D0", "D1", "D2", "D3", "D4")

    def __init__(self, opt):
        super().__init__()
        self.base_channels = opt.base_channels
        self.decoder_channels = int(getattr(opt, "decoder_channels", 128))
        self.num_classes = int(getattr(opt, "num_classes", 2))
        proj_in = {
            "D0": (self.base_channels // 2) * 3,
            "D1": self.base_channels,
            "D2": self.base_channels * 2,
            "D3": self.base_channels * 4,
            "D4": self.base_channels * 4,
        }
        self.input_projections = nn.ModuleDict(
            {name: ConvLNReLU(proj_in[name], self.decoder_channels, kernel_size=1) for name in self.difference_order}
        )

        self.top_down = nn.ModuleDict(
            {
                "D4_to_D3": TopDown(self.decoder_channels),
                "D3_to_D2": TopDown(self.decoder_channels),
                "D2_to_D1": TopDown(self.decoder_channels),
            }
        )
        self.global_scan = build_global_scan(self.decoder_channels, vars(opt) if hasattr(opt, "__dict__") else {})

        self.bottom_up = nn.ModuleDict(
            {
                "scan_D1_to_D0": TopDown(self.decoder_channels),
                "D0_to_D1": BottomUp(self.decoder_channels),
                "D1_to_D2": BottomUp(self.decoder_channels),
                "D2_to_D3": BottomUp(self.decoder_channels),
            }
        )
        self.multi_scale_fusion = MultiScaleFusion(self.decoder_channels, levels=4)
        dropout = float(getattr(opt, "decoder_dropout", 0.1))
        self.prediction_head = nn.Sequential(
            ConvLNReLU(self.decoder_channels, self.decoder_channels, kernel_size=3, dropout=dropout),
            ConvLNReLU(self.decoder_channels, self.decoder_channels // 2, kernel_size=3, dropout=dropout),
            nn.Conv2d(self.decoder_channels // 2, self.num_classes, kernel_size=1),
        )

    def forward(self, D, output_size):
        D = {name: self.input_projections[name](D[name]) for name in self.difference_order}
        F_td3 = self.top_down["D4_to_D3"](D["D4"], D["D3"])
        F_td2 = self.top_down["D3_to_D2"](F_td3, D["D2"])
        F_td1 = self.top_down["D2_to_D1"](F_td2, D["D1"])

        F_bu1 = self.global_scan(F_td1)
        F_bu0 = self.bottom_up["scan_D1_to_D0"](F_bu1, D["D0"])
        F_bu1 = self.bottom_up["D0_to_D1"](F_bu0, F_bu1)
        F_bu2 = self.bottom_up["D1_to_D2"](F_bu1, F_td2)
        F_bu3 = self.bottom_up["D2_to_D3"](F_bu2, F_td3)

        fused = self.multi_scale_fusion([F_bu0, F_bu1, F_bu2, F_bu3])
        logits = self.prediction_head(fused)
        logits = F.interpolate(logits, size=output_size, mode="bilinear", align_corners=False)
        return [F_bu3, F_bu2, F_bu1, logits], logits
