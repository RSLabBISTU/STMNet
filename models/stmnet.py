"""STMNet"""
from __future__ import annotations
import torch
import torch.nn as nn
from models.stm import STMDE
from models.stmfe import build_stmfe
from models.cgad import CGAD
from models.block.drop import DropBlock

class STMNet(nn.Module):
    """Segment-Tree-guided Mamba Network: STMFE -> STMDE -> CGAD."""
    feature_order = ("F0", "F1", "F2", "F3", "F4")

    def __init__(self, opt):
        super().__init__()
        if getattr(opt, "difference_extractor", "stmde") != "stmde":
            raise ValueError("Only the paper's STMDE difference extractor is supported.")
        if hasattr(opt, "stmde_variant"):
            raise ValueError("stmde_variant has been removed; STMDE uses one fixed structure.")
        self.base_channels = opt.base_channels
        self.decoder_channels = int(getattr(opt, "decoder_channels", 128))
        self.num_classes = int(getattr(opt, "num_classes", 2))
        self.global_scan_type = getattr(opt, "global_scan_type", "four_directional")
        self.use_drop = bool(getattr(opt, "use_drop", True))
        self.pretrain = getattr(opt, "pretrain", "") or ""

        self.stmfe = build_stmfe(opt)
        if self.use_drop:
            self.drop = DropBlock(rate=0.15, size=7, step=30)

        stage_channels = [self.base_channels * scale for scale in (1, 2, 4, 8)]
        difference_channels = [self.base_channels * scale for scale in (1, 2, 4, 4)]
        drop_path_rates = [0.01, 0.03, 0.1, 0.15]
        self.stmde = nn.ModuleList([
            STMDE(
                in_channels=stage_channels[i] * 2,
                out_channels=difference_channels[i],
                drop_path_rate=drop_path_rates[i],
                k=opt.k[i],
                x=opt.x[i],
            )
            for i in range(4)
        ])

        self.cgad = CGAD(opt)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        pretrain = getattr(self, "pretrain", "") or ""
        if pretrain.endswith(".pt"):
            ck = torch.load(pretrain, weights_only=False)
            if isinstance(ck, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
                ck = ck.module
            sd = ck.state_dict() if hasattr(ck, "state_dict") else ck
            md = self.state_dict()
            loaded = {k: v for k, v in sd.items() if k in md and md[k].shape == v.shape}
            md.update(loaded)
            self.load_state_dict(md)
            print("Pretrained loaded: {}/{} tensors from {}".format(len(loaded), len(md), pretrain))

    def extract_features(self, image):
        """Apply shared STMFE weights to I_pre or I_post, returning F0--F4."""
        return self.stmfe.forward_features(image)

    def extract_differences(self, F_pre, F_post):
        """Return D0--D4 for CGAD without changing checkpoint computations."""
        D = {}
        for stage, name in enumerate(("F1", "F2", "F3", "F4")):
            D[f"D{stage + 1}"] = self.stmde[stage](F_pre[name], F_post[name])
        D["D0"] = torch.cat(
            [F_pre["F0"], F_post["F0"], torch.abs(F_pre["F0"] - F_post["F0"])], dim=1
        )
        return D

    def forward(self, I_pre, I_post):
        output_size = I_pre.shape[-2:]
        F_pre = self.extract_features(I_pre)
        F_post = self.extract_features(I_post)
        if self.use_drop and self.training:
            drop_features = [F_pre[n] for n in self.feature_order] + [F_post[n] for n in self.feature_order]
            drop_features = self.drop(drop_features)
            for index, name in enumerate(self.feature_order):
                F_pre[name] = drop_features[index]
            for index, name in enumerate(self.feature_order):
                F_post[name] = drop_features[len(self.feature_order) + index]
        D = self.extract_differences(F_pre, F_post)
        return self.cgad(D, output_size)
