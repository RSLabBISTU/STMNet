#!/usr/bin/env python3
"""STMNet.

Architecture
------------
STMFE (Segment-Tree-Mamba Feature Extractor)
-> STMDE (Segment-Tree-Mamba Difference Extractor)
-> CGAD (Cross-Scale Gated Aggregation Decoder).

Usage
-----
Single image pair::

    python infer.py --t1 A.png --t2 B.png --out pred.png

Whole dataset folder (expects ``<root>/<split>/{time1,time2,label}``)::

    python infer.py --data-root /data/LEVIR-CD --split test --out-dir preds/ --eval

``--eval`` additionally reports Precision / Recall / F1 / IoU / OA / Kappa,
which lets you confirm that the environment is set up correctly: the reference
numbers for LEVIR-CD ``test`` are::

    P = 93.00   R = 92.04   F1 = 92.52   IoU = 86.08   OA = 98.36   KC = 91.60
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import types

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from models import STMNet

DEFAULT_CFG = {
    "base_channels": 128,
    "decoder_channels": 128,
    "num_classes": 2,
    "global_scan_type": "four_directional",
    "difference_extractor": "stmde",
    "use_drop": True,
    "decoder_dropout": 0.1,
    "feature_extractor": "stmfe",
    "scan_state_dim": 16,
    "input_size": 256,
    "k": [10, 10, 10, 10],
    "x": [200, 50, 10, 3],
}

REFERENCE = {"P": 93.00, "R": 92.04, "F1": 92.52, "IoU": 86.08, "OA": 98.36, "KC": 91.60}


# --------------------------------------------------------------------------
# Evaluation-time resize pipeline, retained exactly for checkpoint equivalence.
# --------------------------------------------------------------------------
class ScaleInOutput:
    def __init__(self, input_size=256):
        self.input_size = (input_size, input_size) if isinstance(input_size, int) else tuple(input_size)
        self.output_size = None

    def scale_input(self, imgs: tuple):
        imgs = list(imgs)
        self.output_size = imgs[0].shape[2:]
        for i, img in enumerate(imgs):
            imgs[i] = F.interpolate(img, self.input_size, mode="bilinear", align_corners=True)
        return tuple(imgs)

    def scale_output(self, outs: tuple):
        if isinstance(outs, torch.Tensor):
            outs = (outs,)
        outs = list(outs)
        assert self.output_size is not None, "call scale_input() first"
        for i, out in enumerate(outs):
            outs[i] = F.interpolate(out, self.output_size, mode="bilinear", align_corners=True)
        return tuple(outs)


def build_stmnet(cfg: dict, ckpt: str, device: str) -> STMNet:
    if "stmde_variant" in cfg:
        raise ValueError("Remove stmde_variant from the config; STMDE uses one fixed structure.")
    opt = types.SimpleNamespace(
        base_channels=cfg["base_channels"],
        decoder_channels=cfg["decoder_channels"],
        num_classes=cfg["num_classes"],
        global_scan_type=cfg["global_scan_type"],
        difference_extractor=cfg["difference_extractor"],
        use_drop=cfg["use_drop"],
        pretrain="",
        k=list(cfg["k"]),
        x=list(cfg["x"]),
        decoder_dropout=cfg["decoder_dropout"],
        feature_extractor=cfg["feature_extractor"],
        scan_state_dim=cfg["scan_state_dim"],
        input_size=cfg["input_size"],
    )
    model = STMNet(opt)
    try:  # the released file is a plain state_dict, so weights_only is safe
        state = torch.load(ckpt, map_location="cpu", weights_only=True)
    except Exception:
        state = torch.load(ckpt, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    # strict: any mismatch means the config and the checkpoint disagree
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def load_pair(path_pre: str, path_post: str) -> tuple:
    """Exact match of the evaluation-time preprocessing: RGB -> float32 / 255."""
    a = np.asarray(Image.open(path_pre).convert("RGB"), dtype=np.float32) / 255.0
    b = np.asarray(Image.open(path_post).convert("RGB"), dtype=np.float32) / 255.0
    I_pre = torch.from_numpy(a.transpose(2, 0, 1)).unsqueeze(0)
    I_post = torch.from_numpy(b.transpose(2, 0, 1)).unsqueeze(0)
    return I_pre, I_post


@torch.no_grad()
def predict_prob(model, I_pre, I_post, input_size=256, device="cuda") -> np.ndarray:
    """Returns the change probability map at the original image resolution."""
    scale = ScaleInOutput(input_size)
    x1, x2 = scale.scale_input((I_pre.to(device), I_post.to(device)))
    _, outs = model(x1, x2)
    if not isinstance(outs, tuple):
        outs = (outs, outs)
    outs = scale.scale_output(outs)
    prob = torch.softmax(outs[0].float(), dim=1)[:, 1]
    return prob[0].cpu().numpy()


def save_mask(prob: np.ndarray, path: str, threshold: float = 0.5) -> None:
    Image.fromarray(((prob > threshold) * 255).astype(np.uint8)).save(path)


def confusion(prob, label, threshold):
    pred = prob > threshold
    gt = label > 0
    tp = int((pred & gt).sum())
    fp = int((pred & ~gt).sum())
    fn = int((~pred & gt).sum())
    tn = int((~pred & ~gt).sum())
    return tp, fp, fn, tn


def metrics_from_counts(tp, fp, fn, tn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    iou = tp / (tp + fp + fn) if tp + fp + fn else 0.0
    oa = (tp + tn) / (tp + tn + fp + fn)
    pe = ((tp + fp) * (tp + fn) + (fp + tn) * (fn + tn)) / (tp + tn + fp + fn) ** 2
    kc = (oa - pe) / (1 - pe)
    return p, r, f1, iou, oa, kc


def main() -> None:
    ap = argparse.ArgumentParser(description="STMNet inference (LEVIR-CD checkpoint)")
    ap.add_argument("--ckpt", default=os.path.join(HERE, "weights", "stmnet_levir_k10_ms.pth"))
    ap.add_argument("--config", default=os.path.join(HERE, "configs", "levir_k10_ms.json"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--threshold", type=float, default=0.5)
    # single pair
    ap.add_argument("--t1", help="path to the bi-temporal image 1")
    ap.add_argument("--t2", help="path to the bi-temporal image 2")
    ap.add_argument("--out", help="output mask path for the single-pair mode")
    ap.add_argument("--prob", help="optional .npy path to store the probability map")
    # dataset mode
    ap.add_argument("--data-root", help="dataset root containing <split>/{time1,time2,label}")
    ap.add_argument("--split", default="test")
    ap.add_argument("--out-dir", help="folder to write predicted masks into")
    ap.add_argument("--eval", action="store_true", help="compute metrics (requires labels)")
    ap.add_argument("--limit", type=int, default=0, help="only process the first N pairs (0 = all)")
    args = ap.parse_args()

    cfg = dict(DEFAULT_CFG)
    if os.path.isfile(args.config):
        with open(args.config) as fh:
            cfg.update(json.load(fh).get("model", {}))
    model = build_stmnet(cfg, args.ckpt, args.device)
    print("model loaded: {:.2f} M params on {}".format(
        sum(p.numel() for p in model.parameters()) / 1e6, args.device))

    if args.t1 and args.t2:
        I_pre, I_post = load_pair(args.t1, args.t2)

        prob = predict_prob(model, I_pre, I_post, cfg["input_size"], args.device)
        out = args.out or "pred.png"
        save_mask(prob, out, args.threshold)
        if args.prob:
            np.save(args.prob, prob)
        print("saved {}  (change pixels: {:.2%})".format(out, float((prob > args.threshold).mean())))
        return

    if not args.data_root:
        ap.error("give either --t1/--t2 or --data-root")

    d = os.path.join(args.data_root, args.split)
    names = sorted(os.listdir(os.path.join(d, "time1")))
    if args.limit:
        names = names[: args.limit]
    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)

    tot = [0, 0, 0, 0]
    for i, name in enumerate(names):
        I_pre, I_post = load_pair(os.path.join(d, "time1", name), os.path.join(d, "time2", name))
        prob = predict_prob(model, I_pre, I_post, cfg["input_size"], args.device)
        if args.out_dir:
            save_mask(prob, os.path.join(args.out_dir, os.path.splitext(name)[0] + ".png"), args.threshold)
        if args.eval:
            lab_path = os.path.join(d, "label", name)
            if not os.path.isfile(lab_path):
                lab_path = os.path.join(d, "label", os.path.splitext(name)[0] + ".png")
            gt = np.asarray(Image.open(lab_path).convert("L"))
            c = confusion(prob, gt, args.threshold)
            tot = [a + b for a, b in zip(tot, c)]
        if (i + 1) % 50 == 0 or i + 1 == len(names):
            print("  {}/{}".format(i + 1, len(names)), flush=True)

    if args.eval:
        p, r, f1, iou, oa, kc = metrics_from_counts(*tot)
        print("\n--- {} / {} ({} pairs) ---".format(args.data_root, args.split, len(names)))
        print("  Precision = {:.2f}   Recall = {:.2f}   F1 = {:.2f}".format(100 * p, 100 * r, 100 * f1))
        print("  IoU       = {:.2f}   OA     = {:.2f}   Kappa = {:.2f}".format(100 * iou, 100 * oa, 100 * kc))
        print("  reference P={P:.2f} R={R:.2f} F1={F1:.2f} IoU={IoU:.2f} OA={OA:.2f} KC={KC:.2f}".format(**REFERENCE))


if __name__ == "__main__":
    main()
