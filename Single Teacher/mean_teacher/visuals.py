import os.path as osp
from torch.amp import autocast
import csv, logging
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch

import os
import os.path as osp
import shutil
import time
import logging
import csv
import sys
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import torchvision.utils as vutils
from torch.amp import autocast, GradScaler  # modern AMP
import torch.nn.functional as F
from mean_teacher import architectures, losses, ramps, cli
from mean_teacher.run_context import RunContext
from mean_teacher.utils import *  # AverageMeterSet, parameters_string, etc.
from mean_teacher.data import TwoStreamBatchSampler, create_coco_datasets
from mean_teacher.metrics import compute_ap_set, segmentation_metrics, format_eta, get_model_device
from pycocotools.coco import COCO
from math import exp 
import yaml


LOG = logging.getLogger('main')
AMP_DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

@torch.no_grad()
def dump_teacher_pseudomasks(x_u_t, logits_u_t, paths, out_dir, step, max_samples=4, thresh=0.9):
    os.makedirs(out_dir, exist_ok=True)
    probs = torch.sigmoid(logits_u_t)
    binmsk = (probs > thresh).float()
    B = probs.size(0)
    k = min(B, max_samples)

    for j in range(k):
        img = x_u_t[j].detach().cpu().clamp(0, 1)  # [3,H,W]
        pm  = probs[j].detach().cpu()              # [1,H,W]
        bm  = binmsk[j].detach().cpu()             # [1,H,W]

        overlay = img.clone()
        overlay[0] = torch.max(overlay[0], bm[0])

        stem = f"step{step:07d}_{j}"
        base = None
        if paths is not None and j < len(paths):
            base = os.path.splitext(os.path.basename(paths[j]))[0]
        if base:
            stem = f"{base}_{stem}"

        vutils.save_image(img,     osp.join(out_dir, f"{stem}_img.png"))
        vutils.save_image(pm,      osp.join(out_dir, f"{stem}_prob.png"))
        vutils.save_image(bm,      osp.join(out_dir, f"{stem}_bin.png"))
        vutils.save_image(overlay, osp.join(out_dir, f"{stem}_overlay.png"))


def plot_from_csv(csv_path, title, out_dir):
    """
    Reads epoch, IoU, Dice from csv_path and produces two separate PNGs:
      {title_lower}_iou.png and {title_lower}_dice.png
    """
    if not os.path.exists(csv_path):
        LOG.warning(f"No CSV found at {csv_path}, skipping {title} plots.")
        return

    # default to the CSV's directory
    if out_dir is None:
        out_dir = os.path.dirname(csv_path) or "."

    os.makedirs(out_dir, exist_ok=True)

    epochs, ious, dices = [], [], []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(float(row['epoch'])))
            ious.append(float(row['iou']))
            dices.append(float(row['dice']))

    # IoU plot
    plt.figure()
    plt.plot(epochs, ious, label="IoU")
    plt.xlabel("Epoch")
    plt.ylabel("IoU")
    plt.title(f"{title} IoU")
    plt.legend()
    out_iou = os.path.join(out_dir, f"{title.lower()}_iou.png")
    plt.savefig(out_iou, dpi=150)
    plt.close()

    # Dice plot
    plt.figure()
    plt.plot(epochs, dices, label="Dice")
    plt.xlabel("Epoch")
    plt.ylabel("Dice")
    plt.title(f"{title} Dice")
    plt.legend()
    out_dice = os.path.join(out_dir, f"{title.lower()}_dice.png")
    plt.savefig(out_dice, dpi=150)
    plt.close()

    LOG.info("Saved plots: %s, %s", out_iou, out_dice)

@torch.no_grad()
def inspect_unlabeled_batch(train_loader, student, teacher, out_dir, max_images=20, thresh=0.9):
    """
    Saves overlays for up to `max_images` UNLABELED samples from the next train batch:
      - input image (weak)
      - teacher prob/bin/overlay (Mask R-CNN inference)
      - student prob/bin/overlay (Mask R-CNN inference)
    """
    os.makedirs(out_dir, exist_ok=True)

    # Grab one batch from the loader
    try:
        batch = next(iter(train_loader))
    except StopIteration:
        LOG.warning("inspect_unlabeled_batch: empty loader")
        return

    (x_w, x_s), (_y_mask, is_labeled), paths = batch

    # Only keep unlabeled indices
    mask_unlab = ~is_labeled.bool()
    if not mask_unlab.any():
        LOG.warning("inspect_unlabeled_batch: no unlabeled samples in this batch")
        return

    # Pick a device (same as student model)
    device = next(student.parameters()).device

    # Move data
    xw_u = x_w[mask_unlab].to(device, non_blocking=True)  # weak → teacher
    xs_u = x_s[mask_unlab].to(device, non_blocking=True)  # strong → student
    sel_paths = [p for p, lbl in zip(paths, is_labeled.tolist()) if lbl == 0]

    B = xw_u.size(0)
    H, W = xw_u.shape[-2:]

    # ---- Inference-only (no targets) ----
    teacher.eval()
    was_training = student.training
    student.eval()
    with torch.no_grad(), autocast(AMP_DEVICE):
        t_out = teacher([xw_u[j] for j in range(B)])  # list[dict]
        s_out = student([xs_u[j] for j in range(B)])  # list[dict]
    if was_training:
        student.train()

    # Convert instance predictions → semantic probability maps [B,1,H,W]
    t_probs = _preds_to_semantic_prob(t_out, H, W, device=device)
    s_probs = _preds_to_semantic_prob(s_out, H, W, device=device)

    t_bin = (t_probs > thresh).float()
    s_bin = (s_probs > thresh).float()

    k = min(max_images, B)
    for j in range(k):
        img = xw_u[j].detach().clamp(0, 1)  # WEAK view
        tp  = t_probs[j].detach()
        tb  = t_bin[j].detach()
        sp  = s_probs[j].detach()
        sb  = s_bin[j].detach()

        # overlays
        over_t = img.clone(); over_t[0] = torch.max(over_t[0], tb[0])
        over_s = img.clone(); over_s[0] = torch.max(over_s[0], sb[0])

        base = os.path.splitext(os.path.basename(sel_paths[j]))[0] if j < len(sel_paths) else f"unlab{j:02d}"
        vutils.save_image(img,     os.path.join(out_dir, f"{base}_img.png"))
        vutils.save_image(tp,      os.path.join(out_dir, f"{base}_teacher_prob.png"))
        vutils.save_image(tb,      os.path.join(out_dir, f"{base}_teacher_bin.png"))
        vutils.save_image(over_t,  os.path.join(out_dir, f"{base}_teacher_overlay.png"))
        vutils.save_image(sp,      os.path.join(out_dir, f"{base}_student_prob.png"))
        vutils.save_image(sb,      os.path.join(out_dir, f"{base}_student_bin.png"))
        vutils.save_image(over_s,  os.path.join(out_dir, f"{base}_student_overlay.png"))

    LOG.info("Saved %d unlabeled inspections to %s", k, out_dir)

def _preds_to_semantic_prob(outputs, H, W, eps=1e-6, device=None):
    """
    outputs: list of dicts (Mask R-CNN eval output per image)
      expects keys 'masks' [N,1,hm,wm] and 'scores' [N]
    Returns tensor [B,1,H,W] via score-weighted noisy-OR union:
      P = 1 - Π_i (1 - s_i * p_i_up)
    """
    out = []

    # Pick a reference device
    if device is None:
        for o in outputs:
            if "masks" in o and o["masks"].numel() > 0:
                device = o["masks"].device
                break
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for o in outputs:
        if ("masks" in o) and (o["masks"].numel() > 0):
            pm = o["masks"][:, 0:1, :, :].to(device)  # [N,1,hm,wm]
            pm = F.interpolate(pm, size=(H, W), mode="bilinear", align_corners=False)  # [N,1,H,W]
            scores = o.get("scores", torch.ones((pm.shape[0],), device=device)).view(-1, 1, 1, 1).to(device)
            pmw = (scores * pm).clamp(0.0, 1.0)
            one_minus = (1.0 - pmw).clamp(eps, 1.0)
            prob = 1.0 - torch.prod(one_minus, dim=0, keepdim=True)  # [1,1,H,W]
        else:
            prob = torch.zeros((1, 1, H, W), device=device)
        out.append(prob)

    return torch.cat(out, dim=0).to(device)  # [B,1,H,W]


def _preds_to_semantic_prob_max(outputs, H, W):
    out = []
    device = None
    for o in outputs:
        if device is None:
            device = (o.get("masks", torch.empty(0)).device
                      if ("masks" in o and o["masks"].numel() > 0)
                      else torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        if ("masks" in o) and (o["masks"].numel() > 0):
            pm = o["masks"][:, 0:1, :, :]  # [N,1,h,w], probs
            pm = F.interpolate(pm, size=(H, W), mode="bilinear", align_corners=False)  # [N,1,H,W]
            prob = pm.max(dim=0, keepdim=True).values  # [1,1,H,W]
        else:
            prob = torch.zeros((1,1,H,W), device=device)
        out.append(prob)
    return torch.cat(out, dim=0)  # [B,1,H,W]

def _collapse_to_student_space(P_a, P_b, rule="max"):
    if rule == "noisy_or":
        return 1.0 - (1.0 - P_a) * (1.0 - P_b)
    # default: max
    return torch.maximum(P_a, P_b)