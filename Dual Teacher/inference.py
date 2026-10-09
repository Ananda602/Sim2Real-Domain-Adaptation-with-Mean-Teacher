# inference.py
import os
import glob
import json
import argparse
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch.amp import autocast
from PIL import Image
import numpy as np
import torchvision.transforms as T
from pycocotools import mask as maskUtils

from mean_teacher import architectures, datasets
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn

Box = Tuple[int, int, int, int]  # (x, y, w, h)


def _pick_checkpoint(weights_arg: str) -> str:
    if os.path.isfile(weights_arg):
        return weights_arg
    if not os.path.isdir(weights_arg):
        raise FileNotFoundError(f"weights path not found: {weights_arg}")

    for name in ("best.ckpt", "last.ckpt"):
        cand = os.path.join(weights_arg, name)
        if os.path.isfile(cand):
            return cand

    cands = glob.glob(os.path.join(weights_arg, "checkpoint.*.ckpt"))
    if not cands:
        raise FileNotFoundError(f"No checkpoints found in directory: {weights_arg}")

    def _epoch_num(p):
        parts = os.path.basename(p).split(".")
        try:
            return int(parts[1])
        except Exception:
            return -1

    cands.sort(key=_epoch_num, reverse=True)
    return cands[0]


def load_model(arch: str, ckpt_path: str, use_ema: bool, num_classes: int, device: str):
    if arch == "maskrcnn":
        model = maskrcnn_resnet50_fpn(weights=None, num_classes=num_classes)
    else:
        model = architectures.__dict__[arch](num_classes=num_classes, in_channels=3)

    ckpt = torch.load(ckpt_path, map_location="cpu")
    raw_state = ckpt["ema_state_dict"] if (use_ema and "ema_state_dict" in ckpt) else ckpt["state_dict"]

    model.load_state_dict(raw_state, strict=False)  # allow mismatches
    model.to(device).eval()
    return model

def load_coco_annotations(json_path: str):
    with open(json_path, "r") as f:
        data = json.load(f)

    if not isinstance(data, dict) or "images" not in data or "annotations" not in data:
        raise ValueError("Annotation file must be COCO-style with 'images' and 'annotations'.")

    file_to_id: Dict[str, int] = {}
    id_to_filename: Dict[int, str] = {}
    for im in data["images"]:
        iid = int(im["id"])
        fn = im["file_name"]
        file_to_id[fn] = iid
        id_to_filename[iid] = fn

    id_to_anns: Dict[int, List[dict]] = {}
    for ann in data["annotations"]:
        id_to_anns.setdefault(int(ann["image_id"]), []).append(ann)

    return file_to_id, id_to_anns, id_to_filename

@torch.no_grad()
def infer_box_mask(model, image: Image.Image, box: Box, tfm: T.Compose,
                   device: str, threshold: float, amp_device: str) -> np.ndarray:
    x, y, w, h = box
    crop = image.crop((x, y, x + w, y + h)).convert("RGB")
    x_in = tfm(crop).unsqueeze(0).to(device)

    if device == "cuda":
        with autocast(device_type=amp_device):
            outputs = model(x_in)
    else:
        outputs = model(x_in)

    if len(outputs) == 0 or "masks" not in outputs[0]:
        return np.zeros((h, w), dtype=np.uint8)

    masks = outputs[0]["masks"]  # [N,1,H,W]
    scores = outputs[0]["scores"]
    if masks.numel() == 0:
        return np.zeros((h, w), dtype=np.uint8)

    idx = scores.argmax().item()
    probs = masks[idx, 0]  # [H,W]

    probs = F.interpolate(probs.unsqueeze(0).unsqueeze(0),
                          size=(h, w), mode="bilinear", align_corners=False)
    binmask = (probs >= threshold).float()[0, 0].cpu().numpy().astype(np.uint8)
    return binmask


def make_overlay(rgb_img: Image.Image, mask: np.ndarray, color: str) -> Image.Image:
    rgb = np.array(rgb_img).astype(np.float32) / 255.0
    m = mask.astype(np.float32)[..., None]

    if color == "red":
        rgb[..., 0] = np.maximum(rgb[..., 0], m[..., 0])
    elif color == "green":
        rgb[..., 1] = np.maximum(rgb[..., 1], m[..., 0])
    else:
        raise ValueError("color must be 'red' or 'green'")

    rgb = (np.clip(rgb, 0, 1) * 255.0).astype(np.uint8)
    return Image.fromarray(rgb)



def make_side_by_side(gt_overlay: Image.Image, pred_overlay: Image.Image) -> Image.Image:
    W, H = gt_overlay.size
    if pred_overlay.size != (W, H):
        pred_overlay = pred_overlay.resize((W, H))
    panel = Image.new("RGB", (W * 2, H))
    panel.paste(gt_overlay, (0, 0))
    panel.paste(pred_overlay, (W, 0))
    return panel

def main():
    ap = argparse.ArgumentParser(description="Segmentation inference with GT vs Prediction overlays (1x2 view).")
    ap.add_argument("--images", required=True, help="Path to a folder of images.")
    ap.add_argument("--ann", required=True, help="COCO-style annotation JSON file.")
    ap.add_argument("--weights", required=True, help="Checkpoint FILE or DIRECTORY.")
    ap.add_argument("--out", required=True, help="Output directory.")
    ap.add_argument("--arch", default="maskrcnn", help="Architecture (e.g., maskrcnn).")
    ap.add_argument("--num-classes", type=int, default=1, help="Output classes.")
    ap.add_argument("--use-ema", action="store_true", help="Use EMA weights.")
    ap.add_argument("--threshold", type=float, default=0.5, help="Mask binarization threshold.")
    ap.add_argument("--size", type=int, default=128, help="Eval crop size.")
    ap.add_argument("--cpu", action="store_true", help="Force CPU.")
    args = ap.parse_args()

    device = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    amp_device = "cuda" if device == "cuda" else "cpu"
    os.makedirs(args.out, exist_ok=True)

    ckpt_path = _pick_checkpoint(args.weights)
    print(f"[info] using checkpoint: {ckpt_path}")

    tfm = T.Compose([T.Resize((args.size, args.size)), T.ToTensor()])
    model = load_model(args.arch, ckpt_path, args.use_ema, args.num_classes, device)

    file_to_id, id_to_anns, id_to_filename = load_coco_annotations(args.ann)

    image_paths = sorted([p for p in glob.glob(os.path.join(args.images, "*"))
                          if p.lower().endswith((".jpg", ".png", ".jpeg"))])
    print(f"[info] found {len(image_paths)} images")

    for img_path in image_paths:
        img = Image.open(img_path).convert("RGB")

        # --- EXIF orientation ---
        try:
            exif = img.getexif()
        except Exception:
            exif = None
        if exif:
            orientation = exif.get(0x0112, 1)
            if orientation == 3:
                img = img.rotate(180, expand=True)
            elif orientation == 6:
                img = img.rotate(270, expand=True)
            elif orientation == 8:
                img = img.rotate(90, expand=True)

        W0, H0 = img.size
        bn = os.path.basename(img_path)
        img_id = file_to_id.get(bn, None)
        if img_id is None:
            print(f"[warn] no matching ID for {bn}, skipping.")
            continue

        anns = id_to_anns.get(img_id, [])
        if not anns:
            print(f"[warn] no annotations for {bn}, skipping.")
            continue

        # --- GT mask overlay ---
        gt_union = np.zeros((H0, W0), dtype=np.uint8)
        for ann in anns:
            seg = ann.get("segmentation", None)
            if seg is None:
                continue
            if isinstance(seg, list):  # polygon
                rle = maskUtils.frPyObjects(seg, H0, W0)
                m = maskUtils.decode(rle)
                gt_union = np.maximum(gt_union, m.max(axis=2).astype(np.uint8))
            elif isinstance(seg, dict):  # RLE
                m = maskUtils.decode(seg)
                gt_union = np.maximum(gt_union, m.astype(np.uint8))
        gt_overlay = make_overlay(img, gt_union, color="red")

        # --- Prediction mask overlay ---
        pred_union = np.zeros((H0, W0), dtype=np.uint8)
        for ann in anns:
            x, y, w, h = map(int, ann["bbox"])

            # --- clip box to image bounds ---
            x = max(0, min(x, W0 - 1))
            y = max(0, min(y, H0 - 1))
            w = max(0, min(w, W0 - x))
            h = max(0, min(h, H0 - y))
            if w == 0 or h == 0:
                continue

            mask_hw = infer_box_mask(model, img, (x, y, w, h), tfm, device,
                                     args.threshold, amp_device)
            if mask_hw.max() == 0:
                continue
            pred_union[y:y+h, x:x+w] = np.maximum(pred_union[y:y+h, x:x+w], mask_hw)

        pred_overlay = make_overlay(img, pred_union, color= "green")

        # --- 1x2 panel ---
        panel = make_side_by_side(gt_overlay, pred_overlay)
        base = os.path.splitext(os.path.basename(img_path))[0]
        panel_out = os.path.join(args.out, f"{base}_gt_vs_pred.png")
        panel.save(panel_out)

        print(f"[done] saved {panel_out}")


if __name__ == "__main__":
    main()
