# inference.py
import os
import glob
import json
import argparse
import csv
from typing import Dict, List, Tuple
import cv2
import torch
import torch.nn.functional as F
from torch.amp import autocast
from PIL import Image, ImageDraw
import numpy as np
import torchvision.transforms as T
from pycocotools import mask as maskUtils

from mean_teacher import architectures
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn

Box = Tuple[int, int, int, int]  # (x, y, w, h)

EIGEN_FEAT = None

def _eigen_hook(module, input, output):
    global EIGEN_FEAT
    EIGEN_FEAT = output.detach()

def compute_eigencam(feat: torch.Tensor) -> np.ndarray:
    """
    feat: Tensor [C, H, W]
    Return EigenCAM heatmap: (H, W)
    """
    # reshape to (C, H*W)
    C, H, W = feat.shape
    feat = feat.float()
    X = feat.reshape(C, H * W)

    # subtract mean per channel
    X = X - X.mean(dim=1, keepdim=True)

    # SVD for principal direction
    # U: (C x C), S: (C), Vt: (C x HW)
    _, _, Vt = torch.linalg.svd(X, full_matrices=False)

    pc1 = Vt[0]                 # shape (H*W)
    cam = pc1.reshape(H, W)

    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-6)

    return cam.cpu().numpy()


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

    model.load_state_dict(raw_state, strict=False)
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
                   device: str, threshold: float, amp_device: str):
    global EIGEN_FEAT

    x, y, w, h = box
    crop = image.crop((x, y, x + w, y + h)).convert("RGB")
    x_in = tfm(crop).unsqueeze(0).to(device)

    # reset hook storage
    EIGEN_FEAT = None

    # forward pass
    if device == "cuda":
        with autocast(device_type=amp_device):
            outputs = model(x_in)
    else:
        outputs = model(x_in)

    # default CAM result (in case model gives no features)
    cam_map = np.zeros((h, w), dtype=np.float32)

    # ---------------------------
    # Compute CAM if hook captured features
    # ---------------------------
    if EIGEN_FEAT is not None:
        feat = EIGEN_FEAT[0]           # shape [C,H',W']
        cam_small = compute_eigencam(feat)  # shape [H',W']
        # resize CAM to (h, w)
        cam_small_img = Image.fromarray((cam_small * 255).astype(np.uint8))
        cam_map = cam_small_img.resize((w, h))
        cam_map = np.array(cam_map).astype(np.float32)
        cam_map = cam_map / 255.0

    # ---------------------------
    # Handle case with no prediction
    # ---------------------------
    if len(outputs) == 0 or "masks" not in outputs[0]:
        binmask = np.zeros((h, w), dtype=np.uint8)
        return binmask, cam_map

    masks = outputs[0]["masks"]  # [N,1,H,W]
    scores = outputs[0]["scores"]

    if masks.numel() == 0:
        binmask = np.zeros((h, w), dtype=np.uint8)
        return binmask, cam_map

    idx = scores.argmax().item()
    probs = masks[idx, 0]

    # resize mask
    probs = F.interpolate(
        probs.unsqueeze(0).unsqueeze(0),
        size=(h, w),
        mode="bilinear",
        align_corners=False
    )

    binmask = (probs >= threshold).float()[0, 0].cpu().numpy().astype(np.uint8)

    # Return BOTH mask + CAM
    return binmask, cam_map


    # -------------------------
    # EigenCAM for this crop
    # -------------------------
    if EIGEN_FEAT is not None:
        cam_map = compute_eigencam(EIGEN_FEAT[0])  # [C,H,W] → (H,W)
    else:
        cam_map = np.zeros((h, w), dtype=np.float32)
    return binmask, cam_map

# -------------------------
# Visualization helpers
# -------------------------
def draw_bounding_boxes(image: Image.Image, anns: List[dict]) -> Image.Image:
    """Draw bounding boxes from annotations on the RGB image."""
    img_copy = image.copy()
    draw = ImageDraw.Draw(img_copy)

    # --------------------------
    # UPDATED: Dark blue + thick
    # --------------------------
    COLOR = (0, 60, 180)    # strong dark blue
    BORDER = 4              # thicker border

    for ann in anns:
        x, y, w, h = ann["bbox"]
        x2, y2 = x + w, y + h
        draw.rectangle([x, y, x2, y2], outline=COLOR, width=BORDER)
    return img_copy


def make_overlay(rgb_img: Image.Image, mask: np.ndarray, color: str = "red") -> Image.Image:
    rgb = np.array(rgb_img).astype(np.float32) / 255.0
    m = mask.astype(np.float32)[..., None]
    if color == "red":
        rgb[..., 0] = np.maximum(rgb[..., 0], m[..., 0])
    elif color == "green":
        rgb[..., 1] = np.maximum(rgb[..., 1], m[..., 0])
    elif color == "blue":
        rgb[..., 2] = np.maximum(rgb[..., 2], m[..., 0])
    rgb = (np.clip(rgb, 0, 1) * 255.0).astype(np.uint8)
    return Image.fromarray(rgb)

def make_cam_overlay(rgb_img: Image.Image, cam_map: np.ndarray) -> Image.Image:
    """
    rgb_img : PIL RGB image (full image)
    cam_map : float32 heatmap (H, W) in [0,1]
    returns : PIL RGB heatmap overlay
    """

    # Convert RGB to numpy
    rgb_np = np.array(rgb_img).astype(np.float32)

    # Normalize CAM to 0–255 for cv2 colormap
    cam_norm = (cam_map * 255).astype(np.uint8)

    # Apply JET colormap (BGR from OpenCV)
    cam_color = cv2.applyColorMap(cam_norm, cv2.COLORMAP_JET)
    cam_color = cv2.cvtColor(cam_color, cv2.COLOR_BGR2RGB).astype(np.float32)

    # Resize heatmap to match full RGB image size
    cam_color = cv2.resize(cam_color, (rgb_np.shape[1], rgb_np.shape[0]))
    cam_map_resized = cv2.resize(cam_map, (rgb_np.shape[1], rgb_np.shape[0]))
    mask = cam_map_resized[..., None]  # expand to (H, W, 1)

    # Blend only where CAM is strong
    out = (rgb_np * (1 - mask) + cam_color * mask).clip(0, 255).astype(np.uint8)

    return Image.fromarray(out)


def make_four_panel(rgb_with_boxes: Image.Image, 
                    gt_overlay: Image.Image, 
                    pred_overlay: Image.Image,
                    cam_overlay: Image.Image) -> Image.Image:
    """Create a 1x4 (RGB | GT | Pred | CAM) composite panel."""
    W, H = rgb_with_boxes.size
    gt_overlay = gt_overlay.resize((W, H))
    pred_overlay = pred_overlay.resize((W, H))
    cam_overlay = cam_overlay.resize((W, H))

    panel = Image.new("RGB", (W * 4, H))
    panel.paste(rgb_with_boxes, (0, 0))
    panel.paste(gt_overlay, (W, 0))
    panel.paste(pred_overlay, (2 * W, 0))
    panel.paste(cam_overlay, (3 * W, 0))
    return panel

def main():
    ap = argparse.ArgumentParser(description="Segmentation inference (RGB+boxes | GT | Prediction) with CSV stats.")
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

    # -------------------------
    # Register EigenCAM hook
    # (last conv layer)
    # For MaskRCNN, use backbone body layer4
    # -------------------------
    try:
        target_layer = model.backbone.fpn.layer_blocks[1]   # P3 output
        target_layer.register_forward_hook(_eigen_hook)
        print("[EigenCAM] Hook attached to conv5")
    except:
        print("[warn] Could not attach EigenCAM hook")


    file_to_id, id_to_anns, id_to_filename = load_coco_annotations(args.ann)

    image_paths = sorted([p for p in glob.glob(os.path.join(args.images, "*"))
                          if p.lower().endswith((".jpg", ".png", ".jpeg"))])
    print(f"[info] found {len(image_paths)} images")

    # CSV setup
    csv_path = os.path.join(args.out, "inference_stats.csv")
    csv_header = ["image_name", "num_boxes", "num_gt_masks", "num_pred_masks"]
    csv_rows = []

    total_boxes = 0
    total_gt = 0
    total_pred = 0

    for img_path in image_paths:
        img = Image.open(img_path).convert("RGB")

        # Handle EXIF rotation
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

        num_boxes = len(anns)
        total_boxes += num_boxes

        # Draw boxes on RGB
        rgb_with_boxes = draw_bounding_boxes(img, anns)

        # --- GT MASK COUNT (UPDATED) ---
        gt_mask_count = 0
        gt_union = np.zeros((H0, W0), dtype=np.uint8)

        for ann in anns:
            seg = ann.get("segmentation", None)
            if seg is None:
                continue

            if isinstance(seg, list):      # polygon format
                rle = maskUtils.frPyObjects(seg, H0, W0)
                m = maskUtils.decode(rle)
                mask_single = m.max(axis=2).astype(np.uint8)
            else:                           # RLE
                m = maskUtils.decode(seg)
                mask_single = m.astype(np.uint8)

            # Count EACH ground-truth mask
            if mask_single.max() == 1:
                gt_mask_count += 1

            gt_union = np.maximum(gt_union, mask_single)

        total_gt += gt_mask_count
        gt_overlay = make_overlay(img, gt_union, color="red")

        # --- Prediction ---
        pred_union = np.zeros((H0, W0), dtype=np.uint8)
        pred_mask_count = 0
        CAM_full = np.zeros((H0, W0), dtype=np.float32)
        for ann in anns:
            x, y, w, h = map(int, ann["bbox"])
            x = max(0, min(x, W0 - 1))
            y = max(0, min(y, H0 - 1))
            w = max(0, min(w, W0 - x))
            h = max(0, min(h, H0 - y))
            if w == 0 or h == 0:
                continue

            mask_hw, cam_hw = infer_box_mask(model, img, (x, y, w, h), tfm, device, args.threshold, amp_device)
            
            if mask_hw.max() == 0:
                continue
            pred_mask_count += 1
            pred_union[y:y+h, x:x+w] = np.maximum(pred_union[y:y+h, x:x+w], mask_hw)

            CAM_full[y:y+h, x:x+w] = np.maximum(CAM_full[y:y+h, x:x+w], cam_hw)

        total_pred += pred_mask_count
        pred_overlay = make_overlay(img, pred_union, color="green")
        cam_overlay = make_cam_overlay(img, CAM_full)

        # --- 3-panel output ---
        panel = make_four_panel(rgb_with_boxes, gt_overlay, pred_overlay, cam_overlay)
        base = os.path.splitext(bn)[0]
        panel_out = os.path.join(args.out, f"{base}_rgb_gt_pred.png")
        panel.save(panel_out)

        # --- CSV Row ---
        csv_rows.append([bn, num_boxes, gt_mask_count, pred_mask_count])

    # Write CSV file
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(csv_header)
        writer.writerows(csv_rows)
        writer.writerow([])
        writer.writerow(["TOTAL", total_boxes, total_gt, total_pred])
        if len(csv_rows) > 0:
            writer.writerow([
                "AVERAGE",
                round(total_boxes / len(csv_rows), 2),
                round(total_gt / len(csv_rows), 2),
                round(total_pred / len(csv_rows), 2)
            ])

    print(f"[done] Inference complete. CSV saved to {csv_path}")


if __name__ == "__main__":
    main()