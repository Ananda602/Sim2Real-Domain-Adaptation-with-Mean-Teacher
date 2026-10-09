# architectures.py
import torch
import torch.nn as nn
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.backbone_utils import BackboneWithFPN
from torchvision.ops import FeaturePyramidNetwork
from torchvision.models.detection.backbone_utils import LastLevelMaxPool
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.ops import sigmoid_focal_loss
from torchvision.ops import generalized_box_iou_loss
import timm
import torch.nn.functional as F
from mmcv import Config
from mmdet.apis import init_detector, inference_detector
from torch import amp

# new in recent torchvision:
try:
    from torchvision.models import ResNet50_Weights
except Exception:
    ResNet50_Weights = None  # fallback for older versions

def maskrcnn(num_classes=1, in_channels=3, pretrained=True, **kwargs):
    """
    Mask R-CNN with Focal Loss for classification and GIoU Loss for box regression.
    Compatible with Torchvision >= 0.15.
    """
    # Total detection classes (background + foreground)
    num_det_classes = max(2, num_classes + 1)

    # --- Backbone ---
    backbone = resnet_fpn_backbone(
        backbone_name='resnet50',
        weights=ResNet50_Weights.IMAGENET1K_V1 if pretrained else None,
        trainable_layers=3,
        returned_layers=[1, 2, 3, 4]  # C2–C5 → P2–P5
    )

    # --- Mask R-CNN model ---
    model = MaskRCNN(backbone, num_classes=num_det_classes)

    # --- GIoU helper (optional clarity) ---
    def giou_loss(boxes_pred, boxes_gt):
        """Wrapper around torchvision.ops.generalized_box_iou_loss."""
        return generalized_box_iou_loss(boxes_pred, boxes_gt, reduction="mean")

    # --- Focal + GIoU combined loss ---
    def fastrcnn_loss_focal(class_logits, box_regression, labels, regression_targets):
        """
        Replace CE + SmoothL1 with Focal + GIoU.
        """
        num_classes = class_logits.shape[1]

        # Focal classification loss
        loss_cls = sigmoid_focal_loss(
            class_logits,
            F.one_hot(labels, num_classes=num_classes).float(),
            alpha=0.25,
            gamma=2.0,
            reduction="mean",
        )

        # GIoU regression loss
        loss_box_reg = giou_loss(box_regression, regression_targets)
        return loss_cls, loss_box_reg

    # Attach custom loss functions to ROI heads
    model.roi_heads.fastrcnn_loss = fastrcnn_loss_focal
    model.roi_heads.box_loss = giou_loss

    print("[Focal+GIoU] Custom ROI-head losses enabled (γ=2, α=0.25)")

    # --- Anchor configuration (smaller scales for forest dataset) ---
    anchor_generator = AnchorGenerator(
        sizes=((16,), (32,), (64,), (128,), (256,)),  # match 5 FPN levels
        aspect_ratios=((0.5, 1.0, 2.0),) * 5,
    )
    model.rpn.anchor_generator = anchor_generator

    print("\n[Anchor sanity] Using small-scale anchors and full P2–P6 feature pyramid.")
    print("Active FPN strides:", [4, 8, 16, 32, 64], "\n")

    return model

try:
    from mmdet.apis import init_detector, inference_detector
    _HAS_MMDET = True
except Exception:
    _HAS_MMDET = False

class MMDetTeacher(nn.Module):
    def __init__(self, cfg_path: str, ckpt_path: str, device: str = 'cuda', binary=True, conf_thresh=0.0):
        super().__init__()
        assert isinstance(cfg_path, str), "cfg_path must be a file path for MMDet 2.x"

        cfg = Config.fromfile(cfg_path)

        # --- force FP32 inference ---
        if hasattr(cfg, 'fp16'):
            cfg.fp16 = None
        if hasattr(cfg, 'optimizer_config') and isinstance(cfg.optimizer_config, dict):
            # MMDet 2.x uses Fp16OptimizerHook for mixed precision; remove it for inference
            if str(cfg.optimizer_config.get('type', '')).lower().startswith('fp16'):
                cfg.optimizer_config = dict(grad_clip=None)

        # (optional but safe) prevent reloading any backbone pretrain from cfg during init
        if hasattr(cfg, 'model') and isinstance(cfg.model, dict):
            cfg.model.setdefault('pretrained', None)
            if 'backbone' in cfg.model and isinstance(cfg.model['backbone'], dict):
                cfg.model['backbone'].setdefault('init_cfg', None)

        self.det = init_detector(cfg, ckpt_path, device=device)
        self.det.float()          # ensure the whole model is in FP32
        self.det.eval()

        # ---- Compatibility smoke test ----
        try:
            dummy = torch.zeros(1, 3, 128, 128, device=device)  # 1x3xHxW tensor
            dummy_np = (dummy[0].permute(1,2,0).cpu().numpy() * 255).astype("uint8")[:, :, ::-1]
            result = inference_detector(self.det, dummy_np)

            if isinstance(result, tuple):
                bbox_results, segm_results = result
                print(f"[Teacher Check] Loaded OK: {len(bbox_results)} classes, "
                    f"bbox for class[0]: {None if not bbox_results[0] else bbox_results[0].shape}")
            else:
                print(f"[Teacher Check] Loaded OK: Only bbox results, shape={len(result)}")
        except Exception as e:
            print(f"[Teacher Check] FAILED for ckpt={ckpt_path}: {e}")


        for p in self.parameters():
            p.requires_grad_(False)

        self.binary = binary
        self.conf_thresh = float(conf_thresh)

    @torch.no_grad()
    def forward(self, images):
        """
        images: list[Tensor] (B items), each [C,H,W], float in [0,1], RGB
        returns: list[dict] with "masks" [N,1,H,W] and "scores" [N]
        """
        assert isinstance(images, (list, tuple)) and all(torch.is_tensor(x) for x in images), \
            f"Teacher expects list[Tensor]; got {type(images)} with elems " \
            f"{[type(e) for e in (images if isinstance(images,(list,tuple)) else [])]}"

        out = []
        for x in images:
            # --- sanitize channels on tensor side ---
            x = x.detach().clamp(0, 1)  # [C,H,W], float

            C, H, W = x.shape
            if C == 1:
                # gray -> RGB
                x = x.repeat(3, 1, 1)
            elif C == 2:
                # rare LA (luma+alpha) -> use luma as RGB
                luma = x[0:1, ...]
                x = torch.cat([luma, luma, luma], dim=0)
            elif C >= 3:
                x = x[:3, ...]  # keep first 3 channels

            # to HWC uint8 (RGB)
            x_np = (x.permute(1, 2, 0).cpu().numpy() * 255.0).astype("uint8")  # [H,W,3?]

            # --- sanitize channels on numpy side (extra safety) ---
            if x_np.ndim == 2:
                # [H,W] -> [H,W,3]
                x_np = np.repeat(x_np[..., None], 3, axis=2)
            if x_np.shape[2] == 1:
                x_np = np.repeat(x_np, 3, axis=2)
            elif x_np.shape[2] == 2:
                # take first channel as luma and replicate
                l = x_np[..., 0]
                x_np = np.stack([l, l, l], axis=2)
            elif x_np.shape[2] > 3:
                x_np = x_np[:, :, :3]

            # RGB -> BGR for MMDet v2
            x_np = x_np[:, :, ::-1]

            with amp.autocast('cuda', enabled=False):
                result = inference_detector(self.det, x_np)

            # unpack
            if isinstance(result, tuple):
                bbox_results, segm_results = result
            else:
                bbox_results, segm_results = result, None

            masks_all, scores_all = [], []
            num_classes = len(bbox_results)
            for c in range(num_classes):
                b = bbox_results[c]
                if b is None or len(b) == 0:
                    continue
                scores = b[:, -1]
                if self.conf_thresh > 0:
                    keep = scores >= self.conf_thresh
                    b = b[keep]
                    if segm_results is not None and segm_results[c] is not None:
                        segms_c = [seg for seg, k in zip(segm_results[c], keep.tolist()) if k]
                    else:
                        segms_c = None
                    scores = scores[keep]
                else:
                    segms_c = segm_results[c] if segm_results is not None else None

                if segms_c is None:
                    continue

                for m, s in zip(segms_c, scores):
                    if hasattr(m, 'to_ndarray'):
                        m = m.to_ndarray()
                    mask_t = torch.from_numpy(m.astype('float32'))  # [H,W]
                    masks_all.append(mask_t[None, None, ...])  # [1,1,H,W]
                    scores_all.append(float(s))

            if len(masks_all) == 0:
                Ht, Wt = images[0].shape[-2:]
                out.append({
                    "masks": torch.zeros((0, 1, Ht, Wt), dtype=torch.float32, device=images[0].device),
                    "scores": torch.zeros((0,), dtype=torch.float32, device=images[0].device),
                })
                continue

            # move to same device as inputs
            masks = masks.to(images[0].device, non_blocking=True)
            scores = scores.to(images[0].device, non_blocking=True)

            out.append({"masks": masks, "scores": scores})

        return out



# ---- factory function to register in your architectures registry ----
def mmdet_maskrcnn_swin_t(num_classes=1, in_channels=3, cfg_path=None, ckpt_path=None, device="cuda", **kwargs):
    """
    Factory used by create_models() to build a teacher with MMDetection.
    num_classes & in_channels are ignored (model is defined by cfg).
    """
    assert cfg_path is not None and ckpt_path is not None, \
        "Provide cfg_path and ckpt_path for mmdet teacher"
    return MMDetTeacher(cfg_path=cfg_path, ckpt_path=ckpt_path, device=device)



__all__ = ['maskrcnn', 'mmdet_maskrcnn_swin_t']

