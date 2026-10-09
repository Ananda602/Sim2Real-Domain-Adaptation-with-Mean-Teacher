import numpy as np
import torch
import datetime

def _interp_ap(recall, precision):
    """COCO-style area under precision envelope."""
    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))
    # make precision non-increasing (envelope)
    for i in range(mpre.size - 1, 0, -1):
        mpre[i-1] = max(mpre[i-1], mpre[i])
    # integrate where recall changes
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx+1] - mrec[idx]) * mpre[idx+1]))

def average_precision_from_hits(scores, hits, num_gt):
    """
    scores: list/np.array of confidences (higher = better), shape [N]
    hits:   list/np.array of {0,1}, TP(1) if IoU>=thr else FP(0), shape [N]
    num_gt: number of positives (=len(hits) for one-gt-per-sample eval)
    """
    if num_gt == 0 or len(scores) == 0:
        return 0.0
    scores = np.asarray(scores)
    hits   = np.asarray(hits).astype(np.int32)

    order = np.argsort(-scores)
    hits  = hits[order]

    tp = np.cumsum(hits)
    fp = np.cumsum(1 - hits)
    recall    = tp / max(1, num_gt)
    precision = tp / np.maximum(tp + fp, 1e-12)
    return _interp_ap(recall, precision)

def compute_ap_set(ious, scores, iou_thresholds=(0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95)):
    """
    ious:   list of per-sample IoU (float)
    scores: list of per-sample confidence (float)
    returns: dict with 'AP@50', 'mAP@[.5:.95]' and optionally per-threshold APs
    """
    ious   = np.asarray(ious, dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float32)
    num = len(ious)
    out = {}
    aps = []
    for t in iou_thresholds:
        hits = (ious >= t).astype(np.int32)
        ap = average_precision_from_hits(scores, hits, num_gt=num)
        aps.append(ap)
        if abs(t - 0.50) < 1e-6:
            out['mAP@50'] = ap
        if abs(t - 0.75) < 1e-6:
            out['mAP@75'] = ap   # 👈 new line
    out['mAP@[.5:.95]'] = float(np.mean(aps)) if aps else 0.0
    return out

def segmentation_metrics(logits, targets):
    """Binary IoU & Dice for [B,1,H,W] logits vs targets."""
    device = logits.device  # ensure consistency
    targets = targets.to(device)

    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).float()
    preds = preds.view(preds.size(0), -1)
    targets = (targets > 0.5).float().view(targets.size(0), -1)
    inter = (preds * targets).sum(dim=1)
    union = preds.sum(dim=1) + targets.sum(dim=1) - inter
    iou = ((inter + 1e-6) / (union + 1e-6)).mean().item()
    dice = ((2 * inter + 1e-6) / (preds.sum(dim=1) + targets.sum(dim=1) + 1e-6)).mean().item()
    return iou, dice

def format_eta(epoch, i, train_loader, total_epochs, global_batch_times):
    import datetime

    if not global_batch_times:  # nothing recorded yet
        return "estimating..."

    avg_batch_time = sum(global_batch_times) / len(global_batch_times)
    batches_done = epoch * len(train_loader) + (i + 1)
    total_batches = total_epochs * len(train_loader)
    batches_left = total_batches - batches_done
    eta_seconds = avg_batch_time * batches_left
    return str(datetime.timedelta(seconds=int(eta_seconds)))


def get_model_device(model):
    """Get root device (works for DataParallel too)."""
    if isinstance(model, torch.nn.DataParallel):
        return next(model.module.parameters()).device
    return next(model.parameters()).device