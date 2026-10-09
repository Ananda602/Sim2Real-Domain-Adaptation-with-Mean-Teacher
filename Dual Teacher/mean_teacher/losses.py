import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0, eps: float = 1e-7):
        super().__init__()
        self.smooth = smooth
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits, targets: [B,1,H,W]; targets in {0,1}
        """
        probs = torch.sigmoid(logits)
        probs = probs.reshape(probs.size(0), -1)
        targets = targets.reshape(targets.size(0), -1)
        inter = (probs * targets).sum(dim=1)
        denom = probs.sum(dim=1) + targets.sum(dim=1) + self.smooth
        dice = (2 * inter + self.smooth) / (denom + self.eps)
        return 1.0 - dice.mean()

class SegmentationLoss(nn.Module):
    def __init__(self, alpha: float = 0.5, pos_weight: torch.Tensor | None = None):
        """
        alpha: weight for BCE vs Dice (0..1)
        pos_weight: optional tensor for class imbalance (e.g., torch.tensor([2.0], device=device))
        """
        super().__init__()
        self.alpha = alpha
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction='mean')
        self.dice = DiceLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.float()
        return self.alpha * self.bce(logits, targets) + (1 - self.alpha) * self.dice(logits, targets)

def consistency_mse_with_confidence(student_logits: torch.Tensor,
                                    teacher_logits: torch.Tensor,
                                    threshold: float = 0.9) -> torch.Tensor:
    """
    Pixel-wise MSE between student/teacher probabilities, but only where
    the teacher is confident (> threshold). Returns 0 if no valid pixels.
    """
    with torch.no_grad():
        p_t = torch.sigmoid(teacher_logits)
        mask = (p_t > threshold).float()  # [B,1,H,W], 1 = keep
        valid = mask.sum()

    if valid.item() < 1:
        return torch.zeros((), device=student_logits.device)

    p_s = torch.sigmoid(student_logits)
    return ((p_s - p_t) ** 2 * mask).sum() / valid


