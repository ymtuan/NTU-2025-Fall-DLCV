import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def mean_iou_score(preds, labels, num_classes=6):
    """
    Compute mean IoU over num_classes (ignoring 'unknown' if desired).
    preds, labels: np.ndarray of shape (N, H, W)
    """
    mean_iou = 0.0
    for i in range(num_classes):
        tp_fp = np.sum(preds == i)
        tp_fn = np.sum(labels == i)
        tp = np.sum((preds == i) & (labels == i))
        if tp_fp + tp_fn - tp == 0:
            iou = 0.0
        else:
            iou = tp / (tp_fp + tp_fn - tp)
        mean_iou += iou / num_classes
        print(f"class #{i} : {iou:1.5f}")
    print(f"\nmean_iou: {mean_iou:.5f}\n")
    return mean_iou


def save_model(model, path="best_model.pth"):
    torch.save(model.state_dict(), path)

def load_model(model, path="best_model.pth", device="cpu"):
    model.load_state_dict(torch.load(path, map_location=device))
    return model

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, ignore_index=None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.ignore_index = ignore_index

    def forward(self, inputs, targets):
        """
        inputs: (B, C, H, W) logits
        targets: (B, H, W) long
        """
        # Flatten
        B, C, H, W = inputs.shape
        inputs = inputs.permute(0, 2, 3, 1).reshape(-1, C)  # (B*H*W, C)
        targets = targets.view(-1)  # (B*H*W)

        if self.ignore_index is not None:
            mask = targets != self.ignore_index
            inputs = inputs[mask]
            targets = targets[mask]

        logpt = F.log_softmax(inputs, dim=1)
        pt = torch.exp(logpt)
        logpt = logpt.gather(1, targets.unsqueeze(1)).squeeze(1)
        pt = pt.gather(1, targets.unsqueeze(1)).squeeze(1)

        if self.alpha is not None:
            at = self.alpha.gather(0, targets)
            logpt = logpt * at

        loss = -((1 - pt) ** self.gamma) * logpt
        return loss.mean()


class IoULoss(nn.Module):
    def __init__(self, smooth=1.0, ignore_index=None):
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, inputs, targets):
        """
        inputs: (B, C, H, W) logits
        targets: (B, H, W) long
        """
        inputs = F.softmax(inputs, dim=1)  # probabilities
        B, C, H, W = inputs.shape

        # One-hot encode targets
        targets_one_hot = torch.zeros_like(inputs)
        targets_one_hot.scatter_(1, targets.unsqueeze(1), 1)

        if self.ignore_index is not None:
            mask = targets != self.ignore_index
            mask = mask.unsqueeze(1)
            inputs = inputs * mask
            targets_one_hot = targets_one_hot * mask

        # Compute IoU per class
        intersection = (inputs * targets_one_hot).sum(dim=(0, 2, 3))
        union = (inputs + targets_one_hot - inputs * targets_one_hot).sum(dim=(0, 2, 3))
        iou = (intersection + self.smooth) / (union + self.smooth)

        return 1 - iou.mean()