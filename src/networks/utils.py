class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

import torch
import torch.nn as nn
import torchvision.ops as ops  # Importing focal loss from torchvision

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        """
        Wrapper around torchvision's sigmoid_focal_loss.

        Parameters:
        - alpha: Weight for positive class (default: 0.25).
        - gamma: Focusing parameter (default: 2.0).
        - reduction: 'mean' (default) or 'sum' or 'none'.
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, preds, targets):
        """
        Compute focal loss.

        Parameters:
        - preds: Model logits (not probabilities), shape (N, 1, H, W).
        - targets: Ground truth binary labels, shape (N, 1, H, W).

        Returns:
        - Loss value.
        """
        preds = preds  # Ensure shape is (N, H, W)
        targets = targets.float()

        loss = ops.sigmoid_focal_loss(preds, targets, alpha=self.alpha, gamma=self.gamma, reduction=self.reduction)
        return loss
    
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, preds, targets):
        preds = torch.sigmoid(preds)  # Convert logits to probabilities
        targets = targets.float()

        intersection = (preds * targets).sum()
        dice = (2.0 * intersection + self.smooth) / (preds.sum() + targets.sum() + self.smooth)

        return 1 - dice  # Dice loss

class ComboLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=1.5, dice_weight=0.5):
        super(ComboLoss, self).__init__()
        self.focal_loss = FocalLoss(alpha=alpha, gamma=gamma)
        self.dice_loss = DiceLoss()
        self.dice_weight = dice_weight

    def forward(self, preds, targets):
        return self.focal_loss(preds, targets) + self.dice_weight * self.dice_loss(preds, targets)

def init_weights_xavier(m):
    if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Linear, nn.ConvTranspose2d)):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)