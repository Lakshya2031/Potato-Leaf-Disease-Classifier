"""
Advanced Loss Functions for improved model training.
Includes Label Smoothing, Focal Loss, and more.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class LabelSmoothingCrossEntropy(nn.Module):
    """
    Cross Entropy with Label Smoothing.
    Prevents the model from becoming overconfident.
    
    Reference: Szegedy et al., "Rethinking the Inception Architecture for Computer Vision"
    """
    def __init__(self, smoothing: float = 0.1, reduction: str = 'mean'):
        super().__init__()
        self.smoothing = smoothing
        self.reduction = reduction
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        n_classes = pred.size(-1)
        log_preds = F.log_softmax(pred, dim=-1)
        
        # One-hot encode target with smoothing
        with torch.no_grad():
            smooth_target = torch.zeros_like(pred)
            smooth_target.fill_(self.smoothing / (n_classes - 1))
            smooth_target.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)
        
        loss = -smooth_target * log_preds
        loss = loss.sum(dim=-1)
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.
    Focuses training on hard examples by down-weighting easy examples.
    
    Reference: Lin et al., "Focal Loss for Dense Object Detection"
    
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """
    def __init__(self, alpha: Optional[torch.Tensor] = None, gamma: float = 2.0, 
                 reduction: str = 'mean', label_smoothing: float = 0.0):
        super().__init__()
        self.alpha = alpha  # Class weights [num_classes]
        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        n_classes = pred.size(-1)
        
        # Compute softmax probabilities
        log_p = F.log_softmax(pred, dim=-1)
        p = torch.exp(log_p)
        
        # Get probability of true class
        ce_loss = F.nll_loss(log_p, target, reduction='none')
        p_t = p.gather(1, target.unsqueeze(1)).squeeze(1)
        
        # Apply focal modulation
        focal_weight = (1 - p_t) ** self.gamma
        focal_loss = focal_weight * ce_loss
        
        # Apply class weights if provided
        if self.alpha is not None:
            alpha = self.alpha.to(pred.device)
            alpha_t = alpha.gather(0, target)
            focal_loss = alpha_t * focal_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class SoftTargetCrossEntropy(nn.Module):
    """
    Cross Entropy for soft targets (e.g., from MixUp/CutMix).
    """
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        log_preds = F.log_softmax(pred, dim=-1)
        loss = -(target * log_preds).sum(dim=-1)
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class AsymmetricLoss(nn.Module):
    """
    Asymmetric Loss for multi-label classification.
    Better handles positive-negative imbalance.
    
    Reference: Ben-Baruch et al., "Asymmetric Loss For Multi-Label Classification"
    """
    def __init__(self, gamma_neg: float = 4.0, gamma_pos: float = 1.0, 
                 clip: float = 0.05, reduction: str = 'mean'):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.reduction = reduction
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Get probabilities
        p = torch.sigmoid(pred)
        
        # One-hot encode if needed
        if target.dim() == 1:
            target_onehot = F.one_hot(target, num_classes=pred.size(-1)).float()
        else:
            target_onehot = target.float()
        
        # Probability clipping for asymmetric focusing
        p_t = p * target_onehot + (1 - p) * (1 - target_onehot)
        
        # Asymmetric focusing
        p_clipped = (p + self.clip).clamp(max=1)
        
        # Separate positive and negative terms
        pos_loss = target_onehot * torch.log(p.clamp(min=1e-8))
        neg_loss = (1 - target_onehot) * torch.log(p_clipped.clamp(min=1e-8))
        
        # Apply asymmetric gamma
        pos_weight = (1 - p) ** self.gamma_pos
        neg_weight = p ** self.gamma_neg
        
        loss = -pos_weight * pos_loss - neg_weight * neg_loss
        loss = loss.sum(dim=-1)
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class PolyLoss(nn.Module):
    """
    PolyLoss: A Polynomial Expansion Perspective of Classification Loss Functions
    
    Reference: Leng et al., "PolyLoss"
    """
    def __init__(self, epsilon: float = 1.0, reduction: str = 'mean'):
        super().__init__()
        self.epsilon = epsilon
        self.reduction = reduction
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        n_classes = pred.size(-1)
        
        # Compute softmax and CE
        ce = F.cross_entropy(pred, target, reduction='none')
        p = F.softmax(pred, dim=-1)
        
        # Get p_t (probability of true class)
        p_t = p.gather(1, target.unsqueeze(1)).squeeze(1)
        
        # PolyLoss = CE + epsilon * (1 - p_t)
        poly_loss = ce + self.epsilon * (1 - p_t)
        
        if self.reduction == 'mean':
            return poly_loss.mean()
        elif self.reduction == 'sum':
            return poly_loss.sum()
        return poly_loss


class BiTemperedLogisticLoss(nn.Module):
    """
    Bi-Tempered Logistic Loss for robustness to noisy labels.
    
    Reference: Amid et al., "Robust Bi-Tempered Logistic Loss Based on Bregman Divergences"
    """
    def __init__(self, t1: float = 0.8, t2: float = 1.2, label_smoothing: float = 0.0, 
                 num_iters: int = 5, reduction: str = 'mean'):
        super().__init__()
        self.t1 = t1  # Temperature for computing probabilities
        self.t2 = t2  # Temperature for loss computation
        self.label_smoothing = label_smoothing
        self.num_iters = num_iters
        self.reduction = reduction
    
    def _log_t(self, u: torch.Tensor, t: float) -> torch.Tensor:
        """Tempered logarithm."""
        if t == 1.0:
            return torch.log(u)
        return (u ** (1 - t) - 1) / (1 - t)
    
    def _exp_t(self, u: torch.Tensor, t: float) -> torch.Tensor:
        """Tempered exponential."""
        if t == 1.0:
            return torch.exp(u)
        return torch.relu(1 + (1 - t) * u) ** (1 / (1 - t))
    
    def _tempered_softmax(self, logits: torch.Tensor, t: float) -> torch.Tensor:
        """Compute tempered softmax."""
        if t == 1.0:
            return F.softmax(logits, dim=-1)
        
        # Fixed point iteration
        normalization = torch.ones_like(logits[:, :1])
        for _ in range(self.num_iters):
            shifted = logits - normalization
            exp_shifted = self._exp_t(shifted, t)
            normalization = self._log_t(exp_shifted.sum(dim=-1, keepdim=True), t)
        
        shifted = logits - normalization
        return self._exp_t(shifted, t)
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        n_classes = pred.size(-1)
        
        # One-hot encode target
        with torch.no_grad():
            if self.label_smoothing > 0:
                labels = torch.zeros_like(pred)
                labels.fill_(self.label_smoothing / (n_classes - 1))
                labels.scatter_(1, target.unsqueeze(1), 1.0 - self.label_smoothing)
            else:
                labels = F.one_hot(target, n_classes).float()
        
        # Compute tempered probabilities
        probs = self._tempered_softmax(pred, self.t2)
        
        # Compute loss
        loss_values = labels * (self._log_t(labels + 1e-10, self.t1) - self._log_t(probs, self.t1))
        loss = loss_values.sum(dim=-1)
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


def get_class_weights(labels: torch.Tensor, num_classes: int, 
                      method: str = 'inverse') -> torch.Tensor:
    """
    Compute class weights for handling imbalanced datasets.
    
    Args:
        labels: Tensor of all labels in the dataset
        num_classes: Number of classes
        method: 'inverse' or 'effective' (effective number of samples)
    
    Returns:
        Class weights tensor
    """
    counts = torch.bincount(labels, minlength=num_classes).float()
    
    if method == 'inverse':
        # Inverse frequency
        weights = 1.0 / (counts + 1e-6)
        weights = weights / weights.sum() * num_classes
    elif method == 'effective':
        # Effective number of samples (Cui et al., 2019)
        beta = 0.9999
        effective_num = 1.0 - torch.pow(beta, counts)
        weights = (1.0 - beta) / (effective_num + 1e-6)
        weights = weights / weights.sum() * num_classes
    else:
        weights = torch.ones(num_classes)
    
    return weights


class CombinedLoss(nn.Module):
    """
    Combine multiple loss functions with learnable or fixed weights.
    """
    def __init__(self, losses: list, weights: Optional[list] = None, learnable: bool = False):
        super().__init__()
        self.losses = nn.ModuleList(losses)
        
        if weights is None:
            weights = [1.0] * len(losses)
        
        if learnable:
            self.weights = nn.Parameter(torch.tensor(weights))
        else:
            self.register_buffer('weights', torch.tensor(weights))
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        total_loss = 0
        for loss_fn, weight in zip(self.losses, self.weights):
            total_loss += weight * loss_fn(pred, target)
        return total_loss
