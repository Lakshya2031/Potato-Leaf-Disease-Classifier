"""
Training utilities including EMA, schedulers, and optimization helpers.
"""
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import _LRScheduler
from typing import Optional, List, Dict, Callable
import copy
import math


# ============================================================================
# Exponential Moving Average (EMA)
# ============================================================================
class ModelEMA:
    """
    Exponential Moving Average of model weights.
    Maintains a shadow copy of model weights that are updated with EMA.
    Typically improves generalization and reduces noise in final model.
    
    Usage:
        ema = ModelEMA(model, decay=0.9999)
        for batch in dataloader:
            loss.backward()
            optimizer.step()
            ema.update(model)
        
        # Use EMA model for evaluation
        ema.apply_shadow(model)
    """
    def __init__(self, model: nn.Module, decay: float = 0.9999, updates: int = 0):
        self.decay = decay
        self.updates = updates
        # Create shadow parameters
        self.shadow = {}
        self.backup = {}
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self, model: nn.Module):
        """Update shadow weights with EMA."""
        self.updates += 1
        # Apply warmup decay
        d = self.decay * (1 - math.exp(-self.updates / 2000))
        
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad and name in self.shadow:
                    self.shadow[name].mul_(d).add_(param.data, alpha=1 - d)
    
    def apply_shadow(self, model: nn.Module):
        """Apply shadow weights to model (backup original first)."""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])
    
    def restore(self, model: nn.Module):
        """Restore original weights from backup."""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}


# ============================================================================
# Learning Rate Schedulers
# ============================================================================
class WarmupCosineScheduler(_LRScheduler):
    """
    Learning rate scheduler with linear warmup followed by cosine annealing.
    """
    def __init__(self, optimizer, warmup_epochs: int, total_epochs: int, 
                 min_lr: float = 1e-6, last_epoch: int = -1):
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # Linear warmup
            warmup_factor = (self.last_epoch + 1) / self.warmup_epochs
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        else:
            # Cosine annealing
            progress = (self.last_epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            cosine_factor = 0.5 * (1 + math.cos(math.pi * progress))
            return [self.min_lr + (base_lr - self.min_lr) * cosine_factor for base_lr in self.base_lrs]


class OneCycleLR(_LRScheduler):
    """
    1cycle learning rate policy (Smith & Topin, 2019).
    Cycles learning rate between min and max with warmup and cooldown phases.
    """
    def __init__(self, optimizer, max_lr: float, total_steps: int,
                 pct_start: float = 0.3, div_factor: float = 25.0,
                 final_div_factor: float = 1e4, last_epoch: int = -1):
        self.max_lr = max_lr
        self.total_steps = total_steps
        self.pct_start = pct_start
        self.div_factor = div_factor
        self.final_div_factor = final_div_factor
        self.initial_lr = max_lr / div_factor
        self.final_lr = max_lr / final_div_factor
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        step = self.last_epoch
        
        if step < self.total_steps * self.pct_start:
            # Warmup phase
            pct = step / (self.total_steps * self.pct_start)
            lr = self.initial_lr + (self.max_lr - self.initial_lr) * pct
        else:
            # Annealing phase
            pct = (step - self.total_steps * self.pct_start) / (self.total_steps * (1 - self.pct_start))
            lr = self.max_lr - (self.max_lr - self.final_lr) * (1 - math.cos(math.pi * pct)) / 2
        
        return [lr for _ in self.base_lrs]


class PolynomialLRDecay(_LRScheduler):
    """
    Polynomial learning rate decay.
    """
    def __init__(self, optimizer, total_epochs: int, power: float = 0.9,
                 min_lr: float = 1e-6, last_epoch: int = -1):
        self.total_epochs = total_epochs
        self.power = power
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        factor = (1 - self.last_epoch / self.total_epochs) ** self.power
        return [max(self.min_lr, base_lr * factor) for base_lr in self.base_lrs]


# ============================================================================
# Gradient Clipping Utilities
# ============================================================================
class GradientClipper:
    """
    Gradient clipping utilities with adaptive clipping support.
    """
    def __init__(self, max_norm: float = 1.0, adaptive: bool = False):
        self.max_norm = max_norm
        self.adaptive = adaptive
        self.grad_history = []
    
    def clip(self, model: nn.Module) -> float:
        """Clip gradients and return total norm."""
        total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), self.max_norm)
        
        if self.adaptive:
            self.grad_history.append(total_norm.item())
            if len(self.grad_history) > 100:
                self.grad_history.pop(0)
                self.max_norm = sum(self.grad_history) / len(self.grad_history) * 2
        
        return total_norm.item()


# ============================================================================
# SAM Optimizer (Sharpness-Aware Minimization)
# ============================================================================
class SAM:
    """
    Sharpness-Aware Minimization (Foret et al., 2021)
    Seeks parameters in neighborhoods with uniformly low loss.
    
    Usage:
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        sam = SAM(optimizer, rho=0.05)
        
        for batch in dataloader:
            # First forward-backward pass
            loss = criterion(model(inputs), labels)
            loss.backward()
            sam.first_step(zero_grad=True)
            
            # Second forward-backward pass
            criterion(model(inputs), labels).backward()
            sam.second_step(zero_grad=True)
    """
    def __init__(self, optimizer: torch.optim.Optimizer, rho: float = 0.05):
        self.optimizer = optimizer
        self.rho = rho
        self.state = {}
    
    def _grad_norm(self):
        shared_device = None
        norm = torch.tensor(0.0)
        for group in self.optimizer.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                if shared_device is None:
                    shared_device = p.device
                    norm = norm.to(shared_device)
                norm += p.grad.data.norm(2) ** 2
        return norm.sqrt()
    
    def first_step(self, zero_grad: bool = False):
        """Perturb weights toward gradient direction."""
        grad_norm = self._grad_norm()
        
        for group in self.optimizer.param_groups:
            scale = self.rho / (grad_norm + 1e-12)
            for p in group['params']:
                if p.grad is None:
                    continue
                e_w = p.grad.data * scale
                p.add_(e_w)  # Climb to adversarial point
                self.state[p] = e_w
        
        if zero_grad:
            self.optimizer.zero_grad()
    
    def second_step(self, zero_grad: bool = False):
        """Restore weights and perform optimization step."""
        for group in self.optimizer.param_groups:
            for p in group['params']:
                if p in self.state:
                    p.sub_(self.state[p])  # Go back to original
        
        self.optimizer.step()
        
        if zero_grad:
            self.optimizer.zero_grad()
        
        self.state = {}


# ============================================================================
# LARS Optimizer (Layer-wise Adaptive Rate Scaling)
# ============================================================================
class LARS(torch.optim.Optimizer):
    """
    LARS optimizer for large-batch training (You et al., 2017)
    """
    def __init__(self, params, lr: float = 0.1, momentum: float = 0.9,
                 weight_decay: float = 1e-4, trust_coefficient: float = 0.001):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay,
                       trust_coefficient=trust_coefficient)
        super().__init__(params, defaults)
    
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                d_p = p.grad
                
                # Weight decay
                if group['weight_decay'] != 0:
                    d_p = d_p.add(p, alpha=group['weight_decay'])
                
                # LARS scaling
                p_norm = p.norm()
                g_norm = d_p.norm()
                
                if p_norm > 0 and g_norm > 0:
                    local_lr = group['trust_coefficient'] * p_norm / g_norm
                else:
                    local_lr = 1.0
                
                d_p = d_p.mul(local_lr)
                
                # Momentum
                if group['momentum'] != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(group['momentum']).add_(d_p)
                    d_p = buf
                
                p.add_(d_p, alpha=-group['lr'])
        
        return loss


# ============================================================================
# Mixed Precision Training Helper
# ============================================================================
class MixedPrecisionTrainer:
    """
    Helper class for automatic mixed precision training.
    Uses float16 for forward/backward, float32 for optimizer.
    """
    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer,
                 loss_scale: float = 'dynamic'):
        self.model = model
        self.optimizer = optimizer
        self.scaler = torch.cuda.amp.GradScaler(
            init_scale=2**16 if loss_scale == 'dynamic' else loss_scale,
            enabled=True
        )
    
    def train_step(self, inputs: torch.Tensor, targets: torch.Tensor,
                   criterion: nn.Module) -> float:
        """Perform one training step with mixed precision."""
        self.optimizer.zero_grad()
        
        with torch.cuda.amp.autocast():
            outputs = self.model(inputs)
            loss = criterion(outputs, targets)
        
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        
        return loss.item()


# ============================================================================
# Early Stopping
# ============================================================================
class EarlyStopping:
    """
    Early stopping with patience and delta threshold.
    """
    def __init__(self, patience: int = 7, min_delta: float = 0.0,
                 mode: str = 'min', restore_best: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best = restore_best
        
        self.counter = 0
        self.best_score = None
        self.best_state = None
        self.should_stop = False
    
    def __call__(self, score: float, model: nn.Module) -> bool:
        if self.best_score is None:
            self.best_score = score
            self.best_state = copy.deepcopy(model.state_dict())
        elif self._is_improvement(score):
            self.best_score = score
            self.best_state = copy.deepcopy(model.state_dict())
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                if self.restore_best:
                    model.load_state_dict(self.best_state)
        
        return self.should_stop
    
    def _is_improvement(self, score: float) -> bool:
        if self.mode == 'min':
            return score < self.best_score - self.min_delta
        return score > self.best_score + self.min_delta


# ============================================================================
# Stochastic Weight Averaging (SWA)
# ============================================================================
class SWA:
    """
    Stochastic Weight Averaging (Izmailov et al., 2018)
    Averages weights from the final epochs of training.
    """
    def __init__(self, model: nn.Module, start_epoch: int = 0):
        self.model = model
        self.start_epoch = start_epoch
        self.swa_state = None
        self.n_averaged = 0
    
    def update(self, model: nn.Module, epoch: int):
        """Update SWA weights."""
        if epoch < self.start_epoch:
            return
        
        with torch.no_grad():
            if self.swa_state is None:
                self.swa_state = {
                    name: param.data.clone()
                    for name, param in model.named_parameters()
                }
                self.n_averaged = 1
            else:
                self.n_averaged += 1
                for name, param in model.named_parameters():
                    self.swa_state[name].add_(
                        (param.data - self.swa_state[name]) / self.n_averaged
                    )
    
    def apply_swa(self, model: nn.Module):
        """Apply SWA weights to model."""
        if self.swa_state is not None:
            with torch.no_grad():
                for name, param in model.named_parameters():
                    param.data.copy_(self.swa_state[name])
    
    def update_bn(self, model: nn.Module, loader: torch.utils.data.DataLoader, device: torch.device):
        """Update BatchNorm statistics after SWA."""
        model.train()
        with torch.no_grad():
            for inputs, _ in loader:
                model(inputs.to(device))


# ============================================================================
# Knowledge Distillation
# ============================================================================
class KnowledgeDistillation:
    """
    Knowledge Distillation for model compression (Hinton et al., 2015)
    """
    def __init__(self, teacher: nn.Module, temperature: float = 4.0, alpha: float = 0.5):
        self.teacher = teacher
        self.temperature = temperature
        self.alpha = alpha
        self.teacher.eval()
    
    def compute_loss(self, student_logits: torch.Tensor, targets: torch.Tensor,
                    inputs: torch.Tensor, hard_loss_fn: nn.Module) -> torch.Tensor:
        """
        Compute distillation loss.
        
        Args:
            student_logits: Student model outputs
            targets: Ground truth labels
            inputs: Input images
            hard_loss_fn: Loss function for hard labels (e.g., CrossEntropy)
        
        Returns:
            Combined loss
        """
        # Hard loss (student vs ground truth)
        hard_loss = hard_loss_fn(student_logits, targets)
        
        # Soft loss (student vs teacher)
        with torch.no_grad():
            teacher_logits = self.teacher(inputs)
        
        soft_student = torch.nn.functional.log_softmax(student_logits / self.temperature, dim=1)
        soft_teacher = torch.nn.functional.softmax(teacher_logits / self.temperature, dim=1)
        soft_loss = torch.nn.functional.kl_div(soft_student, soft_teacher, reduction='batchmean')
        soft_loss = soft_loss * (self.temperature ** 2)
        
        # Combined loss
        return self.alpha * soft_loss + (1 - self.alpha) * hard_loss


# ============================================================================
# Layer-wise Learning Rate Decay (for fine-tuning)
# ============================================================================
def get_layer_wise_lr_params(model: nn.Module, base_lr: float, 
                              decay_rate: float = 0.9) -> List[Dict]:
    """
    Get parameter groups with layer-wise learning rate decay.
    Lower layers get smaller learning rates.
    
    Args:
        model: PyTorch model
        base_lr: Base learning rate for the head
        decay_rate: Decay factor per layer
    
    Returns:
        Parameter groups for optimizer
    """
    params = []
    num_layers = len(list(model.named_parameters()))
    
    for i, (name, param) in enumerate(model.named_parameters()):
        layer_num = num_layers - i - 1
        lr = base_lr * (decay_rate ** layer_num)
        params.append({'params': param, 'lr': lr, 'name': name})
    
    return params
