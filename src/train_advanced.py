"""
Advanced Training Pipeline for Perfect Potato Leaf Disease Classification.

Features:
- Advanced augmentations (MixUp, CutMix, RandAugment)
- Multiple loss functions (Focal, Label Smoothing, Poly)
- Learning rate scheduling (Cosine, OneCycle)
- EMA (Exponential Moving Average)
- Mixed precision training
- Gradient clipping
- Early stopping with best model restoration
- Test Time Augmentation (TTA)
- Model ensemble support
- Comprehensive logging and metrics

Usage:
    python train_advanced.py --data-dir data --arch efficientnet_attention --epochs 50
"""

import os
import argparse
import time
import json
from typing import List, Tuple, Dict, Optional
import copy

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import torchvision.transforms.functional as TF
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Local imports
from dataset import infer_class_names
from augmentations import (
    MixUp, CutMix, get_advanced_transforms, mixup_criterion,
    TestTimeAugmentation, RandAugment
)
from losses import (
    LabelSmoothingCrossEntropy, FocalLoss, PolyLoss, 
    get_class_weights, CombinedLoss
)
from advanced_models import build_advanced_model
from training_utils import (
    ModelEMA, WarmupCosineScheduler, OneCycleLR, GradientClipper,
    EarlyStopping, SAM, SWA, MixedPrecisionTrainer
)
from xai import GradCAM, get_default_target_layer, overlay_heatmap_on_image


# ============================================================================
# Dataset Building with Proper Splits
# ============================================================================
def build_dataloaders_advanced(
    data_dir: str,
    batch_size: int = 32,
    val_split: float = 0.15,
    test_split: float = 0.15,
    image_size: int = 224,
    num_workers: int = 4,
    use_randaugment: bool = True,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader, DataLoader, List[str]]:
    """
    Build train/val/test dataloaders with stratified splitting.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Load dataset without transforms first for splitting
    temp_transform = transforms.Compose([transforms.Resize((image_size, image_size)), transforms.ToTensor()])
    full_dataset = datasets.ImageFolder(root=data_dir, transform=temp_transform)
    class_names = full_dataset.classes
    
    # Get labels for stratified split
    labels = [label for _, label in full_dataset.samples]
    
    # Stratified split
    indices = np.arange(len(full_dataset))
    np.random.shuffle(indices)
    
    n_test = int(len(full_dataset) * test_split)
    n_val = int(len(full_dataset) * val_split)
    n_train = len(full_dataset) - n_test - n_val
    
    # Simple split (for stratified, use sklearn)
    from sklearn.model_selection import train_test_split
    
    train_idx, temp_idx = train_test_split(
        indices, test_size=(n_val + n_test), stratify=[labels[i] for i in indices], random_state=seed
    )
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=n_test/(n_val + n_test), 
        stratify=[labels[i] for i in temp_idx], random_state=seed
    )
    
    # Create datasets with proper transforms
    train_transform = get_advanced_transforms(image_size, is_train=True, use_randaugment=use_randaugment)
    val_transform = get_advanced_transforms(image_size, is_train=False)
    
    train_dataset = datasets.ImageFolder(root=data_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(root=data_dir, transform=val_transform)
    test_dataset = datasets.ImageFolder(root=data_dir, transform=val_transform)
    
    train_subset = Subset(train_dataset, train_idx)
    val_subset = Subset(val_dataset, val_idx)
    test_subset = Subset(test_dataset, test_idx)
    
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, 
                              num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False,
                           num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    
    print(f"Dataset splits: Train={len(train_subset)}, Val={len(val_subset)}, Test={len(test_subset)}")
    print(f"Classes: {class_names}")
    
    return train_loader, val_loader, test_loader, class_names


# ============================================================================
# Training Functions
# ============================================================================
def train_one_epoch_advanced(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[object] = None,
    ema: Optional[ModelEMA] = None,
    grad_clipper: Optional[GradientClipper] = None,
    mixup: Optional[MixUp] = None,
    cutmix: Optional[CutMix] = None,
    mixup_prob: float = 0.5,
    use_amp: bool = True,
    sam: Optional[SAM] = None
) -> Tuple[float, float]:
    """
    Advanced training loop with all the bells and whistles.
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    scaler = torch.cuda.amp.GradScaler() if use_amp and device.type == 'cuda' else None
    
    pbar = tqdm(loader, desc="Train", leave=False)
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Apply MixUp or CutMix
        use_mix = (mixup is not None or cutmix is not None) and np.random.random() < mixup_prob
        if use_mix:
            if np.random.random() < 0.5 and mixup is not None:
                inputs, labels_a, labels_b, lam = mixup(inputs, labels)
            elif cutmix is not None:
                inputs, labels_a, labels_b, lam = cutmix(inputs, labels)
            else:
                use_mix = False
        
        optimizer.zero_grad()
        
        if use_amp and scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                if use_mix:
                    loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
                else:
                    loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            
            if grad_clipper:
                scaler.unscale_(optimizer)
                grad_clipper.clip(model)
            
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(inputs)
            if use_mix:
                loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
            else:
                loss = criterion(outputs, labels)
            
            loss.backward()
            
            if grad_clipper:
                grad_clipper.clip(model)
            
            if sam is not None:
                sam.first_step(zero_grad=True)
                # Second forward-backward
                outputs2 = model(inputs)
                if use_mix:
                    loss2 = mixup_criterion(criterion, outputs2, labels_a, labels_b, lam)
                else:
                    loss2 = criterion(outputs2, labels)
                loss2.backward()
                sam.second_step(zero_grad=True)
            else:
                optimizer.step()
        
        # Update EMA
        if ema is not None:
            ema.update(model)
        
        # Update scheduler (per-step)
        if scheduler is not None and hasattr(scheduler, 'step'):
            scheduler.step()
        
        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        
        if use_mix:
            # For mixed samples, count accuracy with dominant label
            correct += (lam * (preds == labels_a).float() + (1 - lam) * (preds == labels_b).float()).sum().item()
        else:
            correct += torch.sum(preds == labels).item()
        total += labels.size(0)
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{correct/total:.4f}'})
    
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    use_tta: bool = False,
    tta_augments: int = 5
) -> Tuple[float, float, List[int], List[int], List[float]]:
    """
    Validation with optional TTA.
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_labels = []
    all_preds = []
    all_probs = []
    
    if use_tta:
        tta = TestTimeAugmentation(model, device, tta_augments)
    
    for inputs, labels in tqdm(loader, desc="Val", leave=False):
        inputs, labels = inputs.to(device), labels.to(device)
        
        if use_tta:
            probs = tta.predict(inputs)
            outputs = torch.log(probs + 1e-10)  # Convert back for loss computation
        else:
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
        
        loss = criterion(outputs, labels)
        running_loss += loss.item() * inputs.size(0)
        
        _, preds = torch.max(probs, 1)
        correct += torch.sum(preds == labels).item()
        total += labels.size(0)
        
        all_labels.extend(labels.cpu().tolist())
        all_preds.extend(preds.cpu().tolist())
        all_probs.extend(probs.cpu().tolist())
    
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc, all_labels, all_preds, all_probs


# ============================================================================
# Visualization Functions
# ============================================================================
def plot_training_curves(history: Dict, out_dir: str):
    """Plot training and validation curves."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Loss
    axes[0, 0].plot(history['train_loss'], label='Train Loss', linewidth=2)
    axes[0, 0].plot(history['val_loss'], label='Val Loss', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training & Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[0, 1].plot(history['train_acc'], label='Train Acc', linewidth=2)
    axes[0, 1].plot(history['val_acc'], label='Val Acc', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('Training & Validation Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Learning rate
    if 'lr' in history:
        axes[1, 0].plot(history['lr'], linewidth=2, color='green')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_title('Learning Rate Schedule')
        axes[1, 0].grid(True, alpha=0.3)
    
    # F1 Score
    if 'val_f1' in history:
        axes[1, 1].plot(history['val_f1'], label='Val F1', linewidth=2, color='purple')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('F1 Score')
        axes[1, 1].set_title('Validation F1 Score')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'training_curves.png'), dpi=150)
    plt.close()


def plot_confusion_matrix(cm: np.ndarray, class_names: List[str], out_dir: str, 
                          normalize: bool = True, title: str = 'Confusion Matrix'):
    """Plot confusion matrix with normalization option."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Raw counts
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names, ax=axes[0])
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('True')
    axes[0].set_title('Confusion Matrix (Counts)')
    
    # Normalized
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Greens',
                xticklabels=class_names, yticklabels=class_names, ax=axes[1])
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('True')
    axes[1].set_title('Confusion Matrix (Normalized)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'confusion_matrix.png'), dpi=150)
    plt.close()


def plot_per_class_metrics(report_dict: Dict, class_names: List[str], out_dir: str):
    """Plot per-class precision, recall, F1."""
    metrics = ['precision', 'recall', 'f1-score']
    x = np.arange(len(class_names))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for i, metric in enumerate(metrics):
        values = [report_dict[cls][metric] for cls in class_names]
        ax.bar(x + i * width, values, width, label=metric.capitalize())
    
    ax.set_xlabel('Class')
    ax.set_ylabel('Score')
    ax.set_title('Per-Class Performance Metrics')
    ax.set_xticks(x + width)
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'per_class_metrics.png'), dpi=150)
    plt.close()


# ============================================================================
# Model Ensemble
# ============================================================================
class ModelEnsemble:
    """Ensemble multiple models for improved predictions."""
    
    def __init__(self, models: List[nn.Module], weights: Optional[List[float]] = None):
        self.models = models
        self.weights = weights or [1.0 / len(models)] * len(models)
    
    @torch.no_grad()
    def predict(self, inputs: torch.Tensor) -> torch.Tensor:
        """Get ensemble predictions."""
        all_probs = []
        for model, weight in zip(self.models, self.weights):
            model.eval()
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            all_probs.append(probs * weight)
        
        # Weighted average
        ensemble_probs = torch.stack(all_probs).sum(dim=0)
        return ensemble_probs


def train_ensemble(
    data_dir: str,
    architectures: List[str],
    num_classes: int,
    device: torch.device,
    args
) -> List[nn.Module]:
    """Train multiple models for ensemble."""
    models = []
    
    for i, arch in enumerate(architectures):
        print(f"\n{'='*50}")
        print(f"Training model {i+1}/{len(architectures)}: {arch}")
        print(f"{'='*50}")
        
        model = build_advanced_model(arch, num_classes, pretrained=True)
        model = model.to(device)
        
        # Train each model (simplified - in practice you'd run full training)
        # ... training code ...
        
        models.append(model)
    
    return models


# ============================================================================
# K-Fold Cross Validation
# ============================================================================
def train_with_kfold(
    data_dir: str,
    n_splits: int,
    args,
    device: torch.device
) -> Dict:
    """Train with K-Fold cross validation for robust evaluation."""
    from sklearn.model_selection import StratifiedKFold
    
    # Load full dataset
    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor()
    ])
    full_dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    class_names = full_dataset.classes
    num_classes = len(class_names)
    
    # Get labels
    labels = np.array([label for _, label in full_dataset.samples])
    
    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    fold_results = []
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(np.zeros(len(labels)), labels)):
        print(f"\n{'='*50}")
        print(f"Fold {fold + 1}/{n_splits}")
        print(f"{'='*50}")
        
        # Create datasets for this fold
        train_transform = get_advanced_transforms(args.image_size, is_train=True)
        val_transform = get_advanced_transforms(args.image_size, is_train=False)
        
        train_dataset = datasets.ImageFolder(root=data_dir, transform=train_transform)
        val_dataset = datasets.ImageFolder(root=data_dir, transform=val_transform)
        
        train_subset = Subset(train_dataset, train_idx)
        val_subset = Subset(val_dataset, val_idx)
        
        train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False, num_workers=4)
        
        # Build model
        model = build_advanced_model(args.arch, num_classes, pretrained=True)
        model = model.to(device)
        
        # Train (use your training loop)
        # ... simplified here ...
        
        fold_results.append({
            'fold': fold + 1,
            # Add metrics
        })
    
    return {'folds': fold_results}


# ============================================================================
# Main Training Function
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="Advanced Potato Leaf Disease Classifier Training")
    
    # Data
    parser.add_argument('--data-dir', type=str, required=True, help='Path to data directory')
    parser.add_argument('--image-size', type=int, default=224, help='Input image size')
    parser.add_argument('--val-split', type=float, default=0.15, help='Validation split ratio')
    parser.add_argument('--test-split', type=float, default=0.15, help='Test split ratio')
    
    # Model
    parser.add_argument('--arch', type=str, default='efficientnet_attention',
                       choices=['efficientnet_attention', 'efficientnet_b0', 'efficientnet_b3',
                               'resnet50', 'improved_cnn', 'vit', 'cnn_transformer', 'convnext_tiny'])
    parser.add_argument('--attention', type=str, default='cbam', choices=['cbam', 'se', 'eca', 'none'])
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--no-pretrained', action='store_true')
    
    # Training
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    
    # Advanced training
    parser.add_argument('--loss', type=str, default='focal', 
                       choices=['ce', 'focal', 'label_smooth', 'poly', 'combined'])
    parser.add_argument('--label-smoothing', type=float, default=0.1)
    parser.add_argument('--focal-gamma', type=float, default=2.0)
    parser.add_argument('--scheduler', type=str, default='cosine',
                       choices=['cosine', 'onecycle', 'step', 'none'])
    parser.add_argument('--warmup-epochs', type=int, default=5)
    
    # Augmentation
    parser.add_argument('--use-mixup', action='store_true', default=True)
    parser.add_argument('--use-cutmix', action='store_true', default=True)
    parser.add_argument('--mixup-alpha', type=float, default=0.2)
    parser.add_argument('--cutmix-alpha', type=float, default=1.0)
    parser.add_argument('--mixup-prob', type=float, default=0.5)
    parser.add_argument('--no-randaugment', action='store_true')
    
    # Regularization
    parser.add_argument('--use-ema', action='store_true', default=True)
    parser.add_argument('--ema-decay', type=float, default=0.9999)
    parser.add_argument('--grad-clip', type=float, default=1.0)
    parser.add_argument('--use-swa', action='store_true')
    parser.add_argument('--swa-start', type=int, default=30)
    parser.add_argument('--use-sam', action='store_true')
    
    # Evaluation
    parser.add_argument('--use-tta', action='store_true', default=True, help='Test Time Augmentation')
    parser.add_argument('--tta-augments', type=int, default=5)
    
    # Output
    parser.add_argument('--output-dir', type=str, default='models')
    parser.add_argument('--save-xai-samples', type=int, default=10)
    parser.add_argument('--notes', type=str, default='')
    
    # Hardware
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--use-amp', action='store_true', default=True, help='Mixed precision')
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Build dataloaders
    train_loader, val_loader, test_loader, class_names = build_dataloaders_advanced(
        args.data_dir,
        batch_size=args.batch_size,
        val_split=args.val_split,
        test_split=args.test_split,
        image_size=args.image_size,
        num_workers=args.num_workers,
        use_randaugment=not args.no_randaugment,
        seed=args.seed
    )
    
    num_classes = len(class_names)
    print(f"Number of classes: {num_classes}")
    
    # Build model
    model = build_advanced_model(
        args.arch, num_classes, 
        pretrained=not args.no_pretrained,
        attention=args.attention,
        dropout=args.dropout
    )
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Build loss function
    if args.loss == 'ce':
        criterion = nn.CrossEntropyLoss()
    elif args.loss == 'focal':
        criterion = FocalLoss(gamma=args.focal_gamma)
    elif args.loss == 'label_smooth':
        criterion = LabelSmoothingCrossEntropy(smoothing=args.label_smoothing)
    elif args.loss == 'poly':
        criterion = PolyLoss(epsilon=1.0)
    elif args.loss == 'combined':
        criterion = CombinedLoss([
            FocalLoss(gamma=2.0),
            LabelSmoothingCrossEntropy(smoothing=0.1)
        ], weights=[0.5, 0.5])
    else:
        criterion = nn.CrossEntropyLoss()
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # SAM optimizer wrapper
    sam = SAM(optimizer, rho=0.05) if args.use_sam else None
    
    # Learning rate scheduler
    total_steps = len(train_loader) * args.epochs
    if args.scheduler == 'cosine':
        scheduler = WarmupCosineScheduler(optimizer, args.warmup_epochs, args.epochs)
    elif args.scheduler == 'onecycle':
        scheduler = OneCycleLR(optimizer, max_lr=args.lr * 10, total_steps=total_steps)
    elif args.scheduler == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    else:
        scheduler = None
    
    # EMA
    ema = ModelEMA(model, decay=args.ema_decay) if args.use_ema else None
    
    # SWA
    swa = SWA(model, start_epoch=args.swa_start) if args.use_swa else None
    
    # Gradient clipper
    grad_clipper = GradientClipper(max_norm=args.grad_clip) if args.grad_clip > 0 else None
    
    # MixUp / CutMix
    mixup = MixUp(alpha=args.mixup_alpha) if args.use_mixup else None
    cutmix = CutMix(alpha=args.cutmix_alpha) if args.use_cutmix else None
    
    # Early stopping
    early_stopping = EarlyStopping(patience=args.patience, mode='max', restore_best=True)
    
    # Training history
    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': [],
        'val_f1': [], 'lr': []
    }
    
    best_val_acc = 0.0
    best_val_f1 = 0.0
    best_model_state = None
    
    print(f"\n{'='*60}")
    print("Starting Training")
    print(f"{'='*60}")
    
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        start_time = time.time()
        
        # Train
        train_loss, train_acc = train_one_epoch_advanced(
            model, train_loader, device, criterion, optimizer,
            scheduler=scheduler if args.scheduler == 'onecycle' else None,
            ema=ema, grad_clipper=grad_clipper,
            mixup=mixup, cutmix=cutmix, mixup_prob=args.mixup_prob,
            use_amp=args.use_amp, sam=sam
        )
        
        # Update epoch-level scheduler
        if scheduler is not None and args.scheduler != 'onecycle':
            scheduler.step()
        
        # Update SWA
        if swa is not None:
            swa.update(model, epoch)
        
        # Validate with EMA model if available
        eval_model = model
        if ema is not None:
            ema.apply_shadow(model)
            eval_model = model
        
        val_loss, val_acc, val_labels, val_preds, val_probs = validate(
            eval_model, val_loader, device, criterion,
            use_tta=args.use_tta and epoch > args.epochs // 2,  # TTA in later epochs
            tta_augments=args.tta_augments
        )
        
        # Restore original weights
        if ema is not None:
            ema.restore(model)
        
        # Calculate F1
        val_f1 = f1_score(val_labels, val_preds, average='macro')
        
        # Get current LR
        current_lr = optimizer.param_groups[0]['lr']
        
        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)
        history['lr'].append(current_lr)
        
        elapsed = time.time() - start_time
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")
        print(f"LR: {current_lr:.6f} | Time: {elapsed:.1f}s")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_f1 = val_f1
            
            # Save with EMA weights if available
            if ema is not None:
                ema.apply_shadow(model)
            
            best_model_state = copy.deepcopy(model.state_dict())
            
            torch.save({
                'model_state': model.state_dict(),
                'class_names': class_names,
                'arch': args.arch,
                'epoch': epoch,
                'val_acc': val_acc,
                'val_f1': val_f1,
                'args': vars(args)
            }, os.path.join(args.output_dir, 'best_model.pt'))
            
            if ema is not None:
                ema.restore(model)
            
            print(f"✓ Saved new best model (Acc: {val_acc:.4f}, F1: {val_f1:.4f})")
        
        # Early stopping
        if early_stopping(val_acc, model):
            print(f"\nEarly stopping triggered at epoch {epoch}")
            break
    
    # Apply SWA if used
    if swa is not None and swa.swa_state is not None:
        print("\nApplying SWA weights...")
        swa.apply_swa(model)
        swa.update_bn(model, train_loader, device)
    
    # Load best model for final evaluation
    model.load_state_dict(best_model_state)
    
    # Final evaluation on test set
    print(f"\n{'='*60}")
    print("Final Evaluation on Test Set")
    print(f"{'='*60}")
    
    if ema is not None:
        ema.apply_shadow(model)
    
    test_loss, test_acc, test_labels, test_preds, test_probs = validate(
        model, test_loader, device, criterion,
        use_tta=args.use_tta, tta_augments=args.tta_augments
    )
    
    test_f1 = f1_score(test_labels, test_preds, average='macro')
    
    print(f"\nTest Results:")
    print(f"  Accuracy: {test_acc:.4f}")
    print(f"  F1 Score: {test_f1:.4f}")
    print(f"  Loss: {test_loss:.4f}")
    
    # Detailed classification report
    report = classification_report(test_labels, test_preds, target_names=class_names, output_dict=True)
    print("\nClassification Report:")
    print(classification_report(test_labels, test_preds, target_names=class_names))
    
    # Confusion matrix
    cm = confusion_matrix(test_labels, test_preds)
    
    # Save results
    plot_training_curves(history, args.output_dir)
    plot_confusion_matrix(cm, class_names, args.output_dir)
    plot_per_class_metrics(report, class_names, args.output_dir)
    
    # Save classification report
    with open(os.path.join(args.output_dir, 'classification_report.json'), 'w') as f:
        json.dump(report, f, indent=2)
    
    # Save training history
    with open(os.path.join(args.output_dir, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    
    # Save final results summary
    results_summary = {
        'architecture': args.arch,
        'best_val_acc': best_val_acc,
        'best_val_f1': best_val_f1,
        'test_acc': test_acc,
        'test_f1': test_f1,
        'total_epochs': len(history['train_loss']),
        'args': vars(args)
    }
    with open(os.path.join(args.output_dir, 'results_summary.json'), 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    # Save Grad-CAM samples
    if args.save_xai_samples > 0:
        print("\nGenerating Grad-CAM visualizations...")
        os.makedirs(os.path.join(args.output_dir, 'xai_samples'), exist_ok=True)
        
        try:
            target_layer = get_default_target_layer(model, arch=args.arch)
        except:
            # Fallback for custom models
            for m in reversed(list(model.modules())):
                if isinstance(m, nn.Conv2d):
                    target_layer = m
                    break
        
        cam = GradCAM(model, target_layer)
        saved = 0
        mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
        
        for inputs, labels_batch in test_loader:
            if saved >= args.save_xai_samples:
                break
            
            inputs = inputs.to(device)
            
            for i in range(inputs.size(0)):
                if saved >= args.save_xai_samples:
                    break
                
                img_tensor = inputs[i:i+1]
                heatmap, pred_idx = cam.generate(img_tensor)
                
                # Denormalize
                denorm = (img_tensor * std + mean).clamp(0, 1)[0].cpu()
                pil_img = TF.to_pil_image(denorm)
                overlay = overlay_heatmap_on_image(pil_img, heatmap)
                
                true_label = class_names[labels_batch[i].item()]
                pred_label = class_names[pred_idx]
                status = "correct" if true_label == pred_label else "wrong"
                
                out_name = f"sample_{saved+1:02d}_{status}_true-{true_label}_pred-{pred_label}.jpg"
                overlay.save(os.path.join(args.output_dir, 'xai_samples', out_name))
                saved += 1
        
        cam.close()
        print(f"Saved {saved} Grad-CAM samples")
    
    print(f"\n{'='*60}")
    print("Training Complete!")
    print(f"{'='*60}")
    print(f"Best Validation Accuracy: {best_val_acc:.4f}")
    print(f"Best Validation F1: {best_val_f1:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test F1: {test_f1:.4f}")
    print(f"\nResults saved to: {args.output_dir}")
    
    return results_summary


if __name__ == '__main__':
    main()
