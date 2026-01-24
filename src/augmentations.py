"""
Advanced Data Augmentation techniques for improved model performance.
Includes MixUp, CutMix, RandAugment, and more.
"""
import torch
import torch.nn as nn
import numpy as np
from PIL import Image, ImageEnhance, ImageOps
from torchvision import transforms
from typing import Tuple, Optional, Callable, List
import random


# ============================================================================
# MixUp Augmentation
# ============================================================================
class MixUp:
    """
    MixUp: Beyond Empirical Risk Minimization (Zhang et al., 2018)
    Linearly interpolates between two samples and their labels.
    """
    def __init__(self, alpha: float = 0.2):
        self.alpha = alpha
    
    def __call__(self, images: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """
        Args:
            images: Batch of images [N, C, H, W]
            labels: Batch of labels [N]
        Returns:
            mixed_images, labels_a, labels_b, lambda
        """
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1.0
        
        batch_size = images.size(0)
        index = torch.randperm(batch_size, device=images.device)
        
        mixed_images = lam * images + (1 - lam) * images[index]
        labels_a, labels_b = labels, labels[index]
        
        return mixed_images, labels_a, labels_b, lam


# ============================================================================
# CutMix Augmentation
# ============================================================================
class CutMix:
    """
    CutMix: Regularization Strategy (Yun et al., 2019)
    Cuts and pastes patches between training images.
    """
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
    
    def _rand_bbox(self, size: Tuple, lam: float) -> Tuple[int, int, int, int]:
        W = size[2]
        H = size[3]
        cut_rat = np.sqrt(1.0 - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)
        
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        
        return bbx1, bby1, bbx2, bby2
    
    def __call__(self, images: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """
        Args:
            images: Batch of images [N, C, H, W]
            labels: Batch of labels [N]
        Returns:
            mixed_images, labels_a, labels_b, lambda
        """
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1.0
        
        batch_size = images.size(0)
        index = torch.randperm(batch_size, device=images.device)
        
        bbx1, bby1, bbx2, bby2 = self._rand_bbox(images.size(), lam)
        
        mixed_images = images.clone()
        mixed_images[:, :, bbx1:bbx2, bby1:bby2] = images[index, :, bbx1:bbx2, bby1:bby2]
        
        # Adjust lambda based on actual box area
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (images.size(-1) * images.size(-2)))
        
        labels_a, labels_b = labels, labels[index]
        
        return mixed_images, labels_a, labels_b, lam


# ============================================================================
# CutOut Augmentation
# ============================================================================
class Cutout:
    """
    Improved Regularization of CNN via Cutout (DeVries & Taylor, 2017)
    Randomly masks out square regions of input during training.
    """
    def __init__(self, n_holes: int = 1, length: int = 16):
        self.n_holes = n_holes
        self.length = length
    
    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        """
        Args:
            img: Tensor image [C, H, W]
        Returns:
            Image with n_holes holes of size length x length cut out
        """
        h = img.size(1)
        w = img.size(2)
        
        mask = np.ones((h, w), np.float32)
        
        for _ in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)
            
            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)
            
            mask[y1:y2, x1:x2] = 0.0
        
        mask = torch.from_numpy(mask).expand_as(img)
        img = img * mask
        
        return img


# ============================================================================
# RandAugment
# ============================================================================
class RandAugment:
    """
    RandAugment: Practical automated data augmentation (Cubuk et al., 2020)
    Applies N random augmentations with magnitude M.
    """
    def __init__(self, n: int = 2, m: int = 9, max_magnitude: int = 10):
        self.n = n
        self.m = m
        self.max_magnitude = max_magnitude
        
        self.augment_list = [
            ("Identity", 0, 1),
            ("AutoContrast", 0, 1),
            ("Equalize", 0, 1),
            ("Rotate", -30, 30),
            ("Solarize", 0, 256),
            ("Color", 0.1, 1.9),
            ("Posterize", 4, 8),
            ("Contrast", 0.1, 1.9),
            ("Brightness", 0.1, 1.9),
            ("Sharpness", 0.1, 1.9),
            ("ShearX", -0.3, 0.3),
            ("ShearY", -0.3, 0.3),
            ("TranslateX", -0.3, 0.3),
            ("TranslateY", -0.3, 0.3),
        ]
    
    def _apply_operation(self, img: Image.Image, name: str, magnitude: float) -> Image.Image:
        if name == "Identity":
            return img
        elif name == "AutoContrast":
            return ImageOps.autocontrast(img)
        elif name == "Equalize":
            return ImageOps.equalize(img)
        elif name == "Rotate":
            return img.rotate(magnitude)
        elif name == "Solarize":
            return ImageOps.solarize(img, int(magnitude))
        elif name == "Color":
            return ImageEnhance.Color(img).enhance(magnitude)
        elif name == "Posterize":
            return ImageOps.posterize(img, int(magnitude))
        elif name == "Contrast":
            return ImageEnhance.Contrast(img).enhance(magnitude)
        elif name == "Brightness":
            return ImageEnhance.Brightness(img).enhance(magnitude)
        elif name == "Sharpness":
            return ImageEnhance.Sharpness(img).enhance(magnitude)
        elif name == "ShearX":
            return img.transform(img.size, Image.AFFINE, (1, magnitude, 0, 0, 1, 0))
        elif name == "ShearY":
            return img.transform(img.size, Image.AFFINE, (1, 0, 0, magnitude, 1, 0))
        elif name == "TranslateX":
            return img.transform(img.size, Image.AFFINE, (1, 0, magnitude * img.size[0], 0, 1, 0))
        elif name == "TranslateY":
            return img.transform(img.size, Image.AFFINE, (1, 0, 0, 0, 1, magnitude * img.size[1]))
        return img
    
    def __call__(self, img: Image.Image) -> Image.Image:
        ops = random.choices(self.augment_list, k=self.n)
        for name, min_val, max_val in ops:
            magnitude = (self.m / self.max_magnitude) * (max_val - min_val) + min_val
            img = self._apply_operation(img, name, magnitude)
        return img


# ============================================================================
# GridMask Augmentation
# ============================================================================
class GridMask:
    """
    GridMask Data Augmentation (Chen et al., 2020)
    Drops information by masking in a grid pattern.
    """
    def __init__(self, d1: int = 96, d2: int = 224, rotate: float = 1.0, ratio: float = 0.5, mode: int = 0, prob: float = 0.5):
        self.d1 = d1
        self.d2 = d2
        self.rotate = rotate
        self.ratio = ratio
        self.mode = mode
        self.prob = prob
    
    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        if np.random.random() > self.prob:
            return img
        
        h = img.size(1)
        w = img.size(2)
        
        d = np.random.randint(self.d1, self.d2)
        l = int(d * self.ratio + 0.5)
        
        mask = np.ones((h, w), np.float32)
        st_h = np.random.randint(d)
        st_w = np.random.randint(d)
        
        for i in range(h // d + 1):
            s = d * i + st_h
            t = min(s + l, h)
            mask[s:t, :] *= 0
        
        for i in range(w // d + 1):
            s = d * i + st_w
            t = min(s + l, w)
            mask[:, s:t] *= 0
        
        mask = torch.from_numpy(mask).expand_as(img).float()
        
        if self.mode == 1:
            mask = 1 - mask
        
        return img * mask


# ============================================================================
# Test Time Augmentation (TTA)
# ============================================================================
class TestTimeAugmentation:
    """
    Test Time Augmentation: Average predictions across multiple augmented versions.
    """
    def __init__(self, model: nn.Module, device: torch.device, num_augments: int = 5):
        self.model = model
        self.device = device
        self.num_augments = num_augments
        
        self.transforms_list = [
            transforms.Compose([]),  # Original
            transforms.Compose([transforms.RandomHorizontalFlip(p=1.0)]),
            transforms.Compose([transforms.RandomVerticalFlip(p=1.0)]),
            transforms.Compose([transforms.RandomRotation(degrees=(90, 90))]),
            transforms.Compose([transforms.RandomRotation(degrees=(-90, -90))]),
            transforms.Compose([transforms.RandomRotation(degrees=(180, 180))]),
            transforms.Compose([transforms.GaussianBlur(3, sigma=(0.1, 0.5))]),
            transforms.Compose([transforms.ColorJitter(brightness=0.1)]),
        ]
    
    @torch.no_grad()
    def predict(self, images: torch.Tensor) -> torch.Tensor:
        """
        Run TTA on a batch of images.
        Args:
            images: Normalized tensor [N, C, H, W]
        Returns:
            Averaged softmax predictions [N, num_classes]
        """
        self.model.eval()
        batch_preds = []
        
        for i in range(min(self.num_augments, len(self.transforms_list))):
            transform = self.transforms_list[i]
            augmented = transform(images)
            outputs = self.model(augmented.to(self.device))
            probs = torch.softmax(outputs, dim=1)
            batch_preds.append(probs)
        
        # Average predictions
        avg_preds = torch.stack(batch_preds).mean(dim=0)
        return avg_preds


# ============================================================================
# Advanced Transform Pipeline
# ============================================================================
def get_advanced_transforms(image_size: int = 224, is_train: bool = True, use_randaugment: bool = True):
    """
    Get advanced transform pipeline with optional RandAugment.
    """
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    if is_train:
        transform_list = [
            transforms.Resize((int(image_size * 1.1), int(image_size * 1.1))),
            transforms.RandomCrop(image_size),
        ]
        
        if use_randaugment:
            transform_list.append(RandAugment(n=2, m=9))
        
        transform_list.extend([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.RandomApply([
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05)
            ], p=0.5),
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))
            ], p=0.2),
            transforms.RandomGrayscale(p=0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
            transforms.RandomErasing(p=0.25, scale=(0.02, 0.2), ratio=(0.3, 3.3)),
        ])
        
        return transforms.Compose(transform_list)
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])


# ============================================================================
# MixUp/CutMix Loss Helper
# ============================================================================
def mixup_criterion(criterion: nn.Module, pred: torch.Tensor, 
                    y_a: torch.Tensor, y_b: torch.Tensor, lam: float) -> torch.Tensor:
    """
    Compute mixed loss for MixUp/CutMix.
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ============================================================================
# Progressive Resizing
# ============================================================================
class ProgressiveResizing:
    """
    Start training with smaller images and progressively increase size.
    Helps model learn coarse features first, then fine details.
    """
    def __init__(self, start_size: int = 128, end_size: int = 224, num_stages: int = 3):
        self.start_size = start_size
        self.end_size = end_size
        self.num_stages = num_stages
        self.sizes = np.linspace(start_size, end_size, num_stages).astype(int).tolist()
    
    def get_size(self, epoch: int, total_epochs: int) -> int:
        """Get image size for current epoch."""
        progress = epoch / total_epochs
        stage = min(int(progress * self.num_stages), self.num_stages - 1)
        return self.sizes[stage]
