import os
import random
from typing import Tuple, Dict
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, Subset
import torch
from PIL import Image, ImageFilter, ImageOps

CLASS_NAMES = ["Early_Blight", "Late_Blight", "Healthy"]


class GaussianBlur:
    """Apply Gaussian Blur with random radius."""
    def __init__(self, radius_min=0.1, radius_max=2.0):
        self.radius_min = radius_min
        self.radius_max = radius_max
    
    def __call__(self, img):
        radius = random.uniform(self.radius_min, self.radius_max)
        return img.filter(ImageFilter.GaussianBlur(radius=radius))


class RandomErasing:
    """Custom Random Erasing on PIL images."""
    def __init__(self, p=0.5, scale=(0.02, 0.15), ratio=(0.3, 3.3)):
        self.p = p
        self.scale = scale
        self.ratio = ratio
    
    def __call__(self, img):
        if random.random() > self.p:
            return img
        
        img_array = torch.tensor(list(img.getdata())).reshape(img.size[1], img.size[0], 3)
        h, w = img_array.shape[:2]
        
        area = h * w
        for _ in range(10):
            erase_area = random.uniform(self.scale[0], self.scale[1]) * area
            aspect_ratio = random.uniform(self.ratio[0], self.ratio[1])
            
            eh = int(round((erase_area * aspect_ratio) ** 0.5))
            ew = int(round((erase_area / aspect_ratio) ** 0.5))
            
            if eh < h and ew < w:
                x = random.randint(0, w - ew)
                y = random.randint(0, h - eh)
                
                # Create erased image
                from PIL import ImageDraw
                img_copy = img.copy()
                draw = ImageDraw.Draw(img_copy)
                fill_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                draw.rectangle([x, y, x + ew, y + eh], fill=fill_color)
                return img_copy
        
        return img


class Cutout:
    """Randomly mask out square regions."""
    def __init__(self, n_holes=1, length=40):
        self.n_holes = n_holes
        self.length = length
    
    def __call__(self, img):
        w, h = img.size
        mask = Image.new('RGB', (w, h), (0, 0, 0))
        
        for _ in range(self.n_holes):
            x = random.randint(0, w)
            y = random.randint(0, h)
            
            x1 = max(0, x - self.length // 2)
            y1 = max(0, y - self.length // 2)
            x2 = min(w, x + self.length // 2)
            y2 = min(h, y + self.length // 2)
            
            # Paste gray patch
            from PIL import ImageDraw
            img_copy = img.copy()
            draw = ImageDraw.Draw(img_copy)
            draw.rectangle([x1, y1, x2, y2], fill=(128, 128, 128))
            return img_copy
        
        return img


class RandomSolarize:
    """Randomly solarize the image."""
    def __init__(self, threshold=128, p=0.2):
        self.threshold = threshold
        self.p = p
    
    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img, self.threshold)
        return img


class RandomPosterize:
    """Randomly posterize the image."""
    def __init__(self, bits=4, p=0.2):
        self.bits = bits
        self.p = p
    
    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.posterize(img, self.bits)
        return img


def get_data_transforms(image_size: int = 224, augment_level: str = "strong") -> Dict[str, transforms.Compose]:
    """
    Get data transforms with different augmentation levels.
    
    Args:
        image_size: Target image size
        augment_level: 'light', 'medium', 'strong', or 'extreme'
    """
    
    if augment_level == "light":
        train_tf = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    elif augment_level == "medium":
        train_tf = transforms.Compose([
            transforms.Resize((image_size + 32, image_size + 32)),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomRotation(25),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    elif augment_level == "strong":
        # Strong augmentation for robust training
        train_tf = transforms.Compose([
            transforms.Resize((image_size + 32, image_size + 32)),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(45),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.85, 1.15), shear=15),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            transforms.RandomApply([GaussianBlur(0.1, 2.0)], p=0.3),
            transforms.RandomGrayscale(p=0.1),
            RandomSolarize(threshold=128, p=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.25, scale=(0.02, 0.2), ratio=(0.3, 3.3))
        ])
    
    else:  # extreme
        train_tf = transforms.Compose([
            transforms.Resize((image_size + 48, image_size + 48)),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(90),
            transforms.RandomAffine(degrees=0, translate=(0.15, 0.15), scale=(0.75, 1.25), shear=20),
            transforms.RandomPerspective(distortion_scale=0.3, p=0.4),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.15),
            transforms.RandomApply([GaussianBlur(0.1, 3.0)], p=0.4),
            transforms.RandomGrayscale(p=0.2),
            RandomSolarize(threshold=100, p=0.15),
            RandomPosterize(bits=4, p=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.4, scale=(0.02, 0.3), ratio=(0.3, 3.3))
        ])

    val_tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return {"train": train_tf, "val": val_tf}


def build_dataloaders(data_dir: str, batch_size: int = 32, val_split: float = 0.2, image_size: int = 224,
                      num_workers: int = 0, augment_level: str = "strong") -> Tuple[DataLoader, DataLoader]:
    """
    Build train and validation data loaders with stratified sampling.
    
    Args:
        data_dir: Path to data directory with class subfolders
        batch_size: Batch size for training
        val_split: Fraction of data for validation
        image_size: Target image size
        num_workers: Number of workers for data loading
        augment_level: Augmentation level ('light', 'medium', 'strong', 'extreme')
    """
    tfs = get_data_transforms(image_size, augment_level)
    
    # Validate that data_dir exists and has at least one class subfolder
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Data directory '{data_dir}' does not exist. Expected structure: data/Early_Blight, data/Late_Blight, data/Healthy")
    subdirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    if len(subdirs) == 0:
        raise FileNotFoundError(
            f"No class folders found in '{data_dir}'. Create folders like:\n"
            f"{data_dir}/Early_Blight\n{data_dir}/Late_Blight\n{data_dir}/Healthy\n"
            "and place images inside each.")
    
    # Create full dataset
    full_dataset = datasets.ImageFolder(root=data_dir)
    
    # Stratified split to ensure balanced classes in train/val
    from collections import defaultdict
    class_indices = defaultdict(list)
    for idx, (_, label) in enumerate(full_dataset.samples):
        class_indices[label].append(idx)
    
    train_indices = []
    val_indices = []
    
    for label, indices in class_indices.items():
        random.shuffle(indices)
        n_val = max(1, int(len(indices) * val_split))
        val_indices.extend(indices[:n_val])
        train_indices.extend(indices[n_val:])
    
    # Create train dataset with augmentation
    train_dataset = datasets.ImageFolder(root=data_dir, transform=tfs["train"])
    train_subset = Subset(train_dataset, train_indices)
    
    # Create val dataset without augmentation
    val_dataset = datasets.ImageFolder(root=data_dir, transform=tfs["val"])
    val_subset = Subset(val_dataset, val_indices)
    
    print(f"Dataset split - Train: {len(train_indices)}, Val: {len(val_indices)}")
    print(f"Augmentation level: {augment_level}")
    
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, 
                              num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, 
                            num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader


def infer_class_names(data_dir: str):
    path = os.path.join(data_dir)
    sub = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    return sorted(sub)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Inspect dataset splits and classes")
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--val-split", type=float, default=0.2)
    args = parser.parse_args()
    try:
        train_loader, val_loader = build_dataloaders(args.data_dir, image_size=args.image_size, val_split=args.val_split)
        print(f"Train batches: {len(train_loader)}  Validation batches: {len(val_loader)}")
        print(f"Sample classes: {infer_class_names(args.data_dir)}")
    except FileNotFoundError as e:
        print("ERROR:", e)
        print("Fix the dataset structure and retry.")
