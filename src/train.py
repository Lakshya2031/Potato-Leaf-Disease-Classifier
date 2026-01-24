import os
import argparse
from typing import List, Tuple
import time
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from dataset import build_dataloaders, infer_class_names
from xai import GradCAM, get_default_target_layer, overlay_heatmap_on_image


def build_model(arch: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    arch = arch.lower()
    if arch == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    elif arch == "mobilenet_v2":
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT if pretrained else None)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
    elif arch == "efficientnet_b0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT if pretrained else None)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
    elif arch == "cnn_lstm":
        backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT if pretrained else None)
        model = CNNLSTM(backbone, num_classes)
    elif arch == "custom":
        model = CustomCNN(num_classes)
    else:
        raise ValueError(f"Unsupported architecture: {arch}")
    return model


class CustomCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class CNNLSTM(nn.Module):
    def __init__(self, backbone: nn.Module, num_classes: int, hidden_size: int = 256):
        super().__init__()
        self.backbone = backbone
        self.features = backbone.features  # EfficientNet features
        # determine channels from last conv
        sample = torch.zeros(1,3,224,224)
        with torch.no_grad():
            c = self.features(sample).shape[1]
        self.lstm = nn.LSTM(input_size=c, hidden_size=hidden_size, num_layers=1, batch_first=True, bidirectional=True)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size*2, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        feats = self.features(x)            # [N,C,H,W]
        n, c, h, w = feats.shape
        seq = feats.view(n, c, h*w).permute(0, 2, 1)  # [N,L,C]
        out, (hn, cn) = self.lstm(seq)     # [N,L,2H]
        pooled = out.mean(dim=1)           # [N,2H] average over sequence
        logits = self.classifier(pooled)
        return logits


def train_one_epoch(model: nn.Module, loader: DataLoader, device: torch.device, criterion, optimizer) -> Tuple[float, float]:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for inputs, labels in tqdm(loader, desc="Train", leave=False):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        correct += torch.sum(preds == labels).item()
        total += labels.size(0)
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def validate(model: nn.Module, loader: DataLoader, device: torch.device, criterion) -> Tuple[float, float, List[int], List[int]]:
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc="Val", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels).item()
            total += labels.size(0)
            all_labels.extend(labels.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc, all_labels, all_preds


def plot_curves(history, out_dir):
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.legend(); plt.title('Loss');
    plt.subplot(1,2,2)
    plt.plot(history['train_acc'], label='Train Acc')
    plt.plot(history['val_acc'], label='Val Acc')
    plt.legend(); plt.title('Accuracy');
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'training_curves.png'))
    plt.close()


def plot_confusion(cm, class_names, out_dir):
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted'); plt.ylabel('True'); plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'confusion_matrix.png'))
    plt.close()


def save_classification_report(report_dict, out_dir):
    with open(os.path.join(out_dir, 'classification_report.json'), 'w') as f:
        json.dump(report_dict, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Train potato leaf disease classifier")
    parser.add_argument('--data-dir', type=str, required=True, help='Path with class subfolders')
    parser.add_argument('--arch', type=str, default='efficientnet_b0', choices=['efficientnet_b0','resnet50','mobilenet_v2','cnn_lstm','custom'])
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--val-split', type=float, default=0.2)
    parser.add_argument('--image-size', type=int, default=224)
    parser.add_argument('--output-dir', type=str, default='models')
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--no-pretrained', action='store_true')
    parser.add_argument('--notes', type=str, default='', help='Optional notes for ablation/changes to append to ablation_notes.md')
    parser.add_argument('--save-xai-samples', type=int, default=0, help='Number of Grad-CAM samples to save from validation set after training')
    parser.add_argument('--augment-level', type=str, default='strong', choices=['light', 'medium', 'strong', 'extreme'],
                        help='Data augmentation intensity level')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_loader, val_loader = build_dataloaders(args.data_dir, batch_size=args.batch_size, val_split=args.val_split, 
                                                  image_size=args.image_size, augment_level=args.augment_level)
    print(f"Using {args.augment_level} data augmentation")
    class_names = infer_class_names(args.data_dir)
    num_classes = len(class_names)

    model = build_model(args.arch, num_classes, pretrained=not args.no_pretrained).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_val_acc = 0.0
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        print(f"Epoch {epoch}/{args.epochs}")
        start = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, device, criterion, optimizer)
        val_loss, val_acc, labels, preds = validate(model, val_loader, device, criterion)
        history['train_loss'].append(tr_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(tr_acc)
        history['val_acc'].append(val_acc)
        elapsed = time.time() - start
        print(f"Train Loss: {tr_loss:.4f} Acc: {tr_acc:.4f} | Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} (Time {elapsed:.1f}s)")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({'model_state': model.state_dict(), 'class_names': class_names, 'arch': args.arch}, os.path.join(args.output_dir, 'best_model.pt'))
            patience_counter = 0
            print("Saved new best model.")
        else:
            patience_counter += 1
        if patience_counter >= args.patience:
            print("Early stopping triggered.")
            break

    plot_curves(history, args.output_dir)
    # final evaluation for metrics
    cm = confusion_matrix(labels, preds)
    plot_confusion(cm, class_names, args.output_dir)
    report = classification_report(labels, preds, target_names=class_names, output_dict=True)
    save_classification_report(report, args.output_dir)
    print("Classification Report (last epoch):")
    print(classification_report(labels, preds, target_names=class_names))

    with open(os.path.join(args.output_dir, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=2)

    print(f"Best Validation Accuracy: {best_val_acc:.4f}")

    # Optionally save Grad-CAM samples from validation set
    if args.save_xai_samples > 0:
        os.makedirs(os.path.join(args.output_dir, 'xai_samples'), exist_ok=True)
        try:
            target_layer = get_default_target_layer(model, arch=args.arch)
        except Exception:
            target_layer = get_default_target_layer(model, arch='custom')
        cam = GradCAM(model, target_layer)
        saved = 0
        mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1,3,1,1)
        std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1,3,1,1)
        with torch.no_grad():
            for inputs, labels_batch in val_loader:
                inputs = inputs.to(device)
                outputs = model(inputs)
                _, pred_batch = torch.max(outputs, 1)
                for i in range(inputs.size(0)):
                    if saved >= args.save_xai_samples:
                        break
                    img_tensor = inputs[i:i+1]
                    # Grad-CAM requires backward; call generator without no_grad
                    cam.model.zero_grad()
                    heatmap, _ = cam.generate(img_tensor)
                    # denormalize for visualization
                    denorm = (img_tensor * std + mean).clamp(0,1)[0].cpu()
                    pil_img = TF.to_pil_image(denorm)
                    overlay = overlay_heatmap_on_image(pil_img, heatmap)
                    true_label = class_names[labels_batch[i].item()]
                    pred_label = class_names[pred_batch[i].item()]
                    out_name = f"sample_{saved+1:02d}_true-{true_label}_pred-{pred_label}.jpg"
                    overlay.save(os.path.join(args.output_dir, 'xai_samples', out_name))
                    saved += 1
                if saved >= args.save_xai_samples:
                    break
        cam.close()

    # Append key results to a CSV aggregator
    results_csv = os.path.join(args.output_dir, 'results.csv')
    header = 'arch,epochs,batch_size,lr,val_acc,val_loss,macro_precision,macro_recall,macro_f1\n'
    macro = report.get('macro avg', {})
    row = f"{args.arch},{args.epochs},{args.batch_size},{args.lr},{history['val_acc'][-1]:.6f},{history['val_loss'][-1]:.6f},{macro.get('precision',0):.6f},{macro.get('recall',0):.6f},{macro.get('f1-score',0):.6f}\n"
    need_header = not os.path.exists(results_csv)
    with open(results_csv, 'a') as f:
        if need_header:
            f.write(header)
        f.write(row)

    # Append ablation notes if provided
    if args.notes:
        ablation_path = os.path.join(args.output_dir, 'ablation_notes.md')
        with open(ablation_path, 'a') as f:
            f.write(f"\n## Experiment: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"- Arch: {args.arch}\n- Epochs: {args.epochs}\n- Batch: {args.batch_size}\n- LR: {args.lr}\n- Best Val Acc: {best_val_acc:.4f}\n")
            f.write(f"- Notes: {args.notes}\n")

if __name__ == '__main__':
    main()
