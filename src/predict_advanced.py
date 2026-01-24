"""
Advanced Prediction Script with Ensemble and TTA Support

Features:
- Single model prediction
- Ensemble prediction
- Test Time Augmentation (TTA)
- Grad-CAM visualization
- Batch prediction
- Confidence calibration
- JSON output support
"""

import argparse
import os
import json
from typing import List, Dict, Tuple, Optional
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np

# Local imports
from advanced_models import build_advanced_model
from train import build_model
from augmentations import TestTimeAugmentation
from xai import GradCAM, get_default_target_layer, overlay_heatmap_on_image


# ============================================================================
# Model Loading
# ============================================================================
def load_model(model_path: str, device: torch.device) -> Tuple[nn.Module, List[str], str]:
    """Load a trained model from checkpoint."""
    ckpt = torch.load(model_path, map_location=device)
    
    arch = ckpt.get('arch', 'efficientnet_b0')
    class_names = ckpt['class_names']
    num_classes = len(class_names)
    
    # Try advanced model first, fall back to basic
    try:
        model = build_advanced_model(arch, num_classes, pretrained=False)
    except:
        model = build_model(arch, num_classes, pretrained=False)
    
    model.load_state_dict(ckpt['model_state'])
    model.to(device)
    model.eval()
    
    return model, class_names, arch


def load_ensemble(model_paths: List[str], device: torch.device) -> Tuple[List[nn.Module], List[str]]:
    """Load multiple models for ensemble prediction."""
    models = []
    class_names = None
    
    for path in model_paths:
        model, names, _ = load_model(path, device)
        models.append(model)
        if class_names is None:
            class_names = names
    
    return models, class_names


# ============================================================================
# Image Preprocessing
# ============================================================================
def preprocess_image(
    image_path: str,
    image_size: int = 224
) -> Tuple[torch.Tensor, Image.Image]:
    """Preprocess an image for prediction."""
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    img = Image.open(image_path).convert('RGB')
    tensor = transform(img).unsqueeze(0)
    
    return tensor, img


def preprocess_batch(
    image_paths: List[str],
    image_size: int = 224
) -> Tuple[torch.Tensor, List[Image.Image]]:
    """Preprocess a batch of images."""
    tensors = []
    images = []
    
    for path in image_paths:
        tensor, img = preprocess_image(path, image_size)
        tensors.append(tensor)
        images.append(img)
    
    return torch.cat(tensors, dim=0), images


# ============================================================================
# Prediction Functions
# ============================================================================
@torch.no_grad()
def predict_single(
    model: nn.Module,
    tensor: torch.Tensor,
    device: torch.device,
    class_names: List[str],
    use_tta: bool = False,
    tta_augments: int = 5
) -> Dict:
    """Make prediction for a single image."""
    tensor = tensor.to(device)
    
    if use_tta:
        tta = TestTimeAugmentation(model, device, tta_augments)
        probs = tta.predict(tensor)
    else:
        model.eval()
        outputs = model(tensor)
        probs = F.softmax(outputs, dim=1)
    
    probs = probs.cpu().numpy()[0]
    pred_idx = np.argmax(probs)
    confidence = probs[pred_idx]
    
    # Calculate entropy for uncertainty
    entropy = -np.sum(probs * np.log(probs + 1e-10))
    
    return {
        'prediction': class_names[pred_idx],
        'prediction_index': int(pred_idx),
        'confidence': float(confidence),
        'entropy': float(entropy),
        'probabilities': {name: float(prob) for name, prob in zip(class_names, probs)},
        'all_probs': probs.tolist()
    }


@torch.no_grad()
def predict_ensemble(
    models: List[nn.Module],
    tensor: torch.Tensor,
    device: torch.device,
    class_names: List[str],
    weights: Optional[List[float]] = None,
    use_tta: bool = False
) -> Dict:
    """Make ensemble prediction."""
    tensor = tensor.to(device)
    
    if weights is None:
        weights = [1.0 / len(models)] * len(models)
    
    all_probs = []
    
    for model, weight in zip(models, weights):
        model.eval()
        
        if use_tta:
            tta = TestTimeAugmentation(model, device, num_augments=5)
            probs = tta.predict(tensor)
        else:
            outputs = model(tensor)
            probs = F.softmax(outputs, dim=1)
        
        all_probs.append(probs.cpu().numpy() * weight)
    
    # Weighted average
    ensemble_probs = np.sum(all_probs, axis=0)[0]
    pred_idx = np.argmax(ensemble_probs)
    confidence = ensemble_probs[pred_idx]
    
    # Calculate model agreement
    individual_preds = [np.argmax(p[0]) for p in all_probs]
    agreement = sum(p == pred_idx for p in individual_preds) / len(models)
    
    return {
        'prediction': class_names[pred_idx],
        'prediction_index': int(pred_idx),
        'confidence': float(confidence),
        'model_agreement': float(agreement),
        'probabilities': {name: float(prob) for name, prob in zip(class_names, ensemble_probs)},
        'individual_predictions': [class_names[p] for p in individual_preds]
    }


# ============================================================================
# Explainability
# ============================================================================
def generate_gradcam(
    model: nn.Module,
    tensor: torch.Tensor,
    original_image: Image.Image,
    device: torch.device,
    arch: str,
    save_path: Optional[str] = None
) -> Image.Image:
    """Generate Grad-CAM visualization."""
    tensor = tensor.to(device)
    
    try:
        target_layer = get_default_target_layer(model, arch=arch)
    except:
        # Fallback: find last Conv2d
        for m in reversed(list(model.modules())):
            if isinstance(m, nn.Conv2d):
                target_layer = m
                break
    
    cam = GradCAM(model, target_layer)
    heatmap, _ = cam.generate(tensor)
    cam.close()
    
    overlay = overlay_heatmap_on_image(original_image, heatmap)
    
    if save_path:
        overlay.save(save_path)
    
    return overlay


# ============================================================================
# Batch Processing
# ============================================================================
def predict_directory(
    model: nn.Module,
    directory: str,
    device: torch.device,
    class_names: List[str],
    image_size: int = 224,
    use_tta: bool = False,
    extensions: Tuple = ('.jpg', '.jpeg', '.png', '.bmp')
) -> List[Dict]:
    """Predict all images in a directory."""
    results = []
    
    image_files = [
        f for f in Path(directory).iterdir()
        if f.suffix.lower() in extensions
    ]
    
    print(f"Found {len(image_files)} images")
    
    for img_path in image_files:
        tensor, _ = preprocess_image(str(img_path), image_size)
        result = predict_single(model, tensor, device, class_names, use_tta)
        result['file'] = str(img_path.name)
        results.append(result)
        
        print(f"{img_path.name}: {result['prediction']} ({result['confidence']:.2%})")
    
    return results


# ============================================================================
# Temperature Scaling (Confidence Calibration)
# ============================================================================
class TemperatureScaling:
    """Apply temperature scaling for better calibrated probabilities."""
    
    def __init__(self, temperature: float = 1.0):
        self.temperature = temperature
    
    def calibrate(self, logits: torch.Tensor) -> torch.Tensor:
        return F.softmax(logits / self.temperature, dim=1)


# ============================================================================
# Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description='Advanced Potato Leaf Disease Prediction')
    
    # Model
    parser.add_argument('--model-path', type=str, help='Path to model checkpoint')
    parser.add_argument('--ensemble-paths', type=str, nargs='+', help='Paths to multiple models for ensemble')
    parser.add_argument('--ensemble-weights', type=float, nargs='+', help='Weights for ensemble models')
    
    # Input
    parser.add_argument('--image', type=str, help='Path to a single image')
    parser.add_argument('--directory', type=str, help='Path to directory of images')
    parser.add_argument('--image-size', type=int, default=224)
    
    # Prediction options
    parser.add_argument('--use-tta', action='store_true', help='Use Test Time Augmentation')
    parser.add_argument('--tta-augments', type=int, default=5)
    
    # Explainability
    parser.add_argument('--explain', action='store_true', help='Generate Grad-CAM visualization')
    parser.add_argument('--save-cam', type=str, help='Path to save Grad-CAM image')
    
    # Output
    parser.add_argument('--output-json', type=str, help='Save results to JSON file')
    parser.add_argument('--quiet', action='store_true', help='Minimal output')
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model(s)
    if args.ensemble_paths:
        models, class_names = load_ensemble(args.ensemble_paths, device)
        arch = 'ensemble'
        is_ensemble = True
    else:
        model, class_names, arch = load_model(args.model_path, device)
        models = [model]
        is_ensemble = False
    
    results = []
    
    if args.image:
        # Single image prediction
        tensor, original_img = preprocess_image(args.image, args.image_size)
        
        if is_ensemble:
            result = predict_ensemble(models, tensor, device, class_names, 
                                     args.ensemble_weights, args.use_tta)
        else:
            result = predict_single(models[0], tensor, device, class_names, 
                                   args.use_tta, args.tta_augments)
        
        result['file'] = args.image
        results.append(result)
        
        if not args.quiet:
            print(f"\n{'='*50}")
            print(f"Prediction: {result['prediction']}")
            print(f"Confidence: {result['confidence']:.2%}")
            
            if 'model_agreement' in result:
                print(f"Model Agreement: {result['model_agreement']:.2%}")
            
            print(f"\nClass Probabilities:")
            for name, prob in result['probabilities'].items():
                bar = '█' * int(prob * 30)
                print(f"  {name:15s}: {prob:6.2%} {bar}")
            print(f"{'='*50}\n")
        
        # Generate Grad-CAM if requested
        if args.explain and not is_ensemble:
            save_path = args.save_cam or args.image.replace('.', '_gradcam.')
            overlay = generate_gradcam(models[0], tensor, original_img, device, arch, save_path)
            print(f"Saved Grad-CAM to: {save_path}")
    
    elif args.directory:
        # Batch prediction
        results = predict_directory(
            models[0] if not is_ensemble else models,
            args.directory, device, class_names,
            args.image_size, args.use_tta
        )
        
        # Summary statistics
        if not args.quiet:
            print(f"\n{'='*50}")
            print("Summary:")
            for class_name in class_names:
                count = sum(1 for r in results if r['prediction'] == class_name)
                print(f"  {class_name}: {count} images")
            avg_conf = np.mean([r['confidence'] for r in results])
            print(f"Average Confidence: {avg_conf:.2%}")
            print(f"{'='*50}\n")
    
    # Save results to JSON
    if args.output_json and results:
        with open(args.output_json, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Saved results to: {args.output_json}")


if __name__ == '__main__':
    main()
