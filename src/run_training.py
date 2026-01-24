"""
Quick Training Configurations for Perfect Model Training

This script provides pre-configured training setups for different scenarios.
Choose the configuration that best fits your needs.
"""

import subprocess
import sys
import os

# Base training command
BASE_CMD = [sys.executable, "train_advanced.py"]

# =============================================================================
# CONFIGURATION PRESETS
# =============================================================================

CONFIGS = {
    # -------------------------------------------------------------------------
    # RECOMMENDED: Best balance of performance and training time
    # -------------------------------------------------------------------------
    "recommended": {
        "description": "Recommended configuration - Best balance of accuracy and speed",
        "expected_accuracy": "97-99%",
        "training_time": "~30 minutes on GPU",
        "args": [
            "--data-dir", "data",
            "--arch", "efficientnet_attention",
            "--attention", "cbam",
            "--epochs", "50",
            "--batch-size", "32",
            "--lr", "3e-4",
            "--loss", "focal",
            "--scheduler", "cosine",
            "--warmup-epochs", "5",
            "--use-mixup",
            "--use-cutmix",
            "--mixup-prob", "0.5",
            "--use-ema",
            "--use-tta",
            "--patience", "10",
            "--output-dir", "models/recommended"
        ]
    },
    
    # -------------------------------------------------------------------------
    # QUICK: Fast training for rapid prototyping
    # -------------------------------------------------------------------------
    "quick": {
        "description": "Quick training for rapid prototyping",
        "expected_accuracy": "94-96%",
        "training_time": "~10 minutes on GPU",
        "args": [
            "--data-dir", "data",
            "--arch", "efficientnet_b0",
            "--epochs", "20",
            "--batch-size", "64",
            "--lr", "1e-3",
            "--loss", "label_smooth",
            "--scheduler", "cosine",
            "--warmup-epochs", "2",
            "--patience", "5",
            "--output-dir", "models/quick"
        ]
    },
    
    # -------------------------------------------------------------------------
    # MAXIMUM: Maximum accuracy (longer training)
    # -------------------------------------------------------------------------
    "maximum": {
        "description": "Maximum accuracy configuration - Longer training",
        "expected_accuracy": "98-99.5%",
        "training_time": "~2 hours on GPU",
        "args": [
            "--data-dir", "data",
            "--arch", "efficientnet_attention",
            "--attention", "cbam",
            "--epochs", "100",
            "--batch-size", "16",
            "--lr", "1e-4",
            "--weight-decay", "1e-4",
            "--loss", "combined",
            "--scheduler", "cosine",
            "--warmup-epochs", "10",
            "--use-mixup",
            "--use-cutmix",
            "--mixup-prob", "0.6",
            "--use-ema",
            "--ema-decay", "0.9999",
            "--use-swa",
            "--swa-start", "60",
            "--use-tta",
            "--tta-augments", "7",
            "--grad-clip", "1.0",
            "--patience", "20",
            "--save-xai-samples", "20",
            "--output-dir", "models/maximum"
        ]
    },
    
    # -------------------------------------------------------------------------
    # LIGHTWEIGHT: For deployment on edge devices
    # -------------------------------------------------------------------------
    "lightweight": {
        "description": "Lightweight model for edge deployment",
        "expected_accuracy": "93-95%",
        "training_time": "~15 minutes on GPU",
        "args": [
            "--data-dir", "data",
            "--arch", "improved_cnn",
            "--attention", "eca",
            "--dropout", "0.4",
            "--epochs", "40",
            "--batch-size", "32",
            "--lr", "1e-3",
            "--loss", "focal",
            "--scheduler", "cosine",
            "--use-ema",
            "--patience", "10",
            "--output-dir", "models/lightweight"
        ]
    },
    
    # -------------------------------------------------------------------------
    # TRANSFORMER: Vision Transformer based
    # -------------------------------------------------------------------------
    "transformer": {
        "description": "Vision Transformer hybrid model",
        "expected_accuracy": "96-98%",
        "training_time": "~45 minutes on GPU",
        "args": [
            "--data-dir", "data",
            "--arch", "cnn_transformer",
            "--epochs", "60",
            "--batch-size", "24",
            "--lr", "5e-5",
            "--loss", "label_smooth",
            "--label-smoothing", "0.1",
            "--scheduler", "cosine",
            "--warmup-epochs", "10",
            "--use-mixup",
            "--mixup-alpha", "0.4",
            "--use-ema",
            "--grad-clip", "0.5",
            "--patience", "15",
            "--output-dir", "models/transformer"
        ]
    },
    
    # -------------------------------------------------------------------------
    # ROBUST: Focus on generalization
    # -------------------------------------------------------------------------
    "robust": {
        "description": "Focus on robustness and generalization",
        "expected_accuracy": "95-97%",
        "training_time": "~1 hour on GPU",
        "args": [
            "--data-dir", "data",
            "--arch", "efficientnet_b3",
            "--epochs", "60",
            "--batch-size", "24",
            "--lr", "1e-4",
            "--loss", "focal",
            "--focal-gamma", "2.5",
            "--scheduler", "cosine",
            "--warmup-epochs", "8",
            "--use-mixup",
            "--use-cutmix",
            "--mixup-prob", "0.7",
            "--mixup-alpha", "0.4",
            "--cutmix-alpha", "1.0",
            "--use-ema",
            "--use-sam",
            "--grad-clip", "1.0",
            "--dropout", "0.4",
            "--patience", "15",
            "--output-dir", "models/robust"
        ]
    }
}


def print_configs():
    """Print all available configurations."""
    print("\n" + "="*70)
    print("AVAILABLE TRAINING CONFIGURATIONS")
    print("="*70)
    
    for name, config in CONFIGS.items():
        print(f"\n[{name.upper()}]")
        print(f"  Description: {config['description']}")
        print(f"  Expected Accuracy: {config['expected_accuracy']}")
        print(f"  Training Time: {config['training_time']}")
    
    print("\n" + "="*70)
    print("Usage: python run_training.py <config_name>")
    print("Example: python run_training.py recommended")
    print("="*70 + "\n")


def run_training(config_name: str):
    """Run training with specified configuration."""
    if config_name not in CONFIGS:
        print(f"Error: Unknown configuration '{config_name}'")
        print_configs()
        return
    
    config = CONFIGS[config_name]
    
    print("\n" + "="*70)
    print(f"STARTING TRAINING: {config_name.upper()}")
    print("="*70)
    print(f"Description: {config['description']}")
    print(f"Expected Accuracy: {config['expected_accuracy']}")
    print(f"Expected Time: {config['training_time']}")
    print("="*70 + "\n")
    
    # Build command
    cmd = BASE_CMD + config['args']
    
    # Run training
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    subprocess.run(cmd)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print_configs()
        config = input("Enter configuration name (or 'recommended'): ").strip()
        if not config:
            config = "recommended"
    else:
        config = sys.argv[1].lower()
    
    if config == "list":
        print_configs()
    else:
        run_training(config)
