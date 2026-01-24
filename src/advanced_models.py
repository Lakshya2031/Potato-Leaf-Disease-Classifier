"""
Advanced Model Architectures with Attention Mechanisms
Includes SE-Net, CBAM, ECA, and improved backbones.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import Optional, Tuple


# ============================================================================
# Squeeze-and-Excitation (SE) Block
# ============================================================================
class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block (Hu et al., 2018)
    Adaptively recalibrates channel-wise feature responses.
    """
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


# ============================================================================
# Efficient Channel Attention (ECA)
# ============================================================================
class ECABlock(nn.Module):
    """
    Efficient Channel Attention (Wang et al., 2020)
    Local cross-channel interaction without dimensionality reduction.
    """
    def __init__(self, channels: int, kernel_size: int = 3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)


# ============================================================================
# CBAM: Convolutional Block Attention Module
# ============================================================================
class ChannelAttention(nn.Module):
    """Channel attention sub-module of CBAM."""
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    """Spatial attention sub-module of CBAM."""
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(x))


class CBAMBlock(nn.Module):
    """
    Convolutional Block Attention Module (Woo et al., 2018)
    Sequential channel and spatial attention.
    """
    def __init__(self, channels: int, reduction: int = 16, kernel_size: int = 7):
        super().__init__()
        self.channel_attention = ChannelAttention(channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * self.channel_attention(x)
        x = x * self.spatial_attention(x)
        return x


# ============================================================================
# Self-Attention / Non-Local Block
# ============================================================================
class NonLocalBlock(nn.Module):
    """
    Non-Local Neural Networks (Wang et al., 2018)
    Captures long-range dependencies via self-attention.
    """
    def __init__(self, channels: int, reduction: int = 2, mode: str = 'embedded'):
        super().__init__()
        self.channels = channels
        self.reduction = reduction
        self.inter_channels = channels // reduction
        self.mode = mode
        
        self.theta = nn.Conv2d(channels, self.inter_channels, 1)
        self.phi = nn.Conv2d(channels, self.inter_channels, 1)
        self.g = nn.Conv2d(channels, self.inter_channels, 1)
        self.out = nn.Sequential(
            nn.Conv2d(self.inter_channels, channels, 1),
            nn.BatchNorm2d(channels)
        )
        
        # Initialize to zero for residual
        nn.init.zeros_(self.out[0].weight)
        nn.init.zeros_(self.out[0].bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.size()
        
        theta = self.theta(x).view(b, self.inter_channels, -1).permute(0, 2, 1)  # [B, HW, C']
        phi = self.phi(x).view(b, self.inter_channels, -1)  # [B, C', HW]
        g = self.g(x).view(b, self.inter_channels, -1).permute(0, 2, 1)  # [B, HW, C']
        
        # Attention: softmax(theta @ phi) @ g
        attention = torch.bmm(theta, phi)  # [B, HW, HW]
        attention = F.softmax(attention, dim=-1)
        
        y = torch.bmm(attention, g).permute(0, 2, 1).view(b, self.inter_channels, h, w)
        y = self.out(y)
        
        return x + y


# ============================================================================
# Improved Custom CNN with Attention
# ============================================================================
class ImprovedCNN(nn.Module):
    """
    Custom CNN with attention blocks, residual connections, and dropout.
    """
    def __init__(self, num_classes: int, attention: str = 'cbam', dropout: float = 0.5):
        super().__init__()
        
        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1)
        )
        
        # Stage 1
        self.stage1 = self._make_stage(64, 128, 2, attention)
        
        # Stage 2
        self.stage2 = self._make_stage(128, 256, 2, attention)
        
        # Stage 3
        self.stage3 = self._make_stage(256, 512, 2, attention)
        
        # Stage 4
        self.stage4 = self._make_stage(512, 1024, 2, attention)
        
        # Head
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(512, num_classes)
        )
        
        self._init_weights()
    
    def _make_stage(self, in_ch: int, out_ch: int, num_blocks: int, attention: str):
        layers = []
        
        # First block with stride 2 for downsampling
        layers.append(self._make_block(in_ch, out_ch, stride=2, attention=attention))
        
        # Remaining blocks
        for _ in range(num_blocks - 1):
            layers.append(self._make_block(out_ch, out_ch, stride=1, attention=attention))
        
        return nn.Sequential(*layers)
    
    def _make_block(self, in_ch: int, out_ch: int, stride: int, attention: str):
        """Residual block with attention."""
        return ResidualBlockWithAttention(in_ch, out_ch, stride, attention)
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.head(x)
        return x


class ResidualBlockWithAttention(nn.Module):
    """Residual block with optional attention mechanism."""
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, attention: str = 'cbam'):
        super().__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch)
        )
        
        # Attention
        if attention == 'cbam':
            self.attention = CBAMBlock(out_ch)
        elif attention == 'se':
            self.attention = SEBlock(out_ch)
        elif attention == 'eca':
            self.attention = ECABlock(out_ch)
        else:
            self.attention = nn.Identity()
        
        # Skip connection
        if stride != 1 or in_ch != out_ch:
            self.skip = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch)
            )
        else:
            self.skip = nn.Identity()
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.skip(x)
        
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.attention(out)
        
        out = out + identity
        out = self.relu(out)
        
        return out


# ============================================================================
# EfficientNet with CBAM
# ============================================================================
class EfficientNetWithAttention(nn.Module):
    """EfficientNet-B0 backbone enhanced with CBAM attention."""
    def __init__(self, num_classes: int, pretrained: bool = True, attention: str = 'cbam', dropout: float = 0.3):
        super().__init__()
        
        # Load pretrained EfficientNet
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        self.backbone = models.efficientnet_b0(weights=weights)
        
        # Get feature dimensions
        self.feature_dim = self.backbone.classifier[-1].in_features
        
        # Add attention after features
        if attention == 'cbam':
            self.attention = CBAMBlock(1280)  # EfficientNet-B0 final channel
        elif attention == 'se':
            self.attention = SEBlock(1280)
        elif attention == 'eca':
            self.attention = ECABlock(1280)
        else:
            self.attention = nn.Identity()
        
        # Replace classifier
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(self.feature_dim, num_classes)
        )
        
        # Store features module for Grad-CAM compatibility
        self.features = self.backbone.features
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone.features(x)
        x = self.attention(x)
        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.backbone.classifier(x)
        return x


# ============================================================================
# Vision Transformer (ViT) Simplified
# ============================================================================
class PatchEmbed(nn.Module):
    """Split image into patches and embed them."""
    def __init__(self, img_size: int = 224, patch_size: int = 16, in_chans: int = 3, embed_dim: int = 768):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)  # [B, C, H', W']
        x = x.flatten(2).transpose(1, 2)  # [B, N, C]
        return x


class TransformerBlock(nn.Module):
    """Standard Transformer block with multi-head self-attention."""
    def __init__(self, dim: int, num_heads: int = 8, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(dropout)
        )
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention with residual
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed)
        x = x + self.dropout(attn_out)
        
        # MLP with residual
        x = x + self.mlp(self.norm2(x))
        
        return x


class SimpleViT(nn.Module):
    """
    Simplified Vision Transformer for image classification.
    """
    def __init__(self, img_size: int = 224, patch_size: int = 16, num_classes: int = 3,
                 dim: int = 384, depth: int = 6, num_heads: int = 6, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        
        self.patch_embed = PatchEmbed(img_size, patch_size, 3, dim)
        num_patches = self.patch_embed.num_patches
        
        # Class token and position embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, dim))
        self.pos_drop = nn.Dropout(dropout)
        
        # Transformer blocks
        self.blocks = nn.Sequential(*[
            TransformerBlock(dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)
        
        # Initialize
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)
        
        # Add class token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # Add position embedding
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        # Transformer blocks
        x = self.blocks(x)
        x = self.norm(x)
        
        # Classification head (use class token)
        x = self.head(x[:, 0])
        
        return x


# ============================================================================
# Hybrid CNN-Transformer
# ============================================================================
class CNNTransformerHybrid(nn.Module):
    """
    Hybrid model combining CNN features with Transformer attention.
    Uses CNN as feature extractor, then applies Transformer.
    """
    def __init__(self, num_classes: int, pretrained: bool = True, 
                 transformer_depth: int = 2, num_heads: int = 8, dropout: float = 0.2):
        super().__init__()
        
        # CNN backbone (EfficientNet-B0 features)
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        backbone = models.efficientnet_b0(weights=weights)
        self.features = backbone.features[:-1]  # Remove last block for more spatial info
        
        # Get spatial dimensions after CNN
        with torch.no_grad():
            sample = torch.zeros(1, 3, 224, 224)
            feat = self.features(sample)
            _, c, h, w = feat.shape
        
        self.feat_dim = c
        self.num_patches = h * w
        
        # Project CNN features
        self.proj = nn.Linear(c, 256)
        
        # Position embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, 256))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, 256))
        
        # Transformer blocks
        self.transformer = nn.Sequential(*[
            TransformerBlock(256, num_heads, 4.0, dropout)
            for _ in range(transformer_depth)
        ])
        
        self.norm = nn.LayerNorm(256)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
        
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        
        # CNN features
        x = self.features(x)  # [B, C, H, W]
        x = x.flatten(2).transpose(1, 2)  # [B, HW, C]
        x = self.proj(x)  # [B, HW, 256]
        
        # Add class token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # Add position embedding
        x = x + self.pos_embed
        
        # Transformer
        x = self.transformer(x)
        x = self.norm(x)
        
        # Classification from class token
        x = self.head(x[:, 0])
        
        return x


# ============================================================================
# Model Factory
# ============================================================================
def build_advanced_model(arch: str, num_classes: int, pretrained: bool = True, 
                         attention: str = 'cbam', dropout: float = 0.3) -> nn.Module:
    """
    Build advanced model architectures.
    
    Args:
        arch: Architecture name
        num_classes: Number of output classes
        pretrained: Use pretrained weights
        attention: Attention type ('cbam', 'se', 'eca', 'none')
        dropout: Dropout rate
    
    Returns:
        PyTorch model
    """
    arch = arch.lower()
    
    if arch == 'efficientnet_attention':
        return EfficientNetWithAttention(num_classes, pretrained, attention, dropout)
    
    elif arch == 'improved_cnn':
        return ImprovedCNN(num_classes, attention, dropout)
    
    elif arch == 'vit':
        return SimpleViT(num_classes=num_classes, dim=384, depth=6, dropout=dropout)
    
    elif arch == 'cnn_transformer':
        return CNNTransformerHybrid(num_classes, pretrained, transformer_depth=2, dropout=dropout)
    
    elif arch == 'efficientnet_b0':
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        return model
    
    elif arch == 'efficientnet_b3':
        weights = models.EfficientNet_B3_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b3(weights=weights)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        return model
    
    elif arch == 'resnet50':
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        model = models.resnet50(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    
    elif arch == 'convnext_tiny':
        weights = models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        model = models.convnext_tiny(weights=weights)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        return model
    
    else:
        raise ValueError(f"Unknown architecture: {arch}")
