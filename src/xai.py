import torch
import torch.nn.functional as F
from typing import Tuple, Callable, Optional
import numpy as np
from PIL import Image


def _to_numpy(img_tensor):
    x = img_tensor.detach().cpu().numpy()
    return x


def overlay_heatmap_on_image(img: Image.Image, heatmap: np.ndarray, alpha: float = 0.4) -> Image.Image:
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    heatmap_img = Image.fromarray(np.uint8(255 * heatmap))
    heatmap_img = heatmap_img.resize(img.size, resample=Image.BILINEAR)
    heatmap_color = np.array(heatmap_img.convert('L'))
    # apply simple colormap (jet-like)
    cmap = np.zeros((heatmap_color.shape[0], heatmap_color.shape[1], 3), dtype=np.uint8)
    cmap[..., 0] = np.clip(255 * (heatmap_color / 255.0 - 0.5) * 2, 0, 255)  # R
    cmap[..., 2] = np.clip(255 * (1 - heatmap_color / 255.0) * 2, 0, 255)    # B
    cmap[..., 1] = 255 - np.maximum(cmap[..., 0], cmap[..., 2])              # G
    cmap_img = Image.fromarray(cmap).convert('RGBA')
    base = img.convert('RGBA')
    blended = Image.blend(base, cmap_img, alpha=alpha)
    return blended.convert('RGB')


class GradCAM:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self.hook_a = target_layer.register_forward_hook(self._forward_hook)
        self.hook_g = target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, inp, out):
        self.activations = out.detach()

    def _backward_hook(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def generate(self, input_tensor: torch.Tensor, target_index: Optional[int] = None) -> Tuple[np.ndarray, int]:
        self.model.zero_grad()
        out = self.model(input_tensor)
        if target_index is None:
            target_index = out.argmax(dim=1).item()
        loss = out[:, target_index].sum()
        loss.backward()
        grads = self.gradients  # [N,C,H,W]
        acts = self.activations  # [N,C,H,W]
        weights = grads.mean(dim=(2,3), keepdim=True)  # [N,C,1,1]
        cam = (weights * acts).sum(dim=1, keepdim=True)  # [N,1,H,W]
        cam = F.relu(cam)
        # Normalize per-sample
        cam_np = _to_numpy(cam[0,0])
        cam_np -= cam_np.min()
        cam_np /= (cam_np.max() + 1e-8)
        return cam_np, target_index

    def close(self):
        self.hook_a.remove()
        self.hook_g.remove()


def get_default_target_layer(model: torch.nn.Module, arch: str) -> torch.nn.Module:
    arch = arch.lower()
    if arch.startswith('efficientnet'):
        # last conv block before pooling
        return model.features[-1][0]
    if arch == 'resnet50':
        return model.layer4[-1].conv3 if hasattr(model.layer4[-1], 'conv3') else model.layer4[-1]
    if arch == 'mobilenet_v2':
        return model.features[-1][0]
    if arch in ('custom', 'cnn_lstm'):
        # best-effort: last conv in features
        return model.features[-1]
    # fallback: try to find any Conv2d
    for m in reversed(list(model.modules())):
        if isinstance(m, torch.nn.Conv2d):
            return m
    raise ValueError('No suitable target layer found for Grad-CAM.')


def vanilla_saliency(model: torch.nn.Module, input_tensor: torch.Tensor, target_index: Optional[int] = None) -> Tuple[np.ndarray, int]:
    model.zero_grad()
    input_tensor.requires_grad_(True)
    out = model(input_tensor)
    if target_index is None:
        target_index = out.argmax(dim=1).item()
    loss = out[:, target_index].sum()
    loss.backward()
    sal = input_tensor.grad.abs().max(dim=1)[0]  # [N,H,W]
    sal = sal[0]
    sal = sal - sal.min()
    sal = sal / (sal.max() + 1e-8)
    return _to_numpy(sal), target_index


def integrated_gradients(model: torch.nn.Module, input_tensor: torch.Tensor, baselines: Optional[torch.Tensor] = None, steps: int = 32, target_index: Optional[int] = None) -> Tuple[np.ndarray, int]:
    model.zero_grad()
    if baselines is None:
        baselines = torch.zeros_like(input_tensor)
    if target_index is None:
        with torch.no_grad():
            out = model(input_tensor)
            target_index = out.argmax(dim=1).item()
    scaled_inputs = [baselines + (float(i)/steps) * (input_tensor - baselines) for i in range(0, steps+1)]
    grads = []
    for x in scaled_inputs:
        x.requires_grad_(True)
        out = model(x)
        loss = out[:, target_index].sum()
        model.zero_grad()
        loss.backward()
        grads.append(x.grad.detach())
    avg_grad = torch.stack(grads, dim=0).mean(dim=0)
    ig = (input_tensor - baselines) * avg_grad
    ig = ig.abs().max(dim=1)[0]
    ig = ig[0]
    ig = ig - ig.min()
    ig = ig / (ig.max() + 1e-8)
    return _to_numpy(ig), target_index
