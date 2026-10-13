"""Model loading, pre-processing, prediction and Grad-CAM for the Streamlit app.

The pre-processing is identical to the training notebook:
    any image -> RGB -> resize to IMG_SIZE x IMG_SIZE (area interpolation) -> 0-1 -> ImageNet standardisation
"""
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models

CLASSES = ["glioma", "meningioma", "no_tumor", "pituitary"]


class CustomCNN(nn.Module):
    """Same architecture as in the notebook (Model 1)."""

    def __init__(self, width=16, dropout=0.4, n_classes=len(CLASSES)):
        super().__init__()

        def conv_bn(i, o):
            return [nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True)]

        layers = [nn.Conv2d(3, width, 5, stride=2, padding=2, bias=False), nn.BatchNorm2d(width), nn.ReLU(inplace=True)]
        c_in = width
        for mult in (2, 4, 8, 16):
            c_out = width * mult
            layers += conv_bn(c_in, c_out) + conv_bn(c_out, c_out) + [nn.MaxPool2d(2), nn.Dropout2d(0.1)]
            c_in = c_out
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(dropout), nn.Linear(c_in, n_classes))

    def forward(self, x):
        return self.head(self.features(x))


def build_architecture(arch: str) -> nn.Module:
    """Architecture only (no ImageNet download): the trained weights are loaded afterwards."""
    if arch == "custom_cnn":
        return CustomCNN()
    ctor = {"mobilenet_v2": models.mobilenet_v2, "efficientnet_b0": models.efficientnet_b0,
            "resnet50": models.resnet50}[arch]
    model = ctor(weights=None)
    if arch == "resnet50":
        model.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(model.fc.in_features, len(CLASSES)))
    else:
        model.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(model.classifier[1].in_features, len(CLASSES)))
    return model


def load_bundle(path):
    """Load models/best_model.pt (weights + the settings needed to use them)."""
    info = torch.load(path, map_location="cpu")
    model = build_architecture(info["arch"])
    model.load_state_dict(info["state_dict"])
    model.eval()
    return model, {k: v for k, v in info.items() if k != "state_dict"}


def crop_head(rgb, margin=0.04):
    """Square crop around the non-black region (the head), with a small margin. Same as in the notebook."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    mask = cv2.morphologyEx((gray > 25).astype(np.uint8), cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    ys, xs = np.where(mask > 0)
    if len(ys) < 100:                                        # blank or unusual image: keep it unchanged
        return rgb
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    half = int(max(y1 - y0, x1 - x0) * (1 + 2 * margin)) // 2
    cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
    padded = cv2.copyMakeBorder(rgb, half, half, half, half, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    return padded[cy: cy + 2 * half, cx: cx + 2 * half]


def normalise_intensity(rgb):
    """Map the 99th percentile of head pixels to grey level 200 (removes exposure differences)."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    head = gray[gray > 25]
    p99 = np.percentile(head, 99) if head.size else 255.0
    return np.clip(rgb.astype(np.float32) * (200.0 / max(p99, 1.0)), 0, 255).astype(np.uint8)


def preprocess(pil_image: Image.Image, info: dict):
    """Return (normalised tensor (1,3,S,S), resized uint8 RGB array for display).
    info['preprocess'] is 'plain' (resize only) or 'headcrop' (crop to head + brightness normalisation)."""
    size = info["img_size"]
    variant = info.get("preprocess", "plain")
    rgb = np.array(pil_image.convert("RGB"))
    if variant == "headcrop":
        rgb = crop_head(rgb)
    rgb = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)
    if variant == "headcrop":
        rgb = normalise_intensity(rgb)
    x = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
    mean = torch.tensor(info["mean"]).view(3, 1, 1)
    std = torch.tensor(info["std"]).view(3, 1, 1)
    return ((x - mean) / std).unsqueeze(0), rgb


@torch.no_grad()
def predict_proba(model, x):
    return torch.softmax(model(x), dim=1)[0].numpy()


def _target_layer(model, arch):
    if arch == "custom_cnn":
        return model.features[-3]          # last ReLU before the final max-pool
    if arch == "resnet50":
        return model.layer4[-1]
    return model.features[-1]


def grad_cam(model, arch, x, class_idx=None):
    """Grad-CAM heat-map (values 0..1, same size as the input) for the predicted (or given) class."""
    layer = _target_layer(model, arch)
    store = {}

    def hook(module, inp, out):
        store["act"] = out
        out.register_hook(lambda g: store.__setitem__("grad", g))

    handle = layer.register_forward_hook(hook)
    model.zero_grad()
    logits = model(x)
    handle.remove()
    c = int(logits.argmax(1)) if class_idx is None else class_idx
    logits[0, c].backward()
    w = store["grad"].mean(dim=(2, 3), keepdim=True)
    cam = F.relu((w * store["act"]).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0].detach().numpy()
    return (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)


def overlay(rgb_uint8, cam):
    heat = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)[..., ::-1]
    return (0.55 * rgb_uint8 + 0.45 * heat).astype(np.uint8)
