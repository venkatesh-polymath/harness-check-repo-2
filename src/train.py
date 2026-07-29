"""
Training utilities:
  - ERM supervised training
  - SimCLR self-supervised pretraining
  - Linear probe training + evaluation
  - Evaluation: accuracy, spurious-reliance gap, WGA
"""

import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as T

from models import SimCLRBackbone, LinearProbe, nt_xent_loss, get_resnet18


# ──────────────────────────────────────────────
# Seed
# ──────────────────────────────────────────────

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ──────────────────────────────────────────────
# SimCLR augmentation
# ──────────────────────────────────────────────

def get_simclr_transform():
    """
    SimCLR augmentation for CIFAR-32.
    Color jitter + grayscale intentionally covers background-color variation.
    Patch (top-left) is NOT masked — it survives most crop/flip combos.
    """
    return T.Compose([
        T.RandomResizedCrop(32, scale=(0.2, 1.0)),
        T.RandomHorizontalFlip(),
        T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        T.RandomGrayscale(p=0.2),
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])


class TwoViewDataset(torch.utils.data.Dataset):
    """Wraps a TogglableCIFAR10 to return two augmented views per image."""

    def __init__(self, base_dataset, transform):
        self.base = base_dataset
        self.transform = transform
        # Override the base dataset's transform to return PIL images
        self.base.transform = None

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        # Get raw PIL image from base dataset
        from PIL import Image
        img_np = self.base.images[idx].copy()

        # Apply spurious signals
        from dataset import BG_HUES, PATCH_HUES, apply_background_tint, apply_color_patch
        if self.base.bg_enabled:
            img_np = apply_background_tint(img_np, BG_HUES[self.base.bg_class[idx]])
        if self.base.patch_enabled:
            img_np = apply_color_patch(img_np, PATCH_HUES[self.base.patch_class[idx]])

        img_pil = Image.fromarray(img_np)
        v1 = self.transform(img_pil)
        v2 = self.transform(img_pil)
        return v1, v2


# ──────────────────────────────────────────────
# ERM Training
# ──────────────────────────────────────────────

def train_erm(model, loader, optimizer, device, epochs: int):
    """Standard supervised training."""
    model.train()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    for ep in range(epochs):
        total_loss = 0.0
        n = 0
        for batch in loader:
            x, y = batch[0], batch[1]
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(y)
            n += len(y)
        scheduler.step()
        if (ep + 1) % max(1, epochs // 5) == 0:
            print(f"  ERM ep {ep+1}/{epochs}  loss={total_loss/n:.4f}")
    return model


# ──────────────────────────────────────────────
# SimCLR Pretraining
# ──────────────────────────────────────────────

def train_simclr(backbone, loader, optimizer, device, epochs: int, temperature: float = 0.5):
    """Self-supervised SimCLR pretraining."""
    backbone.train()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    for ep in range(epochs):
        total_loss = 0.0
        n = 0
        for v1, v2 in loader:
            v1, v2 = v1.to(device), v2.to(device)
            optimizer.zero_grad()
            _, z1 = backbone(v1)
            _, z2 = backbone(v2)
            loss = nt_xent_loss(z1, z2, temperature=temperature)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(v1)
            n += len(v1)
        scheduler.step()
        if (ep + 1) % max(1, epochs // 5) == 0:
            print(f"  SimCLR ep {ep+1}/{epochs}  loss={total_loss/n:.4f}")
    return backbone


# ──────────────────────────────────────────────
# Linear Probe Training
# ──────────────────────────────────────────────

def extract_features(backbone, loader, device):
    """Extract frozen backbone features."""
    backbone.eval()
    all_feats, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            x, y = batch[0], batch[1]
            x = x.to(device)
            if isinstance(backbone, SimCLRBackbone):
                h = backbone.encode(x)
            else:
                # ERM ResNet: remove the final FC to get 512-d features
                h = _forward_features(backbone, x)
            all_feats.append(h.cpu())
            all_labels.append(y if isinstance(y, torch.Tensor) else torch.tensor(y))
    return torch.cat(all_feats), torch.cat(all_labels)


def _forward_features(resnet, x):
    """Forward pass of ResNet-18 without the final FC layer."""
    m = resnet
    x = m.conv1(x)
    x = m.bn1(x)
    x = m.relu(x)
    x = m.maxpool(x)
    x = m.layer1(x)
    x = m.layer2(x)
    x = m.layer3(x)
    x = m.layer4(x)
    x = m.avgpool(x)
    x = torch.flatten(x, 1)
    return x


def train_linear_probe(probe, features, labels, device, epochs: int = 50, lr: float = 0.01):
    """Train a linear probe on pre-extracted features."""
    ds = torch.utils.data.TensorDataset(features, labels)
    loader = DataLoader(ds, batch_size=256, shuffle=True)
    probe.to(device)
    optimizer = torch.optim.SGD(probe.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    probe.train()
    for ep in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = probe(xb)
            loss = F.cross_entropy(logits, yb)
            loss.backward()
            optimizer.step()
        scheduler.step()
    return probe


# ──────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────

@torch.no_grad()
def evaluate(model_or_probe, loader, device, is_probe: bool = False, features=None, labels=None):
    """
    Returns (accuracy, per_class_accuracy).
    If is_probe=True, expects (features, labels) instead of a dataloader.
    """
    if is_probe:
        assert features is not None and labels is not None
        model_or_probe.eval()
        model_or_probe.to(device)
        ds = torch.utils.data.TensorDataset(features, labels)
        loader2 = DataLoader(ds, batch_size=512, shuffle=False)
        correct, total = 0, 0
        per_class_correct = torch.zeros(10)
        per_class_total = torch.zeros(10)
        for xb, yb in loader2:
            xb, yb = xb.to(device), yb.to(device)
            preds = model_or_probe(xb).argmax(1)
            correct += (preds == yb).sum().item()
            total += len(yb)
            for c in range(10):
                mask = (yb == c)
                per_class_correct[c] += (preds[mask] == yb[mask]).sum().item()
                per_class_total[c] += mask.sum().item()
        acc = correct / total
        per_class_acc = (per_class_correct / per_class_total.clamp(min=1)).tolist()
        return acc, per_class_acc

    # Regular model + dataloader
    if hasattr(model_or_probe, 'eval'):
        model_or_probe.eval()
    correct, total = 0, 0
    per_class_correct = torch.zeros(10)
    per_class_total = torch.zeros(10)
    for batch in loader:
        x, y = batch[0], batch[1]
        x, y = x.to(device), y.to(device)
        preds = model_or_probe(x).argmax(1)
        correct += (preds == y).sum().item()
        total += len(y)
        for c in range(10):
            mask = (y == c)
            per_class_correct[c] += (preds[mask] == y[mask]).sum().item()
            per_class_total[c] += mask.sum().item()
    acc = correct / total
    per_class_acc = (per_class_correct / per_class_total.clamp(min=1)).tolist()
    return acc, per_class_acc


@torch.no_grad()
def evaluate_with_groups(model_or_probe, loader_or_features, device,
                         is_probe: bool = False, features=None, labels=None, groups=None):
    """
    Evaluate accuracy per group (group = bg_matches*2 + patch_matches).
    Returns: (overall_acc, group_accs, wga)
    """
    if is_probe:
        assert features is not None and labels is not None and groups is not None
        model_or_probe.eval()
        model_or_probe.to(device)
        ds = torch.utils.data.TensorDataset(features, labels, groups)
        loader = DataLoader(ds, batch_size=512, shuffle=False)
        group_correct = torch.zeros(4)
        group_total = torch.zeros(4)
        total_correct = 0
        total_n = 0
        for xb, yb, gb in loader:
            xb, yb, gb = xb.to(device), yb.to(device), gb.to(device)
            preds = model_or_probe(xb).argmax(1)
            is_correct = (preds == yb)
            total_correct += is_correct.sum().item()
            total_n += len(yb)
            for g in range(4):
                gm = (gb == g)
                group_correct[g] += is_correct[gm].sum().item()
                group_total[g] += gm.sum().item()
    else:
        # loader returns (x, y, group)
        model_or_probe.eval()
        group_correct = torch.zeros(4)
        group_total = torch.zeros(4)
        total_correct = 0
        total_n = 0
        for batch in loader_or_features:
            x, y, g = batch[0], batch[1], batch[2]
            x, y = x.to(device), y.to(device)
            preds = model_or_probe(x).argmax(1)
            is_correct = (preds == y)
            total_correct += is_correct.sum().item()
            total_n += len(y)
            g = g.to(device)
            for grp in range(4):
                gm = (g == grp)
                group_correct[grp] += is_correct[gm].sum().item()
                group_total[grp] += gm.sum().item()

    group_accs = {}
    for g in range(4):
        if group_total[g] > 0:
            group_accs[g] = (group_correct[g] / group_total[g]).item()
        else:
            group_accs[g] = float('nan')

    overall_acc = total_correct / total_n
    valid_accs = [v for v in group_accs.values() if not np.isnan(v)]
    wga = min(valid_accs) if valid_accs else float('nan')
    return overall_acc, group_accs, wga
