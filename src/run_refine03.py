"""
refine-03: Control for color-augmentation confound in SimCLR WGA gain.

Reviewer concern: SimCLR's color-jitter/grayscale augmentations MECHANICALLY
DELETE the injected HSV hue signal, so the WGA gain may not reflect a real
invariance mechanism — just signal deletion.

Fix: Add SimCLR-NO-COLOR-AUG arm (crop+flip only, no color-jitter/grayscale).

Three arms:
  1. ERM                 — supervised baseline
  2. SimCLR-full-aug     — SimCLR with color-jitter+grayscale (same as refine-02)
  3. SimCLR-no-color-aug — SimCLR with crop+flip ONLY (cannot delete HSV signal)

If WGA gain persists in arm 3 → real invariance mechanism
If WGA gain vanishes in arm 3 → augmentation deletion was the explanation

Also reports HSV-signal decodability per arm (linear probe on backbone features
to predict bg_class, measuring how much HSV color info is retained).
"""

import sys
import os
import json
import math
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(__file__))

from dataset import (
    TogglableCIFAR10, TRAIN_TRANSFORM, TEST_TRANSFORM, get_hsv_histogram,
    apply_background_tint, apply_color_patch, BG_HUES, PATCH_HUES
)
from models import get_resnet18, SimCLRBackbone, LinearProbe, nt_xent_loss
from sanity_gates import (
    gate_fixed_seed, gate_loss_at_init, gate_dummy_classifier, gate_overfit_one_batch,
    set_seed
)
from train import (
    train_erm, train_simclr, extract_features, train_linear_probe,
    evaluate, get_simclr_transform, TwoViewDataset, set_seed
)
import torchvision.transforms as T

CIFAR_ROOT = "/opt/datasets"
WEIGHTS_DIR = "/workspace/_weights/refine-03"
RESULTS_DIR = "/workspace/results/refine-03"

# ── Probe config ──
SUBSET_TRAIN = 5000
SUBSET_TEST  = 1000
ERM_EPOCHS    = 20
SIMCLR_EPOCHS = 20
PROBE_EPOCHS  = 20
SEEDS = [0, 1, 2]
BATCH_SIZE = 128
LR_ERM     = 0.1
LR_SIMCLR  = 0.03
LR_PROBE   = 0.1
TEMPERATURE = 0.5
NUM_CLASSES = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Device: {DEVICE}")
print(f"Scale: train={SUBSET_TRAIN} test={SUBSET_TEST}")
print(f"Epochs: ERM={ERM_EPOCHS} SimCLR={SIMCLR_EPOCHS} Probe={PROBE_EPOCHS}")
print(f"Seeds: {SEEDS}")


# ═══════════════════════════════════════════════════
# NEW: SimCLR-NO-COLOR-AUG transform
# ═══════════════════════════════════════════════════

def get_simclr_no_color_aug_transform():
    """
    SimCLR WITHOUT color-jitter or grayscale.
    Only random crop + horizontal flip → cannot mechanically delete HSV signal.
    This is the control arm to test whether the color-aug confound explains WGA gains.
    """
    return T.Compose([
        T.RandomResizedCrop(32, scale=(0.2, 1.0)),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])


# ═══════════════════════════════════════════════════
# evaluate_with_groups_v2 (same as refine-02)
# ═══════════════════════════════════════════════════

@torch.no_grad()
def evaluate_with_groups_v2(model_or_probe, loader_or_features, device,
                             is_probe=False, features=None, labels=None,
                             groups=None):
    """Evaluate accuracy per group. Returns (overall_acc, group_accs, group_sizes, wga)."""
    if is_probe:
        assert features is not None and labels is not None and groups is not None
        model_or_probe.eval()
        model_or_probe.to(device)
        ds = torch.utils.data.TensorDataset(features, labels, groups)
        loader = DataLoader(ds, batch_size=512, shuffle=False)
        group_correct = torch.zeros(4)
        group_total   = torch.zeros(4)
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
                group_total[g]   += gm.sum().item()
    else:
        model_or_probe.eval()
        group_correct = torch.zeros(4)
        group_total   = torch.zeros(4)
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
                group_total[grp]   += gm.sum().item()

    group_accs  = {}
    group_sizes = {}
    for g in range(4):
        group_sizes[g] = int(group_total[g].item())
        if group_total[g] > 0:
            group_accs[g] = float((group_correct[g] / group_total[g]).item())
        else:
            group_accs[g] = float('nan')

    overall_acc = total_correct / total_n
    valid_accs  = [v for v in group_accs.values() if not math.isnan(v)]
    wga = min(valid_accs) if valid_accs else float('nan')
    return overall_acc, group_accs, group_sizes, wga


# ═══════════════════════════════════════════════════
# SANITY GATES (unchanged from refine-02)
# ═══════════════════════════════════════════════════

def run_sanity_gates():
    print("\n" + "="*60)
    print("SANITY GATES")
    print("="*60)

    test_ds = TogglableCIFAR10(
        CIFAR_ROOT, train=False, spurious_rate=1.0,
        bg_enabled=True, patch_enabled=True, seed=0,
        subset_size=SUBSET_TEST, transform=TEST_TRANSFORM,
    )
    from collections import defaultdict
    class_indices = defaultdict(list)
    for i, (_, lbl) in enumerate(test_ds):
        class_indices[lbl].append(i)
    balanced_idx = []
    for c in range(10):
        balanced_idx.extend(class_indices[c][:10])
    balanced_subset  = torch.utils.data.Subset(test_ds, balanced_idx)
    balanced_loader  = DataLoader(balanced_subset, batch_size=100, shuffle=False)

    train_ds = TogglableCIFAR10(
        CIFAR_ROOT, train=True, spurious_rate=1.0,
        bg_enabled=True, patch_enabled=True, seed=0,
        subset_size=500, transform=TRAIN_TRANSFORM,
    )
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=2)

    def model_factory():
        return get_resnet18(num_classes=10)

    print("\nG1: Fixed seed reproducibility...")
    g1 = gate_fixed_seed(model_factory, train_loader, DEVICE, seed=42, n_steps=10)
    print(f"  max_diff={g1['max_diff']:.2e}  PASS={g1['passed']}")

    print("\nG2: Loss at init ≈ ln(10) = 2.303...")
    g2 = gate_loss_at_init(model_factory, train_loader, DEVICE, seed=42)
    print(f"  loss_0={g2['loss_0']:.4f}  expected={g2['expected']:.4f}  PASS={g2['passed']}")

    print("\nG3: Dummy classifier → ~10% accuracy...")
    g3 = gate_dummy_classifier(balanced_loader, DEVICE)
    print(f"  dummy_acc={g3['dummy_acc_pct']:.1f}%  PASS={g3['passed']}")

    print("\nG4: Overfit single batch (32 samples, 200 SGD steps)...")
    x_batch, y_batch = next(iter(DataLoader(train_ds, batch_size=32, shuffle=False)))
    g4 = gate_overfit_one_batch(model_factory, (x_batch, y_batch), DEVICE, seed=42)
    print(f"  train_loss@200={g4['train_loss_step200']:.6f}  PASS={g4['passed']}")

    gates = {"G1_fixed_seed": g1, "G2_loss_at_init": g2,
             "G3_dummy_classifier": g3, "G4_overfit_batch": g4}
    all_passed = all(g['passed'] for g in gates.values())
    print(f"\nAll gates passed: {all_passed}")
    return gates, all_passed


# ═══════════════════════════════════════════════════
# SIGNAL SANITY (G5+G6+G7)
# ═══════════════════════════════════════════════════

def run_linear_probe_sanity():
    print("\n" + "="*60)
    print("SIGNAL SANITY (G5+G6+G7)")
    print("="*60)

    from sklearn.linear_model import LogisticRegression
    import cv2

    def compute_probes(spurious_rate):
        ds = TogglableCIFAR10(
            CIFAR_ROOT, train=True, spurious_rate=spurious_rate,
            bg_enabled=True, patch_enabled=True, seed=0,
            subset_size=SUBSET_TRAIN, transform=None,
        )
        bg_feats, patch_feats, lbls = [], [], []
        for i in range(len(ds.images)):
            img = ds.images[i].copy()
            if ds.bg_enabled:
                img = apply_background_tint(img, BG_HUES[ds.bg_class[i]])
            if ds.patch_enabled:
                img = apply_color_patch(img, PATCH_HUES[ds.patch_class[i]])
            img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
            h_hist = np.histogram(img_hsv[:, :, 0], bins=16, range=(0, 180))[0].astype(np.float32)
            h_hist /= (h_hist.sum() + 1e-8)
            bg_feats.append(h_hist)
            patch = img[:4, :4, :]
            ph = cv2.cvtColor(patch, cv2.COLOR_RGB2HSV)
            p_hist = np.histogram(ph[:, :, 0], bins=16, range=(0, 180))[0].astype(np.float32)
            p_hist /= (p_hist.sum() + 1e-8)
            patch_feats.append(p_hist)
            lbls.append(ds.labels[i])

        bg_feats    = np.array(bg_feats)
        patch_feats = np.array(patch_feats)
        lbls        = np.array(lbls)

        clf_bg = LogisticRegression(max_iter=500, C=10.0)
        clf_bg.fit(bg_feats, lbls)
        acc_bg = clf_bg.score(bg_feats, lbls)

        clf_patch = LogisticRegression(max_iter=500, C=10.0)
        clf_patch.fit(patch_feats, lbls)
        acc_patch = clf_patch.score(patch_feats, lbls)
        return acc_bg, acc_patch

    print("  ρ=1.0 ...")
    acc_bg_1, acc_patch_1 = compute_probes(1.0)
    print(f"    BG probe  acc = {acc_bg_1:.3f} (need ≥0.85)")
    print(f"    Patch probe acc = {acc_patch_1:.3f} (need ≥0.85)")

    print("  ρ=0.0 ...")
    acc_bg_0, acc_patch_0 = compute_probes(0.0)
    print(f"    BG probe  acc = {acc_bg_0:.3f} (need ≤0.15)")
    print(f"    Patch probe acc = {acc_patch_0:.3f} (need ≤0.15)")

    g5 = {
        "gate": "lp_background_signal",
        "passed": bool(acc_bg_1 >= 0.85 and acc_bg_0 <= 0.15),
        "lp_bg_acc_rho1": float(acc_bg_1),
        "lp_bg_acc_rho0": float(acc_bg_0),
    }
    g6 = {
        "gate": "lp_patch_signal",
        "passed": bool(acc_patch_1 >= 0.85 and acc_patch_0 <= 0.15),
        "lp_patch_acc_rho1": float(acc_patch_1),
        "lp_patch_acc_rho0": float(acc_patch_0),
    }
    print(f"\nG5 (bg signal): PASS={g5['passed']}")
    print(f"G6 (patch signal): PASS={g6['passed']}")

    # G7: verify balanced 4-group WGA test set
    print("\nG7: Verify balanced 4-group WGA test set has all 4 groups...")
    wga_test_ds = TogglableCIFAR10(
        CIFAR_ROOT, train=False, spurious_rate=1.0,
        bg_enabled=True, patch_enabled=True, seed=0,
        subset_size=SUBSET_TEST, transform=TEST_TRANSFORM,
        return_group=True,
        bg_spurious_rate=0.5, patch_spurious_rate=0.5,
    )
    group_counts = np.bincount(wga_test_ds.group_labels, minlength=4)
    print(f"  Group counts: {dict(enumerate(group_counts.tolist()))}")
    g7 = {
        "gate": "balanced_4group",
        "passed": bool((group_counts > 0).all()),
        "group_counts": group_counts.tolist(),
        "note": "All 4 groups must be non-empty for valid WGA",
    }
    print(f"  PASS={g7['passed']}")
    return g5, g6, g7


# ═══════════════════════════════════════════════════
# WGA eval loaders (same as refine-02)
# ═══════════════════════════════════════════════════

def make_wga_eval_loaders(seed, subset_size=SUBSET_TEST):
    """Create 4 test loaders for proper 4-group WGA evaluation."""
    test_on_ds = TogglableCIFAR10(
        CIFAR_ROOT, train=False, spurious_rate=1.0,
        bg_enabled=True, patch_enabled=True, seed=seed,
        subset_size=subset_size, transform=TEST_TRANSFORM,
        return_group=True,
    )
    test_off_ds = TogglableCIFAR10(
        CIFAR_ROOT, train=False, spurious_rate=1.0,
        bg_enabled=False, patch_enabled=False, seed=seed,
        subset_size=subset_size, transform=TEST_TRANSFORM,
        return_group=True,
    )
    test_bgoff_wga_ds = TogglableCIFAR10(
        CIFAR_ROOT, train=False, spurious_rate=1.0,
        bg_enabled=False, patch_enabled=True, seed=seed,
        subset_size=subset_size, transform=TEST_TRANSFORM,
        return_group=True,
        bg_spurious_rate=0.5, patch_spurious_rate=0.5,
    )
    test_patchoff_wga_ds = TogglableCIFAR10(
        CIFAR_ROOT, train=False, spurious_rate=1.0,
        bg_enabled=True, patch_enabled=False, seed=seed,
        subset_size=subset_size, transform=TEST_TRANSFORM,
        return_group=True,
        bg_spurious_rate=0.5, patch_spurious_rate=0.5,
    )

    loader_on         = DataLoader(test_on_ds,         batch_size=256, shuffle=False, num_workers=2)
    loader_off        = DataLoader(test_off_ds,        batch_size=256, shuffle=False, num_workers=2)
    loader_bgoff_wga  = DataLoader(test_bgoff_wga_ds,  batch_size=256, shuffle=False, num_workers=2)
    loader_patchoff_wga = DataLoader(test_patchoff_wga_ds, batch_size=256, shuffle=False, num_workers=2)

    gc_bgoff   = np.bincount(test_bgoff_wga_ds.group_labels, minlength=4)
    gc_patchoff = np.bincount(test_patchoff_wga_ds.group_labels, minlength=4)
    print(f"    bgoff  group counts: {gc_bgoff.tolist()}")
    print(f"    patchoff group counts: {gc_patchoff.tolist()}")

    return (loader_on, loader_off, loader_bgoff_wga, loader_patchoff_wga,
            test_on_ds, test_off_ds, test_bgoff_wga_ds, test_patchoff_wga_ds)


# ═══════════════════════════════════════════════════
# NEW: HSV decodability measurement
# ═══════════════════════════════════════════════════

def measure_hsv_decodability(backbone, train_ds, device):
    """
    Train a linear probe on backbone features to predict bg_class (the injected hue class).
    High accuracy → backbone retains HSV color information.
    Low accuracy → backbone has suppressed/lost the HSV signal.

    Returns: float accuracy (train set)
    """
    # Extract backbone features
    probe_loader = DataLoader(train_ds, batch_size=256, shuffle=False, num_workers=2)
    feats, _ = extract_features(backbone, probe_loader, device)
    # bg_class labels (which hue class was assigned to each training example)
    bg_class_labels = torch.tensor(train_ds.bg_class, dtype=torch.long)

    # Train linear probe to predict bg_class from backbone features
    feat_dim = feats.shape[1]
    probe = LinearProbe(feat_dim=feat_dim, num_classes=10)
    train_linear_probe(probe, feats, bg_class_labels, device, epochs=PROBE_EPOCHS, lr=LR_PROBE)

    # Evaluate on training set (sufficient for decodability — we want to know if info is there)
    probe.eval()
    probe.to(device)
    ds_eval = torch.utils.data.TensorDataset(feats, bg_class_labels)
    loader_eval = DataLoader(ds_eval, batch_size=512, shuffle=False)
    correct, total = 0, 0
    with torch.no_grad():
        for xb, yb in loader_eval:
            xb, yb = xb.to(device), yb.to(device)
            preds = probe(xb).argmax(1)
            correct += (preds == yb).sum().item()
            total += len(yb)
    acc = correct / total
    return float(acc)


# ═══════════════════════════════════════════════════
# ERM: one seed
# ═══════════════════════════════════════════════════

def run_erm_seed(seed):
    print(f"\n  [ERM seed={seed}]")
    set_seed(seed)

    train_ds = TogglableCIFAR10(
        CIFAR_ROOT, train=True, spurious_rate=1.0,
        bg_enabled=True, patch_enabled=True, seed=seed,
        subset_size=SUBSET_TRAIN, transform=TRAIN_TRANSFORM,
        return_group=False,
    )
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=True)

    model = get_resnet18(num_classes=NUM_CLASSES).to(DEVICE)
    optimizer = torch.optim.SGD(model.parameters(), lr=LR_ERM, momentum=0.9, weight_decay=1e-4)
    train_erm(model, train_loader, optimizer, DEVICE, ERM_EPOCHS)

    # Measure HSV decodability from ERM features
    train_ds_feat = TogglableCIFAR10(
        CIFAR_ROOT, train=True, spurious_rate=1.0,
        bg_enabled=True, patch_enabled=True, seed=seed,
        subset_size=SUBSET_TRAIN, transform=TEST_TRANSFORM,
        return_group=False,
    )
    # Wrap ERM model as a backbone-like encoder
    class ERMEncoder:
        """Wraps ResNet-18 to extract penultimate features."""
        def __init__(self, model):
            self.model = model
            self.enc_dim = 512
        def eval(self):
            self.model.eval()
        def encode(self, x):
            from train import _forward_features
            return _forward_features(self.model, x)
        def __call__(self, x):
            return self.model(x)

    erm_encoder = ERMEncoder(model)
    # Extract features using the existing infrastructure
    probe_loader = DataLoader(train_ds_feat, batch_size=256, shuffle=False, num_workers=2)
    erm_encoder.model.eval()
    all_feats = []
    with torch.no_grad():
        for batch in probe_loader:
            x = batch[0].to(DEVICE)
            from train import _forward_features
            h = _forward_features(model, x)
            all_feats.append(h.cpu())
    feats = torch.cat(all_feats)
    bg_class_labels = torch.tensor(train_ds_feat.bg_class, dtype=torch.long)

    # Train decodability probe
    probe_hsv = LinearProbe(feat_dim=512, num_classes=10)
    train_linear_probe(probe_hsv, feats, bg_class_labels, DEVICE, epochs=PROBE_EPOCHS, lr=LR_PROBE)
    probe_hsv.eval().to(DEVICE)
    ds_eval = torch.utils.data.TensorDataset(feats, bg_class_labels)
    loader_eval = DataLoader(ds_eval, batch_size=512, shuffle=False)
    correct, total = 0, 0
    with torch.no_grad():
        for xb, yb in loader_eval:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            preds = probe_hsv(xb).argmax(1)
            correct += (preds == yb).sum().item()
            total += len(yb)
    hsv_decodability = correct / total
    print(f"    ERM HSV decodability (bg_class predict acc): {hsv_decodability:.3f}")

    print(f"    Creating WGA eval loaders (seed={seed}) ...")
    (loader_on, loader_off, loader_bgoff_wga, loader_patchoff_wga,
     ds_on, ds_off, ds_bgoff_wga, ds_patchoff_wga) = make_wga_eval_loaders(seed)

    acc_on,  _ = evaluate(model, loader_on,  DEVICE)
    acc_off, _ = evaluate(model, loader_off, DEVICE)
    spurious_gap = acc_on - acc_off

    _, group_accs_bgoff, group_sizes_bgoff, wga_bgoff = evaluate_with_groups_v2(
        model, loader_bgoff_wga, DEVICE)
    _, group_accs_patchoff, group_sizes_patchoff, wga_patchoff = evaluate_with_groups_v2(
        model, loader_patchoff_wga, DEVICE)

    print(f"    acc_on={acc_on:.3f}  acc_off={acc_off:.3f}  gap={spurious_gap:.3f}")
    print(f"    WGA bgoff:    {wga_bgoff:.3f}  groups={group_accs_bgoff}")
    print(f"    WGA patchoff: {wga_patchoff:.3f}  groups={group_accs_patchoff}")

    return {
        "seed": seed,
        "acc_on": float(acc_on),
        "acc_off": float(acc_off),
        "spurious_gap": float(spurious_gap),
        "wga_bgoff": float(wga_bgoff),
        "wga_patchoff": float(wga_patchoff),
        "hsv_decodability": float(hsv_decodability),
        "group_accs_bgoff":   {str(k): float(v) for k, v in group_accs_bgoff.items()},
        "group_sizes_bgoff":  {str(k): v for k, v in group_sizes_bgoff.items()},
        "group_accs_patchoff":{str(k): float(v) for k, v in group_accs_patchoff.items()},
        "group_sizes_patchoff":{str(k): v for k, v in group_sizes_patchoff.items()},
    }


# ═══════════════════════════════════════════════════
# SimCLR: one seed (unified for full-aug and no-color-aug)
# ═══════════════════════════════════════════════════

def run_simclr_seed(seed, color_aug=True):
    arm_name = "SimCLR-full-aug" if color_aug else "SimCLR-no-color-aug"
    print(f"\n  [{arm_name} seed={seed}]")
    set_seed(seed)

    pretrain_base = TogglableCIFAR10(
        CIFAR_ROOT, train=True, spurious_rate=1.0,
        bg_enabled=True, patch_enabled=True, seed=seed,
        subset_size=SUBSET_TRAIN, transform=None,
    )
    if color_aug:
        aug_transform = get_simclr_transform()
    else:
        aug_transform = get_simclr_no_color_aug_transform()

    pretrain_ds = TwoViewDataset(pretrain_base, aug_transform)
    pretrain_loader = DataLoader(pretrain_ds, batch_size=BATCH_SIZE, shuffle=True,
                                 num_workers=4, pin_memory=True, drop_last=True)

    backbone = SimCLRBackbone(proj_dim=128).to(DEVICE)
    optimizer = torch.optim.SGD(backbone.parameters(), lr=LR_SIMCLR, momentum=0.9, weight_decay=1e-4)
    train_simclr(backbone, pretrain_loader, optimizer, DEVICE, SIMCLR_EPOCHS, TEMPERATURE)

    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    aug_tag = "full" if color_aug else "nocolor"
    torch.save(backbone.state_dict(),
               os.path.join(WEIGHTS_DIR, f"simclr_{aug_tag}_seed{seed}.pt"))

    # ── Linear probe for classification ──
    probe_ds = TogglableCIFAR10(
        CIFAR_ROOT, train=True, spurious_rate=1.0,
        bg_enabled=True, patch_enabled=True, seed=seed,
        subset_size=SUBSET_TRAIN, transform=TEST_TRANSFORM,
        return_group=False,
    )
    probe_loader = DataLoader(probe_ds, batch_size=256, shuffle=False,
                              num_workers=2, pin_memory=True)
    feats_train, labels_train = extract_features(backbone, probe_loader, DEVICE)

    probe = LinearProbe(feat_dim=backbone.enc_dim, num_classes=NUM_CLASSES)
    train_linear_probe(probe, feats_train, labels_train, DEVICE, PROBE_EPOCHS, LR_PROBE)

    # ── HSV decodability: predict bg_class from backbone features ──
    bg_class_labels = torch.tensor(probe_ds.bg_class, dtype=torch.long)
    probe_hsv = LinearProbe(feat_dim=backbone.enc_dim, num_classes=10)
    train_linear_probe(probe_hsv, feats_train, bg_class_labels, DEVICE, epochs=PROBE_EPOCHS, lr=LR_PROBE)
    probe_hsv.eval().to(DEVICE)
    ds_eval = torch.utils.data.TensorDataset(feats_train, bg_class_labels)
    loader_eval = DataLoader(ds_eval, batch_size=512, shuffle=False)
    correct, total = 0, 0
    with torch.no_grad():
        for xb, yb in loader_eval:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            preds = probe_hsv(xb).argmax(1)
            correct += (preds == yb).sum().item()
            total += len(yb)
    hsv_decodability = correct / total
    print(f"    {arm_name} HSV decodability: {hsv_decodability:.3f}")

    # ── WGA evaluation ──
    print(f"    Creating WGA eval loaders (seed={seed}) ...")
    (loader_on, loader_off, loader_bgoff_wga, loader_patchoff_wga,
     ds_on, ds_off, ds_bgoff_wga, ds_patchoff_wga) = make_wga_eval_loaders(seed)

    def get_feats_groups(ds):
        loader = DataLoader(ds, batch_size=256, shuffle=False, num_workers=2)
        feats, labels = extract_features(backbone, loader, DEVICE)
        groups = torch.tensor(ds.group_labels)
        return feats, labels, groups

    feats_on,      labels_on,      groups_on      = get_feats_groups(ds_on)
    feats_off,     labels_off,     groups_off     = get_feats_groups(ds_off)
    feats_bgoff,   labels_bgoff,   groups_bgoff   = get_feats_groups(ds_bgoff_wga)
    feats_patchoff,labels_patchoff,groups_patchoff = get_feats_groups(ds_patchoff_wga)

    acc_on,  _ = evaluate(probe, None, DEVICE, is_probe=True,
                          features=feats_on, labels=labels_on)
    acc_off, _ = evaluate(probe, None, DEVICE, is_probe=True,
                          features=feats_off, labels=labels_off)
    spurious_gap = acc_on - acc_off

    _, group_accs_bgoff, group_sizes_bgoff, wga_bgoff = evaluate_with_groups_v2(
        probe, None, DEVICE, is_probe=True,
        features=feats_bgoff, labels=labels_bgoff, groups=groups_bgoff)
    _, group_accs_patchoff, group_sizes_patchoff, wga_patchoff = evaluate_with_groups_v2(
        probe, None, DEVICE, is_probe=True,
        features=feats_patchoff, labels=labels_patchoff, groups=groups_patchoff)

    print(f"    acc_on={acc_on:.3f}  acc_off={acc_off:.3f}  gap={spurious_gap:.3f}")
    print(f"    WGA bgoff:    {wga_bgoff:.3f}  groups={group_accs_bgoff}")
    print(f"    WGA patchoff: {wga_patchoff:.3f}  groups={group_accs_patchoff}")

    return {
        "seed": seed,
        "color_aug": color_aug,
        "acc_on": float(acc_on),
        "acc_off": float(acc_off),
        "spurious_gap": float(spurious_gap),
        "wga_bgoff": float(wga_bgoff),
        "wga_patchoff": float(wga_patchoff),
        "hsv_decodability": float(hsv_decodability),
        "group_accs_bgoff":   {str(k): float(v) for k, v in group_accs_bgoff.items()},
        "group_sizes_bgoff":  {str(k): v for k, v in group_sizes_bgoff.items()},
        "group_accs_patchoff":{str(k): float(v) for k, v in group_accs_patchoff.items()},
        "group_sizes_patchoff":{str(k): v for k, v in group_sizes_patchoff.items()},
    }


# ═══════════════════════════════════════════════════
# Aggregate helpers
# ═══════════════════════════════════════════════════

def aggregate(results, key):
    vals = [r[key] for r in results if not math.isnan(r[key])]
    if not vals:
        return {"mean": float("nan"), "std": float("nan"), "per_seed": []}
    return {
        "mean": float(np.mean(vals)),
        "std":  float(np.std(vals)),
        "per_seed": [float(v) for v in vals],
    }


def aggregate_group_accs(results, key):
    all_groups = {}
    for r in results:
        for g, v in r[key].items():
            if g not in all_groups:
                all_groups[g] = []
            if not math.isnan(v):
                all_groups[g].append(v)
    return {g: {"mean": float(np.mean(vs)) if vs else float("nan"),
                "std":  float(np.std(vs)) if len(vs) > 1 else 0.0,
                "per_seed": [float(v) for v in vs]}
            for g, vs in all_groups.items()}


# ═══════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════

def main():
    t0 = time.time()

    # Step 1: Sanity gates
    gates, all_gates_passed = run_sanity_gates()

    # Step 2: Signal sanity + balanced-group check
    try:
        import cv2
        g5, g6, g7 = run_linear_probe_sanity()
    except ImportError:
        print("  [WARNING] cv2 not available, skipping G5/G6")
        g5 = {"gate": "lp_background_signal", "passed": None, "note": "cv2 not available"}
        g6 = {"gate": "lp_patch_signal",       "passed": None, "note": "cv2 not available"}
        g7 = {"gate": "balanced_4group",        "passed": None, "note": "cv2 not available"}

    gates["G5_bg_signal_probe"] = g5
    gates["G6_patch_signal_probe"] = g6
    gates["G7_balanced_groups"] = g7

    # Step 3: ERM baseline
    print("\n" + "="*60)
    print("ARM 1: ERM BASELINE (3 seeds)")
    print("="*60)
    erm_results = [run_erm_seed(s) for s in SEEDS]

    # Step 4: SimCLR full-aug
    print("\n" + "="*60)
    print("ARM 2: SimCLR-FULL-AUG (crop+flip+colorjitter+grayscale, 3 seeds)")
    print("="*60)
    simclr_full_results = [run_simclr_seed(s, color_aug=True) for s in SEEDS]

    # Step 5: SimCLR no-color-aug (THE NEW CONTROL)
    print("\n" + "="*60)
    print("ARM 3: SimCLR-NO-COLOR-AUG (crop+flip ONLY, 3 seeds)")
    print("="*60)
    simclr_nocolor_results = [run_simclr_seed(s, color_aug=False) for s in SEEDS]

    # Step 6: Aggregate
    keys = ["acc_on", "acc_off", "spurious_gap", "wga_bgoff", "wga_patchoff", "hsv_decodability"]
    erm_agg        = {k: aggregate(erm_results, k)              for k in keys}
    simclr_full_agg = {k: aggregate(simclr_full_results, k)    for k in keys}
    simclr_nc_agg   = {k: aggregate(simclr_nocolor_results, k) for k in keys}

    # Deltas
    delta_bgoff_full   = simclr_full_agg["wga_bgoff"]["mean"]   - erm_agg["wga_bgoff"]["mean"]
    delta_bgoff_nocolor = simclr_nc_agg["wga_bgoff"]["mean"]    - erm_agg["wga_bgoff"]["mean"]
    delta_patchoff_full  = simclr_full_agg["wga_patchoff"]["mean"] - erm_agg["wga_patchoff"]["mean"]
    delta_patchoff_nocolor = simclr_nc_agg["wga_patchoff"]["mean"] - erm_agg["wga_patchoff"]["mean"]

    elapsed = time.time() - t0

    # ── Print summary ──
    print("\n" + "="*60)
    print("RESULTS SUMMARY — refine-03")
    print("="*60)
    print(f"\n{'Arm':<25} {'wga_bgoff':>10} {'wga_patchoff':>13} {'hsv_decode':>11} {'acc_on':>8}")
    print("-"*70)
    for arm_name, agg in [("ERM", erm_agg),
                           ("SimCLR-full-aug", simclr_full_agg),
                           ("SimCLR-no-color-aug", simclr_nc_agg)]:
        print(f"{arm_name:<25} "
              f"{agg['wga_bgoff']['mean']:>10.3f} "
              f"{agg['wga_patchoff']['mean']:>13.3f} "
              f"{agg['hsv_decodability']['mean']:>11.3f} "
              f"{agg['acc_on']['mean']:>8.3f}")

    print(f"\nΔwga_bgoff   (SimCLR-full − ERM):         {delta_bgoff_full:+.3f}")
    print(f"Δwga_bgoff   (SimCLR-no-color − ERM):     {delta_bgoff_nocolor:+.3f}")
    print(f"Δwga_patchoff (SimCLR-full − ERM):        {delta_patchoff_full:+.3f}")
    print(f"Δwga_patchoff (SimCLR-no-color − ERM):    {delta_patchoff_nocolor:+.3f}")

    print(f"\nHSV decodability (higher = bg color info retained in representation):")
    print(f"  ERM:               {erm_agg['hsv_decodability']['mean']:.3f}")
    print(f"  SimCLR-full-aug:   {simclr_full_agg['hsv_decodability']['mean']:.3f}")
    print(f"  SimCLR-no-color:   {simclr_nc_agg['hsv_decodability']['mean']:.3f}")

    # Conclusion
    bgoff_gain_persists   = delta_bgoff_nocolor > 0.03
    bgoff_gain_full       = delta_bgoff_full    > 0.03
    hsv_decode_full_lower = (simclr_full_agg['hsv_decodability']['mean'] <
                             erm_agg['hsv_decodability']['mean'] - 0.10)
    hsv_decode_nc_retained = (simclr_nc_agg['hsv_decodability']['mean'] >
                              simclr_full_agg['hsv_decodability']['mean'] + 0.10)

    if bgoff_gain_full and not bgoff_gain_persists:
        conclusion = "DELETION: WGA gain vanishes without color-aug → augmentation deletion explains SimCLR gain"
    elif bgoff_gain_full and bgoff_gain_persists:
        conclusion = "REAL_INVARIANCE: WGA gain persists without color-aug → genuine invariance mechanism"
    elif not bgoff_gain_full:
        conclusion = "NO_GAIN: SimCLR-full did not improve WGA at this probe scale"
    else:
        conclusion = "AMBIGUOUS"

    print(f"\nConclusion: {conclusion}")
    print(f"  bgoff gain PERSISTS without color aug: {bgoff_gain_persists}")
    print(f"  SimCLR-full HSV decode lower: {hsv_decode_full_lower}")
    print(f"  SimCLR-no-color HSV decode retained: {hsv_decode_nc_retained}")

    # ── Write RESULTS.json ──
    gates_ok = all(v.get('passed', True) is not False
                   for v in gates.values() if isinstance(v, dict) and isinstance(v.get('passed'), bool))
    status = "SUCCESS" if gates_ok else "PARTIAL"

    results_json = {
        "status": status,
        "scale": "probe",
        "subject_executed": (
            "refine-03: Control for color-aug confound. "
            "ERM vs SimCLR-full-aug vs SimCLR-no-color-aug on Togglable-Signal CIFAR-10. "
            f"{SUBSET_TRAIN} train / {SUBSET_TEST} test / {SEEDS} seeds. "
            f"ERM {ERM_EPOCHS}ep, SimCLR {SIMCLR_EPOCHS}ep pretrain + {PROBE_EPOCHS}ep probe."
        ),
        "metrics": {
            "erm": {
                "wga_bgoff":        erm_agg["wga_bgoff"],
                "wga_patchoff":     erm_agg["wga_patchoff"],
                "hsv_decodability": erm_agg["hsv_decodability"],
                "acc_on":           erm_agg["acc_on"],
                "acc_off":          erm_agg["acc_off"],
                "spurious_gap":     erm_agg["spurious_gap"],
                "per_group_bgoff":  aggregate_group_accs(erm_results, "group_accs_bgoff"),
                "per_group_patchoff": aggregate_group_accs(erm_results, "group_accs_patchoff"),
                "per_seed_results": erm_results,
            },
            "simclr_full_aug": {
                "wga_bgoff":        simclr_full_agg["wga_bgoff"],
                "wga_patchoff":     simclr_full_agg["wga_patchoff"],
                "hsv_decodability": simclr_full_agg["hsv_decodability"],
                "acc_on":           simclr_full_agg["acc_on"],
                "acc_off":          simclr_full_agg["acc_off"],
                "spurious_gap":     simclr_full_agg["spurious_gap"],
                "per_group_bgoff":  aggregate_group_accs(simclr_full_results, "group_accs_bgoff"),
                "per_group_patchoff": aggregate_group_accs(simclr_full_results, "group_accs_patchoff"),
                "per_seed_results": simclr_full_results,
            },
            "simclr_no_color_aug": {
                "wga_bgoff":        simclr_nc_agg["wga_bgoff"],
                "wga_patchoff":     simclr_nc_agg["wga_patchoff"],
                "hsv_decodability": simclr_nc_agg["hsv_decodability"],
                "acc_on":           simclr_nc_agg["acc_on"],
                "acc_off":          simclr_nc_agg["acc_off"],
                "spurious_gap":     simclr_nc_agg["spurious_gap"],
                "per_group_bgoff":  aggregate_group_accs(simclr_nocolor_results, "group_accs_bgoff"),
                "per_group_patchoff": aggregate_group_accs(simclr_nocolor_results, "group_accs_patchoff"),
                "per_seed_results": simclr_nocolor_results,
            },
            "deltas": {
                "wga_bgoff_full_minus_erm":    float(delta_bgoff_full),
                "wga_bgoff_nocolor_minus_erm": float(delta_bgoff_nocolor),
                "wga_patchoff_full_minus_erm": float(delta_patchoff_full),
                "wga_patchoff_nocolor_minus_erm": float(delta_patchoff_nocolor),
            },
            "conclusion": conclusion,
            "interpretation": {
                "bgoff_gain_full_aug":       bool(bgoff_gain_full),
                "bgoff_gain_no_color_aug":   bool(bgoff_gain_persists),
                "hsv_decode_full_lower":     bool(hsv_decode_full_lower),
                "hsv_decode_nc_retained":    bool(hsv_decode_nc_retained),
                "mechanism": conclusion,
            },
            "refine02_reference": {
                "erm_wga_bgoff_mean":    0.0394,
                "erm_wga_patchoff_mean": 0.0157,
                "simclr_wga_bgoff_mean": 0.1925,
                "simclr_wga_patchoff_mean": 0.0970,
                "delta_bgoff": 0.1530,
                "note": "refine-02 showed SimCLR wga_bgoff gain +15.3pp; reviewer flagged color-aug confound",
            },
        },
        "sanity_gates": {
            "G1_fixed_seed":  gates["G1_fixed_seed"]["passed"],
            "G2_loss_at_init":gates["G2_loss_at_init"]["passed"],
            "G3_dummy":       gates["G3_dummy_classifier"]["passed"],
            "G4_overfit":     gates["G4_overfit_batch"]["passed"],
            "G5_bg_probe":    g5.get("passed"),
            "G6_patch_probe": g6.get("passed"),
            "G7_balanced_groups": g7.get("passed"),
            "all_gates_passed": all_gates_passed,
        },
        "config": {
            "subset_train": SUBSET_TRAIN,
            "subset_test":  SUBSET_TEST,
            "erm_epochs":    ERM_EPOCHS,
            "simclr_epochs": SIMCLR_EPOCHS,
            "probe_epochs":  PROBE_EPOCHS,
            "batch_size":    BATCH_SIZE,
            "seeds":         SEEDS,
            "device":        DEVICE,
            "arms": ["ERM", "SimCLR-full-aug (crop+flip+colorjitter+grayscale)",
                     "SimCLR-no-color-aug (crop+flip ONLY)"],
        },
        "elapsed_seconds": round(elapsed, 1),
        "notes": (
            "refine-03: Added SimCLR-no-color-aug control arm to test whether "
            "SimCLR's WGA gain is due to augmentation deletion of HSV signal or "
            "a genuine invariance mechanism. "
            "HSV decodability = linear probe accuracy predicting bg_class from "
            "backbone features (high=color info retained, low=color suppressed). "
            "Interpretation: if WGA gain persists without color aug → real invariance; "
            "if vanishes → deletion mechanism."
        ),
    }

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "RESULTS.json")
    with open(out_path, "w") as f:
        json.dump(results_json, f, indent=2)
    print(f"\nResults written to {out_path}")
    print(f"Total elapsed: {elapsed:.1f}s")
    print("\n" + "="*60)
    print(json.dumps(results_json, indent=2))
    print("="*60)
    return results_json


if __name__ == "__main__":
    main()
