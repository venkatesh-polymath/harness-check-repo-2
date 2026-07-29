"""
refine-02: Fixed WGA evaluation.

Bug in main-01: spurious_rate=1.0 for test sets → all samples in group 3
(both signals correlated), so groups 0-2 are empty → WGA collapses to single
group accuracy (not true WGA).

Fix: Use independent bg_spurious_rate=0.5 and patch_spurious_rate=0.5 for
WGA evaluation datasets → ~25% per group → all 4 groups populated.

Proper group semantics for bgoff/patchoff:
  bgoff  (bg=False, patch=True):
    - Groups 0,2 have patch_matches=0 → no active shortcut → HARD
    - Groups 1,3 have patch_matches=1 → patch shortcut present → EASY
    - WGA_bgoff = min over all 4 non-empty groups = min(acc_g0, acc_g2)
  patchoff (bg=True, patch=False):
    - Groups 0,1 have bg_matches=0 → no active shortcut → HARD
    - Groups 2,3 have bg_matches=1 → bg shortcut present → EASY
    - WGA_patchoff = min over all 4 non-empty groups = min(acc_g0, acc_g1)
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

CIFAR_ROOT = "/opt/datasets"
WEIGHTS_DIR = "/workspace/_weights/refine-02"
RESULTS_DIR = "/workspace/results/refine-02"

# ── Probe config (same as main-01) ──
SUBSET_TRAIN = 5000
SUBSET_TEST = 1000
ERM_EPOCHS = 20
SIMCLR_EPOCHS = 20
PROBE_EPOCHS = 20
SEEDS = [0, 1, 2]
BATCH_SIZE = 128
LR_ERM = 0.1
LR_SIMCLR = 0.03
LR_PROBE = 0.1
TEMPERATURE = 0.5
NUM_CLASSES = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Device: {DEVICE}")
print(f"Probe scale: train={SUBSET_TRAIN} test={SUBSET_TEST}")
print(f"Epochs: ERM={ERM_EPOCHS} SimCLR={SIMCLR_EPOCHS} Probe={PROBE_EPOCHS}")
print(f"Seeds: {SEEDS}")


# ═══════════════════════════════════════════════════
# FIXED evaluate_with_groups
# ═══════════════════════════════════════════════════

@torch.no_grad()
def evaluate_with_groups_v2(model_or_probe, loader_or_features, device,
                             is_probe=False, features=None, labels=None,
                             groups=None):
    """
    Evaluate accuracy per group (4-group: bg_matches*2 + patch_matches).
    Returns: (overall_acc, group_accs, group_sizes, wga)

    This version returns group_sizes so we can verify all groups are populated.
    WGA = min over non-empty groups.
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
    group_sizes = {}
    for g in range(4):
        group_sizes[g] = int(group_total[g].item())
        if group_total[g] > 0:
            group_accs[g] = float((group_correct[g] / group_total[g]).item())
        else:
            group_accs[g] = float('nan')

    overall_acc = total_correct / total_n
    valid_accs = [v for v in group_accs.values() if not math.isnan(v)]
    wga = min(valid_accs) if valid_accs else float('nan')
    return overall_acc, group_accs, group_sizes, wga


# ═══════════════════════════════════════════════════
# SANITY GATES
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
    balanced_subset = torch.utils.data.Subset(test_ds, balanced_idx)
    balanced_loader = DataLoader(balanced_subset, batch_size=100, shuffle=False)

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
# SIGNAL SANITY (G5+G6)
# ═══════════════════════════════════════════════════

def run_linear_probe_sanity():
    print("\n" + "="*60)
    print("SIGNAL SANITY (G5+G6)")
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

        bg_feats = np.array(bg_feats)
        patch_feats = np.array(patch_feats)
        lbls = np.array(lbls)

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

    # NEW: G7 — verify balanced 4-group WGA test set has all groups populated
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
# HELPER: create WGA evaluation datasets
# ═══════════════════════════════════════════════════

def make_wga_eval_loaders(seed, subset_size=SUBSET_TEST):
    """
    Create the 4 test loaders needed for proper WGA evaluation.
    Key: use bg_spurious_rate=0.5, patch_spurious_rate=0.5 INDEPENDENTLY
    so all 4 groups are populated.
    """
    # acc_on: both signals ON, correlated (matches training distribution)
    test_on_ds = TogglableCIFAR10(
        CIFAR_ROOT, train=False, spurious_rate=1.0,
        bg_enabled=True, patch_enabled=True, seed=seed,
        subset_size=subset_size, transform=TEST_TRANSFORM,
        return_group=True,
    )

    # acc_off: both signals OFF (control — only semantic content)
    test_off_ds = TogglableCIFAR10(
        CIFAR_ROOT, train=False, spurious_rate=1.0,
        bg_enabled=False, patch_enabled=False, seed=seed,
        subset_size=subset_size, transform=TEST_TRANSFORM,
        return_group=True,
    )

    # WGA bgoff: bg=False, patch=True, INDEPENDENT 50/50 rates → all 4 groups
    # Groups 0,2 (patch anti-correlated) = HARD; groups 1,3 (patch correlated) = EASY
    test_bgoff_wga_ds = TogglableCIFAR10(
        CIFAR_ROOT, train=False, spurious_rate=1.0,
        bg_enabled=False, patch_enabled=True, seed=seed,
        subset_size=subset_size, transform=TEST_TRANSFORM,
        return_group=True,
        bg_spurious_rate=0.5, patch_spurious_rate=0.5,
    )

    # WGA patchoff: bg=True, patch=False, INDEPENDENT 50/50 rates → all 4 groups
    # Groups 0,1 (bg anti-correlated) = HARD; groups 2,3 (bg correlated) = EASY
    test_patchoff_wga_ds = TogglableCIFAR10(
        CIFAR_ROOT, train=False, spurious_rate=1.0,
        bg_enabled=True, patch_enabled=False, seed=seed,
        subset_size=subset_size, transform=TEST_TRANSFORM,
        return_group=True,
        bg_spurious_rate=0.5, patch_spurious_rate=0.5,
    )

    loader_on = DataLoader(test_on_ds, batch_size=256, shuffle=False, num_workers=2)
    loader_off = DataLoader(test_off_ds, batch_size=256, shuffle=False, num_workers=2)
    loader_bgoff_wga = DataLoader(test_bgoff_wga_ds, batch_size=256, shuffle=False, num_workers=2)
    loader_patchoff_wga = DataLoader(test_patchoff_wga_ds, batch_size=256, shuffle=False, num_workers=2)

    # Log group distributions
    gc_bgoff = np.bincount(test_bgoff_wga_ds.group_labels, minlength=4)
    gc_patchoff = np.bincount(test_patchoff_wga_ds.group_labels, minlength=4)
    print(f"    bgoff  group counts: {gc_bgoff.tolist()}")
    print(f"    patchoff group counts: {gc_patchoff.tolist()}")

    return (loader_on, loader_off, loader_bgoff_wga, loader_patchoff_wga,
            test_on_ds, test_off_ds, test_bgoff_wga_ds, test_patchoff_wga_ds)


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

    print(f"    Creating WGA eval loaders (seed={seed}) ...")
    (loader_on, loader_off, loader_bgoff_wga, loader_patchoff_wga,
     ds_on, ds_off, ds_bgoff_wga, ds_patchoff_wga) = make_wga_eval_loaders(seed)

    # Basic accuracies (signal ON vs OFF)
    acc_on, _ = evaluate(model, loader_on, DEVICE)
    acc_off, _ = evaluate(model, loader_off, DEVICE)
    spurious_gap = acc_on - acc_off

    # TRUE WGA on bgoff balanced test set
    _, group_accs_bgoff, group_sizes_bgoff, wga_bgoff = evaluate_with_groups_v2(
        model, loader_bgoff_wga, DEVICE)

    # TRUE WGA on patchoff balanced test set
    _, group_accs_patchoff, group_sizes_patchoff, wga_patchoff = evaluate_with_groups_v2(
        model, loader_patchoff_wga, DEVICE)

    print(f"    acc_on={acc_on:.3f}  acc_off={acc_off:.3f}  gap={spurious_gap:.3f}")
    print(f"    TRUE WGA bgoff:    {wga_bgoff:.3f}  groups={group_accs_bgoff}")
    print(f"    TRUE WGA patchoff: {wga_patchoff:.3f}  groups={group_accs_patchoff}")

    return {
        "seed": seed,
        "acc_on": float(acc_on),
        "acc_off": float(acc_off),
        "spurious_gap": float(spurious_gap),
        "wga_bgoff": float(wga_bgoff),
        "wga_patchoff": float(wga_patchoff),
        "group_accs_bgoff": {str(k): float(v) for k, v in group_accs_bgoff.items()},
        "group_sizes_bgoff": {str(k): v for k, v in group_sizes_bgoff.items()},
        "group_accs_patchoff": {str(k): float(v) for k, v in group_accs_patchoff.items()},
        "group_sizes_patchoff": {str(k): v for k, v in group_sizes_patchoff.items()},
    }


# ═══════════════════════════════════════════════════
# SimCLR: one seed
# ═══════════════════════════════════════════════════

def run_simclr_seed(seed):
    print(f"\n  [SimCLR seed={seed}]")
    set_seed(seed)

    pretrain_base = TogglableCIFAR10(
        CIFAR_ROOT, train=True, spurious_rate=1.0,
        bg_enabled=True, patch_enabled=True, seed=seed,
        subset_size=SUBSET_TRAIN, transform=None,
    )
    simclr_transform = get_simclr_transform()
    pretrain_ds = TwoViewDataset(pretrain_base, simclr_transform)
    pretrain_loader = DataLoader(pretrain_ds, batch_size=BATCH_SIZE, shuffle=True,
                                 num_workers=4, pin_memory=True, drop_last=True)

    backbone = SimCLRBackbone(proj_dim=128).to(DEVICE)
    optimizer = torch.optim.SGD(backbone.parameters(), lr=LR_SIMCLR, momentum=0.9, weight_decay=1e-4)
    train_simclr(backbone, pretrain_loader, optimizer, DEVICE, SIMCLR_EPOCHS, TEMPERATURE)

    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    torch.save(backbone.state_dict(),
               os.path.join(WEIGHTS_DIR, f"simclr_seed{seed}.pt"))

    # Linear probe: extract features
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

    # Build WGA eval loaders
    print(f"    Creating WGA eval loaders (seed={seed}) ...")
    (loader_on, loader_off, loader_bgoff_wga, loader_patchoff_wga,
     ds_on, ds_off, ds_bgoff_wga, ds_patchoff_wga) = make_wga_eval_loaders(seed)

    # Extract features for each test split
    def get_feats_groups(ds):
        loader = DataLoader(ds, batch_size=256, shuffle=False, num_workers=2)
        feats, labels = extract_features(backbone, loader, DEVICE)
        groups = torch.tensor(ds.group_labels)
        return feats, labels, groups

    feats_on, labels_on, groups_on = get_feats_groups(ds_on)
    feats_off, labels_off, groups_off = get_feats_groups(ds_off)
    feats_bgoff, labels_bgoff, groups_bgoff = get_feats_groups(ds_bgoff_wga)
    feats_patchoff, labels_patchoff, groups_patchoff = get_feats_groups(ds_patchoff_wga)

    # Basic accuracies
    acc_on, _ = evaluate(probe, None, DEVICE, is_probe=True,
                         features=feats_on, labels=labels_on)
    acc_off, _ = evaluate(probe, None, DEVICE, is_probe=True,
                          features=feats_off, labels=labels_off)
    spurious_gap = acc_on - acc_off

    # TRUE WGA on balanced test sets
    _, group_accs_bgoff, group_sizes_bgoff, wga_bgoff = evaluate_with_groups_v2(
        probe, None, DEVICE, is_probe=True,
        features=feats_bgoff, labels=labels_bgoff, groups=groups_bgoff)
    _, group_accs_patchoff, group_sizes_patchoff, wga_patchoff = evaluate_with_groups_v2(
        probe, None, DEVICE, is_probe=True,
        features=feats_patchoff, labels=labels_patchoff, groups=groups_patchoff)

    print(f"    acc_on={acc_on:.3f}  acc_off={acc_off:.3f}  gap={spurious_gap:.3f}")
    print(f"    TRUE WGA bgoff:    {wga_bgoff:.3f}  groups={group_accs_bgoff}")
    print(f"    TRUE WGA patchoff: {wga_patchoff:.3f}  groups={group_accs_patchoff}")

    return {
        "seed": seed,
        "acc_on": float(acc_on),
        "acc_off": float(acc_off),
        "spurious_gap": float(spurious_gap),
        "wga_bgoff": float(wga_bgoff),
        "wga_patchoff": float(wga_patchoff),
        "group_accs_bgoff": {str(k): float(v) for k, v in group_accs_bgoff.items()},
        "group_sizes_bgoff": {str(k): v for k, v in group_sizes_bgoff.items()},
        "group_accs_patchoff": {str(k): float(v) for k, v in group_accs_patchoff.items()},
        "group_sizes_patchoff": {str(k): v for k, v in group_sizes_patchoff.items()},
    }


# ═══════════════════════════════════════════════════
# AGGREGATE
# ═══════════════════════════════════════════════════

def aggregate(results, key):
    vals = [r[key] for r in results if not math.isnan(r[key])]
    if not vals:
        return {"mean": float("nan"), "std": float("nan"), "per_seed": []}
    return {
        "mean": float(np.mean(vals)),
        "std": float(np.std(vals)),
        "per_seed": [float(v) for v in vals],
    }


def aggregate_group_accs(results, key):
    """Aggregate per-group accuracies across seeds."""
    all_groups = {}
    for r in results:
        d = r[key]
        for g, v in d.items():
            if g not in all_groups:
                all_groups[g] = []
            if not math.isnan(v):
                all_groups[g].append(v)
    return {g: {"mean": float(np.mean(vs)) if vs else float("nan"),
                "std": float(np.std(vs)) if len(vs) > 1 else 0.0,
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
        print("  [WARNING] cv2 not available, skipping G5/G6 signal probes")
        g5 = {"gate": "lp_background_signal", "passed": None, "note": "cv2 not available"}
        g6 = {"gate": "lp_patch_signal", "passed": None, "note": "cv2 not available"}
        g7 = {"gate": "balanced_4group", "passed": None, "note": "cv2 not available"}

    gates["G5_bg_signal_probe"] = g5
    gates["G6_patch_signal_probe"] = g6
    gates["G7_balanced_groups"] = g7

    # Step 3: ERM baseline (3 seeds)
    print("\n" + "="*60)
    print("ERM BASELINE (3 seeds, fixed WGA evaluation)")
    print("="*60)
    erm_results = [run_erm_seed(s) for s in SEEDS]

    # Step 4: SimCLR (3 seeds)
    print("\n" + "="*60)
    print("SimCLR + LINEAR PROBE (3 seeds, fixed WGA evaluation)")
    print("="*60)
    simclr_results = [run_simclr_seed(s) for s in SEEDS]

    # Step 5: Aggregate
    erm_agg = {k: aggregate(erm_results, k)
               for k in ["acc_on", "acc_off", "spurious_gap", "wga_bgoff", "wga_patchoff"]}
    ssl_agg = {k: aggregate(simclr_results, k)
               for k in ["acc_on", "acc_off", "spurious_gap", "wga_bgoff", "wga_patchoff"]}

    erm_group_bgoff = aggregate_group_accs(erm_results, "group_accs_bgoff")
    ssl_group_bgoff = aggregate_group_accs(simclr_results, "group_accs_bgoff")
    erm_group_patchoff = aggregate_group_accs(erm_results, "group_accs_patchoff")
    ssl_group_patchoff = aggregate_group_accs(simclr_results, "group_accs_patchoff")

    delta_wga_bgoff = ssl_agg["wga_bgoff"]["mean"] - erm_agg["wga_bgoff"]["mean"]
    delta_wga_patchoff = erm_agg["wga_patchoff"]["mean"] - ssl_agg["wga_patchoff"]["mean"]
    delta_spurious_gap = ssl_agg["spurious_gap"]["mean"] - erm_agg["spurious_gap"]["mean"]

    elapsed = time.time() - t0

    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    print(f"\nERM:")
    print(f"  spurious_gap = {erm_agg['spurious_gap']['mean']:.3f} ± {erm_agg['spurious_gap']['std']:.3f}")
    print(f"  TRUE wga_bgoff    = {erm_agg['wga_bgoff']['mean']:.3f} ± {erm_agg['wga_bgoff']['std']:.3f}")
    print(f"  TRUE wga_patchoff = {erm_agg['wga_patchoff']['mean']:.3f} ± {erm_agg['wga_patchoff']['std']:.3f}")
    print(f"  ERM per-group bgoff:    {erm_group_bgoff}")
    print(f"  ERM per-group patchoff: {erm_group_patchoff}")

    print(f"\nSimCLR:")
    print(f"  spurious_gap = {ssl_agg['spurious_gap']['mean']:.3f} ± {ssl_agg['spurious_gap']['std']:.3f}")
    print(f"  TRUE wga_bgoff    = {ssl_agg['wga_bgoff']['mean']:.3f} ± {ssl_agg['wga_bgoff']['std']:.3f}")
    print(f"  TRUE wga_patchoff = {ssl_agg['wga_patchoff']['mean']:.3f} ± {ssl_agg['wga_patchoff']['std']:.3f}")
    print(f"  SSL per-group bgoff:    {ssl_group_bgoff}")
    print(f"  SSL per-group patchoff: {ssl_group_patchoff}")

    print(f"\nDeltas (SSL − ERM):")
    print(f"  Δwga_bgoff   = {delta_wga_bgoff:+.3f}  (predict: +5..+8pp)")
    print(f"  Δwga_patchoff = {-delta_wga_patchoff:+.3f}  (predict: ERM better by +3..+7pp)")
    print(f"  Δspurious_gap = {delta_spurious_gap:+.3f}")

    h9_pass = 0.05 <= delta_wga_bgoff <= 0.08
    h10_pass = 0.03 <= delta_wga_patchoff <= 0.07
    h11_pass = (abs(delta_wga_bgoff - delta_wga_patchoff) < 0.03 and
                delta_wga_bgoff > 0.03 and delta_wga_patchoff > 0.03)

    print(f"\nHypothesis checks (updated with TRUE WGA):")
    print(f"  H9  (SSL+5-8pp on bg WGA):    {h9_pass}  Δ={delta_wga_bgoff:+.3f}")
    print(f"  H10 (ERM+3-7pp on patch WGA): {h10_pass}  Δ={-delta_wga_patchoff:+.3f}")
    print(f"  H11 (shift balance <3pp):     {h11_pass}")

    gates_ok = all(v.get('passed', True) is not False
                   for v in gates.values() if isinstance(v, dict) and isinstance(v.get('passed'), bool))
    status = "SUCCESS" if gates_ok else "PARTIAL"

    results_json = {
        "status": status,
        "scale": "probe",
        "subject_executed": (
            f"refine-02: Fixed 4-group WGA; ERM vs SimCLR on Togglable-Signal CIFAR-10; "
            f"5k train, 1k test, {SEEDS} seeds; "
            f"ERM {ERM_EPOCHS}ep, SimCLR {SIMCLR_EPOCHS}ep pretrain + {PROBE_EPOCHS}ep probe; "
            f"WGA test sets: bg_spurious_rate=0.5, patch_spurious_rate=0.5 (independent)"
        ),
        "metrics": {
            "erm": {
                "spurious_gap": erm_agg["spurious_gap"],
                "wga_bgoff": erm_agg["wga_bgoff"],
                "wga_patchoff": erm_agg["wga_patchoff"],
                "acc_on": erm_agg["acc_on"],
                "acc_off": erm_agg["acc_off"],
                "per_group_bgoff": erm_group_bgoff,
                "per_group_patchoff": erm_group_patchoff,
                "per_seed_results": erm_results,
            },
            "simclr": {
                "spurious_gap": ssl_agg["spurious_gap"],
                "wga_bgoff": ssl_agg["wga_bgoff"],
                "wga_patchoff": ssl_agg["wga_patchoff"],
                "acc_on": ssl_agg["acc_on"],
                "acc_off": ssl_agg["acc_off"],
                "per_group_bgoff": ssl_group_bgoff,
                "per_group_patchoff": ssl_group_patchoff,
                "per_seed_results": simclr_results,
            },
            "delta_wga_bgoff_ssl_minus_erm": float(delta_wga_bgoff),
            "delta_wga_patchoff_erm_minus_ssl": float(delta_wga_patchoff),
            "delta_spurious_gap_ssl_minus_erm": float(delta_spurious_gap),
            "h9_ssl_better_on_bg": bool(h9_pass),
            "h10_erm_better_on_patch": bool(h10_pass),
            "h11_shift_balance": bool(h11_pass),
            "baseline_main01": {
                "erm_wga_bgoff_degenerate": 0.071,
                "erm_wga_patchoff_degenerate": 0.904,
                "simclr_wga_bgoff_degenerate": 0.240,
                "simclr_wga_patchoff_degenerate": 0.811,
                "note": "main-01 WGA was degenerate (only group 3 populated)"
            }
        },
        "sanity_gates": {
            "G1_fixed_seed": gates["G1_fixed_seed"]["passed"],
            "G2_loss_at_init": gates["G2_loss_at_init"]["passed"],
            "G3_dummy_classifier": gates["G3_dummy_classifier"]["passed"],
            "G4_overfit_batch": gates["G4_overfit_batch"]["passed"],
            "G5_bg_probe": g5.get("passed"),
            "G6_patch_probe": g6.get("passed"),
            "G7_balanced_groups": g7.get("passed"),
            "all_gates_passed": all_gates_passed,
            "details": {k: {kk: vv for kk, vv in v.items()
                             if kk not in ('losses_run1', 'losses_run2')}
                        for k, v in gates.items() if isinstance(v, dict)},
        },
        "config": {
            "subset_train": SUBSET_TRAIN,
            "subset_test": SUBSET_TEST,
            "erm_epochs": ERM_EPOCHS,
            "simclr_epochs": SIMCLR_EPOCHS,
            "probe_epochs": PROBE_EPOCHS,
            "batch_size": BATCH_SIZE,
            "seeds": SEEDS,
            "device": DEVICE,
            "wga_eval_bg_spurious_rate": 0.5,
            "wga_eval_patch_spurious_rate": 0.5,
            "fix_description": (
                "WGA test sets now use independent bg/patch spurious rates=0.5 "
                "so all 4 groups are populated (~25% each). "
                "This is the key fix vs main-01 where spurious_rate=1.0 put all "
                "test samples in group 3 only."
            )
        },
        "elapsed_seconds": round(elapsed, 1),
        "notes": (
            "refine-02: Fixed degenerate WGA. "
            "WGA test sets use independent bg/patch spurious_rate=0.5 for balanced 4-group coverage. "
            "bgoff WGA: group 0 (no shortcut) expected lowest acc; "
            "patchoff WGA: groups 0,1 (bg anti-correlated) expected lowest acc. "
            "ERM with strong spurious signals → low WGA on hard groups. "
            "SimCLR with color-jitter suppresses bg → better wga_bgoff, "
            "may shift reliance to patch → lower wga_patchoff."
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
