"""
Main experiment runner for main-01 (probe round).

Runs:
  1. Sanity gates (G1–G4)
  2. ERM baseline (3 seeds, reduced scale)
  3. SimCLR + linear probe (3 seeds, reduced scale)

Probe scale:
  - 5000 train / 1000 test samples
  - ERM: 20 epochs
  - SimCLR pretrain: 20 epochs
  - Linear probe: 20 epochs
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
from torch.utils.data import DataLoader
import torchvision.transforms as T

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
    evaluate, evaluate_with_groups, get_simclr_transform, TwoViewDataset, set_seed
)

CIFAR_ROOT = "/opt/datasets"
WEIGHTS_DIR = "/workspace/_weights"
RESULTS_DIR = "/workspace/results/main-01"

# ── Probe config ──
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
# SANITY GATES
# ═══════════════════════════════════════════════════

def run_sanity_gates():
    print("\n" + "="*60)
    print("SANITY GATES")
    print("="*60)

    # Load a small balanced test set for gate checks
    # Use signal-ON dataset
    test_ds = TogglableCIFAR10(
        CIFAR_ROOT, train=False, spurious_rate=1.0,
        bg_enabled=True, patch_enabled=True, seed=0,
        subset_size=SUBSET_TEST, transform=TEST_TRANSFORM,
    )
    # Build a perfectly balanced subset of 100 samples (10 per class)
    from collections import defaultdict
    class_indices = defaultdict(list)
    for i, (_, lbl) in enumerate(test_ds):
        class_indices[lbl].append(i)
    balanced_idx = []
    for c in range(10):
        balanced_idx.extend(class_indices[c][:10])
    balanced_subset = torch.utils.data.Subset(test_ds, balanced_idx)
    balanced_loader = DataLoader(balanced_subset, batch_size=100, shuffle=False)

    # Small train dataset for gate checks
    train_ds = TogglableCIFAR10(
        CIFAR_ROOT, train=True, spurious_rate=1.0,
        bg_enabled=True, patch_enabled=True, seed=0,
        subset_size=500, transform=TRAIN_TRANSFORM,
    )
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=2)

    def model_factory():
        return get_resnet18(num_classes=10)

    # G1: Fixed seed
    print("\nG1: Fixed seed reproducibility...")
    g1 = gate_fixed_seed(model_factory, train_loader, DEVICE, seed=42, n_steps=10)
    print(f"  max_diff={g1['max_diff']:.2e}  PASS={g1['passed']}")

    # G2: Loss at init
    print("\nG2: Loss at init ≈ ln(10) = 2.303...")
    g2 = gate_loss_at_init(model_factory, train_loader, DEVICE, seed=42)
    print(f"  loss_0={g2['loss_0']:.4f}  expected={g2['expected']:.4f}  PASS={g2['passed']}")

    # G3: Dummy classifier
    print("\nG3: Dummy classifier → ~10% accuracy...")
    g3 = gate_dummy_classifier(balanced_loader, DEVICE)
    print(f"  dummy_acc={g3['dummy_acc_pct']:.1f}%  PASS={g3['passed']}")

    # G4: Overfit one batch
    print("\nG4: Overfit single batch (32 samples, 200 SGD steps)...")
    x_batch, y_batch = next(iter(DataLoader(train_ds, batch_size=32, shuffle=False)))
    g4 = gate_overfit_one_batch(model_factory, (x_batch, y_batch), DEVICE, seed=42)
    print(f"  train_loss@200={g4['train_loss_step200']:.6f}  PASS={g4['passed']}")

    gates = {
        "G1_fixed_seed": g1,
        "G2_loss_at_init": g2,
        "G3_dummy_classifier": g3,
        "G4_overfit_batch": g4,
    }
    all_passed = all(g['passed'] for g in gates.values())
    print(f"\nAll gates passed: {all_passed}")
    if not all_passed:
        failed = [k for k, v in gates.items() if not v['passed']]
        print(f"  FAILED gates: {failed}")
    return gates, all_passed


# ═══════════════════════════════════════════════════
# LINEAR PROBE SANITY (G5+G6): Background & patch
# ═══════════════════════════════════════════════════

def run_linear_probe_sanity():
    """
    G5: HSV background histogram → linear probe ≥85% at ρ=1.0, ≤15% at ρ=0.0
    G6: Patch histogram → linear probe ≥85% at ρ=1.0, ≤15% at ρ=0.0
    """
    print("\n" + "="*60)
    print("SIGNAL SANITY (G5+G6: Linear Probe on Raw Features)")
    print("="*60)

    from sklearn.linear_model import LogisticRegression

    def compute_probes(spurious_rate):
        ds = TogglableCIFAR10(
            CIFAR_ROOT, train=True, spurious_rate=spurious_rate,
            bg_enabled=True, patch_enabled=True, seed=0,
            subset_size=SUBSET_TRAIN, transform=None,  # raw numpy
        )
        import cv2
        bg_feats, patch_feats, lbls = [], [], []
        for i in range(len(ds.images)):
            img = ds.images[i].copy()
            if ds.bg_enabled:
                img = apply_background_tint(img, BG_HUES[ds.bg_class[i]])
            if ds.patch_enabled:
                img = apply_color_patch(img, PATCH_HUES[ds.patch_class[i]])
            # Background histogram (full image HSV)
            img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
            h_hist = np.histogram(img_hsv[:, :, 0], bins=16, range=(0, 180))[0].astype(np.float32)
            h_hist /= (h_hist.sum() + 1e-8)
            bg_feats.append(h_hist)
            # Patch histogram (top-left 4×4)
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

    print("  Computing probes at ρ=1.0 ...")
    acc_bg_1, acc_patch_1 = compute_probes(1.0)
    print(f"    BG probe acc = {acc_bg_1:.3f} (need ≥0.85)")
    print(f"    Patch probe acc = {acc_patch_1:.3f} (need ≥0.85)")

    print("  Computing probes at ρ=0.0 ...")
    acc_bg_0, acc_patch_0 = compute_probes(0.0)
    print(f"    BG probe acc = {acc_bg_0:.3f} (need ≤0.15)")
    print(f"    Patch probe acc = {acc_patch_0:.3f} (need ≤0.15)")

    g5 = {
        "gate": "lp_background_signal",
        "passed": acc_bg_1 >= 0.85 and acc_bg_0 <= 0.15,
        "lp_bg_acc_rho1": float(acc_bg_1),
        "lp_bg_acc_rho0": float(acc_bg_0),
    }
    g6 = {
        "gate": "lp_patch_signal",
        "passed": acc_patch_1 >= 0.85 and acc_patch_0 <= 0.15,
        "lp_patch_acc_rho1": float(acc_patch_1),
        "lp_patch_acc_rho0": float(acc_patch_0),
    }
    print(f"\nG5 (bg signal): PASS={g5['passed']}")
    print(f"G6 (patch signal): PASS={g6['passed']}")
    return g5, g6


# ═══════════════════════════════════════════════════
# ONE SEED: ERM
# ═══════════════════════════════════════════════════

def run_erm_seed(seed: int):
    print(f"\n  [ERM seed={seed}]")
    set_seed(seed)

    # Train dataset: signals ON, spurious_rate=1.0
    train_ds = TogglableCIFAR10(
        CIFAR_ROOT, train=True, spurious_rate=1.0,
        bg_enabled=True, patch_enabled=True, seed=seed,
        subset_size=SUBSET_TRAIN, transform=TRAIN_TRANSFORM,
        return_group=True,
    )
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=True)

    # Test datasets: signal-ON and signal-OFF
    test_on_ds = TogglableCIFAR10(
        CIFAR_ROOT, train=False, spurious_rate=1.0,
        bg_enabled=True, patch_enabled=True, seed=seed,
        subset_size=SUBSET_TEST, transform=TEST_TRANSFORM,
        return_group=True,
    )
    test_off_ds = TogglableCIFAR10(
        CIFAR_ROOT, train=False, spurious_rate=1.0,
        bg_enabled=False, patch_enabled=False, seed=seed,
        subset_size=SUBSET_TEST, transform=TEST_TRANSFORM,
        return_group=True,
    )
    # Background-flip: bg OFF, patch ON
    test_bgoff_ds = TogglableCIFAR10(
        CIFAR_ROOT, train=False, spurious_rate=1.0,
        bg_enabled=False, patch_enabled=True, seed=seed,
        subset_size=SUBSET_TEST, transform=TEST_TRANSFORM,
        return_group=True,
    )
    # Patch-flip: bg ON, patch OFF
    test_patchoff_ds = TogglableCIFAR10(
        CIFAR_ROOT, train=False, spurious_rate=1.0,
        bg_enabled=True, patch_enabled=False, seed=seed,
        subset_size=SUBSET_TEST, transform=TEST_TRANSFORM,
        return_group=True,
    )

    test_on_loader = DataLoader(test_on_ds, batch_size=256, shuffle=False,
                                num_workers=2, pin_memory=True)
    test_off_loader = DataLoader(test_off_ds, batch_size=256, shuffle=False,
                                 num_workers=2, pin_memory=True)
    test_bgoff_loader = DataLoader(test_bgoff_ds, batch_size=256, shuffle=False,
                                   num_workers=2, pin_memory=True)
    test_patchoff_loader = DataLoader(test_patchoff_ds, batch_size=256, shuffle=False,
                                      num_workers=2, pin_memory=True)

    # Model
    model = get_resnet18(num_classes=NUM_CLASSES).to(DEVICE)
    optimizer = torch.optim.SGD(model.parameters(), lr=LR_ERM,
                                momentum=0.9, weight_decay=1e-4)
    train_erm(model, train_loader, optimizer, DEVICE, ERM_EPOCHS)

    # Evaluate
    acc_on, _ = evaluate(model, test_on_loader, DEVICE)
    acc_off, _ = evaluate(model, test_off_loader, DEVICE)
    _, group_accs_bgoff, wga_bgoff = evaluate_with_groups(model, test_bgoff_loader, DEVICE)
    _, group_accs_patchoff, wga_patchoff = evaluate_with_groups(model, test_patchoff_loader, DEVICE)
    _, group_accs_on, wga_on = evaluate_with_groups(model, test_on_loader, DEVICE)

    spurious_gap = acc_on - acc_off
    print(f"    acc_on={acc_on:.3f}  acc_off={acc_off:.3f}  gap={spurious_gap:.3f}")
    print(f"    wga_bgoff={wga_bgoff:.3f}  wga_patchoff={wga_patchoff:.3f}  wga_on={wga_on:.3f}")

    return {
        "seed": seed,
        "acc_on": float(acc_on),
        "acc_off": float(acc_off),
        "spurious_gap": float(spurious_gap),
        "wga_bgoff": float(wga_bgoff),   # WGA when background is flipped
        "wga_patchoff": float(wga_patchoff),  # WGA when patch is flipped
        "wga_on": float(wga_on),
        "group_accs_bgoff": {str(k): float(v) for k, v in group_accs_bgoff.items()},
        "group_accs_patchoff": {str(k): float(v) for k, v in group_accs_patchoff.items()},
    }


# ═══════════════════════════════════════════════════
# ONE SEED: SimCLR + Linear Probe
# ═══════════════════════════════════════════════════

def run_simclr_seed(seed: int):
    print(f"\n  [SimCLR seed={seed}]")
    set_seed(seed)

    # Pretrain dataset (two-view, spurious_rate=1.0)
    pretrain_base = TogglableCIFAR10(
        CIFAR_ROOT, train=True, spurious_rate=1.0,
        bg_enabled=True, patch_enabled=True, seed=seed,
        subset_size=SUBSET_TRAIN, transform=None,
    )
    simclr_transform = get_simclr_transform()
    pretrain_ds = TwoViewDataset(pretrain_base, simclr_transform)
    pretrain_loader = DataLoader(pretrain_ds, batch_size=BATCH_SIZE, shuffle=True,
                                 num_workers=4, pin_memory=True, drop_last=True)

    # SimCLR pretraining
    backbone = SimCLRBackbone(proj_dim=128).to(DEVICE)
    optimizer = torch.optim.SGD(backbone.parameters(), lr=LR_SIMCLR,
                                momentum=0.9, weight_decay=1e-4)
    train_simclr(backbone, pretrain_loader, optimizer, DEVICE, SIMCLR_EPOCHS, TEMPERATURE)

    # Save backbone weights
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    torch.save(backbone.state_dict(),
               os.path.join(WEIGHTS_DIR, f"simclr_seed{seed}.pt"))

    # Linear probe: extract features from frozen backbone
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

    # Test sets
    def get_test_feats(bg_enabled, patch_enabled):
        ds = TogglableCIFAR10(
            CIFAR_ROOT, train=False, spurious_rate=1.0,
            bg_enabled=bg_enabled, patch_enabled=patch_enabled, seed=seed,
            subset_size=SUBSET_TEST, transform=TEST_TRANSFORM,
            return_group=True,
        )
        loader = DataLoader(ds, batch_size=256, shuffle=False,
                            num_workers=2, pin_memory=True)
        feats, labels = extract_features(backbone, loader, DEVICE)
        groups = torch.tensor(ds.group_labels)
        return feats, labels, groups

    feats_on, labels_on, groups_on = get_test_feats(True, True)
    feats_off, labels_off, groups_off = get_test_feats(False, False)
    feats_bgoff, labels_bgoff, groups_bgoff = get_test_feats(False, True)
    feats_patchoff, labels_patchoff, groups_patchoff = get_test_feats(True, False)

    acc_on, _ = evaluate(probe, None, DEVICE, is_probe=True,
                         features=feats_on, labels=labels_on)
    acc_off, _ = evaluate(probe, None, DEVICE, is_probe=True,
                          features=feats_off, labels=labels_off)
    _, group_accs_bgoff, wga_bgoff = evaluate_with_groups(
        probe, None, DEVICE, is_probe=True,
        features=feats_bgoff, labels=labels_bgoff, groups=groups_bgoff)
    _, group_accs_patchoff, wga_patchoff = evaluate_with_groups(
        probe, None, DEVICE, is_probe=True,
        features=feats_patchoff, labels=labels_patchoff, groups=groups_patchoff)
    _, group_accs_on, wga_on = evaluate_with_groups(
        probe, None, DEVICE, is_probe=True,
        features=feats_on, labels=labels_on, groups=groups_on)

    spurious_gap = acc_on - acc_off
    print(f"    acc_on={acc_on:.3f}  acc_off={acc_off:.3f}  gap={spurious_gap:.3f}")
    print(f"    wga_bgoff={wga_bgoff:.3f}  wga_patchoff={wga_patchoff:.3f}  wga_on={wga_on:.3f}")

    return {
        "seed": seed,
        "acc_on": float(acc_on),
        "acc_off": float(acc_off),
        "spurious_gap": float(spurious_gap),
        "wga_bgoff": float(wga_bgoff),
        "wga_patchoff": float(wga_patchoff),
        "wga_on": float(wga_on),
        "group_accs_bgoff": {str(k): float(v) for k, v in group_accs_bgoff.items()},
        "group_accs_patchoff": {str(k): float(v) for k, v in group_accs_patchoff.items()},
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


# ═══════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════

def main():
    t0 = time.time()

    # Step 1: Sanity gates
    gates, all_gates_passed = run_sanity_gates()

    # Step 2: Signal sanity gates G5/G6
    # Skip cv2-based ones if not available; mark as skipped
    try:
        import cv2
        g5, g6 = run_linear_probe_sanity()
    except ImportError:
        print("  [WARNING] cv2 not available, skipping G5/G6 signal probes")
        g5 = {"gate": "lp_background_signal", "passed": None, "note": "cv2 not available"}
        g6 = {"gate": "lp_patch_signal", "passed": None, "note": "cv2 not available"}

    gates["G5_bg_signal_probe"] = g5
    gates["G6_patch_signal_probe"] = g6

    # Step 3: ERM baseline (3 seeds)
    print("\n" + "="*60)
    print("ERM BASELINE (3 seeds)")
    print("="*60)
    erm_results = []
    for seed in SEEDS:
        r = run_erm_seed(seed)
        erm_results.append(r)

    # Step 4: SimCLR (3 seeds)
    print("\n" + "="*60)
    print("SimCLR + LINEAR PROBE (3 seeds)")
    print("="*60)
    simclr_results = []
    for seed in SEEDS:
        r = run_simclr_seed(seed)
        simclr_results.append(r)

    # Step 5: Aggregate and compute deltas
    erm_agg = {k: aggregate(erm_results, k)
               for k in ["acc_on", "acc_off", "spurious_gap", "wga_bgoff", "wga_patchoff", "wga_on"]}
    ssl_agg = {k: aggregate(simclr_results, k)
               for k in ["acc_on", "acc_off", "spurious_gap", "wga_bgoff", "wga_patchoff", "wga_on"]}

    # Key metrics (from the hypothesis chain):
    # H8: lp_simclr_background_acc < lp_erm_background_acc − 10pp (suppression)
    # H9: wga_ssl_bgoff − wga_erm_bgoff ∈ [+5pp, +8pp]  (SSL better on bg-flip)
    # H10: wga_erm_patchoff − wga_ssl_patchoff ∈ [+3pp, +7pp]  (SSL worse on patch-flip)

    delta_wga_bgoff = ssl_agg["wga_bgoff"]["mean"] - erm_agg["wga_bgoff"]["mean"]
    delta_wga_patchoff = erm_agg["wga_patchoff"]["mean"] - ssl_agg["wga_patchoff"]["mean"]
    delta_spurious_gap = ssl_agg["spurious_gap"]["mean"] - erm_agg["spurious_gap"]["mean"]

    elapsed = time.time() - t0

    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    print(f"\nERM:")
    print(f"  spurious_gap = {erm_agg['spurious_gap']['mean']:.3f} ± {erm_agg['spurious_gap']['std']:.3f}")
    print(f"  wga_bgoff    = {erm_agg['wga_bgoff']['mean']:.3f} ± {erm_agg['wga_bgoff']['std']:.3f}")
    print(f"  wga_patchoff = {erm_agg['wga_patchoff']['mean']:.3f} ± {erm_agg['wga_patchoff']['std']:.3f}")

    print(f"\nSimCLR:")
    print(f"  spurious_gap = {ssl_agg['spurious_gap']['mean']:.3f} ± {ssl_agg['spurious_gap']['std']:.3f}")
    print(f"  wga_bgoff    = {ssl_agg['wga_bgoff']['mean']:.3f} ± {ssl_agg['wga_bgoff']['std']:.3f}")
    print(f"  wga_patchoff = {ssl_agg['wga_patchoff']['mean']:.3f} ± {ssl_agg['wga_patchoff']['std']:.3f}")

    print(f"\nDeltas (SSL − ERM):")
    print(f"  Δwga_bgoff   = {delta_wga_bgoff:+.3f}  (predict: +5..+8pp)")
    print(f"  Δwga_patchoff = {-delta_wga_patchoff:+.3f}  (predict: ERM better by +3..+7pp → SSL worse)")
    print(f"  Δspurious_gap = {delta_spurious_gap:+.3f}")

    # Gate outcomes for hypothesis checks
    h8_pass = None  # can't check without ERM feature-level probes here
    h9_pass = 0.05 <= delta_wga_bgoff <= 0.08
    h10_pass = 0.03 <= delta_wga_patchoff <= 0.07
    h11_pass = (abs(delta_wga_bgoff - delta_wga_patchoff) < 0.03 and
                delta_wga_bgoff > 0.03 and delta_wga_patchoff > 0.03)

    print(f"\nHypothesis checks:")
    print(f"  H9  (SSL+5-8pp on bg):    {h9_pass}")
    print(f"  H10 (ERM+3-7pp on patch): {h10_pass}")
    print(f"  H11 (shift balance):      {h11_pass}")

    # Determine status
    gates_ok = all(v.get('passed', True) is not False
                   for v in gates.values() if isinstance(v.get('passed'), bool))
    status = "SUCCESS" if gates_ok else "PARTIAL"

    results_json = {
        "status": status,
        "scale": "probe",
        "subject_executed": (
            f"SSL Spurious-Correlation Suppress-vs-Shift Audit; "
            f"ERM vs SimCLR on Togglable-Signal CIFAR-10; "
            f"5k train, 1k test, {SEEDS} seeds; "
            f"ERM {ERM_EPOCHS}ep, SimCLR {SIMCLR_EPOCHS}ep pretrain + {PROBE_EPOCHS}ep linear probe"
        ),
        "metrics": {
            "erm": {
                "spurious_gap": erm_agg["spurious_gap"],
                "wga_bgoff": erm_agg["wga_bgoff"],
                "wga_patchoff": erm_agg["wga_patchoff"],
                "acc_on": erm_agg["acc_on"],
                "acc_off": erm_agg["acc_off"],
                "wga_on": erm_agg["wga_on"],
                "per_seed_results": erm_results,
            },
            "simclr": {
                "spurious_gap": ssl_agg["spurious_gap"],
                "wga_bgoff": ssl_agg["wga_bgoff"],
                "wga_patchoff": ssl_agg["wga_patchoff"],
                "acc_on": ssl_agg["acc_on"],
                "acc_off": ssl_agg["acc_off"],
                "wga_on": ssl_agg["wga_on"],
                "per_seed_results": simclr_results,
            },
            "delta_wga_bgoff_ssl_minus_erm": float(delta_wga_bgoff),
            "delta_wga_patchoff_erm_minus_ssl": float(delta_wga_patchoff),
            "delta_spurious_gap_ssl_minus_erm": float(delta_spurious_gap),
            "h9_ssl_better_on_bg": h9_pass,
            "h10_erm_better_on_patch": h10_pass,
            "h11_shift_balance": h11_pass,
        },
        "sanity_gates": {
            "G1_fixed_seed": gates["G1_fixed_seed"]["passed"],
            "G2_loss_at_init": gates["G2_loss_at_init"]["passed"],
            "G3_dummy_classifier": gates["G3_dummy_classifier"]["passed"],
            "G4_overfit_batch": gates["G4_overfit_batch"]["passed"],
            "G5_bg_probe": g5.get("passed"),
            "G6_patch_probe": g6.get("passed"),
            "all_gates_passed": all_gates_passed,
            "details": {k: {kk: vv for kk, vv in v.items() if kk not in ('losses_run1', 'losses_run2')}
                        for k, v in gates.items()},
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
        },
        "elapsed_seconds": round(elapsed, 1),
        "notes": (
            "Probe run: 5k train / 1k test, few epochs. "
            "ERM uses standard supervised training with color augmented CIFAR-10 "
            "tinted per-class (bg) + color patch (corner). "
            "SimCLR uses strong color-jitter augmentation which should suppress BG color. "
            "Patch is preserved in random crops. "
            "wga_bgoff = WGA when background color signal is removed at test time. "
            "wga_patchoff = WGA when color patch signal is removed at test time."
        ),
    }

    # Save results
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
