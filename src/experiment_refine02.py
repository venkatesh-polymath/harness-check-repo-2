"""
experiment_refine02.py — Probe round refine-02 for:
  Deep Ensemble vs. MC-Dropout vs. Temperature-Scaled MSP
  Pseudo-Label Precision Pareto at 50 labels/class on CIFAR-10.

FIXES vs. main-01:
  1. LR = 0.1 (was 3e-3 — too low for SGD on CIFAR-10 → only 31% at 50 epochs).
     With LR=0.1 + cosine annealing, 100 epochs should reach 55–65% accuracy,
     putting AUPPC in the spec sanity band [0.70, 0.88].
  2. Added Temperature Scaling arm (missing from main-01).
     Fits a scalar T on 10% held-out labeled val split;
     reports AUPPC / acc / ECE for TS-MSP alongside MSP, MCD, Ensemble.

Probe settings (refine-02):
  - 50 labels/class (500 total labeled), hold out 50 for TS val split
  - 100 epochs (reduced from spec 200; sufficient with LR=0.1)
  - K=3 ensemble members (spec K=5; probe)
  - T=20 MC-Dropout passes, p=0.1
  - 3 seeds
"""

import os
import sys
import json
import time
import random
import logging
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
from sklearn.metrics import auc as sk_auc

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log.info(f"Device: {DEVICE}")

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT    = Path("/workspace")
RESULTS = ROOT / "results" / "refine-02"
WEIGHTS = ROOT / "_weights"
DATA_DIR = Path("/opt/datasets")
RESULTS.mkdir(parents=True, exist_ok=True)
WEIGHTS.mkdir(parents=True, exist_ok=True)

# ── Hyperparameters ───────────────────────────────────────────────────────────
LABELS_PER_CLASS  = 50
NUM_CLASSES       = 10
BATCH_SIZE        = 64
PROBE_EPOCHS      = 100          # was 50 in main-01; 100 with LR=0.1 → ~60% acc
ENSEMBLE_K        = 3
MC_PASSES         = 20
MC_DROPOUT_P      = 0.1
LR                = 0.1          # KEY FIX: was 3e-3 → now 0.1 (standard SGD CIFAR-10)
WEIGHT_DECAY      = 5e-4
THRESHOLD_GRID    = np.linspace(0.50, 0.99, 20)
SEEDS             = [42, 7, 123]
TS_VAL_FRACTION   = 0.10         # 10% of labeled set for temperature scaling calibration

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD  = (0.247,  0.243,  0.261)


# ── Seed helper ───────────────────────────────────────────────────────────────
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ── ResNet-8 for CIFAR-10 ─────────────────────────────────────────────────────
class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, dropout_p=0.0):
        super().__init__()
        self.dropout_p = dropout_p
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)
        self.skip  = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.skip = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        if self.dropout_p > 0:
            out = F.dropout(out, p=self.dropout_p, training=True)  # always-on for MC-Dropout
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.skip(x))


class ResNet8(nn.Module):
    """
    ResNet-8: conv(16) → block(16,16) → block(16,32,s=2) → block(32,64,s=2) → GAP → fc(10)
    dropout_p=0 for baseline/ensemble; dropout_p>0 for MC-Dropout (kept at inference time).
    """
    def __init__(self, dropout_p=0.0):
        super().__init__()
        self.dropout_p = dropout_p
        self.stem = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )
        self.layer1 = ResBlock(16, 16, dropout_p=dropout_p)
        self.layer2 = ResBlock(16, 32, stride=2, dropout_p=dropout_p)
        self.layer3 = ResBlock(32, 64, stride=2, dropout_p=dropout_p)
        self.pool   = nn.AdaptiveAvgPool2d(1)
        self.fc     = nn.Linear(64, NUM_CLASSES)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


# ── Data ──────────────────────────────────────────────────────────────────────
def get_datasets():
    train_tf = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    train_full = torchvision.datasets.CIFAR10(DATA_DIR, train=True,  download=False, transform=train_tf)
    train_notf = torchvision.datasets.CIFAR10(DATA_DIR, train=True,  download=False, transform=test_tf)
    test_ds    = torchvision.datasets.CIFAR10(DATA_DIR, train=False, download=False, transform=test_tf)
    return train_full, train_notf, test_ds


def stratified_split(dataset, labels_per_class: int, seed: int, val_fraction: float = 0.0):
    """
    Return (train_labeled_indices, val_labeled_indices, unlabeled_indices).
    val_fraction: fraction of labeled per class held out for calibration (TS arm).
    """
    rng = np.random.RandomState(seed)
    targets = np.array(dataset.targets)
    train_lbl, val_lbl, unlabeled = [], [], []
    for c in range(NUM_CLASSES):
        idx = np.where(targets == c)[0]
        chosen = rng.choice(idx, size=labels_per_class, replace=False)
        rng.shuffle(chosen)
        n_val = max(1, int(len(chosen) * val_fraction))  # at least 1 per class
        val_lbl.extend(chosen[:n_val].tolist())
        train_lbl.extend(chosen[n_val:].tolist())
        unlabeled.extend(list(set(idx.tolist()) - set(chosen.tolist())))
    return train_lbl, val_lbl, unlabeled


# ── Training ──────────────────────────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, scheduler=None):
    model.train()
    total_loss, total_correct, n = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        logits = model(x)
        loss   = F.cross_entropy(logits, y)
        loss.backward()
        optimizer.step()
        total_loss    += loss.item() * x.size(0)
        total_correct += (logits.argmax(1) == y).sum().item()
        n             += x.size(0)
    if scheduler is not None:
        scheduler.step()
    return total_loss / n, total_correct / n


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    correct, n = 0, 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        pred  = model(x).argmax(1)
        correct += (pred == y).sum().item()
        n       += x.size(0)
    return correct / n


def train_model(model, labeled_loader, test_loader, epochs, lr=LR, wd=WEIGHT_DECAY):
    """Train model, return final test accuracy."""
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=wd)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    for ep in range(1, epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, labeled_loader, optimizer, scheduler)
        if ep % 20 == 0 or ep == 1:
            te_acc = evaluate(model, test_loader)
            log.info(f"  ep {ep:3d}/{epochs}: loss={tr_loss:.4f} tr_acc={tr_acc:.3f} te_acc={te_acc:.3f}")
    return evaluate(model, test_loader)


# ── Temperature Scaling ───────────────────────────────────────────────────────
def fit_temperature(model, val_loader):
    """
    Fit scalar temperature T on held-out labeled val set via NLL minimization.
    Returns the optimal temperature scalar (float).
    """
    model.eval()
    logits_list, labels_list = [], []
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(DEVICE)
            logits_list.append(model(x).cpu())
            labels_list.append(y)
    logits = torch.cat(logits_list).to(DEVICE)
    labels = torch.cat(labels_list).to(DEVICE)

    temperature = nn.Parameter(torch.ones(1, device=DEVICE) * 1.5)
    optimizer   = optim.LBFGS([temperature], lr=0.05, max_iter=100)

    def eval_fn():
        optimizer.zero_grad()
        scaled_logits = logits / temperature.clamp(min=0.05)
        loss = F.cross_entropy(scaled_logits, labels)
        loss.backward()
        return loss

    optimizer.step(eval_fn)
    T = float(temperature.clamp(min=0.05).item())
    log.info(f"  Temperature scaling: T = {T:.4f}")
    return T


# ── Pseudo-label scoring helpers ──────────────────────────────────────────────
@torch.no_grad()
def msp_scores(model, loader):
    """Maximum Softmax Probability scores."""
    model.eval()
    probs_all, preds_all, labels_all = [], [], []
    for x, y in loader:
        x = x.to(DEVICE)
        p = F.softmax(model(x), dim=1)
        probs_all.append(p.cpu())
        preds_all.append(p.argmax(1).cpu())
        labels_all.append(y)
    probs  = torch.cat(probs_all)
    preds  = torch.cat(preds_all)
    labels = torch.cat(labels_all)
    return probs.max(1).values.numpy(), preds.numpy(), labels.numpy()


@torch.no_grad()
def ts_scores(model, loader, temperature: float):
    """Temperature-scaled MSP scores (same model, scaled logits)."""
    model.eval()
    probs_all, preds_all, labels_all = [], [], []
    for x, y in loader:
        x = x.to(DEVICE)
        p = F.softmax(model(x) / temperature, dim=1)
        probs_all.append(p.cpu())
        preds_all.append(p.argmax(1).cpu())
        labels_all.append(y)
    probs  = torch.cat(probs_all)
    preds  = torch.cat(preds_all)
    labels = torch.cat(labels_all)
    return probs.max(1).values.numpy(), preds.numpy(), labels.numpy()


def mc_dropout_scores(model, loader, T=20):
    """MC-Dropout: mean softmax over T stochastic forward passes."""
    model.train()  # keep dropout on
    all_probs_runs = []
    labels_run = None
    for _ in range(T):
        probs_run, labs = [], []
        with torch.no_grad():
            for x, y in loader:
                x = x.to(DEVICE)
                p = F.softmax(model(x), dim=1)
                probs_run.append(p.cpu())
                labs.append(y)
        all_probs_runs.append(torch.cat(probs_run))
        labels_run = labs
    mean_probs = torch.stack(all_probs_runs).mean(0)
    preds  = mean_probs.argmax(1).numpy()
    labels = torch.cat(labels_run).numpy()
    return mean_probs.max(1).values.numpy(), preds, labels


def ensemble_scores(models, loader):
    """Deep Ensemble: average softmax across K models."""
    all_probs = []
    labels_run = None
    for model in models:
        model.eval()
        run_probs, labs = [], []
        with torch.no_grad():
            for x, y in loader:
                x = x.to(DEVICE)
                p = F.softmax(model(x), dim=1)
                run_probs.append(p.cpu())
                labs.append(y)
        all_probs.append(torch.cat(run_probs))
        labels_run = labs
    mean_probs = torch.stack(all_probs).mean(0)
    preds  = mean_probs.argmax(1).numpy()
    labels = torch.cat(labels_run).numpy()
    return mean_probs.max(1).values.numpy(), preds, labels


# ── Metrics ───────────────────────────────────────────────────────────────────
def compute_auppc(scores, preds, labels, thresholds=THRESHOLD_GRID):
    """
    Area Under Precision-Coverage curve.
    At each threshold τ: retain samples where score >= τ.
    precision = fraction correct among retained.
    coverage  = fraction retained.
    """
    precisions, coverages = [], []
    for tau in thresholds:
        mask = scores >= tau
        if mask.sum() == 0:
            precisions.append(1.0)
            coverages.append(0.0)
        else:
            precision = (preds[mask] == labels[mask]).mean()
            coverage  = mask.mean()
            precisions.append(float(precision))
            coverages.append(float(coverage))
    coverages  = np.array(coverages)
    precisions = np.array(precisions)
    idx = np.argsort(coverages)
    coverages  = coverages[idx]
    precisions = precisions[idx]
    if len(np.unique(coverages)) < 2:
        return float(np.mean(precisions))
    return float(sk_auc(coverages, precisions))


def compute_ece(scores, preds, labels, n_bins=10):
    """ECE using max-softmax confidence."""
    accuracies = (preds == labels).astype(float)
    bin_edges  = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n   = len(labels)
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (scores >= lo) & (scores < hi)
        if mask.sum() == 0:
            continue
        acc  = accuracies[mask].mean()
        conf = scores[mask].mean()
        ece  += mask.sum() / n * abs(acc - conf)
    return float(ece)


# ═══════════════════════════════════════════════════════════════════════════════
# SANITY GATES
# ═══════════════════════════════════════════════════════════════════════════════

def gate_loss_at_init(train_full):
    """Gate 2: Initial CE loss ≈ ln(10) ≈ 2.3026."""
    log.info("── Gate: Loss at init ──")
    set_seed(42)
    model  = ResNet8().to(DEVICE)
    lbl, _, _ = stratified_split(train_full, LABELS_PER_CLASS, seed=42)
    loader = DataLoader(Subset(train_full, lbl), batch_size=len(lbl),
                        shuffle=False, num_workers=2)
    model.eval()
    x, y = next(iter(loader))
    x, y = x.to(DEVICE), y.to(DEVICE)
    with torch.no_grad():
        loss = F.cross_entropy(model(x), y).item()
    passed = 2.28 <= loss <= 2.33
    log.info(f"  Init CE loss = {loss:.6f}  (expect [2.28, 2.33]) → {'PASS' if passed else 'FAIL'}")
    return {"name": "loss_at_init", "passed": passed, "init_ce_loss": loss}


def gate_uniform_baseline(test_ds):
    """Gate 3: Uniform logit model → ~10% accuracy."""
    log.info("── Gate: Uniform baseline ──")
    class UniformModel(nn.Module):
        def forward(self, x):
            return torch.zeros(x.size(0), NUM_CLASSES, device=x.device)
    model  = UniformModel().to(DEVICE)
    loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=2)
    acc    = evaluate(model, loader)
    passed = 0.098 <= acc <= 0.102
    log.info(f"  Uniform accuracy = {acc:.4f}  (expect [0.098, 0.102]) → {'PASS' if passed else 'FAIL'}")
    return {"name": "uniform_baseline", "passed": passed, "uniform_accuracy": float(acc)}


def gate_overfit_one_batch(train_full):
    """Gate 4: Overfit single batch of 64 to CE < 0.01 within 200 steps."""
    log.info("── Gate: Overfit one batch ──")
    set_seed(42)
    model  = ResNet8().to(DEVICE)
    loader = DataLoader(train_full, batch_size=64, shuffle=True, num_workers=2)
    x, y   = next(iter(loader))
    x, y   = x.to(DEVICE), y.to(DEVICE)
    optimizer  = optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
    final_loss = None
    for step in range(200):
        model.train()
        optimizer.zero_grad()
        loss = F.cross_entropy(model(x), y)
        loss.backward()
        optimizer.step()
        final_loss = loss.item()
        if final_loss < 0.01:
            log.info(f"  Reached CE < 0.01 at step {step+1}: loss={final_loss:.6f}")
            break
    passed = final_loss < 0.01
    log.info(f"  Final batch loss at step 200: {final_loss:.6f} → {'PASS' if passed else 'FAIL'}")
    return {"name": "overfit_one_batch", "passed": passed,
            "single_batch_loss_at_step200": float(final_loss)}


def gate_reproducibility(train_full):
    """Gate 1: Fixed seed → bit-identical loss at step 1."""
    log.info("── Gate: Reproducibility ──")
    results = []
    for run in range(2):
        set_seed(42)
        model  = ResNet8().to(DEVICE)
        lbl, _, _ = stratified_split(train_full, LABELS_PER_CLASS, seed=42)
        loader = DataLoader(Subset(train_full, lbl), batch_size=BATCH_SIZE,
                            shuffle=True, num_workers=2)
        optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=WEIGHT_DECAY)
        model.train()
        x, y   = next(iter(loader))
        x, y   = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        loss = F.cross_entropy(model(x), y)
        loss.backward()
        optimizer.step()
        results.append(loss.item())
        log.info(f"  Run {run+1} first-batch loss: {loss.item():.8f}")
    diff   = abs(results[0] - results[1])
    passed = diff == 0.0
    log.info(f"  |Δ loss| = {diff:.2e}  → {'PASS' if passed else 'FAIL'}")
    return {"name": "reproducibility", "passed": passed, "diff": float(diff),
            "run1_loss": float(results[0]), "run2_loss": float(results[1])}


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN EXPERIMENT (3 seeds × 4 methods)
# ═══════════════════════════════════════════════════════════════════════════════

def run_one_seed(seed, train_full, train_notf, test_ds):
    """
    For a single seed:
      1. Stratified labeled split (50/class); 10% held out as val for TS.
      2. Train MSP baseline.
      3. Fit temperature on val set → TS-MSP arm (no extra training).
      4. Train MC-Dropout model.
      5. Train Deep Ensemble (K=3 models).
      6. Compute AUPPC / ECE for all 4 methods on unlabeled pool.
    """
    log.info(f"\n{'='*60}")
    log.info(f"SEED {seed}")
    log.info(f"{'='*60}")
    set_seed(seed)

    train_lbl_idx, val_lbl_idx, unl_idx = stratified_split(
        train_full, LABELS_PER_CLASS, seed=seed, val_fraction=TS_VAL_FRACTION
    )
    log.info(f"  Train-labeled: {len(train_lbl_idx)}, Val-labeled: {len(val_lbl_idx)}, "
             f"Unlabeled: {len(unl_idx)}, Test: {len(test_ds)}")

    lbl_loader  = DataLoader(Subset(train_full, train_lbl_idx), batch_size=BATCH_SIZE,
                             shuffle=True, num_workers=4, pin_memory=True)
    val_loader  = DataLoader(Subset(train_notf, val_lbl_idx), batch_size=256,
                             shuffle=False, num_workers=2, pin_memory=True)
    unl_loader  = DataLoader(Subset(train_notf, unl_idx), batch_size=256,
                             shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False,
                             num_workers=4, pin_memory=True)

    results = {"seed": seed}

    # ── 1. MSP Baseline ──────────────────────────────────────────────────────
    log.info(f"\n  [MSP Baseline]")
    set_seed(seed)
    msp_model = ResNet8(dropout_p=0.0).to(DEVICE)
    msp_acc   = train_model(msp_model, lbl_loader, test_loader, PROBE_EPOCHS)
    log.info(f"  MSP test accuracy: {msp_acc:.4f}")

    s_msp, p_msp, l_msp = msp_scores(msp_model, unl_loader)
    auppc_msp = compute_auppc(s_msp, p_msp, l_msp)
    ece_msp   = compute_ece(s_msp, p_msp, l_msp)
    log.info(f"  MSP AUPPC={auppc_msp:.4f}  ECE={ece_msp:.4f}")

    results["msp"] = {
        "test_acc": float(msp_acc),
        "auppc":    float(auppc_msp),
        "ece":      float(ece_msp),
    }

    # ── 2. Temperature Scaling ────────────────────────────────────────────────
    log.info(f"\n  [Temperature Scaling (reuses MSP model)]")
    temperature = fit_temperature(msp_model, val_loader)

    s_ts, p_ts, l_ts = ts_scores(msp_model, unl_loader, temperature)
    auppc_ts = compute_auppc(s_ts, p_ts, l_ts)
    ece_ts   = compute_ece(s_ts, p_ts, l_ts)
    log.info(f"  TS  AUPPC={auppc_ts:.4f}  ECE={ece_ts:.4f}  T={temperature:.4f}")

    # ECE on val set (calibration quality)
    s_ts_val, p_ts_val, l_ts_val = ts_scores(msp_model, val_loader, temperature)
    ece_ts_val = compute_ece(s_ts_val, p_ts_val, l_ts_val)
    s_msp_val, p_msp_val, l_msp_val = msp_scores(msp_model, val_loader)
    ece_msp_val = compute_ece(s_msp_val, p_msp_val, l_msp_val)

    results["ts"] = {
        "test_acc":        float(msp_acc),  # same model, same accuracy
        "auppc":           float(auppc_ts),
        "ece":             float(ece_ts),
        "ece_val":         float(ece_ts_val),
        "ece_msp_val":     float(ece_msp_val),
        "temperature":     float(temperature),
        "ece_improvement": float(ece_msp_val - ece_ts_val),
    }
    log.info(f"  TS val ECE:  pre={ece_msp_val:.4f}  post={ece_ts_val:.4f}  "
             f"improvement={ece_msp_val - ece_ts_val:.4f}")

    # ── 3. MC-Dropout ────────────────────────────────────────────────────────
    log.info(f"\n  [MC-Dropout p={MC_DROPOUT_P}, T={MC_PASSES}]")
    set_seed(seed)
    mcd_model = ResNet8(dropout_p=MC_DROPOUT_P).to(DEVICE)
    mcd_acc   = train_model(mcd_model, lbl_loader, test_loader, PROBE_EPOCHS)
    log.info(f"  MCD test accuracy: {mcd_acc:.4f}")

    s_mcd, p_mcd, l_mcd = mc_dropout_scores(mcd_model, unl_loader, T=MC_PASSES)
    auppc_mcd = compute_auppc(s_mcd, p_mcd, l_mcd)
    ece_mcd   = compute_ece(s_mcd, p_mcd, l_mcd)
    log.info(f"  MCD AUPPC={auppc_mcd:.4f}  ECE={ece_mcd:.4f}")

    results["mc_dropout"] = {
        "test_acc": float(mcd_acc),
        "auppc":    float(auppc_mcd),
        "ece":      float(ece_mcd),
    }

    # ── 4. Deep Ensemble ─────────────────────────────────────────────────────
    log.info(f"\n  [Deep Ensemble K={ENSEMBLE_K}]")
    ens_models, ens_accs = [], []
    for k in range(ENSEMBLE_K):
        member_seed = seed + k * 1000
        set_seed(member_seed)
        log.info(f"    Training ensemble member {k+1}/{ENSEMBLE_K} (seed={member_seed})")
        member = ResNet8(dropout_p=0.0).to(DEVICE)
        acc_k  = train_model(member, lbl_loader, test_loader, PROBE_EPOCHS)
        ens_models.append(member)
        ens_accs.append(float(acc_k))
        log.info(f"    Member {k+1} acc: {acc_k:.4f}")

    s_ens, p_ens, l_ens = ensemble_scores(ens_models, unl_loader)
    auppc_ens = compute_auppc(s_ens, p_ens, l_ens)
    ece_ens   = compute_ece(s_ens, p_ens, l_ens)
    ens_acc   = float(np.mean(ens_accs))
    log.info(f"  Ensemble mean acc: {ens_acc:.4f}")
    log.info(f"  Ensemble AUPPC={auppc_ens:.4f}  ECE={ece_ens:.4f}")

    results["ensemble"] = {
        "test_acc":    ens_acc,
        "member_accs": ens_accs,
        "auppc":       float(auppc_ens),
        "ece":         float(ece_ens),
    }

    # ── Pairwise deltas ───────────────────────────────────────────────────────
    results["delta_ens_vs_mcd"]  = float(auppc_ens  - auppc_mcd)
    results["delta_mcd_vs_msp"]  = float(auppc_mcd  - auppc_msp)
    results["delta_ts_vs_msp"]   = float(auppc_ts   - auppc_msp)
    results["delta_ens_vs_msp"]  = float(auppc_ens  - auppc_msp)
    results["delta_ens_vs_ts"]   = float(auppc_ens  - auppc_ts)

    # TS recovery fraction: (ts_auppc - msp_auppc) / (ens_auppc - msp_auppc)
    denom = auppc_ens - auppc_msp
    results["ts_recovery_fraction"] = float((auppc_ts - auppc_msp) / denom) if abs(denom) > 1e-6 else None

    log.info(f"\n  AUPPC deltas:")
    log.info(f"    Ensemble − MCD   = {results['delta_ens_vs_mcd']:+.4f}")
    log.info(f"    MCD − MSP        = {results['delta_mcd_vs_msp']:+.4f}")
    log.info(f"    TS − MSP         = {results['delta_ts_vs_msp']:+.4f}")
    log.info(f"    Ensemble − MSP   = {results['delta_ens_vs_msp']:+.4f}")
    log.info(f"    TS recovery frac = {results['ts_recovery_fraction']}")

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    t_start = time.time()
    log.info("Loading CIFAR-10 …")
    train_full, train_notf, test_ds = get_datasets()

    # ── Sanity gates ──────────────────────────────────────────────────────────
    log.info("\n" + "="*60)
    log.info("SANITY GATES")
    log.info("="*60)
    gates = []
    gates.append(gate_loss_at_init(train_full))
    gates.append(gate_uniform_baseline(test_ds))
    gates.append(gate_overfit_one_batch(train_full))
    gates.append(gate_reproducibility(train_full))

    all_gates_passed = all(g["passed"] for g in gates)
    log.info(f"\nAll sanity gates passed: {all_gates_passed}")
    for g in gates:
        log.info(f"  {g['name']}: {'PASS' if g['passed'] else 'FAIL'}")
    if not all_gates_passed:
        log.warning("Some sanity gates FAILED — proceeding anyway for probe.")

    # ── Main experiment ───────────────────────────────────────────────────────
    log.info("\n" + "="*60)
    log.info(f"MAIN EXPERIMENT  ({len(SEEDS)} seeds × 4 methods)")
    log.info("="*60)

    seed_results = []
    for seed in SEEDS:
        r = run_one_seed(seed, train_full, train_notf, test_ds)
        seed_results.append(r)

    # ── Aggregate ─────────────────────────────────────────────────────────────
    def agg(key, subkey):
        vals = [r[key][subkey] for r in seed_results]
        return {"values": [float(v) for v in vals],
                "mean":   float(np.mean(vals)),
                "std":    float(np.std(vals))}

    auppc_msp = agg("msp",        "auppc")
    auppc_ts  = agg("ts",         "auppc")
    auppc_mcd = agg("mc_dropout", "auppc")
    auppc_ens = agg("ensemble",   "auppc")
    acc_msp   = agg("msp",        "test_acc")
    acc_ts    = agg("ts",         "test_acc")
    acc_mcd   = agg("mc_dropout", "test_acc")
    acc_ens   = agg("ensemble",   "test_acc")
    ece_msp   = agg("msp",        "ece")
    ece_ts    = agg("ts",         "ece")
    ece_mcd   = agg("mc_dropout", "ece")
    ece_ens   = agg("ensemble",   "ece")

    delta_ens_mcd_vals = [r["delta_ens_vs_mcd"] for r in seed_results]
    delta_mcd_msp_vals = [r["delta_mcd_vs_msp"] for r in seed_results]
    delta_ts_msp_vals  = [r["delta_ts_vs_msp"]  for r in seed_results]
    delta_ens_msp_vals = [r["delta_ens_vs_msp"] for r in seed_results]
    delta_ens_ts_vals  = [r["delta_ens_vs_ts"]  for r in seed_results]
    ts_recovery_vals   = [r["ts_recovery_fraction"] for r in seed_results if r["ts_recovery_fraction"] is not None]

    log.info("\n" + "="*60)
    log.info("AGGREGATE RESULTS")
    log.info("="*60)
    log.info(f"  AUPPC MSP:       {auppc_msp['mean']:.4f} ± {auppc_msp['std']:.4f}  {auppc_msp['values']}")
    log.info(f"  AUPPC TS-MSP:    {auppc_ts['mean']:.4f} ± {auppc_ts['std']:.4f}  {auppc_ts['values']}")
    log.info(f"  AUPPC MC-Drop:   {auppc_mcd['mean']:.4f} ± {auppc_mcd['std']:.4f}  {auppc_mcd['values']}")
    log.info(f"  AUPPC Ensemble:  {auppc_ens['mean']:.4f} ± {auppc_ens['std']:.4f}  {auppc_ens['values']}")
    log.info(f"  Acc MSP:         {acc_msp['mean']:.4f} ± {acc_msp['std']:.4f}")
    log.info(f"  Acc TS-MSP:      {acc_ts['mean']:.4f} ± {acc_ts['std']:.4f}  (same model as MSP)")
    log.info(f"  Acc MC-Drop:     {acc_mcd['mean']:.4f} ± {acc_mcd['std']:.4f}")
    log.info(f"  Acc Ensemble:    {acc_ens['mean']:.4f} ± {acc_ens['std']:.4f}")
    log.info(f"  ECE MSP:         {ece_msp['mean']:.4f}")
    log.info(f"  ECE TS-MSP:      {ece_ts['mean']:.4f}")
    log.info(f"  ECE MC-Drop:     {ece_mcd['mean']:.4f}")
    log.info(f"  ECE Ensemble:    {ece_ens['mean']:.4f}")
    log.info(f"  Δ(Ens−MCD):      {np.mean(delta_ens_mcd_vals):+.4f}")
    log.info(f"  Δ(MCD−MSP):      {np.mean(delta_mcd_msp_vals):+.4f}")
    log.info(f"  Δ(TS−MSP):       {np.mean(delta_ts_msp_vals):+.4f}")
    log.info(f"  Δ(Ens−MSP):      {np.mean(delta_ens_msp_vals):+.4f}")
    log.info(f"  Δ(Ens−TS):       {np.mean(delta_ens_ts_vals):+.4f}")
    if ts_recovery_vals:
        log.info(f"  TS recovery frac: {np.mean(ts_recovery_vals):.4f}")

    # ── Hypothesis checks ─────────────────────────────────────────────────────
    cond_a = np.mean(delta_ens_mcd_vals) >= 0.030
    cond_b_seeds = sum(1 for r in seed_results if r["delta_mcd_vs_msp"] >= 0.015)
    cond_b = cond_b_seeds >= 2
    cond_c_seeds = sum(1 for r in seed_results
                       if r["ensemble"]["test_acc"] - r["msp"]["test_acc"] >= 0.005)
    cond_c = cond_c_seeds >= 2

    ref_a_seeds = sum(1 for r in seed_results if r["delta_ens_vs_mcd"] <= 0.010)
    ref_a = ref_a_seeds >= 2
    ref_b_seeds = sum(1 for r in seed_results if r["msp"]["auppc"] > 0.90)
    ref_b = ref_b_seeds >= 2
    ref_c_seeds = sum(1 for r in seed_results if (1 - r["msp"]["test_acc"]) > 0.25)
    ref_c = ref_c_seeds >= 2

    confirmed = cond_a and cond_b and cond_c
    refuted   = ref_a or ref_b or ref_c

    # Sanity band check
    msp_in_band = sum(1 for r in seed_results if 0.70 <= r["msp"]["auppc"] <= 0.88)

    log.info(f"\n  HYPOTHESIS CHECKS:")
    log.info(f"    (a) Ens−MCD AUPPC >= 0.030: {cond_a}  ({np.mean(delta_ens_mcd_vals):.4f})")
    log.info(f"    (b) MCD−MSP >= 0.015 in >=2/3 seeds: {cond_b}  ({cond_b_seeds}/3)")
    log.info(f"    (c) Ensemble acc > MSP+0.5% in >=2/3: {cond_c}  ({cond_c_seeds}/3)")
    log.info(f"    CONFIRMED: {confirmed}")
    log.info(f"    REFUTED:   {refuted}")
    log.info(f"    MSP AUPPC in sanity band [0.70,0.88]: {msp_in_band}/3 seeds")

    elapsed = time.time() - t_start
    log.info(f"\nTotal runtime: {elapsed:.0f}s  ({elapsed/60:.1f} min)")

    # ── RESULTS.json ──────────────────────────────────────────────────────────
    ts_rec_mean = float(np.mean(ts_recovery_vals)) if ts_recovery_vals else None

    results_json = {
        "status":  "SUCCESS",
        "scale":   "probe",
        "subject_executed": (
            f"MSP / Temperature-Scaled MSP / MC-Dropout (T={MC_PASSES}, p={MC_DROPOUT_P}) "
            f"/ Deep Ensemble (K={ENSEMBLE_K}) on CIFAR-10 50 labels/class, "
            f"{PROBE_EPOCHS} epochs (LR={LR}), {len(SEEDS)} seeds"
        ),
        "notes": (
            f"refine-02: fixed LR (3e-3→{LR}), added TS arm, {PROBE_EPOCHS} epochs. "
            f"MSP AUPPC sanity band [0.70,0.88]: {msp_in_band}/3 seeds. "
            f"Hypothesis confirmed={confirmed}, refuted={refuted}. "
            f"Runtime {elapsed:.0f}s ({elapsed/60:.1f} min)."
        ),
        "sanity_gates": gates,
        "all_sanity_gates_passed": bool(all_gates_passed),
        "per_seed_results": seed_results,
        "metrics": {
            "auppc_msp":          auppc_msp,
            "auppc_ts_msp":       auppc_ts,
            "auppc_mc_dropout":   auppc_mcd,
            "auppc_ensemble":     auppc_ens,
            "acc_msp":            acc_msp,
            "acc_ts_msp":         acc_ts,
            "acc_mc_dropout":     acc_mcd,
            "acc_ensemble":       acc_ens,
            "ece_msp":            ece_msp,
            "ece_ts_msp":         ece_ts,
            "ece_mc_dropout":     ece_mcd,
            "ece_ensemble":       ece_ens,
            "delta_ensemble_vs_mcd_mean":  float(np.mean(delta_ens_mcd_vals)),
            "delta_ensemble_vs_mcd_std":   float(np.std(delta_ens_mcd_vals)),
            "delta_mcd_vs_msp_mean":       float(np.mean(delta_mcd_msp_vals)),
            "delta_mcd_vs_msp_std":        float(np.std(delta_mcd_msp_vals)),
            "delta_ts_vs_msp_mean":        float(np.mean(delta_ts_msp_vals)),
            "delta_ts_vs_msp_std":         float(np.std(delta_ts_msp_vals)),
            "delta_ensemble_vs_msp_mean":  float(np.mean(delta_ens_msp_vals)),
            "delta_ensemble_vs_msp_std":   float(np.std(delta_ens_msp_vals)),
            "delta_ensemble_vs_ts_mean":   float(np.mean(delta_ens_ts_vals)),
            "delta_ensemble_vs_ts_std":    float(np.std(delta_ens_ts_vals)),
            "ts_recovery_fraction_mean":   ts_rec_mean,
            "msp_auppc_in_sanity_band_count": msp_in_band,
        },
        "hypothesis": {
            "confirmed": bool(confirmed),
            "refuted":   bool(refuted),
            "cond_a_ens_minus_mcd_ge_0030":       bool(cond_a),
            "cond_b_mcd_minus_msp_ge_2of3":       bool(cond_b),
            "cond_c_ens_acc_gt_msp_2of3":         bool(cond_c),
            "refuted_a_ens_le_mcd_plus_010_ge_2of3": bool(ref_a),
            "refuted_b_msp_auppc_gt_090_ge_2of3":    bool(ref_b),
            "refuted_c_msp_error_gt_25pct_ge_2of3":  bool(ref_c),
        },
        "config": {
            "labels_per_class":  LABELS_PER_CLASS,
            "probe_epochs":      PROBE_EPOCHS,
            "lr":                LR,
            "weight_decay":      WEIGHT_DECAY,
            "ensemble_k":        ENSEMBLE_K,
            "mc_passes":         MC_PASSES,
            "mc_dropout_p":      MC_DROPOUT_P,
            "ts_val_fraction":   TS_VAL_FRACTION,
            "seeds":             SEEDS,
            "thresholds_n":      len(THRESHOLD_GRID),
        },
        "runtime_seconds": float(elapsed),
    }

    out_path = RESULTS / "RESULTS.json"
    with open(out_path, "w") as f:
        json.dump(results_json, f, indent=2)
    log.info(f"\nResults written to {out_path}")
    print("\n" + json.dumps(results_json, indent=2))
    return results_json


if __name__ == "__main__":
    main()
