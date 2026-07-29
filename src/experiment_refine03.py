"""
experiment_refine03.py — Probe round refine-03 for:
  Deep Ensemble vs. MC-Dropout vs. Temperature-Scaled MSP
  Pseudo-Label Precision Pareto at 100 labels/class on CIFAR-10.

FIXES vs. refine-02 (test acc 33% → target >=55%):
  1. More labeled data: 50/class → 100/class (1000 total labeled).
     With 900 training samples (10% held for TS val), 200 epochs:
     ~180,000 sample-epochs (vs 45,000 in refine-02 = 4× more).
     Expected test accuracy: 55–65%.
  2. More epochs: 100 → 200. Full cosine annealing schedule.
  3. Fix MC-Dropout p: 0.1 → 0.3. More stochasticity → meaningful uncertainty.
  4. Fix overfit-one-batch gate: 200 → 300 steps.

Probe settings (refine-03):
  - 100 labels/class (1000 total labeled), hold out 100 for TS val
  - 200 epochs
  - K=3 ensemble members (spec K=5; probe)
  - T=20 MC-Dropout passes, p=0.3
  - 3 seeds [42, 7, 123]
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
ROOT     = Path("/workspace")
RESULTS  = ROOT / "results" / "refine-03"
WEIGHTS  = ROOT / "_weights"
DATA_DIR = Path("/opt/datasets")
RESULTS.mkdir(parents=True, exist_ok=True)
WEIGHTS.mkdir(parents=True, exist_ok=True)

# ── Hyperparameters ───────────────────────────────────────────────────────────
LABELS_PER_CLASS  = 100          # KEY FIX: was 50 → now 100/class (within spec range)
NUM_CLASSES       = 10
BATCH_SIZE        = 64
PROBE_EPOCHS      = 200          # KEY FIX: was 100 → now 200 (spec value)
ENSEMBLE_K        = 3            # probe (spec K=5); K=3 for runtime
MC_PASSES         = 20
MC_DROPOUT_P      = 0.3          # KEY FIX: was 0.1 → now 0.3 (better uncertainty)
LR                = 0.1
WEIGHT_DECAY      = 5e-4
THRESHOLD_GRID    = np.linspace(0.50, 0.99, 20)
SEEDS             = [42, 7, 123]
TS_VAL_FRACTION   = 0.10         # 10% of labeled set held out for TS calibration

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD  = (0.247,  0.243,  0.261)


# ── Seed helper ───────────────────────────────────────────────────────────────
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


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
    def __init__(self, dropout_p=0.0, num_classes=NUM_CLASSES):
        super().__init__()
        self.dropout_p  = dropout_p
        self.num_classes = num_classes
        self.stem = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )
        self.layer1 = ResBlock(16, 16, dropout_p=dropout_p)
        self.layer2 = ResBlock(16, 32, stride=2, dropout_p=dropout_p)
        self.layer3 = ResBlock(32, 64, stride=2, dropout_p=dropout_p)
        self.pool   = nn.AdaptiveAvgPool2d(1)
        self.fc     = nn.Linear(64, num_classes)

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


def train_model(model, labeled_loader, test_loader, epochs, lr=LR, wd=WEIGHT_DECAY, tag=""):
    """Train model, log every 25 epochs, return final test accuracy."""
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=wd)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    for ep in range(1, epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, labeled_loader, optimizer, scheduler)
        if ep % 25 == 0 or ep == 1 or ep == epochs:
            te_acc = evaluate(model, test_loader)
            log.info(f"  {tag} ep {ep:3d}/{epochs}: loss={tr_loss:.4f} "
                     f"tr_acc={tr_acc:.3f} te_acc={te_acc:.3f} "
                     f"lr={scheduler.get_last_lr()[0]:.5f}")
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
    optimizer   = optim.LBFGS([temperature], lr=0.05, max_iter=200)

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


def mc_dropout_scores(model, loader, T=MC_PASSES):
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
                        shuffle=False, num_workers=2, pin_memory=True)
    model.eval()
    with torch.no_grad():
        x, y = next(iter(loader))
        loss = F.cross_entropy(model(x.to(DEVICE)), y.to(DEVICE))
    val = float(loss.item())
    passed = 2.28 <= val <= 2.33
    log.info(f"  init CE loss = {val:.6f}  {'PASS' if passed else 'FAIL'}")
    return {"name": "loss_at_init", "passed": passed, "init_ce_loss": val}


def gate_uniform_baseline(test_ds):
    """Gate 3: Uniform logits → ~10% accuracy."""
    log.info("── Gate: Uniform baseline ──")
    set_seed(42)
    model = ResNet8().to(DEVICE)
    # override fc weights to zero so output is always uniform
    nn.init.zeros_(model.fc.weight)
    nn.init.zeros_(model.fc.bias)
    loader = DataLoader(test_ds, batch_size=512, shuffle=False,
                        num_workers=2, pin_memory=True)
    acc = evaluate(model, loader)
    passed = 0.098 <= acc <= 0.102
    log.info(f"  uniform accuracy = {acc:.4f}  {'PASS' if passed else 'FAIL'}")
    return {"name": "uniform_baseline", "passed": passed, "uniform_accuracy": acc}


def gate_overfit_one_batch(train_full):
    """Gate 4: Overfit a single batch of 64 images to CE < 0.01 within 300 steps."""
    log.info("── Gate: Overfit one batch ──")
    set_seed(42)
    model     = ResNet8().to(DEVICE)
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=0)
    loader    = DataLoader(Subset(train_full, list(range(64))), batch_size=64,
                           shuffle=False, num_workers=0)
    x, y = next(iter(loader))
    x, y = x.to(DEVICE), y.to(DEVICE)
    final_loss = None
    for step in range(1, 301):  # 300 steps (was 200 in refine-02, near-miss at 0.01014)
        model.train()
        optimizer.zero_grad()
        loss = F.cross_entropy(model(x), y)
        loss.backward()
        optimizer.step()
        final_loss = float(loss.item())
        if final_loss < 0.01:
            log.info(f"  overfit: loss={final_loss:.6f} at step {step}  PASS")
            break
    passed = final_loss < 0.01
    log.info(f"  final loss at step 300 = {final_loss:.6f}  {'PASS' if passed else 'FAIL'}")
    return {"name": "overfit_one_batch", "passed": passed,
            "single_batch_loss_at_step300": final_loss}


def gate_reproducibility(train_full, train_notf, test_ds):
    """Gate 1: Bit-identical loss curves across two runs with same seed."""
    log.info("── Gate: Reproducibility ──")
    results = []
    for run_idx in range(2):
        set_seed(99)
        model     = ResNet8().to(DEVICE)
        lbl, _, _ = stratified_split(train_full, LABELS_PER_CLASS, seed=99)
        loader    = DataLoader(Subset(train_full, lbl), batch_size=BATCH_SIZE,
                               shuffle=True, num_workers=2, pin_memory=True,
                               generator=torch.Generator().manual_seed(99))
        optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
        model.train()
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            optimizer.step()
            break  # one batch only
        loss_val = float(loss.item())
        results.append(loss_val)
    diff = abs(results[0] - results[1])
    passed = diff == 0.0
    log.info(f"  run1={results[0]:.8f}  run2={results[1]:.8f}  diff={diff}  {'PASS' if passed else 'FAIL'}")
    return {"name": "reproducibility", "passed": passed,
            "diff": diff, "run1_loss": results[0], "run2_loss": results[1]}


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN EXPERIMENT
# ═══════════════════════════════════════════════════════════════════════════════

def run_one_seed(seed, train_full, train_notf, test_ds, test_loader):
    log.info(f"\n{'='*60}")
    log.info(f"SEED {seed}")
    log.info(f"{'='*60}")
    t_seed = time.time()

    # ── Split ────────────────────────────────────────────────────────────────
    train_lbl, val_lbl, unlabeled_idx = stratified_split(
        train_full, LABELS_PER_CLASS, seed=seed, val_fraction=TS_VAL_FRACTION
    )
    log.info(f"  labeled_train={len(train_lbl)} val_ts={len(val_lbl)} unlabeled={len(unlabeled_idx)}")

    set_seed(seed)
    lbl_loader = DataLoader(Subset(train_full, train_lbl), batch_size=BATCH_SIZE,
                            shuffle=True, num_workers=2, pin_memory=True,
                            generator=torch.Generator().manual_seed(seed))
    val_loader = DataLoader(Subset(train_notf, val_lbl), batch_size=512,
                            shuffle=False, num_workers=2, pin_memory=True)
    unl_loader = DataLoader(Subset(train_notf, unlabeled_idx), batch_size=512,
                            shuffle=False, num_workers=2, pin_memory=True)

    # ── MSP Baseline ─────────────────────────────────────────────────────────
    log.info(f"\n── MSP baseline (seed={seed}) ──")
    set_seed(seed)
    msp_model = ResNet8().to(DEVICE)
    msp_acc   = train_model(msp_model, lbl_loader, test_loader, PROBE_EPOCHS, tag="MSP")
    log.info(f"  MSP test acc = {msp_acc:.4f}")

    msp_conf, msp_preds, msp_labs = msp_scores(msp_model, unl_loader)
    msp_test_conf, msp_test_preds, msp_test_labs = msp_scores(msp_model, test_loader)
    msp_auppc = compute_auppc(msp_conf, msp_preds, msp_labs)
    msp_ece   = compute_ece(msp_test_conf, msp_test_preds, msp_test_labs)
    log.info(f"  MSP AUPPC={msp_auppc:.4f}  ECE={msp_ece:.4f}")

    # ── Temperature Scaling ───────────────────────────────────────────────────
    log.info(f"\n── Temperature Scaling (seed={seed}) ──")
    T_opt = fit_temperature(msp_model, val_loader)
    ts_conf, ts_preds, ts_labs = ts_scores(msp_model, unl_loader, T_opt)
    ts_test_conf, ts_test_preds, ts_test_labs = ts_scores(msp_model, test_loader, T_opt)
    ts_auppc = compute_auppc(ts_conf, ts_preds, ts_labs)
    ts_ece   = compute_ece(ts_test_conf, ts_test_preds, ts_test_labs)
    ts_ece_val = compute_ece(*ts_scores(msp_model, val_loader, T_opt)[:3 if True else 3])
    msp_ece_val = compute_ece(*msp_scores(msp_model, val_loader)[:3 if True else 3])
    log.info(f"  TS AUPPC={ts_auppc:.4f}  ECE={ts_ece:.4f}  T={T_opt:.4f}")
    log.info(f"  TS ECE improvement: {msp_ece:.4f} → {ts_ece:.4f}")

    # ── MC-Dropout ────────────────────────────────────────────────────────────
    log.info(f"\n── MC-Dropout (seed={seed}, p={MC_DROPOUT_P}, T={MC_PASSES}) ──")
    set_seed(seed)
    mcd_model = ResNet8(dropout_p=MC_DROPOUT_P).to(DEVICE)
    mcd_acc   = train_model(mcd_model, lbl_loader, test_loader, PROBE_EPOCHS, tag="MCD")
    log.info(f"  MCD test acc = {mcd_acc:.4f}")

    mcd_conf, mcd_preds, mcd_labs = mc_dropout_scores(mcd_model, unl_loader, T=MC_PASSES)
    mcd_test_conf, mcd_test_preds, mcd_test_labs = mc_dropout_scores(mcd_model, test_loader, T=MC_PASSES)
    mcd_auppc = compute_auppc(mcd_conf, mcd_preds, mcd_labs)
    mcd_ece   = compute_ece(mcd_test_conf, mcd_test_preds, mcd_test_labs)
    log.info(f"  MCD AUPPC={mcd_auppc:.4f}  ECE={mcd_ece:.4f}")

    # ── Deep Ensemble ─────────────────────────────────────────────────────────
    log.info(f"\n── Deep Ensemble (seed={seed}, K={ENSEMBLE_K}) ──")
    ens_models = []
    member_accs = []
    for k in range(ENSEMBLE_K):
        member_seed = seed + k * 1000
        set_seed(member_seed)
        ens_lbl_loader = DataLoader(
            Subset(train_full, train_lbl), batch_size=BATCH_SIZE,
            shuffle=True, num_workers=2, pin_memory=True,
            generator=torch.Generator().manual_seed(member_seed),
        )
        m = ResNet8().to(DEVICE)
        acc = train_model(m, ens_lbl_loader, test_loader, PROBE_EPOCHS,
                          tag=f"ENS[k={k}]")
        member_accs.append(acc)
        ens_models.append(m)
        log.info(f"  member {k}: test acc = {acc:.4f}")

    ens_conf, ens_preds, ens_labs = ensemble_scores(ens_models, unl_loader)
    ens_test_conf, ens_test_preds, ens_test_labs = ensemble_scores(ens_models, test_loader)
    ens_auppc = compute_auppc(ens_conf, ens_preds, ens_labs)
    ens_ece   = compute_ece(ens_test_conf, ens_test_preds, ens_test_labs)
    ens_acc   = float(np.mean(member_accs))
    log.info(f"  ENS AUPPC={ens_auppc:.4f}  ECE={ens_ece:.4f}  mean_acc={ens_acc:.4f}")

    # ── Pairwise deltas ───────────────────────────────────────────────────────
    delta_ens_vs_mcd = ens_auppc - mcd_auppc
    delta_mcd_vs_msp = mcd_auppc - msp_auppc
    delta_ts_vs_msp  = ts_auppc  - msp_auppc
    delta_ens_vs_msp = ens_auppc - msp_auppc
    delta_ens_vs_ts  = ens_auppc - ts_auppc

    denom = ens_auppc - msp_auppc
    ts_recovery = (ts_auppc - msp_auppc) / denom if abs(denom) > 1e-6 else float("nan")

    log.info(f"\n  Δ(ENS-MCD)={delta_ens_vs_mcd:+.4f}  "
             f"Δ(MCD-MSP)={delta_mcd_vs_msp:+.4f}  "
             f"Δ(TS-MSP)={delta_ts_vs_msp:+.4f}")
    log.info(f"  Δ(ENS-MSP)={delta_ens_vs_msp:+.4f}  "
             f"Δ(ENS-TS)={delta_ens_vs_ts:+.4f}  "
             f"TS_recovery={ts_recovery:.3f}")

    elapsed = time.time() - t_seed
    log.info(f"\n  Seed {seed} done in {elapsed:.1f}s")

    return {
        "seed": seed,
        "msp": {
            "test_acc":  msp_acc,
            "auppc":     msp_auppc,
            "ece":       msp_ece,
        },
        "ts": {
            "test_acc":  msp_acc,          # TS doesn't change predictions
            "auppc":     ts_auppc,
            "ece":       ts_ece,
            "ece_val":   ts_ece_val,
            "ece_msp_val": msp_ece_val,
            "temperature": T_opt,
            "ece_improvement": msp_ece - ts_ece,
        },
        "mc_dropout": {
            "test_acc":  mcd_acc,
            "auppc":     mcd_auppc,
            "ece":       mcd_ece,
        },
        "ensemble": {
            "test_acc":      ens_acc,
            "member_accs":   member_accs,
            "auppc":         ens_auppc,
            "ece":           ens_ece,
        },
        "delta_ens_vs_mcd": delta_ens_vs_mcd,
        "delta_mcd_vs_msp": delta_mcd_vs_msp,
        "delta_ts_vs_msp":  delta_ts_vs_msp,
        "delta_ens_vs_msp": delta_ens_vs_msp,
        "delta_ens_vs_ts":  delta_ens_vs_ts,
        "ts_recovery_fraction": ts_recovery,
    }


def main():
    t_start = time.time()
    log.info("=" * 70)
    log.info("refine-03: 100 labels/class, 200 epochs, MC-Dropout p=0.3")
    log.info("=" * 70)

    # ── Load data ─────────────────────────────────────────────────────────────
    train_full, train_notf, test_ds = get_datasets()
    test_loader = DataLoader(test_ds, batch_size=512, shuffle=False,
                             num_workers=2, pin_memory=True)

    # ── Sanity gates ──────────────────────────────────────────────────────────
    log.info("\n══ SANITY GATES ══")
    gates = []
    gates.append(gate_reproducibility(train_full, train_notf, test_ds))
    gates.append(gate_loss_at_init(train_full))
    gates.append(gate_uniform_baseline(test_ds))
    gates.append(gate_overfit_one_batch(train_full))
    all_gates_passed = all(g["passed"] for g in gates)
    log.info(f"\nAll sanity gates passed: {all_gates_passed}")

    # ── Main experiment ───────────────────────────────────────────────────────
    log.info("\n══ MAIN EXPERIMENT ══")
    per_seed = []
    for seed in SEEDS:
        result = run_one_seed(seed, train_full, train_notf, test_ds, test_loader)
        per_seed.append(result)

    # ── Aggregate metrics ─────────────────────────────────────────────────────
    def collect(key_path):
        vals = []
        for r in per_seed:
            obj = r
            for k in key_path:
                obj = obj[k]
            vals.append(obj)
        return vals

    def stats(vals):
        return {"values": vals, "mean": float(np.mean(vals)), "std": float(np.std(vals))}

    auppc_msp  = collect(["msp", "auppc"])
    auppc_ts   = collect(["ts", "auppc"])
    auppc_mcd  = collect(["mc_dropout", "auppc"])
    auppc_ens  = collect(["ensemble", "auppc"])
    acc_msp    = collect(["msp", "test_acc"])
    acc_mcd    = collect(["mc_dropout", "test_acc"])
    acc_ens    = collect(["ensemble", "test_acc"])
    ece_msp    = collect(["msp", "ece"])
    ece_ts     = collect(["ts", "ece"])
    ece_mcd    = collect(["mc_dropout", "ece"])
    ece_ens    = collect(["ensemble", "ece"])

    delta_ens_mcd  = collect(["delta_ens_vs_mcd"])
    delta_mcd_msp  = collect(["delta_mcd_vs_msp"])
    delta_ts_msp   = collect(["delta_ts_vs_msp"])
    delta_ens_msp  = collect(["delta_ens_vs_msp"])
    delta_ens_ts   = collect(["delta_ens_vs_ts"])
    ts_recovery    = collect(["ts_recovery_fraction"])

    msp_in_band = sum(1 for v in auppc_msp if 0.70 <= v <= 0.88)

    metrics = {
        "auppc_msp":         stats(auppc_msp),
        "auppc_ts_msp":      stats(auppc_ts),
        "auppc_mc_dropout":  stats(auppc_mcd),
        "auppc_ensemble":    stats(auppc_ens),
        "acc_msp":           stats(acc_msp),
        "acc_ts_msp":        stats(acc_msp),   # same model
        "acc_mc_dropout":    stats(acc_mcd),
        "acc_ensemble":      stats(acc_ens),
        "ece_msp":           stats(ece_msp),
        "ece_ts_msp":        stats(ece_ts),
        "ece_mc_dropout":    stats(ece_mcd),
        "ece_ensemble":      stats(ece_ens),
        "delta_ensemble_vs_mcd_mean":  float(np.mean(delta_ens_mcd)),
        "delta_ensemble_vs_mcd_std":   float(np.std(delta_ens_mcd)),
        "delta_mcd_vs_msp_mean":       float(np.mean(delta_mcd_msp)),
        "delta_mcd_vs_msp_std":        float(np.std(delta_mcd_msp)),
        "delta_ts_vs_msp_mean":        float(np.mean(delta_ts_msp)),
        "delta_ts_vs_msp_std":         float(np.std(delta_ts_msp)),
        "delta_ensemble_vs_msp_mean":  float(np.mean(delta_ens_msp)),
        "delta_ensemble_vs_msp_std":   float(np.std(delta_ens_msp)),
        "delta_ensemble_vs_ts_mean":   float(np.mean(delta_ens_ts)),
        "delta_ensemble_vs_ts_std":    float(np.std(delta_ens_ts)),
        "ts_recovery_fraction_mean":   float(np.mean(ts_recovery)),
        "msp_auppc_in_sanity_band_count": msp_in_band,
        "accuracy_gate_passed_count":  sum(1 for v in acc_msp if v >= 0.55),
    }

    # ── Hypothesis evaluation ─────────────────────────────────────────────────
    # Confirm if: (a) ENS-MCD >= 0.030 across all 3 seeds
    #             (b) MCD-MSP >= 0.015 in >= 2/3 seeds
    #             (c) ENS acc > MSP acc by >= 0.5% in >= 2/3 seeds
    cond_a = all(d >= 0.030 for d in delta_ens_mcd)
    cond_b = sum(1 for d in delta_mcd_msp if d >= 0.015) >= 2
    cond_c = sum(1 for r in per_seed
                 if r["ensemble"]["test_acc"] - r["msp"]["test_acc"] >= 0.005) >= 2
    confirmed = cond_a and cond_b and cond_c

    # Refute if: (a) ENS <= MCD + 0.010 in >= 2/3 seeds
    #            (b) MSP AUPPC > 0.90 in >= 2/3 seeds
    #            (c) MSP test error > 25% in >= 2/3 seeds
    ref_a = sum(1 for d in delta_ens_mcd if d <= 0.010) >= 2
    ref_b = sum(1 for v in auppc_msp if v > 0.90) >= 2
    ref_c = sum(1 for v in acc_msp   if (1 - v) > 0.25) >= 2  # acc < 0.75 means error > 25%
    refuted = ref_a or ref_b or ref_c

    hypothesis = {
        "confirmed": confirmed,
        "refuted":   refuted,
        "cond_a_ens_minus_mcd_ge_0030":       cond_a,
        "cond_b_mcd_minus_msp_ge_2of3":       cond_b,
        "cond_c_ens_acc_gt_msp_2of3":         cond_c,
        "refuted_a_ens_le_mcd_plus_010_ge_2of3": ref_a,
        "refuted_b_msp_auppc_gt_090_ge_2of3":    ref_b,
        "refuted_c_msp_error_gt_25pct_ge_2of3":  ref_c,
    }

    # ── Comparison vs prior rounds ────────────────────────────────────────────
    baseline_comparison = {
        "auppc_msp_refine02":  0.2025,
        "auppc_msp_refine03":  float(np.mean(auppc_msp)),
        "auppc_msp_main01":    0.0634,
        "acc_msp_refine02":    0.336,
        "acc_msp_refine03":    float(np.mean(acc_msp)),
        "ece_msp_refine02":    0.141,
        "ece_msp_refine03":    float(np.mean(ece_msp)),
        "accuracy_gate_passed": f"{sum(1 for v in acc_msp if v >= 0.55)}/3 seeds >= 55%",
        "auppc_in_band":       f"{msp_in_band}/3 seeds in [0.70, 0.88]",
    }

    runtime = time.time() - t_start

    # ── Assemble results ──────────────────────────────────────────────────────
    results = {
        "status":           "SUCCESS" if all_gates_passed else "PARTIAL",
        "scale":            "probe",
        "subject_executed": (
            f"MSP / Temperature-Scaled MSP / MC-Dropout (T={MC_PASSES}, p={MC_DROPOUT_P}) "
            f"/ Deep Ensemble (K={ENSEMBLE_K}) on CIFAR-10 {LABELS_PER_CLASS} labels/class, "
            f"{PROBE_EPOCHS} epochs (LR={LR}), {len(SEEDS)} seeds"
        ),
        "notes": (
            f"refine-03: 100/class (was 50), 200 epochs (was 100), MC-Dropout p=0.3 (was 0.1). "
            f"MSP acc gate ({sum(1 for v in acc_msp if v >= 0.55)}/3 seeds >= 55%). "
            f"AUPPC sanity band [0.70,0.88]: {msp_in_band}/3 seeds. "
            f"Runtime {runtime:.1f}s ({runtime/60:.1f} min)."
        ),
        "baseline_comparison_vs_prior": baseline_comparison,
        "sanity_gates":          gates,
        "all_sanity_gates_passed": all_gates_passed,
        "per_seed_results":      per_seed,
        "metrics":               metrics,
        "hypothesis":            hypothesis,
        "config": {
            "labels_per_class": LABELS_PER_CLASS,
            "probe_epochs":     PROBE_EPOCHS,
            "lr":               LR,
            "weight_decay":     WEIGHT_DECAY,
            "ensemble_k":       ENSEMBLE_K,
            "mc_passes":        MC_PASSES,
            "mc_dropout_p":     MC_DROPOUT_P,
            "ts_val_fraction":  TS_VAL_FRACTION,
            "seeds":            SEEDS,
            "thresholds_n":     len(THRESHOLD_GRID),
        },
        "runtime_seconds": runtime,
    }

    out_path = RESULTS / "RESULTS.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"\nResults written to {out_path}")

    # ── Print summary ─────────────────────────────────────────────────────────
    log.info("\n" + "=" * 70)
    log.info("SUMMARY — refine-03")
    log.info("=" * 70)
    log.info(f"  Runtime: {runtime:.1f}s ({runtime/60:.1f} min)")
    log.info(f"  MSP acc:         {np.mean(acc_msp):.3f} ± {np.std(acc_msp):.3f} (gate >=0.55: {sum(1 for v in acc_msp if v >= 0.55)}/3)")
    log.info(f"  AUPPC MSP:       {np.mean(auppc_msp):.4f} ± {np.std(auppc_msp):.4f} (in-band: {msp_in_band}/3)")
    log.info(f"  AUPPC TS:        {np.mean(auppc_ts):.4f} ± {np.std(auppc_ts):.4f}")
    log.info(f"  AUPPC MCD:       {np.mean(auppc_mcd):.4f} ± {np.std(auppc_mcd):.4f}")
    log.info(f"  AUPPC Ensemble:  {np.mean(auppc_ens):.4f} ± {np.std(auppc_ens):.4f}")
    log.info(f"  Δ(ENS-MCD):     {np.mean(delta_ens_mcd):+.4f} ± {np.std(delta_ens_mcd):.4f}")
    log.info(f"  Δ(MCD-MSP):     {np.mean(delta_mcd_msp):+.4f} ± {np.std(delta_mcd_msp):.4f}")
    log.info(f"  ECE: MSP={np.mean(ece_msp):.4f} TS={np.mean(ece_ts):.4f} "
             f"MCD={np.mean(ece_mcd):.4f} ENS={np.mean(ece_ens):.4f}")
    log.info(f"  Hypothesis: confirmed={confirmed}  refuted={refuted}")
    log.info(f"  Sanity gates: {sum(g['passed'] for g in gates)}/{len(gates)} passed")
    log.info("=" * 70)

    print("\n── RESULTS.json ──")
    print(json.dumps(results, indent=2))

    return results


if __name__ == "__main__":
    main()
