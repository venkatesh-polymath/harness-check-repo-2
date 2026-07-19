#!/usr/bin/env python3
"""
run_pmnist.py — Online Permuted-MNIST plasticity experiment
======================================================
100 tasks × 5 seeds, 3-layer MLP (784→2000→2000→10, ReLU), SGD+momentum.
No plasticity repair. Tests for genuine plasticity loss in a HEALTHY network.

Key checks:
  (a) Network HEALTHY at task 1: dead_after_task1 < 20%, task1_acc > 0.9
  (b) New-task accuracy DECLINES over the 100-task stream
  (c) Temporal precedence of 4 leading indicators

Reference: Dohare et al. (2023) "Loss of Plasticity in Deep Continual Learning"
"""
import os, sys, json, math, time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import wilcoxon as scipy_wilcoxon
import torchvision

# ─────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────
MNIST_ROOT     = "/tmp/mnist"
RESULTS_DIR    = "results/pmnist"
TRAJ_PATH      = os.path.join(RESULTS_DIR, "trajectories.json")
RESULTS_PATH   = os.path.join(RESULTS_DIR, "RESULTS.json")
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")

N_SEEDS        = 5
N_TASKS        = 100
STEPS_PER_TASK = 1000       # ~1.7 epochs (60k / 128 ≈ 469 steps/epoch)
BATCH_SIZE     = 128
PROBE_SIZE     = 1000       # probe samples for observables
N_CLASSES      = 10
IN_DIM         = 784        # 28×28 grayscale
HIDDEN         = 2000       # Dohare et al. architecture
MOMENTUM       = 0.9
WEIGHT_DECAY   = 0.0

# Collapse: acc drops >= THRESH below task-1 acc for >= MIN_TASKS consecutive tasks
COLLAPSE_THRESH_PP = 15.0
COLLAPSE_MIN_TASKS = 2

# Onset: 50% of range crossing, moving-avg window 2
ONSET_FRAC   = 0.50
ONSET_WINDOW = 2

# GNS: two independent mini-batches of GNS_BATCH samples (>= 20 each)
GNS_BATCH = 64

# Bootstrap CI
N_BOOT     = 2000
BOOT_ALPHA = 0.05

# LR grid to tune (want task1_acc > 0.9 AND dead_after_task1 < 0.20)
LR_CANDIDATES = [0.05, 0.01, 0.005, 0.001]

print(f"Device: {DEVICE}", flush=True)
print(f"Config: {N_SEEDS} seeds × {N_TASKS} tasks × {STEPS_PER_TASK} steps/task", flush=True)
print(f"MLP: {IN_DIM}→{HIDDEN}→{HIDDEN}→{N_CLASSES} (ReLU)", flush=True)
print(f"SGD, momentum={MOMENTUM}, WD={WEIGHT_DECAY}", flush=True)


# ─────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────
class MLP(nn.Module):
    def __init__(self, in_dim=IN_DIM, hidden=HIDDEN, n_out=N_CLASSES):
        super().__init__()
        self.fc1  = nn.Linear(in_dim, hidden)
        self.fc2  = nn.Linear(hidden, hidden)
        self.head = nn.Linear(hidden, n_out)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.head(self.relu(self.fc2(self.relu(self.fc1(x)))))

    def penultimate(self, x):
        """Returns (h1, h2): activations after both hidden ReLUs."""
        h1 = self.relu(self.fc1(x))
        h2 = self.relu(self.fc2(h1))
        return h1, h2


# ─────────────────────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────────────────────
def load_mnist():
    ds_tr = torchvision.datasets.MNIST(MNIST_ROOT, train=True,  download=False)
    ds_te = torchvision.datasets.MNIST(MNIST_ROOT, train=False, download=False)
    X_tr = ds_tr.data.numpy().reshape(-1, IN_DIM).astype(np.float32) / 255.0
    y_tr = ds_tr.targets.numpy().astype(np.int64)
    X_te = ds_te.data.numpy().reshape(-1, IN_DIM).astype(np.float32) / 255.0
    y_te = ds_te.targets.numpy().astype(np.int64)
    print(f"MNIST loaded: train={X_tr.shape}, test={X_te.shape}", flush=True)
    return X_tr, y_tr, X_te, y_te


def make_perms(n_tasks, perm_seed=42):
    """Return list of n_tasks permutations (task 0 = identity)."""
    rng = np.random.default_rng(perm_seed)
    perms = [np.arange(IN_DIM)]
    for _ in range(n_tasks - 1):
        perms.append(rng.permutation(IN_DIM))
    return perms


# ─────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────
def train_task(model, optimizer, X_tr_perm, y_tr, criterion, steps, batch_size):
    """Train on (X_tr_perm, y_tr) for `steps` gradient steps."""
    model.train()
    N = len(y_tr)
    # Move to GPU once per task call
    X_gpu = torch.from_numpy(X_tr_perm).to(DEVICE)
    y_gpu = torch.from_numpy(y_tr).to(DEVICE)
    idx = np.arange(N)
    np.random.shuffle(idx)
    ptr = 0
    for _ in range(steps):
        if ptr + batch_size > N:
            np.random.shuffle(idx)
            ptr = 0
        b = idx[ptr:ptr + batch_size]
        ptr += batch_size
        optimizer.zero_grad()
        criterion(model(X_gpu[b]), y_gpu[b]).backward()
        optimizer.step()


# ─────────────────────────────────────────────────────────────
# Observables
# ─────────────────────────────────────────────────────────────
@torch.no_grad()
def compute_dead(model, probe_X):
    """
    Dead-unit fraction: fraction of ReLU units inactive on >95% of probe samples.
    Averaged over both hidden layers.
    """
    model.eval()
    h1, h2 = model.penultimate(probe_X)
    # A unit is "dead" if <= 0 for > 95% of probe samples
    d1 = ((h1 <= 0).float().mean(0) > 0.95).float().mean().item()
    d2 = ((h2 <= 0).float().mean(0) > 0.95).float().mean().item()
    return (d1 + d2) / 2.0


@torch.no_grad()
def compute_erank(model, probe_X):
    """
    Effective rank of penultimate activations: exp(H(p)) where p = singular values / sum.
    Always >= 1.0 (never 0).
    """
    model.eval()
    _, h2 = model.penultimate(probe_X)
    try:
        S = torch.linalg.svdvals(h2.float())
        S = S[S > 1e-10]
        if len(S) == 0:
            return 1.0   # degenerate: all activations zero
        p = S / S.sum()
        # Shannon entropy → effective rank
        H = -(p * torch.log(p + 1e-12)).sum().item()
        er = math.exp(H)
        return max(1.0, er)   # mathematically >= 1 but guard numerics
    except Exception:
        return 1.0


def compute_gns(model, X_tr_perm, y_tr, criterion, rng_state=None):
    """
    Gradient-noise scale (McCandlish B_opt estimator).
    Two independent mini-batches of GNS_BATCH samples each (>> 20).
    Returns noise/signal ratio ≈ B_opt / B (finite, positive).
    """
    model.train()
    N = len(y_tr)
    if N < 2 * GNS_BATCH:
        return float("nan")

    idx = np.random.permutation(N)
    b1_idx, b2_idx = idx[:GNS_BATCH], idx[GNS_BATCH:2*GNS_BATCH]

    X_gpu = torch.from_numpy(X_tr_perm).to(DEVICE)
    y_gpu = torch.from_numpy(y_tr).to(DEVICE)

    def batch_grad(bidx):
        model.zero_grad()
        loss = criterion(model(X_gpu[bidx]), y_gpu[bidx])
        loss.backward()
        return torch.cat([p.grad.detach().flatten()
                          for p in model.parameters() if p.grad is not None])

    g1 = batch_grad(b1_idx)
    g2 = batch_grad(b2_idx)

    g_avg  = (g1 + g2) / 2.0
    noise  = (g1 - g2).norm(2).pow(2).item() / 2.0   # ≈ trace(Σ) / B
    signal = g_avg.norm(2).pow(2).item()              # ≈ ||G||²

    if signal < 1e-20 or not math.isfinite(noise) or not math.isfinite(signal):
        return float("nan")

    gns_val = noise / signal    # ≈ B_opt / B
    # Sanity check: should be positive and finite
    if gns_val <= 0 or not math.isfinite(gns_val):
        return float("nan")
    return float(gns_val)


@torch.no_grad()
def compute_wdrift(model, init_norms):
    """
    Per-layer relative weight-norm drift from init, averaged across layers.
    Formula: mean over layers of |‖w_t‖ / ‖w_0‖ - 1|
    """
    drifts = []
    for (name, p), w0 in zip(model.named_parameters(), init_norms.values()):
        if w0 > 1e-12:
            drifts.append(abs(p.data.norm(2).item() / w0 - 1.0))
    return float(np.mean(drifts)) if drifts else float("nan")


@torch.no_grad()
def eval_acc(model, X_te_perm, y_te):
    """Evaluate accuracy on (X_te_perm, y_te)."""
    model.eval()
    X_gpu = torch.from_numpy(X_te_perm).to(DEVICE)
    y_gpu = torch.from_numpy(y_te).to(DEVICE)
    N = len(y_te)
    correct = 0
    for i in range(0, N, 1024):
        xb = X_gpu[i:i+1024]
        yb = y_gpu[i:i+1024]
        preds = model(xb).argmax(1)
        correct += (preds == yb).sum().item()
    return correct / N if N > 0 else 0.0


def make_probe(X_te_perm, y_te, seed=0):
    """Random probe subset from test data."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X_te_perm), size=min(PROBE_SIZE, len(X_te_perm)), replace=False)
    X = torch.from_numpy(X_te_perm[idx]).to(DEVICE)
    y = torch.from_numpy(y_te[idx]).to(DEVICE)
    return X, y


# ─────────────────────────────────────────────────────────────
# Collapse detection
# ─────────────────────────────────────────────────────────────
def detect_collapse(accs, thresh_pp=COLLAPSE_THRESH_PP, min_tasks=COLLAPSE_MIN_TASKS):
    """
    First task t (0-indexed) where new-task acc drops >= thresh_pp below task-0 acc
    AND stays below for >= min_tasks consecutive tasks.
    """
    if len(accs) < min_tasks + 1:
        return None
    ref = accs[0]
    count = 0
    first = None
    for t, a in enumerate(accs):
        if (ref - a) * 100.0 >= thresh_pp:
            count += 1
            if first is None:
                first = t
        else:
            count = 0
            first = None
        if count >= min_tasks:
            return first
    return None


# ─────────────────────────────────────────────────────────────
# Onset detection (50% of range, moving-avg window 2)
# ─────────────────────────────────────────────────────────────
def moving_avg(vals, window=ONSET_WINDOW):
    """Apply a causal moving average of size `window`."""
    out = []
    for i in range(len(vals)):
        chunk = [v for v in vals[max(0, i-window+1):i+1] if math.isfinite(v)]
        out.append(float(np.mean(chunk)) if chunk else float("nan"))
    return out


def compute_onset(vals_raw, frac=ONSET_FRAC, window=ONSET_WINDOW):
    """
    Onset: first task where the smoothed signal crosses 50% of its total range.
    Robust non-monotone: range defined by [first valid, last valid] values.
    Returns 0-indexed task or None.
    """
    vals = moving_avg(vals_raw, window)
    clean = [(i, v) for i, v in enumerate(vals) if math.isfinite(v)]
    if len(clean) < 4:
        return None
    v0 = clean[0][1]
    vf = clean[-1][1]
    if abs(vf - v0) < 1e-10:
        return None  # no change
    thr = v0 + frac * (vf - v0)
    direction = 1 if vf > v0 else -1
    for i, v in clean:
        if direction == 1 and v >= thr:
            return i
        if direction == -1 and v <= thr:
            return i
    return None


# ─────────────────────────────────────────────────────────────
# Bootstrap CI
# ─────────────────────────────────────────────────────────────
def bootstrap_ci(data, n_boot=N_BOOT, alpha=BOOT_ALPHA, seed=0):
    """Percentile bootstrap CI for the mean."""
    arr = np.array([x for x in data if x is not None and math.isfinite(x)], dtype=float)
    if len(arr) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    stats = [np.mean(rng.choice(arr, size=len(arr), replace=True)) for _ in range(n_boot)]
    return (float(np.percentile(stats, 100*alpha/2)),
            float(np.percentile(stats, 100*(1-alpha/2))))


# ─────────────────────────────────────────────────────────────
# LR tuning (quick 1-seed pass)
# ─────────────────────────────────────────────────────────────
def tune_lr(X_tr, y_tr, X_te, y_te, perms, candidates):
    """
    Try each LR candidate on seed 0 for task 0.
    Select the LR with highest task1_acc subject to dead_after_task1 < 0.20.
    """
    print("\n=== LR Tuning (seed=0, task=0) ===", flush=True)
    criterion = nn.CrossEntropyLoss()

    # Build task-0 arrays (identity permutation)
    X_tr0 = X_tr[:, perms[0]]
    X_te0 = X_te[:, perms[0]]

    best_lr  = candidates[-1]   # fallback
    best_acc = -1.0
    dead_at_init_global = None

    for lr in candidates:
        torch.manual_seed(0)
        np.random.seed(0)
        model = MLP().to(DEVICE)
        optimizer = optim.SGD(model.parameters(), lr=lr,
                              momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
        pr_X, pr_y = make_probe(X_te0, y_te, seed=99)

        # Measure dead_at_init (same for all LRs; record from first pass)
        dead_init = compute_dead(model, pr_X)
        if dead_at_init_global is None:
            dead_at_init_global = dead_init

        # Sanity: CE at init ≈ log(10)
        model.eval()
        with torch.no_grad():
            ce_init = criterion(model(pr_X), pr_y).item()

        train_task(model, optimizer, X_tr0, y_tr, criterion, STEPS_PER_TASK, BATCH_SIZE)

        acc1  = eval_acc(model, X_te0, y_te)
        dead1 = compute_dead(model, pr_X)

        health_ok = (acc1 > 0.9) and (dead1 < 0.20)
        print(f"  lr={lr:.4f}: init_CE={ce_init:.4f} (log10={math.log(10):.4f}), "
              f"task1_acc={acc1:.4f}, dead_init={dead_init:.4f}, "
              f"dead_after_t1={dead1:.4f}  {'✓ HEALTHY' if health_ok else '✗'}", flush=True)

        if health_ok and acc1 > best_acc:
            best_acc = acc1
            best_lr  = lr

    print(f"  → Selected LR = {best_lr}  (task1_acc = {best_acc:.4f})", flush=True)
    return best_lr, dead_at_init_global


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
def main():
    t_start = time.time()
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # 1. Load MNIST
    X_tr, y_tr, X_te, y_te = load_mnist()

    # 2. Create task permutations (fixed across seeds)
    perms = make_perms(N_TASKS, perm_seed=42)
    print(f"Created {N_TASKS} task permutations (task 0 = identity)", flush=True)

    # 3. LR tuning
    LR, dead_at_init_est = tune_lr(X_tr, y_tr, X_te, y_te, perms, LR_CANDIDATES)

    print(f"\n=== Main Experiment ===", flush=True)
    print(f"LR={LR}, momentum={MOMENTUM}, WD={WEIGHT_DECAY}", flush=True)
    print(f"{N_SEEDS} seeds × {N_TASKS} tasks × {STEPS_PER_TASK} steps\n", flush=True)

    criterion  = nn.CrossEntropyLoss()
    all_results = []

    for seed in range(N_SEEDS):
        print(f"\n{'='*70}\nSEED {seed}\n{'='*70}", flush=True)
        torch.manual_seed(seed)
        np.random.seed(seed)

        model     = MLP().to(DEVICE)
        optimizer = optim.SGD(model.parameters(), lr=LR,
                              momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
        init_norms = {n: p.data.norm(2).item() for n, p in model.named_parameters()}

        # dead_at_init (before any training, using task-0 probe)
        X_te0     = X_te[:, perms[0]]
        pr0_X, pr0_y = make_probe(X_te0, y_te, seed=seed)
        dead_init = compute_dead(model, pr0_X)

        # Sanity gate: CE at init ≈ log(N_CLASSES)
        model.eval()
        with torch.no_grad():
            ce_init = criterion(model(pr0_X), pr0_y).item()
        exp_ce    = math.log(N_CLASSES)
        sanity_ok = abs(ce_init - exp_ce) < 0.05
        print(f"  [sanity] init CE = {ce_init:.4f}, expected = {exp_ce:.4f}, "
              f"diff = {abs(ce_init - exp_ce):.4f}  {'PASS' if sanity_ok else 'WARN'}",
              flush=True)
        print(f"  dead_at_init = {dead_init:.4f}", flush=True)

        # Per-task trajectories
        accs, deads, eranks, gnss, wdrifts = [], [], [], [], []

        for t_idx in range(N_TASKS):
            perm       = perms[t_idx]
            X_tr_perm  = X_tr[:, perm]
            X_te_perm  = X_te[:, perm]

            # Train on current task
            train_task(model, optimizer, X_tr_perm, y_tr, criterion,
                       STEPS_PER_TASK, BATCH_SIZE)

            # Probe from current task's test data
            pr_X, pr_y = make_probe(X_te_perm, y_te, seed=seed * 1000 + t_idx)

            # New-task accuracy
            acc   = eval_acc(model, X_te_perm, y_te)
            # Dead-unit fraction
            dead  = compute_dead(model, pr_X)
            # Effective rank (penultimate, always >= 1)
            er    = compute_erank(model, pr_X)
            # Gradient-noise scale (McCandlish B_opt, GNS_BATCH >= 20 per batch)
            gns   = compute_gns(model, X_tr_perm, y_tr, criterion)
            # Weight-norm drift from initialization
            wd    = compute_wdrift(model, init_norms)

            accs.append(acc); deads.append(dead); eranks.append(er)
            gnss.append(gns); wdrifts.append(wd)

            # Print every task for first 5, then every 10th
            if t_idx < 5 or (t_idx + 1) % 10 == 0 or t_idx == N_TASKS - 1:
                print(f"  Task {t_idx+1:3d}/{N_TASKS} | "
                      f"acc={acc:.4f}  dead={dead:.4f}  "
                      f"erank={er:7.3f}  gns={gns:8.4f}  wdrift={wd:.4f}",
                      flush=True)

        dead_after_t1 = deads[0]
        t_col = detect_collapse(accs)
        healthy = (accs[0] > 0.9) and (dead_after_t1 < 0.20)

        print(f"\n  --- Seed {seed} summary ---", flush=True)
        print(f"  dead_at_init={dead_init:.4f}", flush=True)
        print(f"  dead_after_task1={dead_after_t1:.4f}", flush=True)
        print(f"  task1_acc={accs[0]:.4f}  late10_acc={np.mean(accs[-10:]):.4f}", flush=True)
        print(f"  t_collapse={t_col}  healthy={healthy}", flush=True)

        all_results.append({
            "seed":            seed,
            "dead_at_init":    dead_init,
            "dead_after_task1": dead_after_t1,
            "task1_acc":       accs[0],
            "healthy":         healthy,
            "t_collapse":      t_col,
            "accs":            accs,
            "dead":            deads,
            "erank":           eranks,
            "gns":             gnss,
            "wdrift":          wdrifts,
        })

        # Incremental RESULTS.json after each seed
        partial = {
            "status":      "RUNNING",
            "seeds_done":  seed + 1,
            "per_seed_summary": [
                {"seed": r["seed"], "task1_acc": r["task1_acc"],
                 "dead_at_init": r["dead_at_init"],
                 "dead_after_task1": r["dead_after_task1"],
                 "healthy": r["healthy"], "t_collapse": r["t_collapse"]}
                for r in all_results
            ]
        }
        with open(RESULTS_PATH, "w") as f:
            json.dump(partial, f, indent=2)

    # ─────────────────────────────────────────────────────────
    # Analysis
    # ─────────────────────────────────────────────────────────
    print("\n\n" + "="*70, flush=True)
    print("ANALYSIS", flush=True)
    print("="*70, flush=True)

    mean_dead_init  = float(np.mean([r["dead_at_init"] for r in all_results]))
    mean_dead_t1    = float(np.mean([r["dead_after_task1"] for r in all_results]))
    mean_task1_acc  = float(np.mean([r["task1_acc"] for r in all_results]))
    mean_late_acc   = float(np.mean([np.mean(r["accs"][-10:]) for r in all_results]))
    acc_drop_pp     = (mean_task1_acc - mean_late_acc) * 100.0
    all_t_collapse  = [r["t_collapse"] for r in all_results]
    n_collapsed     = sum(t is not None for t in all_t_collapse)
    plasticity_loss = n_collapsed >= 3
    healthy_all     = all(r["healthy"] for r in all_results)

    print(f"dead_at_init (mean across seeds):     {mean_dead_init:.4f}", flush=True)
    print(f"dead_after_task1 (mean across seeds): {mean_dead_t1:.4f}", flush=True)
    print(f"task1_acc (mean across seeds):        {mean_task1_acc:.4f}", flush=True)
    print(f"late_acc tasks 91-100 (mean):         {mean_late_acc:.4f}", flush=True)
    print(f"acc_drop_pp:                          {acc_drop_pp:.2f}pp", flush=True)
    print(f"healthy_all:                          {healthy_all}", flush=True)
    print(f"n_collapsed:                          {n_collapsed}/{N_SEEDS}", flush=True)
    print(f"t_collapse_per_seed:                  {all_t_collapse}", flush=True)
    print(f"plasticity_loss_occurred:             {plasticity_loss}", flush=True)

    # ── Lead times ──
    # Observable keys and display names (matching RESULTS.json template)
    obs_keys   = ["dead",           "erank",          "gns",                    "wdrift"          ]
    obs_names  = ["dead_unit_fraction", "effective_rank", "gradient_noise_scale", "weight_norm_drift"]

    lead_per_seed = {k: [] for k in obs_keys}   # list of lead times (or None)

    print("\n--- Onset detection ---", flush=True)
    for r in all_results:
        t_col = r["t_collapse"]
        print(f"  Seed {r['seed']}, t_collapse={t_col}:", flush=True)
        for k, nm in zip(obs_keys, obs_names):
            if t_col is None:
                lead_per_seed[k].append(None)
                print(f"    {nm}: onset=N/A (no collapse)", flush=True)
                continue
            onset_t = compute_onset(r[k])
            lt = (t_col - onset_t) if onset_t is not None else None
            lead_per_seed[k].append(lt)
            print(f"    {nm}: onset={onset_t}, lead={lt}", flush=True)

    # ── Summarize lead times ──
    lead_summary = {}
    print("\n--- Lead-time summary ---", flush=True)
    for k, nm in zip(obs_keys, obs_names):
        lts   = [x for x in lead_per_seed[k] if x is not None]
        n_val = len(lts)
        if n_val == 0:
            lead_summary[nm] = {
                "mean": None, "median": None,
                "iqr": [None, None], "ci95": [None, None], "n_valid": 0
            }
            print(f"  {nm}: no valid lead times", flush=True)
            continue
        arr  = np.array(lts, dtype=float)
        ci   = bootstrap_ci(lts, seed=42)
        lead_summary[nm] = {
            "mean":    float(np.mean(arr)),
            "median":  float(np.median(arr)),
            "iqr":     [float(np.percentile(arr, 25)), float(np.percentile(arr, 75))],
            "ci95":    [float(ci[0]), float(ci[1])],
            "n_valid": n_val,
        }
        print(f"  {nm}: {lts} → mean={np.mean(arr):.2f}, "
              f"median={np.median(arr):.2f}, "
              f"IQR=[{np.percentile(arr,25):.1f},{np.percentile(arr,75):.1f}], "
              f"CI95={[round(c,2) for c in ci]}", flush=True)

    # ── Precedence order (highest mean lead time first) ──
    sortable = [(nm, lead_summary[nm]["mean"])
                for nm in obs_names if lead_summary[nm]["mean"] is not None]
    sortable.sort(key=lambda x: x[1], reverse=True)
    precedence_order = [nm for nm, _ in sortable]
    print(f"\n  Precedence order: {precedence_order}", flush=True)

    # ── Wilcoxon: paired erank vs GNS lead-time difference ──
    paired_diffs = []
    for er_lt, gns_lt in zip(lead_per_seed["erank"], lead_per_seed["gns"]):
        if er_lt is not None and gns_lt is not None:
            paired_diffs.append(er_lt - gns_lt)

    wilcoxon_stat = None
    wilcoxon_p    = None
    median_diff   = float(np.median(paired_diffs)) if paired_diffs else None

    print(f"\n  erank - gns lead diffs (per seed): {paired_diffs}", flush=True)
    print(f"  median_diff: {median_diff}", flush=True)

    if len(paired_diffs) >= 3:
        diffs_arr = np.array(paired_diffs, dtype=float)
        nonzero = diffs_arr[diffs_arr != 0]
        if len(nonzero) >= 3:
            try:
                stat, p = scipy_wilcoxon(diffs_arr, alternative="two-sided",
                                         zero_method="wilcox")
                wilcoxon_stat = float(stat)
                wilcoxon_p    = float(p)
                print(f"  Wilcoxon signed-rank: stat={wilcoxon_stat:.3f}, p={wilcoxon_p:.4f}",
                      flush=True)
            except Exception as e:
                print(f"  Wilcoxon failed: {e}", flush=True)
        else:
            print(f"  Wilcoxon: too few nonzero diffs ({len(nonzero)}) — skipped", flush=True)
    else:
        print(f"  Wilcoxon: too few pairs ({len(paired_diffs)}) — skipped", flush=True)

    elapsed = time.time() - t_start
    print(f"\nWall-clock: {elapsed:.1f}s ({elapsed/60:.1f} min)", flush=True)

    # ─────────────────────────────────────────────────────────
    # Save trajectories
    # ─────────────────────────────────────────────────────────
    def nan_to_null(x):
        if isinstance(x, float) and not math.isfinite(x):
            return None
        return x

    trajectories = {
        "n_seeds": N_SEEDS,
        "n_tasks": N_TASKS,
        "lr":      LR,
        "tasks":   list(range(1, N_TASKS + 1)),
        "per_seed": [
            {
                "seed":      r["seed"],
                "accs":      r["accs"],
                "dead":      r["dead"],
                "erank":     [nan_to_null(v) for v in r["erank"]],
                "gns":       [nan_to_null(v) for v in r["gns"]],
                "wdrift":    r["wdrift"],
                "t_collapse": r["t_collapse"],
                "dead_at_init": r["dead_at_init"],
                "dead_after_task1": r["dead_after_task1"],
            }
            for r in all_results
        ],
    }
    with open(TRAJ_PATH, "w") as f:
        json.dump(trajectories, f, indent=2)
    print(f"\nTrajectories → {TRAJ_PATH}", flush=True)

    # ─────────────────────────────────────────────────────────
    # Final RESULTS.json
    # ─────────────────────────────────────────────────────────
    results = {
        "status":                 "DONE",
        "benchmark":              "permuted_mnist",
        "n_tasks":                N_TASKS,
        "n_seeds":                N_SEEDS,
        "lr_used":                LR,
        "momentum":               MOMENTUM,
        "weight_decay":           WEIGHT_DECAY,
        "hidden_size":            HIDDEN,
        "steps_per_task":         STEPS_PER_TASK,
        "gns_batch_size":         GNS_BATCH,
        # Health
        "dead_at_init":           mean_dead_init,
        "dead_after_task1":       mean_dead_t1,
        "task1_acc":              mean_task1_acc,
        "healthy":                healthy_all,
        # Plasticity loss
        "plasticity_loss_occurred": plasticity_loss,
        "acc_task1_mean":         mean_task1_acc,
        "acc_late_mean":          mean_late_acc,
        "acc_drop_pp":            acc_drop_pp,
        "t_collapse_per_seed":    all_t_collapse,
        "n_seeds_collapsed":      n_collapsed,
        # Lead times
        "lead_times":             lead_summary,
        "precedence_order":       precedence_order,
        # Paired Wilcoxon
        "erank_vs_gns_paired": {
            "diffs_per_seed": paired_diffs,
            "median_diff":    median_diff,
            "wilcoxon_stat":  wilcoxon_stat,
            "wilcoxon_p":     wilcoxon_p,
        },
        # Per-seed details
        "per_seed_summary": [
            {
                "seed":            r["seed"],
                "dead_at_init":    r["dead_at_init"],
                "dead_after_task1": r["dead_after_task1"],
                "task1_acc":       r["task1_acc"],
                "healthy":         r["healthy"],
                "t_collapse":      r["t_collapse"],
                "acc_task1":       r["accs"][0],
                "acc_task50":      r["accs"][49] if len(r["accs"]) > 49 else None,
                "acc_task100":     r["accs"][-1],
                "lead_dead":       lead_per_seed["dead"][r["seed"]],
                "lead_erank":      lead_per_seed["erank"][r["seed"]],
                "lead_gns":        lead_per_seed["gns"][r["seed"]],
                "lead_wdrift":     lead_per_seed["wdrift"][r["seed"]],
            }
            for r in all_results
        ],
        "wall_clock_sec": round(elapsed, 1),
        "subject_executed": (
            f"Online Permuted-MNIST, 3-layer MLP ({IN_DIM}→{HIDDEN}→{HIDDEN}→{N_CLASSES}, ReLU), "
            f"SGD lr={LR} momentum={MOMENTUM} WD={WEIGHT_DECAY}, no plasticity repair, "
            f"{N_TASKS} tasks × {STEPS_PER_TASK} steps/task × {N_SEEDS} seeds. "
            f"4 observables per task: dead_unit_fraction (>95% threshold), "
            f"effective_rank (penultimate, always>=1), "
            f"gradient_noise_scale (McCandlish B_opt, {GNS_BATCH} samples/batch), "
            f"weight_norm_drift. LR tuned on seed=0 task=0."
        ),
        "notes": (
            f"Network healthy={healthy_all}: dead_at_init={mean_dead_init:.3f}, "
            f"dead_after_task1={mean_dead_t1:.3f} (<0.20), "
            f"task1_acc={mean_task1_acc:.3f} (>0.90). "
            f"Plasticity loss occurred={plasticity_loss}: "
            f"{n_collapsed}/{N_SEEDS} seeds show collapse (>={COLLAPSE_THRESH_PP}pp drop, "
            f">={COLLAPSE_MIN_TASKS} consecutive tasks). "
            f"acc_drop={acc_drop_pp:.1f}pp (task1→late). "
            f"Precedence order={precedence_order}. "
            f"erank_vs_gns Wilcoxon p={wilcoxon_p}. "
            f"GNS batch={GNS_BATCH} samples (>20). "
            f"Effective rank always>=1 (handled degenerate case)."
        ),
    }

    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults → {RESULTS_PATH}", flush=True)
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
