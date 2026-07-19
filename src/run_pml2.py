#!/usr/bin/env python3
"""
run_pml2.py — DEFINITIVE Permuted-MNIST 300-task plasticity experiment (pml2)
==========================================================================
300 tasks × 3 seeds, 3-layer MLP (784→2000→2000→10, ReLU), SGD+momentum.
No plasticity repair.  Tests whether loss of plasticity emerges at 300 tasks
in a HEALTHY network (it was absent at 100 tasks in prior pmnist run).

Key question:
  Does new-task accuracy DECLINE from early tasks (first 20) to late tasks
  (last 20) at 300 tasks?  Report acc_first20 vs acc_last20.

Reference: Dohare et al. (2023) "Loss of Plasticity in Deep Continual Learning"
Builds on: results/pmnist/ (100-task run, lr=0.05, healthy net, no collapse)
"""
import os, sys, json, math, time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import wilcoxon as scipy_wilcoxon
import torchvision

# ─────────────────────────────────────────────────────────────
# Config (changes vs pmnist: N_TASKS=300, N_SEEDS=3, dir=pml2)
# ─────────────────────────────────────────────────────────────
MNIST_ROOT     = "/tmp/mnist"
RESULTS_DIR    = "results/pml2"
TRAJ_PATH      = os.path.join(RESULTS_DIR, "trajectories.json")
RESULTS_PATH   = os.path.join(RESULTS_DIR, "RESULTS.json")
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")

N_SEEDS        = 3
N_TASKS        = 300
STEPS_PER_TASK = 1000        # ~1.7 epochs (60k / 128 ≈ 469 steps/epoch)
BATCH_SIZE     = 128
PROBE_SIZE     = 1000        # probe samples for observables
N_CLASSES      = 10
IN_DIM         = 784         # 28×28 grayscale
HIDDEN         = 2000        # Dohare et al. architecture
LR             = 0.05        # From prior pmnist tuning (task1_acc=0.975, dead<0.023)
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

# Incremental save frequency (every SAVE_EVERY tasks per seed)
SAVE_EVERY = 25

print(f"Device: {DEVICE}", flush=True)
print(f"Config: {N_SEEDS} seeds × {N_TASKS} tasks × {STEPS_PER_TASK} steps/task", flush=True)
print(f"MLP: {IN_DIM}→{HIDDEN}→{HIDDEN}→{N_CLASSES} (ReLU)", flush=True)
print(f"SGD, lr={LR}, momentum={MOMENTUM}, WD={WEIGHT_DECAY}", flush=True)
print(f"(LR from prior pmnist run: task1_acc≈0.975, dead_after_t1≈0.023)", flush=True)


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
    ds_tr = torchvision.datasets.MNIST(MNIST_ROOT, train=True,  download=True)
    ds_te = torchvision.datasets.MNIST(MNIST_ROOT, train=False, download=True)
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
            return 1.0
        p = S / S.sum()
        H = -(p * torch.log(p + 1e-12)).sum().item()
        er = math.exp(H)
        return max(1.0, er)
    except Exception:
        return 1.0


def compute_gns(model, X_tr_perm, y_tr, criterion):
    """
    Gradient-noise scale (McCandlish B_opt estimator).
    Two independent mini-batches of GNS_BATCH samples each.
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
    noise  = (g1 - g2).norm(2).pow(2).item() / 2.0
    signal = g_avg.norm(2).pow(2).item()

    if signal < 1e-20 or not math.isfinite(noise) or not math.isfinite(signal):
        return float("nan")

    gns_val = noise / signal
    if gns_val <= 0 or not math.isfinite(gns_val):
        return float("nan")
    return float(gns_val)


@torch.no_grad()
def compute_wdrift(model, init_norms):
    """
    Per-layer relative weight-norm drift from init, averaged across layers.
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
    out = []
    for i in range(len(vals)):
        chunk = [v for v in vals[max(0, i-window+1):i+1] if math.isfinite(v)]
        out.append(float(np.mean(chunk)) if chunk else float("nan"))
    return out


def compute_onset(vals_raw, frac=ONSET_FRAC, window=ONSET_WINDOW):
    """
    Onset: first task where the smoothed signal crosses 50% of its total range.
    """
    vals = moving_avg(vals_raw, window)
    clean = [(i, v) for i, v in enumerate(vals) if math.isfinite(v)]
    if len(clean) < 4:
        return None
    v0 = clean[0][1]
    vf = clean[-1][1]
    if abs(vf - v0) < 1e-10:
        return None
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
    arr = np.array([x for x in data if x is not None and math.isfinite(x)], dtype=float)
    if len(arr) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    stats = [np.mean(rng.choice(arr, size=len(arr), replace=True)) for _ in range(n_boot)]
    return (float(np.percentile(stats, 100*alpha/2)),
            float(np.percentile(stats, 100*(1-alpha/2))))


# ─────────────────────────────────────────────────────────────
# Incremental save helper
# ─────────────────────────────────────────────────────────────
def save_partial(all_results, seeds_done, extra=None):
    partial = {
        "status":       "RUNNING",
        "seeds_done":   seeds_done,
        "per_seed_so_far": [
            {
                "seed":           r["seed"],
                "tasks_done":     len(r["accs"]),
                "task1_acc":      r["accs"][0] if r["accs"] else None,
                "dead_at_init":   r["dead_at_init"],
                "dead_after_task1": r["dead_after_task1"],
                "healthy":        r["healthy"],
                "t_collapse":     r["t_collapse"],
            }
            for r in all_results
        ]
    }
    if extra:
        partial.update(extra)
    with open(RESULTS_PATH, "w") as f:
        json.dump(partial, f, indent=2)


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
def main():
    t_start = time.time()
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Write initial LOG.md
    log_path = os.path.join(RESULTS_DIR, "LOG.md")
    with open(log_path, "w") as f:
        f.write(f"# pml2 Experiment Log\n\n")
        f.write(f"## Setup\n")
        f.write(f"- Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"- Script: src/run_pml2.py\n")
        f.write(f"- N_TASKS=300, N_SEEDS=3, HIDDEN=2000, LR=0.05\n")
        f.write(f"- Device: {DEVICE}\n")
        f.write(f"- Extending prior pmnist 100-task run (no collapse found)\n\n")
        f.write(f"## Decisions\n")
        f.write(f"- LR=0.05 reused from pmnist tuning (task1_acc=0.975, dead_after_t1=0.023)\n")
        f.write(f"- N_SEEDS=3 (experiment brief specifies 3)\n")
        f.write(f"- STEPS_PER_TASK=1000 (same as pmnist, fast+healthy)\n")
        f.write(f"- GNS_BATCH=64 (two mini-batches of 64, > 20 required)\n")
        f.write(f"- Incremental save every {SAVE_EVERY} tasks\n\n")

    # 1. Load MNIST
    X_tr, y_tr, X_te, y_te = load_mnist()

    # 2. Create task permutations (SAME seed as pmnist for reproducibility)
    perms = make_perms(N_TASKS, perm_seed=42)
    print(f"Created {N_TASKS} task permutations (task 0 = identity)", flush=True)

    # 3. Verify LR on seed=0 / task=0
    print(f"\n=== LR Verification (seed=0, task=0) ===", flush=True)
    criterion = nn.CrossEntropyLoss()
    {
        # Quick sanity: CE at init
    }
    torch.manual_seed(0)
    np.random.seed(0)
    _model_check = MLP().to(DEVICE)
    _pr_X, _pr_y = make_probe(X_te[:, perms[0]], y_te, seed=99)
    with torch.no_grad():
        ce_init = criterion(_model_check(_pr_X), _pr_y).item()
    exp_ce = math.log(N_CLASSES)
    dead_check = compute_dead(_model_check, _pr_X)
    print(f"  CE at init = {ce_init:.4f}  (expected {exp_ce:.4f}, diff={abs(ce_init-exp_ce):.4f})",
          flush=True)
    print(f"  dead_at_init = {dead_check:.4f}", flush=True)
    print(f"  Using LR={LR} (from prior run; expected task1_acc≈0.975)", flush=True)
    del _model_check

    print(f"\n=== Main Experiment ===", flush=True)
    print(f"LR={LR}, momentum={MOMENTUM}, WD={WEIGHT_DECAY}", flush=True)
    print(f"{N_SEEDS} seeds × {N_TASKS} tasks × {STEPS_PER_TASK} steps\n", flush=True)

    all_results = []

    for seed in range(N_SEEDS):
        print(f"\n{'='*70}\nSEED {seed}\n{'='*70}", flush=True)
        torch.manual_seed(seed)
        np.random.seed(seed)

        model     = MLP().to(DEVICE)
        optimizer = optim.SGD(model.parameters(), lr=LR,
                              momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
        init_norms = {n: p.data.norm(2).item() for n, p in model.named_parameters()}

        # dead_at_init (before any training)
        X_te0 = X_te[:, perms[0]]
        pr0_X, pr0_y = make_probe(X_te0, y_te, seed=seed)
        dead_init = compute_dead(model, pr0_X)

        # Sanity gate: CE at init ≈ log(N_CLASSES)
        model.eval()
        with torch.no_grad():
            ce_init_s = criterion(model(pr0_X), pr0_y).item()
        sanity_ok = abs(ce_init_s - exp_ce) < 0.05
        print(f"  [sanity] init CE = {ce_init_s:.4f}, expected = {exp_ce:.4f}  "
              f"{'PASS' if sanity_ok else 'WARN'}", flush=True)
        print(f"  dead_at_init = {dead_init:.4f}", flush=True)

        # Per-task trajectories
        accs, deads, eranks, gnss, wdrifts = [], [], [], [], []
        dead_after_t1 = None
        t_col_running = None

        # Seed result dict for incremental saves
        seed_rec = {
            "seed":            seed,
            "dead_at_init":    dead_init,
            "dead_after_task1": 0.0,
            "healthy":         False,
            "t_collapse":      None,
            "accs":            accs,
            "dead":            deads,
            "erank":           eranks,
            "gns":             gnss,
            "wdrift":          wdrifts,
        }
        all_results.append(seed_rec)

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
            acc  = eval_acc(model, X_te_perm, y_te)
            # Dead-unit fraction
            dead = compute_dead(model, pr_X)
            # Effective rank
            er   = compute_erank(model, pr_X)
            # Gradient-noise scale
            gns  = compute_gns(model, X_tr_perm, y_tr, criterion)
            # Weight-norm drift
            wd   = compute_wdrift(model, init_norms)

            accs.append(acc); deads.append(dead); eranks.append(er)
            gnss.append(gns); wdrifts.append(wd)

            if t_idx == 0:
                dead_after_t1 = dead
                seed_rec["dead_after_task1"] = dead
                seed_rec["healthy"] = (acc > 0.9) and (dead < 0.20)

            # Running collapse detection
            t_col_running = detect_collapse(accs)
            seed_rec["t_collapse"] = t_col_running

            # Print every task for first 5, then every 10th
            if t_idx < 5 or (t_idx + 1) % 10 == 0 or t_idx == N_TASKS - 1:
                print(f"  Task {t_idx+1:3d}/{N_TASKS} | "
                      f"acc={acc:.4f}  dead={dead:.4f}  "
                      f"erank={er:7.3f}  gns={gns:8.4f}  wdrift={wd:.4f}",
                      flush=True)

            # Incremental save every SAVE_EVERY tasks
            if (t_idx + 1) % SAVE_EVERY == 0:
                save_partial(all_results, seed,
                             extra={"current_seed": seed, "current_task": t_idx + 1})
                print(f"  [saved incremental at task {t_idx+1}]", flush=True)

        # End of seed
        healthy = (accs[0] > 0.9) and (dead_after_t1 < 0.20)
        seed_rec["healthy"] = healthy
        seed_rec["t_collapse"] = t_col_running

        print(f"\n  --- Seed {seed} summary ---", flush=True)
        print(f"  dead_at_init={dead_init:.4f}", flush=True)
        print(f"  dead_after_task1={dead_after_t1:.4f}", flush=True)
        print(f"  task1_acc={accs[0]:.4f}", flush=True)
        print(f"  acc_first20_mean={np.mean(accs[:20]):.4f}", flush=True)
        print(f"  acc_last20_mean={np.mean(accs[-20:]):.4f}", flush=True)
        print(f"  acc_drop_first20_last20={( np.mean(accs[:20]) - np.mean(accs[-20:]) )*100:.2f}pp",
              flush=True)
        print(f"  t_collapse={t_col_running}  healthy={healthy}", flush=True)

        save_partial(all_results, seed + 1,
                     extra={"seed_done": seed, "task1_acc": accs[0],
                            "acc_first20": float(np.mean(accs[:20])),
                            "acc_last20":  float(np.mean(accs[-20:]))})

    # ─────────────────────────────────────────────────────────
    # Analysis
    # ─────────────────────────────────────────────────────────
    print("\n\n" + "="*70, flush=True)
    print("ANALYSIS", flush=True)
    print("="*70, flush=True)

    mean_dead_init  = float(np.mean([r["dead_at_init"]    for r in all_results]))
    mean_dead_t1    = float(np.mean([r["dead_after_task1"] for r in all_results]))
    mean_task1_acc  = float(np.mean([r["accs"][0]          for r in all_results]))

    # First 20 and last 20 task accuracies (the core question)
    first20_per_seed = [np.mean(r["accs"][:20])  for r in all_results]
    last20_per_seed  = [np.mean(r["accs"][-20:]) for r in all_results]
    mean_first20 = float(np.mean(first20_per_seed))
    mean_last20  = float(np.mean(last20_per_seed))
    acc_drop_pp  = (mean_first20 - mean_last20) * 100.0

    all_t_collapse  = [r["t_collapse"] for r in all_results]
    n_collapsed     = sum(t is not None for t in all_t_collapse)
    plasticity_loss = n_collapsed >= 2   # >= 2 of 3 seeds
    healthy_all     = all(r["healthy"] for r in all_results)

    print(f"dead_at_init (mean):            {mean_dead_init:.4f}", flush=True)
    print(f"dead_after_task1 (mean):        {mean_dead_t1:.4f}", flush=True)
    print(f"task1_acc (mean):               {mean_task1_acc:.4f}", flush=True)
    print(f"acc_first20_mean (tasks 1-20):  {mean_first20:.4f}", flush=True)
    print(f"acc_last20_mean  (tasks 281-300):{mean_last20:.4f}", flush=True)
    print(f"acc_drop_pp (first20→last20):   {acc_drop_pp:.2f}pp", flush=True)
    print(f"healthy_all:                    {healthy_all}", flush=True)
    print(f"n_collapsed:                    {n_collapsed}/{N_SEEDS}", flush=True)
    print(f"t_collapse_per_seed:            {all_t_collapse}", flush=True)
    print(f"plasticity_loss_occurred:       {plasticity_loss}", flush=True)

    for r in all_results:
        print(f"\n  Seed {r['seed']}: "
              f"first20={np.mean(r['accs'][:20]):.4f}  "
              f"last20={np.mean(r['accs'][-20:]):.4f}  "
              f"drop={(np.mean(r['accs'][:20])-np.mean(r['accs'][-20:]))*100:.2f}pp  "
              f"t_collapse={r['t_collapse']}", flush=True)

    # ── Observable keys ──
    obs_keys  = ["dead",              "erank",          "gns",                  "wdrift"          ]
    obs_names = ["dead_unit_fraction","effective_rank",  "gradient_noise_scale", "weight_norm_drift"]

    lead_per_seed = {k: [] for k in obs_keys}

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

    # ── Precedence order ──
    sortable = [(nm, lead_summary[nm]["mean"])
                for nm in obs_names if lead_summary[nm]["mean"] is not None]
    sortable.sort(key=lambda x: x[1], reverse=True)
    precedence_order = [nm for nm, _ in sortable]
    print(f"\n  Precedence order: {precedence_order}", flush=True)

    # ── Wilcoxon: paired erank vs GNS ──
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
        nonzero   = diffs_arr[diffs_arr != 0]
        if len(nonzero) >= 3:
            try:
                stat, p = scipy_wilcoxon(diffs_arr, alternative="two-sided",
                                         zero_method="wilcox")
                wilcoxon_stat = float(stat)
                wilcoxon_p    = float(p)
                print(f"  Wilcoxon: stat={wilcoxon_stat:.3f}, p={wilcoxon_p:.4f}", flush=True)
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
                "seed":             r["seed"],
                "accs":             r["accs"],
                "dead":             r["dead"],
                "erank":            [nan_to_null(v) for v in r["erank"]],
                "gns":              [nan_to_null(v) for v in r["gns"]],
                "wdrift":           r["wdrift"],
                "t_collapse":       r["t_collapse"],
                "dead_at_init":     r["dead_at_init"],
                "dead_after_task1": r["dead_after_task1"],
                "first20_acc":      float(np.mean(r["accs"][:20])),
                "last20_acc":       float(np.mean(r["accs"][-20:])),
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

    # Build per_seed detail
    per_seed_summary = []
    for r in all_results:
        ps = {
            "seed":             r["seed"],
            "dead_at_init":     r["dead_at_init"],
            "dead_after_task1": r["dead_after_task1"],
            "task1_acc":        r["accs"][0],
            "healthy":          r["healthy"],
            "t_collapse":       r["t_collapse"],
            "acc_task1":        r["accs"][0],
            "acc_task50":       r["accs"][49]  if len(r["accs"]) > 49  else None,
            "acc_task100":      r["accs"][99]  if len(r["accs"]) > 99  else None,
            "acc_task200":      r["accs"][199] if len(r["accs"]) > 199 else None,
            "acc_task300":      r["accs"][-1],
            "acc_first20":      float(np.mean(r["accs"][:20])),
            "acc_last20":       float(np.mean(r["accs"][-20:])),
            "lead_dead":        lead_per_seed["dead"][r["seed"]],
            "lead_erank":       lead_per_seed["erank"][r["seed"]],
            "lead_gns":         lead_per_seed["gns"][r["seed"]],
            "lead_wdrift":      lead_per_seed["wdrift"][r["seed"]],
        }
        per_seed_summary.append(ps)

    # Decide on notes
    if plasticity_loss:
        note_plasticity = (
            f"YES — plasticity loss at 300 tasks: {n_collapsed}/{N_SEEDS} seeds collapsed. "
            f"acc dropped {acc_drop_pp:.1f}pp from first-20 to last-20 tasks."
        )
    else:
        note_plasticity = (
            f"NO plasticity loss even at 300 tasks: {n_collapsed}/{N_SEEDS} seeds collapsed. "
            f"acc_drop={acc_drop_pp:.1f}pp (first20→last20). Network remains plastic. "
            f"Permuted-MNIST with SGD momentum is a robust benchmark where plasticity is NOT lost."
        )

    # Precedence note
    if precedence_order:
        note_precedence = (
            f"Precedence order (by mean lead time): {precedence_order}. "
            f"erank_vs_gns Wilcoxon p={wilcoxon_p}."
        )
    else:
        note_precedence = "No precedence ordering (no collapses detected)."

    results = {
        "status":                   "DONE",
        "benchmark":                "online_permuted_mnist",
        "n_tasks":                  N_TASKS,
        "n_seeds":                  N_SEEDS,
        "lr_used":                  LR,
        "momentum":                 MOMENTUM,
        "weight_decay":             WEIGHT_DECAY,
        "hidden_size":              HIDDEN,
        "steps_per_task":           STEPS_PER_TASK,
        # Health
        "dead_at_init":             mean_dead_init,
        "dead_after_task1":         mean_dead_t1,
        "task1_acc":                mean_task1_acc,
        "healthy":                  healthy_all,
        # Core result
        "acc_first20_mean":         mean_first20,
        "acc_last20_mean":          mean_last20,
        "acc_drop_pp":              acc_drop_pp,
        "plasticity_loss_occurred": plasticity_loss,
        # Collapse details
        "t_collapse_per_seed":      all_t_collapse,
        "n_seeds_collapsed":        n_collapsed,
        # Lead times
        "lead_times":               lead_summary,
        "precedence_order":         precedence_order,
        # Paired Wilcoxon
        "erank_vs_gns_paired": {
            "diffs_per_seed": paired_diffs,
            "median_diff":    median_diff,
            "wilcoxon_stat":  wilcoxon_stat,
            "wilcoxon_p":     wilcoxon_p,
        },
        # Per-seed details
        "per_seed_summary":         per_seed_summary,
        # Comparison with 100-task run
        "comparison_100task_run": {
            "acc_drop_pp_at_100tasks": 0.16,   # from pmnist RESULTS.json
            "plasticity_loss_at_100tasks": False,
        },
        "wall_clock_sec":  round(elapsed, 1),
        "subject_executed": (
            f"Online Permuted-MNIST, 3-layer MLP ({IN_DIM}→{HIDDEN}→{HIDDEN}→{N_CLASSES}, ReLU), "
            f"SGD lr={LR} momentum={MOMENTUM} WD={WEIGHT_DECAY}, no plasticity repair, "
            f"{N_TASKS} tasks × {STEPS_PER_TASK} steps/task × {N_SEEDS} seeds. "
            f"4 observables per task: dead_unit_fraction, effective_rank, "
            f"gradient_noise_scale (McCandlish B_opt), weight_norm_drift."
        ),
        "notes": (
            f"{note_plasticity} "
            f"Network healthy={healthy_all}: dead_at_init={mean_dead_init:.3f}, "
            f"dead_after_task1={mean_dead_t1:.3f} (<0.20), task1_acc={mean_task1_acc:.3f} (>0.90). "
            f"{note_precedence}"
        ),
    }

    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults → {RESULTS_PATH}", flush=True)
    print(json.dumps(results, indent=2), flush=True)

    # Update LOG.md
    with open(log_path, "a") as f:
        f.write(f"\n## Results\n")
        f.write(f"- Wall-clock: {elapsed:.1f}s ({elapsed/60:.1f} min)\n")
        f.write(f"- healthy={healthy_all}, task1_acc={mean_task1_acc:.4f}\n")
        f.write(f"- acc_first20={mean_first20:.4f}, acc_last20={mean_last20:.4f}, "
                f"drop={acc_drop_pp:.2f}pp\n")
        f.write(f"- plasticity_loss_occurred={plasticity_loss} "
                f"({n_collapsed}/{N_SEEDS} seeds collapsed)\n")
        f.write(f"- t_collapse_per_seed={all_t_collapse}\n")
        f.write(f"- precedence_order={precedence_order}\n")
        f.write(f"\n## Conclusion\n")
        f.write(f"{note_plasticity}\n")


if __name__ == "__main__":
    main()
