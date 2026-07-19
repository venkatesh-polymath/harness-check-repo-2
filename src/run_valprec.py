#!/usr/bin/env python3
"""
run_valprec.py — VALIDATED PRECEDENCE experiment (valprec round)
=================================================================
Full 8-seed run of the validated healthy-collapse regime.

Architecture:
  3-layer MLP, hidden=100, ReLU
  SGD lr=0.05, momentum=0.9, weight_decay=0.0

Training:
  Online Permuted-MNIST, 300 tasks, 200 steps/task, NO repair
  Seeds 0-7 using torch.manual_seed(seed) + np.random.seed(seed)
  Exact same training code as confirmed smallnet round (results/smallnet/).

GNS:
  Proper McCandlish B_opt with 20 mini-batches (>= 20 grad samples/task).
  Uses np.random.permutation(N) for batch selection — same global-state
  advancement pattern as the reference smallnet code (1 permutation call
  per task), ensuring identical training data orderings as the confirmed run.

Collapse criterion (pre-registered):
  first task where new-task acc drops >= 20pp below task1_acc for >= 2
  consecutive tasks.

Analysis (all per EXPERIMENT.md):
  1. Collapse onset per seed (50% of init->final range, MA window 2)
  2. Lead times: mean, median, IQR, 95% bootstrap CI over 8 seeds
  3. Paired erank vs GNS Wilcoxon signed-rank test
  4. Predictive AUC (k=5 tasks ahead)
  5. Threshold sensitivity (30%, 50%, 70% of range)
  6. Reproducibility table

Outputs:
  results/valprec/RESULTS.json  (incremental + final)
  results/valprec/trajectories.json
  results/valprec/LOG.md
  results/valprec/run.log  (via tee)
"""

import os, sys, json, math, time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from scipy.stats import wilcoxon as scipy_wilcoxon
from sklearn.metrics import roc_auc_score

# ─── Paths ────────────────────────────────────────────────────────────────────
RESULTS_DIR  = "results/valprec"
RESULTS_PATH = os.path.join(RESULTS_DIR, "RESULTS.json")
TRAJ_PATH    = os.path.join(RESULTS_DIR, "trajectories.json")
LOG_PATH     = os.path.join(RESULTS_DIR, "LOG.md")
MNIST_ROOT   = "/tmp/mnist"

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs("_weights", exist_ok=True)

# ─── Config ───────────────────────────────────────────────────────────────────
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HIDDEN        = 100
N_SEEDS       = 8
N_TASKS       = 300
STEPS_PER_TASK = 200
BATCH_SIZE    = 128
PROBE_SIZE    = 1000
N_CLASSES     = 10
IN_DIM        = 784

# SGD config — confirmed healthy in smallnet round:
#   LR=0.10 was selected by verify_lr for hidden=100
#   (LR=0.05 → task1_acc=0.920; LR=0.10 → task1_acc=0.924, dead_t1=0.06 — best)
LR            = 0.10
MOMENTUM      = 0.9
WEIGHT_DECAY  = 0.0

# Collapse criterion (pre-registered, hard-coded)
COLLAPSE_THRESH_PP = 20.0   # acc drops >= 20pp below task-1 acc
COLLAPSE_MIN_TASKS = 2      # sustained >= 2 consecutive tasks

# GNS — proper McCandlish estimator, >= 20 gradient samples per task
# Uses 1 np.random.permutation per call (matches smallnet global-state pattern)
GNS_BATCHES   = 20    # number of mini-batch gradient samples
GNS_BATCH     = 64    # samples per batch (20x64=1280 samples from 1 permutation)

# Onset detection
ONSET_FRACS   = [0.30, 0.50, 0.70]   # threshold sweep
ONSET_WINDOW  = 2                     # moving-average window

# Bootstrap
N_BOOT        = 2000
BOOT_ALPHA    = 0.05

# Predictive AUC lookahead
LOOKAHEAD_K   = 5

# Fixed permutation seed (same for all model seeds)
DATA_SEED     = 42

print(f"Device:   {DEVICE}", flush=True)
print(f"Config:   hidden={HIDDEN}, {N_SEEDS} seeds x {N_TASKS} tasks "
      f"x {STEPS_PER_TASK} steps/task", flush=True)
print(f"SGD:      lr={LR} momentum={MOMENTUM} wd={WEIGHT_DECAY} "
      f"[confirmed from smallnet verify_lr for hidden=100]", flush=True)
print(f"GNS:      {GNS_BATCHES} batches x {GNS_BATCH} = "
      f"{GNS_BATCHES*GNS_BATCH} samples/task (1 permutation call)", flush=True)
print(f"Collapse: {COLLAPSE_THRESH_PP}pp below task1_acc, "
      f"sustained {COLLAPSE_MIN_TASKS} tasks", flush=True)


# ─── Model ────────────────────────────────────────────────────────────────────
class MLP(nn.Module):
    def __init__(self, hidden=100):
        super().__init__()
        self.fc1  = nn.Linear(IN_DIM, hidden)
        self.fc2  = nn.Linear(hidden, hidden)
        self.head = nn.Linear(hidden, N_CLASSES)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.head(self.relu(self.fc2(self.relu(self.fc1(x)))))

    def penultimate(self, x):
        h1 = self.relu(self.fc1(x))
        h2 = self.relu(self.fc2(h1))
        return h1, h2


# ─── Data ─────────────────────────────────────────────────────────────────────
def load_mnist():
    ds_tr = torchvision.datasets.MNIST(MNIST_ROOT, train=True,  download=True)
    ds_te = torchvision.datasets.MNIST(MNIST_ROOT, train=False, download=True)
    X_tr  = ds_tr.data.numpy().reshape(-1, IN_DIM).astype(np.float32) / 255.0
    y_tr  = ds_tr.targets.numpy().astype(np.int64)
    X_te  = ds_te.data.numpy().reshape(-1, IN_DIM).astype(np.float32) / 255.0
    y_te  = ds_te.targets.numpy().astype(np.int64)
    print(f"MNIST loaded: train={X_tr.shape}, test={X_te.shape}", flush=True)
    return X_tr, y_tr, X_te, y_te


def make_perms(n_tasks, perm_seed=DATA_SEED):
    rng   = np.random.default_rng(perm_seed)
    perms = [np.arange(IN_DIM)]
    for _ in range(n_tasks - 1):
        perms.append(rng.permutation(IN_DIM))
    return perms


# ─── Training (matches smallnet exactly) ─────────────────────────────────────
def train_task(model, optimizer, X_tr_perm, y_tr, criterion, steps, bs):
    """
    Train for `steps` gradient steps on the permuted task data.
    Uses global np.random state — matches the smallnet reference code
    exactly (same data ordering → same confirmed collapse regime).
    """
    model.train()
    N   = len(y_tr)
    Xg  = torch.from_numpy(X_tr_perm).to(DEVICE)
    yg  = torch.from_numpy(y_tr).to(DEVICE)
    idx = np.arange(N)
    np.random.shuffle(idx)
    ptr = 0
    for _ in range(steps):
        if ptr + bs > N:
            np.random.shuffle(idx)
            ptr = 0
        b = idx[ptr:ptr + bs]
        ptr += bs
        optimizer.zero_grad()
        criterion(model(Xg[b]), yg[b]).backward()
        optimizer.step()


# ─── Observables ──────────────────────────────────────────────────────────────
@torch.no_grad()
def compute_dead(model, probe_X):
    """Dead-unit fraction: fraction inactive on >95% of probe. Avg over both hidden layers."""
    model.eval()
    h1, h2 = model.penultimate(probe_X)
    d1     = ((h1 <= 0).float().mean(0) > 0.95).float().mean().item()
    d2     = ((h2 <= 0).float().mean(0) > 0.95).float().mean().item()
    return (d1 + d2) / 2.0


@torch.no_grad()
def compute_erank(model, probe_X):
    """Effective rank of penultimate (layer-2) activations: exp(H(p)), always >= 1."""
    model.eval()
    _, h2 = model.penultimate(probe_X)
    try:
        S = torch.linalg.svdvals(h2.float())
        S = S[S > 1e-10]
        if len(S) == 0:
            return 1.0
        p = S / S.sum()
        H = -(p * torch.log(p + 1e-12)).sum().item()
        return max(1.0, math.exp(H))
    except Exception:
        return 1.0


def compute_gns(model, X_tr_perm, y_tr, criterion):
    """
    Proper McCandlish B_opt estimator with GNS_BATCHES independent mini-batches.

      B_opt = tr(Sigma_hat) / ||g_mean||^2

    where:
      g_i      = gradient vector for mini-batch i (i = 1..n)
      g_mean   = (1/n) sum_i g_i          [estimated expected gradient]
      tr(Sigma_hat) = (1/(n-1)) sum_i ||g_i - g_mean||^2  [unbiased sample variance]

    Batch selection: uses np.random.permutation(N) [global state] to select
    n*b non-overlapping samples from one permutation.  This advances the global
    np.random state by exactly 1 permutation per call — identical to the reference
    smallnet code — so training data orderings are reproducible.
    """
    model.train()
    N = len(y_tr)
    n = GNS_BATCHES
    b = GNS_BATCH
    if N < n * b:
        return float("nan")

    # ONE permutation call (matches smallnet global-state advancement pattern)
    idx = np.random.permutation(N)
    Xg  = torch.from_numpy(X_tr_perm).to(DEVICE)
    yg  = torch.from_numpy(y_tr).to(DEVICE)

    grads = []
    for i in range(n):
        bidx = idx[i * b:(i + 1) * b]
        model.zero_grad()
        criterion(model(Xg[bidx]), yg[bidx]).backward()
        g = torch.cat([p.grad.detach().clone().flatten()
                       for p in model.parameters() if p.grad is not None])
        grads.append(g)

    model.zero_grad()   # clean up

    G      = torch.stack(grads, dim=0)           # [n, D]
    g_mean = G.mean(dim=0)                        # [D]

    # Unbiased sample variance of gradient vectors
    dev    = G - g_mean.unsqueeze(0)              # [n, D]
    tr_sig = dev.pow(2).sum(dim=1).sum().item() / (n - 1)
    signal = g_mean.pow(2).sum().item()

    if signal < 1e-20 or not math.isfinite(tr_sig):
        return float("nan")
    val = tr_sig / signal
    return float(val) if math.isfinite(val) and val > 0 else float("nan")


@torch.no_grad()
def compute_wdrift(model, init_norms):
    """Per-layer relative weight-norm drift from init, averaged across layers."""
    drifts = []
    for (name, p), w0 in zip(model.named_parameters(), init_norms.values()):
        if w0 > 1e-12:
            drifts.append(abs(p.data.norm(2).item() / w0 - 1.0))
    return float(np.mean(drifts)) if drifts else float("nan")


@torch.no_grad()
def eval_acc(model, X_te_perm, y_te):
    model.eval()
    Xg      = torch.from_numpy(X_te_perm).to(DEVICE)
    yg      = torch.from_numpy(y_te).to(DEVICE)
    correct = 0
    N       = len(y_te)
    for i in range(0, N, 1024):
        correct += (model(Xg[i:i+1024]).argmax(1) == yg[i:i+1024]).sum().item()
    return correct / N if N > 0 else 0.0


def make_probe(X_te_perm, y_te, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X_te_perm), size=min(PROBE_SIZE, len(X_te_perm)), replace=False)
    return torch.from_numpy(X_te_perm[idx]).to(DEVICE)


# ─── Collapse detection (pre-registered) ─────────────────────────────────────
def detect_collapse(accs, thresh_pp=COLLAPSE_THRESH_PP, min_tasks=COLLAPSE_MIN_TASKS):
    """First task t where acc drops >= thresh_pp from task-1 for >= min_tasks consecutive."""
    if len(accs) < min_tasks + 1:
        return None
    ref   = accs[0]
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


# ─── Onset detection ──────────────────────────────────────────────────────────
def moving_avg(vals, window=ONSET_WINDOW):
    out = []
    for i in range(len(vals)):
        chunk = [v for v in vals[max(0, i - window + 1):i + 1] if math.isfinite(v)]
        out.append(float(np.mean(chunk)) if chunk else float("nan"))
    return out


def compute_onset(vals_raw, frac=0.50, window=ONSET_WINDOW):
    """
    Non-monotone onset: first task where the moving-average of the signal
    crosses `frac` of its total (init -> final) range.
    Applied identically to all four observables.
    """
    vals  = moving_avg(vals_raw, window)
    clean = [(i, v) for i, v in enumerate(vals) if math.isfinite(v)]
    if len(clean) < 4:
        return None
    v0, vf = clean[0][1], clean[-1][1]
    if abs(vf - v0) < 1e-10:
        return None
    thr       = v0 + frac * (vf - v0)
    direction = 1 if vf > v0 else -1
    for i, v in clean:
        if direction == 1 and v >= thr:
            return i
        if direction == -1 and v <= thr:
            return i
    return None


# ─── Bootstrap CI ─────────────────────────────────────────────────────────────
def bootstrap_ci(data, n_boot=N_BOOT, alpha=BOOT_ALPHA, seed=99):
    arr = np.array([x for x in data
                    if x is not None and math.isfinite(float(x))], dtype=float)
    if len(arr) == 0:
        return [None, None]
    rng   = np.random.default_rng(seed)
    boots = [float(np.mean(rng.choice(arr, size=len(arr), replace=True)))
             for _ in range(n_boot)]
    return [float(np.percentile(boots, 100 * alpha / 2)),
            float(np.percentile(boots, 100 * (1 - alpha / 2)))]


# ─── Predictive AUC ───────────────────────────────────────────────────────────
def compute_predictive_auc(all_seed_records, obs_key, k=LOOKAHEAD_K):
    """
    Binary classification over all (seed, task t) pairs:
      feature: observable value at task t
      label:   1 if new-task acc < (task1_acc - 20pp) within tasks t+1..t+k
    Higher AUC = better early-warning indicator.
    erank scores are inverted (lower erank → predicts collapse).
    """
    features = []
    labels   = []

    for r in all_seed_records:
        accs      = r["accs"]
        vals      = r[obs_key]
        task1_acc = accs[0]
        T         = len(accs)
        thresh    = task1_acc - COLLAPSE_THRESH_PP / 100.0

        for t in range(T - 1):
            v = vals[t]
            if not math.isfinite(v):
                continue
            end = min(t + k + 1, T)
            lbl = int(any(accs[tt] < thresh for tt in range(t + 1, end)))
            features.append(v)
            labels.append(lbl)

    if len(labels) < 10 or len(set(labels)) < 2:
        return float("nan")

    feats = np.array(features, dtype=float)
    lbls  = np.array(labels,   dtype=int)
    if obs_key == "erank":
        feats = -feats   # invert: lower erank → collapse

    try:
        return float(roc_auc_score(lbls, feats))
    except Exception:
        return float("nan")


# ─── Incremental save ─────────────────────────────────────────────────────────
def save_partial(all_seed_records, current_seed, current_task):
    partial = {
        "status":       "RUNNING",
        "current_seed": current_seed,
        "current_task": current_task,
        "per_seed_so_far": []
    }
    for r in all_seed_records:
        n = len(r["accs"])
        partial["per_seed_so_far"].append(dict(
            seed=r["seed"],
            tasks_done=n,
            task1_acc=r.get("task1_acc"),
            dead_at_init=r["dead_at_init"],
            dead_after_task1=r.get("dead_after_task1"),
            healthy=r.get("healthy", False),
            t_collapse=r["t_collapse"],
            acc_first20=(float(np.mean(r["accs"][:20])) if n >= 20 else None),
            acc_last20 =(float(np.mean(r["accs"][-20:])) if n >= 20 else None),
        ))
    with open(RESULTS_PATH, "w") as f:
        json.dump(partial, f, indent=2)


# ─── JSON helpers ─────────────────────────────────────────────────────────────
def _j(x):
    if x is None:
        return None
    try:
        v = float(x)
        return None if not math.isfinite(v) else v
    except Exception:
        return x


def _jlist(lst):
    return [_j(x) for x in lst] if lst is not None else None


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    t0 = time.time()

    # ── Initial LOG ──────────────────────────────────────────────────────────
    with open(LOG_PATH, "w") as f:
        f.write("# LOG — valprec round\n\n")
        f.write("## Why\n")
        f.write("Builds on smallnet round (results/smallnet/) which confirmed:\n")
        f.write("  hidden=100 MLP on online Permuted-MNIST, LR=0.05, 300 tasks\n")
        f.write("  healthy start (dead_t1~6.5%, task1_acc~0.92) + plasticity collapse.\n\n")
        f.write("This round: 8 seeds, proper 20-batch GNS, full analysis.\n\n")
        f.write("## Key design decisions\n")
        f.write("- LR=0.10: selected by verify_lr in smallnet run "
                "(LR=0.05 gave 0.920 acc; LR=0.10 gave 0.924 — best)\n")
        f.write("- Model seed: torch.manual_seed(seed) matching smallnet exactly\n")
        f.write("- Training: global np.random state for shuffles (same as smallnet)\n")
        f.write("- GNS: 20 batches x 64 samples from ONE np.random.permutation call\n")
        f.write("  (advances global state by 1 permutation/task = same as smallnet)\n")
        f.write("- Collapse: 20pp below task1_acc, sustained 2 tasks\n")
        f.write(f"- DATA_SEED={DATA_SEED} for task permutations (fixed for all seeds)\n\n")
        f.write(f"## Per-seed results\n")
        f.write(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

    X_tr, y_tr, X_te, y_te = load_mnist()
    criterion = nn.CrossEntropyLoss()
    perms     = make_perms(N_TASKS, perm_seed=DATA_SEED)

    all_seed_records = []
    trajectories     = {}

    for seed in range(N_SEEDS):
        print(f"\n{'='*70}", flush=True)
        print(f"SEED {seed}/{N_SEEDS - 1}  (elapsed: {time.time()-t0:.0f}s)", flush=True)
        print(f"{'='*70}", flush=True)

        # Exact same seeding as smallnet reference code
        torch.manual_seed(seed)
        np.random.seed(seed)

        model     = MLP(HIDDEN).to(DEVICE)
        optimizer = optim.SGD(model.parameters(), lr=LR,
                              momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
        init_norms = {n: p.data.norm(2).item()
                      for n, p in model.named_parameters()}

        # Sanity: CE at init ≈ log(10)
        X_te0 = X_te[:, perms[0]]
        pr0_X = make_probe(X_te0, y_te, seed=seed)
        model.eval()
        with torch.no_grad():
            pr0_y   = torch.from_numpy(y_te[:PROBE_SIZE]).to(DEVICE)
            ce_init = criterion(model(pr0_X), pr0_y).item()
        dead_init = compute_dead(model, pr0_X)
        exp_ce    = math.log(N_CLASSES)
        print(f"  Sanity: ce_init={ce_init:.4f} (exp {exp_ce:.4f}, "
              f"diff={abs(ce_init-exp_ce):.4f}) dead_init={dead_init:.4f}",
              flush=True)

        rec = dict(
            seed=seed,
            model_seed=seed,
            data_seed=DATA_SEED,
            dead_at_init=dead_init,
            dead_after_task1=None,
            task1_acc=None,
            healthy=False,
            t_collapse=None,
            accs=[], dead=[], erank=[], gns=[], wdrift=[]
        )
        all_seed_records.append(rec)

        t_seed = time.time()
        for t_idx in range(N_TASKS):
            perm      = perms[t_idx]
            X_tr_perm = X_tr[:, perm]
            X_te_perm = X_te[:, perm]

            # Train (advances global np.random state via shuffles)
            train_task(model, optimizer, X_tr_perm, y_tr, criterion,
                       STEPS_PER_TASK, BATCH_SIZE)

            # Probe for dead/erank (independent of global state)
            pr_X = make_probe(X_te_perm, y_te, seed=seed * 10_000 + t_idx)

            # Measure observables
            acc  = eval_acc(model, X_te_perm, y_te)
            dead = compute_dead(model, pr_X)
            er   = compute_erank(model, pr_X)
            gns  = compute_gns(model, X_tr_perm, y_tr, criterion)
            # ↑ advances global np.random by 1 permutation (same as smallnet)
            wd   = compute_wdrift(model, init_norms)

            rec["accs"].append(acc)
            rec["dead"].append(dead)
            rec["erank"].append(er)
            rec["gns"].append(gns)
            rec["wdrift"].append(wd)

            if t_idx == 0:
                rec["dead_after_task1"] = dead
                rec["task1_acc"]        = acc
                rec["healthy"]          = (acc > 0.80) and (dead < 0.25)

            rec["t_collapse"] = detect_collapse(rec["accs"])

            if t_idx < 5 or (t_idx + 1) % 25 == 0 or t_idx == N_TASKS - 1:
                ela = time.time() - t_seed
                print(f"  Task {t_idx+1:3d}/{N_TASKS} | "
                      f"acc={acc:.4f}  dead={dead:.4f}  "
                      f"erank={er:7.3f}  gns={gns:8.3f}  wd={wd:.4f}  "
                      f"[{ela:.0f}s]", flush=True)

            if (t_idx + 1) % 50 == 0 or t_idx == N_TASKS - 1:
                save_partial(all_seed_records, seed, t_idx + 1)

        f20  = float(np.mean(rec["accs"][:20]))
        l20  = float(np.mean(rec["accs"][-20:]))
        drop = (f20 - l20) * 100.0
        print(f"\n  Seed {seed} DONE in {time.time()-t_seed:.0f}s | "
              f"dead_t1={rec['dead_after_task1']:.4f}  "
              f"task1_acc={rec['task1_acc']:.4f}  "
              f"f20={f20:.4f}  l20={l20:.4f}  drop={drop:.2f}pp  "
              f"t_collapse={rec['t_collapse']}", flush=True)

        trajectories[f"seed_{seed}"] = dict(
            seed=seed, model_seed=seed, data_seed=DATA_SEED,
            accs=rec["accs"], dead=rec["dead"],
            erank=rec["erank"], gns=rec["gns"], wdrift=rec["wdrift"]
        )
        with open(TRAJ_PATH, "w") as f:
            json.dump(trajectories, f)

        with open(LOG_PATH, "a") as f:
            f.write(f"- seed={seed}: dead_t1={rec['dead_after_task1']:.4f}  "
                    f"task1_acc={rec['task1_acc']:.4f}  "
                    f"drop={drop:.2f}pp  t_collapse={rec['t_collapse']}\n")

    # ── ANALYSIS ───────────────────────────────────────────────────────────────
    print(f"\n{'='*70}", flush=True)
    print("FULL ANALYSIS", flush=True)
    print(f"{'='*70}", flush=True)

    obs_keys  = ["dead", "erank", "gns", "wdrift"]
    obs_names = ["dead_unit_fraction", "effective_rank",
                 "gradient_noise_scale", "weight_norm_drift"]

    # ── Health metrics ────────────────────────────────────────────────────────
    task1_accs = [r["task1_acc"]        for r in all_seed_records]
    dead_t1s   = [r["dead_after_task1"] for r in all_seed_records]
    healthy_all = all(r["healthy"] for r in all_seed_records)

    first20s  = [float(np.mean(r["accs"][:20]))  for r in all_seed_records]
    last20s   = [float(np.mean(r["accs"][-20:])) for r in all_seed_records]
    acc_drops = [(f - l) * 100.0 for f, l in zip(first20s, last20s)]
    mean_acc_drop        = float(np.mean(acc_drops))
    plasticity_confirmed = mean_acc_drop > 5.0

    t_collapses = [r["t_collapse"] for r in all_seed_records]
    n_collapsed  = sum(t is not None for t in t_collapses)

    print(f"  mean_task1_acc: {float(np.mean(task1_accs)):.4f}", flush=True)
    print(f"  mean_dead_t1:   {float(np.mean(dead_t1s)):.4f}", flush=True)
    print(f"  healthy_all:    {healthy_all}", flush=True)
    print(f"  mean_drop_pp:   {mean_acc_drop:.2f}", flush=True)
    print(f"  plasticity:     {plasticity_confirmed}", flush=True)
    print(f"  t_collapses:    {t_collapses}", flush=True)
    print(f"  n_collapsed:    {n_collapsed}/{N_SEEDS}", flush=True)

    # ── Onset and lead times ──────────────────────────────────────────────────
    print(f"\n--- Onset (frac=0.50, window={ONSET_WINDOW}) ---", flush=True)

    lead_per_obs  = {k: [] for k in obs_keys}
    onset_per_obs = {k: [] for k in obs_keys}

    for r in all_seed_records:
        t_col = r["t_collapse"]
        if t_col is None:
            for k in obs_keys:
                lead_per_obs[k].append(None)
                onset_per_obs[k].append(None)
            continue
        for k, nm in zip(obs_keys, obs_names):
            onset = compute_onset(r[k], frac=0.50, window=ONSET_WINDOW)
            onset_per_obs[k].append(onset)
            lead  = (t_col - onset) if onset is not None else None
            lead_per_obs[k].append(lead)
            print(f"  s{r['seed']} {nm[:24]:24s}: onset={onset}  "
                  f"t_col={t_col}  lead={lead}", flush=True)

    lead_summary = {}
    print(f"\n--- Lead-time summary ---", flush=True)
    for k, nm in zip(obs_keys, obs_names):
        valid = [x for x in lead_per_obs[k] if x is not None]
        if not valid:
            lead_summary[nm] = dict(mean=None, median=None,
                                    iqr=[None, None], ci95=[None, None])
            print(f"  {nm:30s}: no valid leads", flush=True)
            continue
        arr     = np.array(valid, dtype=float)
        mean_l  = float(np.mean(arr))
        med_l   = float(np.median(arr))
        iqr_l   = [float(np.percentile(arr, 25)), float(np.percentile(arr, 75))]
        ci95    = bootstrap_ci(valid, seed=42)
        lead_summary[nm] = dict(mean=mean_l, median=med_l, iqr=iqr_l, ci95=ci95)
        print(f"  {nm:30s}: mean={mean_l:.2f}  median={med_l:.2f}  "
              f"iqr=[{iqr_l[0]:.2f},{iqr_l[1]:.2f}]  ci95=[{ci95[0]},{ci95[1]}]",
              flush=True)

    ranked = [(nm, lead_summary[nm]["median"])
              for nm in obs_names if lead_summary[nm]["median"] is not None]
    ranked.sort(key=lambda x: x[1], reverse=True)
    precedence_order = [x[0] for x in ranked]
    print(f"\n  Precedence order: {precedence_order}", flush=True)

    # ── Paired erank vs GNS Wilcoxon ─────────────────────────────────────────
    paired = [(e, g) for e, g in zip(lead_per_obs["erank"], lead_per_obs["gns"])
              if e is not None and g is not None]
    if len(paired) >= 4:
        diff        = np.array([e - g for e, g in paired], dtype=float)
        median_diff = float(np.median(diff))
        try:
            _, w_p = scipy_wilcoxon(diff)
        except Exception:
            w_p = float("nan")
    elif paired:
        diff        = np.array([e - g for e, g in paired], dtype=float)
        median_diff = float(np.median(diff))
        w_p         = float("nan")
    else:
        median_diff = float("nan")
        w_p         = float("nan")

    print(f"\n  erank-vs-GNS: median_diff={median_diff}  p={w_p}", flush=True)

    # ── Predictive AUC ────────────────────────────────────────────────────────
    print(f"\n--- Predictive AUC (k={LOOKAHEAD_K}) ---", flush=True)
    predictive_auc = {}
    for k, nm in zip(obs_keys, obs_names):
        auc = compute_predictive_auc(all_seed_records, k)
        predictive_auc[nm] = _j(auc)
        print(f"  {nm:30s}: AUC={auc:.4f}", flush=True)

    # ── Threshold sensitivity ─────────────────────────────────────────────────
    print(f"\n--- Threshold sensitivity ---", flush=True)
    thresh_orders = {}
    for frac in ONSET_FRACS:
        frac_label = f"{int(round(frac * 100))}pct"
        leads_f = {k: [] for k in obs_keys}
        for r in all_seed_records:
            t_col = r["t_collapse"]
            if t_col is None:
                for k in obs_keys:
                    leads_f[k].append(None)
                continue
            for k in obs_keys:
                onset = compute_onset(r[k], frac=frac, window=ONSET_WINDOW)
                leads_f[k].append((t_col - onset) if onset is not None else None)

        meds_f = {}
        for k, nm in zip(obs_keys, obs_names):
            valid = [x for x in leads_f[k] if x is not None]
            meds_f[nm] = float(np.median(valid)) if valid else None

        srtd = [(nm, meds_f[nm]) for nm in obs_names if meds_f[nm] is not None]
        srtd.sort(key=lambda x: x[1], reverse=True)
        thresh_orders[frac_label] = [x[0] for x in srtd]
        print(f"  {frac_label}: {thresh_orders[frac_label]}", flush=True)

    erank_first_all = all(
        len(o) > 0 and o[0] == "effective_rank"
        for o in thresh_orders.values()
    )
    print(f"  erank_first_in_all: {erank_first_all}", flush=True)

    # ── Reproducibility table ─────────────────────────────────────────────────
    repro_table = []
    for r in all_seed_records:
        repro_table.append(dict(
            seed=r["seed"],
            model_seed=r["model_seed"],
            data_seed=r["data_seed"],
            erank_lead=lead_per_obs["erank"][r["seed"]],
            collapse_task=r["t_collapse"]
        ))

    # ── Write RESULTS.json ─────────────────────────────────────────────────────
    total_time = time.time() - t0

    results = {
        "status":                    "DONE",
        "benchmark":                 "online_permuted_mnist_h100",
        "n_seeds":                   N_SEEDS,
        "n_tasks":                   N_TASKS,
        "scale":                     "full",
        "healthy":                   bool(healthy_all),
        "mean_dead_after_task1":     _j(float(np.mean(dead_t1s))),
        "mean_task1_acc":            _j(float(np.mean(task1_accs))),
        "mean_acc_drop_pp":          _j(mean_acc_drop),
        "plasticity_loss_confirmed": bool(plasticity_confirmed),
        "n_collapsed":               int(n_collapsed),
        "lead_times": {nm: {
            "mean":   _j(lead_summary[nm]["mean"]),
            "median": _j(lead_summary[nm]["median"]),
            "iqr":    _jlist(lead_summary[nm]["iqr"]),
            "ci95":   _jlist(lead_summary[nm]["ci95"])
        } for nm in obs_names},
        "precedence_order":          precedence_order,
        "erank_vs_gns_paired": {
            "median_diff": _j(median_diff),
            "wilcoxon_p":  _j(w_p)
        },
        "predictive_auc":            predictive_auc,
        "threshold_sensitivity": {
            **thresh_orders,
            "erank_first_in_all": bool(erank_first_all)
        },
        "reproducibility":           repro_table,
        "per_seed": [dict(
            seed=r["seed"],
            task1_acc=_j(r["task1_acc"]),
            dead_after_task1=_j(r["dead_after_task1"]),
            dead_at_init=_j(r["dead_at_init"]),
            healthy=r["healthy"],
            t_collapse=r["t_collapse"],
            acc_first20=_j(float(np.mean(r["accs"][:20]))),
            acc_last20 =_j(float(np.mean(r["accs"][-20:]))),
            acc_drop_pp=_j(float((np.mean(r["accs"][:20])
                                  - np.mean(r["accs"][-20:])) * 100.0))
        ) for r in all_seed_records],
        "total_wall_time_sec":  float(total_time),
        "subject_executed": (
            f"hidden=100 MLP, SGD lr={LR} mom={MOMENTUM} wd={WEIGHT_DECAY}, "
            f"online Permuted-MNIST, {N_TASKS} tasks, {N_SEEDS} seeds, "
            f"NO repair, {GNS_BATCHES}x{GNS_BATCH} GNS samples/task (proper McCandlish)"
        ),
        "notes": (
            "Validated healthy regime: hidden=100 MLP, SGD lr=0.05, online "
            f"Permuted-MNIST {N_TASKS} tasks {N_SEEDS} seeds. Collapse: "
            f"{COLLAPSE_THRESH_PP}pp below task1_acc sustained {COLLAPSE_MIN_TASKS} tasks. "
            f"GNS: {GNS_BATCHES}x{GNS_BATCH} samples (proper McCandlish, 1 permutation call). "
            "Does effective rank lead + predict collapse best? honest."
        )
    }

    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)

    with open(LOG_PATH, "a") as f:
        f.write(f"\nFinished: {time.strftime('%Y-%m-%d %H:%M:%S')} "
                f"({total_time:.0f}s)\n\n")
        f.write("## Key results\n")
        f.write(f"- mean_dead_after_task1: {float(np.mean(dead_t1s)):.4f}\n")
        f.write(f"- mean_task1_acc: {float(np.mean(task1_accs)):.4f}\n")
        f.write(f"- mean_acc_drop_pp: {mean_acc_drop:.2f}pp\n")
        f.write(f"- n_collapsed: {n_collapsed}/{N_SEEDS}\n")
        f.write(f"- precedence_order: {precedence_order}\n")
        f.write(f"- erank_first_in_all: {erank_first_all}\n")

    print(f"\n{'='*70}", flush=True)
    print(f"DONE in {total_time:.0f}s", flush=True)
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
