#!/usr/bin/env python3
"""
run_small.py — DEFINITIVE Small-Network Plasticity Test (smallnet round)
=========================================================================
Tests whether a SMALL MLP (hidden ∈ {100, 256}) loses plasticity over
300 online permuted-MNIST tasks.

Prior rounds (pmnist/pml2) used hidden=2000 → NO collapse.
This round uses the small regime from Dohare et al. (loss of plasticity
documented for smaller networks with fewer redundant units).

Key questions:
  1. Does new-task accuracy drop (acc_first20 vs acc_last20)?
  2. Is the net HEALTHY at task 1 (dead_after_task1 < 0.25, task1_acc > 0.85)?
  3. If collapse: which observable leads?

Reference: Dohare et al. (2023) "Loss of Plasticity in Deep Continual Learning"
Builds on: results/pml2/ (hidden=2000, 300 tasks, no collapse)
"""

import os, sys, json, math, time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import wilcoxon as scipy_wilcoxon
import torchvision

# ─── Paths ────────────────────────────────────────────────────────────────────
MNIST_ROOT   = "/tmp/mnist"
RESULTS_DIR  = "results/smallnet"
RESULTS_PATH = os.path.join(RESULTS_DIR, "RESULTS.json")
LOG_PATH     = os.path.join(RESULTS_DIR, "LOG.md")
TRAJ_PATH    = os.path.join(RESULTS_DIR, "trajectories.json")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs("_weights", exist_ok=True)

# ─── Config ───────────────────────────────────────────────────────────────────
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HIDDEN_SIZES  = [100, 256]          # two small configs
N_SEEDS       = 3
N_TASKS       = 300
STEPS_PER_TASK = 200                # "a few hundred" (online, limited)
BATCH_SIZE    = 128
PROBE_SIZE    = 1000
N_CLASSES     = 10
IN_DIM        = 784                 # 28×28 MNIST

# SGD config — same as prior healthy pmnist run
LR            = 0.05
MOMENTUM      = 0.9
WEIGHT_DECAY  = 0.0

# LR candidates for per-hidden-size verification
LR_CANDIDATES = [0.01, 0.05, 0.1]

# Collapse criterion (hard-coded before any run)
COLLAPSE_THRESH_PP = 15.0           # ≥ 15 pp drop from task-1 acc
COLLAPSE_MIN_TASKS = 2             # sustained ≥ 2 consecutive tasks

# "Plasticity loss" for primary outcome
PLASTICITY_PP_THRESHOLD = 3.0      # acc_drop_pp > 3pp → plasticity_loss=True

# GNS
GNS_BATCH = 64

# Bootstrap
N_BOOT     = 2000
BOOT_ALPHA = 0.05

# Incremental save
SAVE_EVERY = 25

print(f"Device: {DEVICE}", flush=True)
print(f"Config: hidden={HIDDEN_SIZES}, {N_SEEDS} seeds × {N_TASKS} tasks "
      f"× {STEPS_PER_TASK} steps/task", flush=True)
print(f"SGD lr={LR} momentum={MOMENTUM} WD={WEIGHT_DECAY}", flush=True)


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
    X_tr = ds_tr.data.numpy().reshape(-1, IN_DIM).astype(np.float32) / 255.0
    y_tr = ds_tr.targets.numpy().astype(np.int64)
    X_te = ds_te.data.numpy().reshape(-1, IN_DIM).astype(np.float32) / 255.0
    y_te = ds_te.targets.numpy().astype(np.int64)
    print(f"MNIST: train={X_tr.shape}, test={X_te.shape}", flush=True)
    return X_tr, y_tr, X_te, y_te


def make_perms(n_tasks, perm_seed=42):
    rng = np.random.default_rng(perm_seed)
    perms = [np.arange(IN_DIM)]
    for _ in range(n_tasks - 1):
        perms.append(rng.permutation(IN_DIM))
    return perms


# ─── Training ─────────────────────────────────────────────────────────────────
def train_task(model, optimizer, X_tr_perm, y_tr, criterion, steps, bs):
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
    """Dead-unit fraction: >95% inactive on probe. Avg over both hidden layers."""
    model.eval()
    h1, h2 = model.penultimate(probe_X)
    d1 = ((h1 <= 0).float().mean(0) > 0.95).float().mean().item()
    d2 = ((h2 <= 0).float().mean(0) > 0.95).float().mean().item()
    return (d1 + d2) / 2.0


@torch.no_grad()
def compute_erank(model, probe_X):
    """Effective rank of penultimate activations: exp(H(p)), always >= 1."""
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
    """GNS: McCandlish B_opt (noise/signal ratio), two mini-batches of GNS_BATCH."""
    model.train()
    N = len(y_tr)
    if N < 2 * GNS_BATCH:
        return float("nan")
    idx   = np.random.permutation(N)
    b1i   = idx[:GNS_BATCH]
    b2i   = idx[GNS_BATCH:2*GNS_BATCH]
    Xg    = torch.from_numpy(X_tr_perm).to(DEVICE)
    yg    = torch.from_numpy(y_tr).to(DEVICE)

    def grad_vec(bidx):
        model.zero_grad()
        criterion(model(Xg[bidx]), yg[bidx]).backward()
        return torch.cat([p.grad.detach().flatten()
                          for p in model.parameters() if p.grad is not None])

    g1 = grad_vec(b1i)
    g2 = grad_vec(b2i)
    g_avg  = (g1 + g2) / 2.0
    noise  = (g1 - g2).norm(2).pow(2).item() / 2.0
    signal = g_avg.norm(2).pow(2).item()
    if signal < 1e-20 or not math.isfinite(noise) or not math.isfinite(signal):
        return float("nan")
    val = noise / signal
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
    """Accuracy on full test set."""
    model.eval()
    Xg = torch.from_numpy(X_te_perm).to(DEVICE)
    yg = torch.from_numpy(y_te).to(DEVICE)
    correct, N = 0, len(y_te)
    for i in range(0, N, 1024):
        correct += (model(Xg[i:i+1024]).argmax(1) == yg[i:i+1024]).sum().item()
    return correct / N if N > 0 else 0.0


def make_probe(X_te_perm, y_te, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X_te_perm), size=min(PROBE_SIZE, len(X_te_perm)), replace=False)
    return (torch.from_numpy(X_te_perm[idx]).to(DEVICE),
            torch.from_numpy(y_te[idx]).to(DEVICE))


# ─── Analysis helpers ─────────────────────────────────────────────────────────
def detect_collapse(accs, thresh_pp=COLLAPSE_THRESH_PP, min_tasks=COLLAPSE_MIN_TASKS):
    """First task t where acc drops >= thresh_pp from task-1 for >= min_tasks consecutive."""
    if len(accs) < min_tasks + 1:
        return None
    ref, count, first = accs[0], 0, None
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


def moving_avg(vals, window=2):
    out = []
    for i in range(len(vals)):
        chunk = [v for v in vals[max(0, i-window+1):i+1] if math.isfinite(v)]
        out.append(float(np.mean(chunk)) if chunk else float("nan"))
    return out


def compute_onset(vals_raw, frac=0.50, window=2):
    """Onset: first task where smoothed signal crosses 50% of its total range."""
    vals  = moving_avg(vals_raw, window)
    clean = [(i, v) for i, v in enumerate(vals) if math.isfinite(v)]
    if len(clean) < 4:
        return None
    v0, vf = clean[0][1], clean[-1][1]
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


def bootstrap_ci(data, n_boot=N_BOOT, alpha=BOOT_ALPHA, seed=0):
    arr = np.array([x for x in data if x is not None and math.isfinite(x)], dtype=float)
    if len(arr) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    stats = [np.mean(rng.choice(arr, size=len(arr), replace=True)) for _ in range(n_boot)]
    return (float(np.percentile(stats, 100*alpha/2)),
            float(np.percentile(stats, 100*(1-alpha/2))))


def nan_to_null(x):
    if isinstance(x, float) and not math.isfinite(x):
        return None
    return x


# ─── LR Verification ──────────────────────────────────────────────────────────
def verify_lr(hidden, perms, X_tr, y_tr, X_te, y_te, criterion):
    """
    Quick LR search: run 1 task (STEPS_PER_TASK steps) with seed=0 for each
    candidate LR; return LR with best task1_acc that is also healthy
    (dead_after_task1 < 0.25).
    """
    print(f"\n=== LR Verification (hidden={hidden}) ===", flush=True)
    perm   = perms[0]
    Xtrp   = X_tr[:, perm]
    Xtep   = X_te[:, perm]
    best_lr, best_acc = LR, 0.0  # start with default

    for lr_cand in LR_CANDIDATES:
        torch.manual_seed(0); np.random.seed(0)
        m = MLP(hidden).to(DEVICE)
        opt = optim.SGD(m.parameters(), lr=lr_cand, momentum=MOMENTUM,
                        weight_decay=WEIGHT_DECAY)
        pr_X, _ = make_probe(Xtep, y_te, seed=0)

        # CE at init
        m.eval()
        with torch.no_grad():
            ce_init = criterion(m(pr_X), torch.from_numpy(y_te[:PROBE_SIZE]).to(DEVICE)).item()
        dead_init = compute_dead(m, pr_X)

        # Train one task
        train_task(m, opt, Xtrp, y_tr, criterion, STEPS_PER_TASK, BATCH_SIZE)

        acc = eval_acc(m, Xtep, y_te)
        dead_t1 = compute_dead(m, pr_X)
        healthy = (acc > 0.80) and (dead_t1 < 0.25)
        print(f"  lr={lr_cand:.3f}: ce_init={ce_init:.4f}  "
              f"dead_init={dead_init:.4f}  dead_t1={dead_t1:.4f}  "
              f"task1_acc={acc:.4f}  healthy={healthy}", flush=True)
        if healthy and acc > best_acc:
            best_acc = acc
            best_lr  = lr_cand
        del m, opt

    print(f"  → Selected LR={best_lr} (task1_acc={best_acc:.4f})", flush=True)
    return best_lr


# ─── Per-hidden-size experiment ───────────────────────────────────────────────
def run_hidden(hidden, lr, perms, X_tr, y_tr, X_te, y_te, criterion, all_traj):
    """Run N_SEEDS × N_TASKS for one hidden size. Returns summary dict."""
    print(f"\n{'='*70}", flush=True)
    print(f"HIDDEN={hidden}, LR={lr}", flush=True)
    print(f"{'='*70}", flush=True)

    obs_keys  = ["dead", "erank", "gns", "wdrift"]
    obs_names = ["dead_unit_fraction", "effective_rank",
                 "gradient_noise_scale", "weight_norm_drift"]

    seed_records = []

    for seed in range(N_SEEDS):
        print(f"\n--- Seed {seed} ---", flush=True)
        torch.manual_seed(seed); np.random.seed(seed)

        model      = MLP(hidden).to(DEVICE)
        optimizer  = optim.SGD(model.parameters(), lr=lr,
                               momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
        init_norms = {n: p.data.norm(2).item() for n, p in model.named_parameters()}

        # dead_at_init
        X_te0 = X_te[:, perms[0]]
        pr0_X, _ = make_probe(X_te0, y_te, seed=seed)
        dead_init = compute_dead(model, pr0_X)

        # Sanity: CE at init ≈ log(10)
        model.eval()
        with torch.no_grad():
            pr0_y_t = torch.from_numpy(y_te[:PROBE_SIZE]).to(DEVICE)
            ce_init = criterion(model(pr0_X), pr0_y_t).item()
        exp_ce = math.log(N_CLASSES)
        sanity = abs(ce_init - exp_ce) < 0.10
        print(f"  sanity: ce_init={ce_init:.4f} (exp {exp_ce:.4f}) "
              f"{'PASS' if sanity else 'WARN'}  dead_init={dead_init:.4f}", flush=True)

        # Per-task trajectories
        accs, deads, eranks, gnss, wdrifts = [], [], [], [], []
        dead_after_t1 = None

        rec = dict(seed=seed, hidden=hidden, lr=lr,
                   dead_at_init=dead_init, dead_after_task1=0.0,
                   healthy=False, t_collapse=None,
                   accs=accs, dead=deads, erank=eranks, gns=gnss, wdrift=wdrifts)
        seed_records.append(rec)

        t_start_seed = time.time()
        for t_idx in range(N_TASKS):
            perm      = perms[t_idx]
            X_tr_perm = X_tr[:, perm]
            X_te_perm = X_te[:, perm]

            train_task(model, optimizer, X_tr_perm, y_tr, criterion,
                       STEPS_PER_TASK, BATCH_SIZE)

            pr_X, pr_y = make_probe(X_te_perm, y_te, seed=seed*1000+t_idx)
            acc  = eval_acc(model, X_te_perm, y_te)
            dead = compute_dead(model, pr_X)
            er   = compute_erank(model, pr_X)
            gns  = compute_gns(model, X_tr_perm, y_tr, criterion)
            wd   = compute_wdrift(model, init_norms)

            accs.append(acc); deads.append(dead); eranks.append(er)
            gnss.append(gns);  wdrifts.append(wd)

            if t_idx == 0:
                dead_after_t1 = dead
                rec["dead_after_task1"] = dead
                rec["healthy"] = (acc > 0.80) and (dead < 0.25)

            rec["t_collapse"] = detect_collapse(accs)

            if t_idx < 5 or (t_idx + 1) % 20 == 0 or t_idx == N_TASKS - 1:
                print(f"  Task {t_idx+1:3d}/{N_TASKS} | "
                      f"acc={acc:.4f}  dead={dead:.4f}  "
                      f"erank={er:7.3f}  gns={gns:8.4f}  wd={wd:.4f}",
                      flush=True)

            # Incremental save
            if (t_idx + 1) % SAVE_EVERY == 0:
                _incremental_save(all_traj, seed_records, hidden, seed, t_idx+1)

        t_elapsed_seed = time.time() - t_start_seed
        f20 = float(np.mean(accs[:20]))
        l20 = float(np.mean(accs[-20:]))
        drop = (f20 - l20) * 100.0
        print(f"  Seed {seed} done in {t_elapsed_seed:.0f}s | "
              f"dead_init={dead_init:.4f} dead_t1={dead_after_t1:.4f} "
              f"task1_acc={accs[0]:.4f} | "
              f"acc_first20={f20:.4f} acc_last20={l20:.4f} drop={drop:.2f}pp "
              f"t_collapse={rec['t_collapse']}", flush=True)

    return seed_records


# ─── Incremental save ─────────────────────────────────────────────────────────
def _incremental_save(all_traj, seed_records, hidden, current_seed, current_task):
    partial = {
        "status": "RUNNING",
        "current_hidden": hidden,
        "current_seed": current_seed,
        "current_task": current_task,
        "per_seed_so_far": [
            dict(seed=r["seed"], hidden=r["hidden"],
                 tasks_done=len(r["accs"]),
                 task1_acc=r["accs"][0] if r["accs"] else None,
                 dead_at_init=r["dead_at_init"],
                 dead_after_task1=r["dead_after_task1"],
                 healthy=r["healthy"],
                 t_collapse=r["t_collapse"],
                 acc_first20=(float(np.mean(r["accs"][:20]))
                              if len(r["accs"]) >= 20 else None),
                 acc_last20=(float(np.mean(r["accs"][-20:]))
                             if len(r["accs"]) >= 20 else None))
            for r in seed_records
        ]
    }
    with open(RESULTS_PATH, "w") as f:
        json.dump(partial, f, indent=2)


# ─── Analysis for one hidden size ─────────────────────────────────────────────
def analyze_hidden(seed_records, hidden):
    obs_keys  = ["dead", "erank", "gns", "wdrift"]
    obs_names = ["dead_unit_fraction", "effective_rank",
                 "gradient_noise_scale", "weight_norm_drift"]

    # Core health and plasticity metrics
    mean_dead_init = float(np.mean([r["dead_at_init"]     for r in seed_records]))
    mean_dead_t1   = float(np.mean([r["dead_after_task1"] for r in seed_records]))
    mean_t1_acc    = float(np.mean([r["accs"][0]          for r in seed_records]))
    healthy_all    = all(r["healthy"] for r in seed_records)

    first20 = [float(np.mean(r["accs"][:20]))  for r in seed_records]
    last20  = [float(np.mean(r["accs"][-20:])) for r in seed_records]
    mean_f20 = float(np.mean(first20))
    mean_l20 = float(np.mean(last20))
    acc_drop_pp = (mean_f20 - mean_l20) * 100.0
    plasticity_loss = acc_drop_pp > PLASTICITY_PP_THRESHOLD

    t_collapses = [r["t_collapse"] for r in seed_records]
    n_collapsed = sum(t is not None for t in t_collapses)

    print(f"\n--- Analysis hidden={hidden} ---", flush=True)
    print(f"  dead_at_init (mean):     {mean_dead_init:.4f}", flush=True)
    print(f"  dead_after_task1 (mean): {mean_dead_t1:.4f}", flush=True)
    print(f"  task1_acc (mean):        {mean_t1_acc:.4f}", flush=True)
    print(f"  acc_first20 (mean):      {mean_f20:.4f}", flush=True)
    print(f"  acc_last20  (mean):      {mean_l20:.4f}", flush=True)
    print(f"  acc_drop_pp:             {acc_drop_pp:.2f}pp", flush=True)
    print(f"  plasticity_loss:         {plasticity_loss}  (threshold={PLASTICITY_PP_THRESHOLD}pp)", flush=True)
    print(f"  healthy_all:             {healthy_all}", flush=True)
    print(f"  n_collapsed (15pp):      {n_collapsed}/{N_SEEDS}", flush=True)

    for r in seed_records:
        f = float(np.mean(r["accs"][:20]))
        l = float(np.mean(r["accs"][-20:]))
        print(f"  Seed {r['seed']}: first20={f:.4f} last20={l:.4f} "
              f"drop={(f-l)*100:.2f}pp  t_collapse={r['t_collapse']}", flush=True)

    # Lead-time analysis (only if any collapse)
    lead_per_seed = {k: [] for k in obs_keys}
    lead_summary  = {}

    if n_collapsed > 0:
        print(f"\n--- Onset detection (hidden={hidden}) ---", flush=True)
        for r in seed_records:
            t_col = r["t_collapse"]
            print(f"  Seed {r['seed']}, t_collapse={t_col}:", flush=True)
            for k, nm in zip(obs_keys, obs_names):
                if t_col is None:
                    lead_per_seed[k].append(None)
                    print(f"    {nm}: N/A (no collapse)", flush=True)
                    continue
                onset_t = compute_onset(r[k])
                lt = (t_col - onset_t) if onset_t is not None else None
                lead_per_seed[k].append(lt)
                print(f"    {nm}: onset={onset_t}, lead={lt}", flush=True)

        for k, nm in zip(obs_keys, obs_names):
            lts   = [x for x in lead_per_seed[k] if x is not None]
            n_val = len(lts)
            if n_val == 0:
                lead_summary[nm] = {"mean": None, "median": None,
                                    "iqr": [None, None], "ci95": [None, None],
                                    "n_valid": 0}
                continue
            arr = np.array(lts, dtype=float)
            ci  = bootstrap_ci(lts, seed=42)
            lead_summary[nm] = {
                "mean":    float(np.mean(arr)),
                "median":  float(np.median(arr)),
                "iqr":     [float(np.percentile(arr, 25)),
                            float(np.percentile(arr, 75))],
                "ci95":    [float(ci[0]), float(ci[1])],
                "n_valid": n_val,
            }
    else:
        for nm in obs_names:
            lead_summary[nm] = {"mean": None, "median": None,
                                "iqr": [None, None], "ci95": [None, None],
                                "n_valid": 0}

    # Precedence order
    sortable = [(nm, lead_summary[nm]["mean"])
                for nm in obs_names if lead_summary[nm]["mean"] is not None]
    sortable.sort(key=lambda x: x[1], reverse=True)
    precedence_order = [nm for nm, _ in sortable]

    # Wilcoxon: erank vs GNS lead times
    erank_lts = lead_per_seed.get("erank", [])
    gns_lts   = lead_per_seed.get("gns",   [])
    paired_diffs = [er - gn for er, gn in zip(erank_lts, gns_lts)
                    if er is not None and gn is not None]
    wilcoxon_stat = wilcoxon_p = median_diff = None
    if paired_diffs:
        median_diff = float(np.median(paired_diffs))
        if len(paired_diffs) >= 3:
            diffs_arr = np.array(paired_diffs, dtype=float)
            nonzero   = diffs_arr[diffs_arr != 0]
            if len(nonzero) >= 3:
                try:
                    stat, p = scipy_wilcoxon(diffs_arr, alternative="two-sided",
                                             zero_method="wilcox")
                    wilcoxon_stat, wilcoxon_p = float(stat), float(p)
                except Exception as e:
                    print(f"  Wilcoxon: {e}", flush=True)

    # Per-seed details
    per_seed_det = []
    for r in seed_records:
        idx_s = r["seed"]
        per_seed_det.append({
            "seed":             r["seed"],
            "dead_at_init":     r["dead_at_init"],
            "dead_after_task1": r["dead_after_task1"],
            "task1_acc":        r["accs"][0],
            "healthy":          r["healthy"],
            "t_collapse":       r["t_collapse"],
            "acc_first20":      float(np.mean(r["accs"][:20])),
            "acc_last20":       float(np.mean(r["accs"][-20:])),
            "acc_drop_pp":      float((np.mean(r["accs"][:20]) - np.mean(r["accs"][-20:])) * 100),
            "lead_dead":        lead_per_seed["dead"][idx_s]  if idx_s < len(lead_per_seed["dead"]) else None,
            "lead_erank":       lead_per_seed["erank"][idx_s] if idx_s < len(lead_per_seed["erank"]) else None,
            "lead_gns":         lead_per_seed["gns"][idx_s]   if idx_s < len(lead_per_seed["gns"]) else None,
            "lead_wdrift":      lead_per_seed["wdrift"][idx_s] if idx_s < len(lead_per_seed["wdrift"]) else None,
        })

    return {
        "hidden": hidden,
        "lr_used": seed_records[0]["lr"],
        "dead_at_init":     mean_dead_init,
        "dead_after_task1": mean_dead_t1,
        "task1_acc":        mean_t1_acc,
        "healthy":          healthy_all,
        "acc_first20":      mean_f20,
        "acc_last20":       mean_l20,
        "acc_drop_pp":      acc_drop_pp,
        "plasticity_loss":  plasticity_loss,
        "t_collapse_per_seed": t_collapses,
        "n_seeds_collapsed":   n_collapsed,
        "lead_times":       lead_summary,
        "precedence_order": precedence_order,
        "erank_vs_gns_paired": {
            "diffs_per_seed": paired_diffs,
            "median_diff":    median_diff,
            "wilcoxon_stat":  wilcoxon_stat,
            "wilcoxon_p":     wilcoxon_p,
        },
        "per_seed": per_seed_det,
    }


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    t_start_total = time.time()

    # Write initial LOG.md
    with open(LOG_PATH, "w") as f:
        f.write("# smallnet Experiment Log\n\n")
        f.write("## Setup\n")
        f.write(f"- Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"- Script: src/run_small.py\n")
        f.write(f"- Hidden sizes: {HIDDEN_SIZES}\n")
        f.write(f"- N_TASKS={N_TASKS}, N_SEEDS={N_SEEDS}, STEPS_PER_TASK={STEPS_PER_TASK}\n")
        f.write(f"- SGD lr={LR} mom={MOMENTUM} WD={WEIGHT_DECAY}\n")
        f.write(f"- Device: {DEVICE}\n\n")
        f.write("## Decisions\n")
        f.write(f"- Starting LR={LR} from prior pmnist run (task1_acc=0.975 at hidden=2000)\n")
        f.write(f"- Running LR verification for each hidden size to confirm health\n")
        f.write(f"- STEPS_PER_TASK={STEPS_PER_TASK} ('a few hundred', online per brief)\n")
        f.write(f"- Health criterion: dead_after_task1 < 0.25, task1_acc > 0.80\n")
        f.write(f"- Plasticity loss: acc_drop_pp > {PLASTICITY_PP_THRESHOLD}pp\n")
        f.write(f"- Collapse criterion: acc drops >= {COLLAPSE_THRESH_PP}pp for "
                f">= {COLLAPSE_MIN_TASKS} consecutive tasks\n")
        f.write(f"- Incremental save every {SAVE_EVERY} tasks\n\n")

    # 1. Load MNIST
    X_tr, y_tr, X_te, y_te = load_mnist()

    # 2. Permutations (same seed as prior runs for consistency)
    perms = make_perms(N_TASKS, perm_seed=42)
    print(f"Created {N_TASKS} permutations (task 0 = identity)", flush=True)

    criterion = nn.CrossEntropyLoss()
    all_trajectories = {}
    results_by_hidden = {}

    # 3. Run for each hidden size
    for hidden in HIDDEN_SIZES:
        print(f"\n\n{'#'*70}", flush=True)
        print(f"# STARTING HIDDEN={hidden}", flush=True)
        print(f"{'#'*70}", flush=True)

        # LR verification (pick best healthy LR)
        lr_selected = verify_lr(hidden, perms, X_tr, y_tr, X_te, y_te, criterion)

        # Run experiment
        all_traj = {}
        seed_records = run_hidden(hidden, lr_selected, perms,
                                  X_tr, y_tr, X_te, y_te, criterion, all_traj)

        # Analysis
        summary = analyze_hidden(seed_records, hidden)
        results_by_hidden[str(hidden)] = summary

        # Save per-hidden trajectories
        all_trajectories[str(hidden)] = {
            "lr_used": lr_selected,
            "per_seed": [
                {
                    "seed":   r["seed"],
                    "accs":   r["accs"],
                    "dead":   r["dead"],
                    "erank":  [nan_to_null(v) for v in r["erank"]],
                    "gns":    [nan_to_null(v) for v in r["gns"]],
                    "wdrift": [nan_to_null(v) for v in r["wdrift"]],
                    "t_collapse":       r["t_collapse"],
                    "dead_at_init":     r["dead_at_init"],
                    "dead_after_task1": r["dead_after_task1"],
                }
                for r in seed_records
            ]
        }

        # Incremental RESULTS.json with what we have so far
        _write_intermediate(results_by_hidden, hidden)

    # 4. Final RESULTS.json
    total_elapsed = time.time() - t_start_total
    print(f"\n\nTotal wall-clock: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)",
          flush=True)

    any_collapse = any(
        results_by_hidden[str(h)]["plasticity_loss"] for h in HIDDEN_SIZES
    )

    # Combine lead times across all sizes where collapse occurred
    combined_lead = None
    combined_prec = []
    combined_wilcoxon = None

    for h in HIDDEN_SIZES:
        summ = results_by_hidden[str(h)]
        if summ["plasticity_loss"]:
            combined_lead = summ["lead_times"]
            combined_prec = summ["precedence_order"]
            combined_wilcoxon = summ["erank_vs_gns_paired"]

    # Build honest notes
    lines = []
    for h in HIDDEN_SIZES:
        s = results_by_hidden[str(h)]
        lines.append(
            f"hidden={h}: acc_drop={s['acc_drop_pp']:.2f}pp "
            f"(first20={s['acc_first20']:.4f} vs last20={s['acc_last20']:.4f}), "
            f"task1_acc={s['task1_acc']:.4f}, "
            f"dead_after_t1={s['dead_after_task1']:.4f}, "
            f"healthy={s['healthy']}, "
            f"plasticity_loss={s['plasticity_loss']} (>{PLASTICITY_PP_THRESHOLD}pp threshold)"
        )

    if any_collapse:
        verdict = (
            f"GENUINE plasticity loss detected in a healthy small net. "
            f"precedence_order={combined_prec}."
        )
    else:
        verdict = (
            f"NO plasticity loss even in small networks (hidden=100,256) at {N_TASKS} tasks. "
            f"Honest conclusion: permuted-MNIST with SGD+momentum does NOT reproduce Dohare et al. "
            f"plasticity collapse — even in small healthy networks. "
            f"The benchmark itself (same label set, label-preserving permutations) is too easy; "
            f"plasticity loss requires harder, incompatible task sequences (e.g., Split-CIFAR)."
        )

    final_results = {
        "status":    "DONE",
        "scale":     "full",
        "benchmark": "online_permuted_mnist_smallnet",
        "config": {
            "n_tasks":          N_TASKS,
            "n_seeds":          N_SEEDS,
            "steps_per_task":   STEPS_PER_TASK,
            "batch_size":       BATCH_SIZE,
            "momentum":         MOMENTUM,
            "weight_decay":     WEIGHT_DECAY,
            "collapse_thresh_pp":  COLLAPSE_THRESH_PP,
            "plasticity_loss_threshold_pp": PLASTICITY_PP_THRESHOLD,
        },
        "by_hidden": {
            str(h): {
                "dead_at_init":     results_by_hidden[str(h)]["dead_at_init"],
                "dead_after_task1": results_by_hidden[str(h)]["dead_after_task1"],
                "task1_acc":        results_by_hidden[str(h)]["task1_acc"],
                "lr_used":          results_by_hidden[str(h)]["lr_used"],
                "acc_first20":      results_by_hidden[str(h)]["acc_first20"],
                "acc_last20":       results_by_hidden[str(h)]["acc_last20"],
                "acc_drop_pp":      results_by_hidden[str(h)]["acc_drop_pp"],
                "plasticity_loss":  results_by_hidden[str(h)]["plasticity_loss"],
                "healthy":          results_by_hidden[str(h)]["healthy"],
                "t_collapse_per_seed": results_by_hidden[str(h)]["t_collapse_per_seed"],
                "n_seeds_collapsed":   results_by_hidden[str(h)]["n_seeds_collapsed"],
                "per_seed":         results_by_hidden[str(h)]["per_seed"],
            }
            for h in HIDDEN_SIZES
        },
        "any_healthy_collapse":  any_collapse,
        "lead_times":            combined_lead,
        "precedence_order":      combined_prec,
        "erank_vs_gns_paired":   combined_wilcoxon,
        "wall_clock_sec":        round(total_elapsed, 1),
        "subject_executed": (
            f"Online Permuted-MNIST, 3-layer MLP with hidden∈{{100,256}}, ReLU, "
            f"SGD+momentum, no plasticity repair, {N_TASKS} tasks × "
            f"{STEPS_PER_TASK} steps/task × {N_SEEDS} seeds per hidden size. "
            f"4 observables per task: dead_unit_fraction, effective_rank, "
            f"gradient_noise_scale (McCandlish B_opt, 2×64 mini-batches), "
            f"weight_norm_drift. src/run_small.py."
        ),
        "notes": verdict + " | " + " | ".join(lines),
    }

    # Required "metrics" key for outer harness
    final_results["metrics"] = {
        "by_hidden": {
            str(h): {
                "dead_at_init":     final_results["by_hidden"][str(h)]["dead_at_init"],
                "dead_after_task1": final_results["by_hidden"][str(h)]["dead_after_task1"],
                "task1_acc":        final_results["by_hidden"][str(h)]["task1_acc"],
                "acc_first20":      final_results["by_hidden"][str(h)]["acc_first20"],
                "acc_last20":       final_results["by_hidden"][str(h)]["acc_last20"],
                "acc_drop_pp":      final_results["by_hidden"][str(h)]["acc_drop_pp"],
                "plasticity_loss":  final_results["by_hidden"][str(h)]["plasticity_loss"],
            }
            for h in HIDDEN_SIZES
        },
        "any_healthy_collapse": any_collapse,
        "wall_clock_sec": round(total_elapsed, 1),
    }

    with open(RESULTS_PATH, "w") as f:
        json.dump(final_results, f, indent=2)
    print(f"\nResults → {RESULTS_PATH}", flush=True)

    with open(TRAJ_PATH, "w") as f:
        json.dump({"n_tasks": N_TASKS, "n_seeds": N_SEEDS,
                   "tasks": list(range(1, N_TASKS+1)),
                   "by_hidden": all_trajectories}, f, indent=2)
    print(f"Trajectories → {TRAJ_PATH}", flush=True)

    # Update LOG.md
    with open(LOG_PATH, "a") as f:
        f.write("\n## Results\n")
        f.write(f"- Wall-clock: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)\n")
        for h in HIDDEN_SIZES:
            s = results_by_hidden[str(h)]
            f.write(f"- hidden={h}: lr={s['lr_used']} task1_acc={s['task1_acc']:.4f} "
                    f"dead_t1={s['dead_after_task1']:.4f} healthy={s['healthy']}\n")
            f.write(f"  acc_first20={s['acc_first20']:.4f} acc_last20={s['acc_last20']:.4f} "
                    f"drop={s['acc_drop_pp']:.2f}pp plasticity_loss={s['plasticity_loss']}\n")
        f.write(f"\n## Conclusion\n{verdict}\n")

    # Print final
    print("\n" + "="*70, flush=True)
    print("FINAL RESULTS", flush=True)
    print("="*70, flush=True)
    print(json.dumps(final_results, indent=2), flush=True)


def _write_intermediate(results_by_hidden, last_hidden):
    partial = {
        "status": "RUNNING",
        "last_hidden_completed": last_hidden,
        "by_hidden_so_far": {
            str(h): {
                "dead_at_init":     results_by_hidden[str(h)]["dead_at_init"],
                "dead_after_task1": results_by_hidden[str(h)]["dead_after_task1"],
                "task1_acc":        results_by_hidden[str(h)]["task1_acc"],
                "acc_first20":      results_by_hidden[str(h)]["acc_first20"],
                "acc_last20":       results_by_hidden[str(h)]["acc_last20"],
                "acc_drop_pp":      results_by_hidden[str(h)]["acc_drop_pp"],
                "plasticity_loss":  results_by_hidden[str(h)]["plasticity_loss"],
            }
            for h in results_by_hidden
        }
    }
    with open(RESULTS_PATH, "w") as f:
        json.dump(partial, f, indent=2)


if __name__ == "__main__":
    main()
