"""
run_grid2.py — Scale-out experiment (scale_grid2 round): generalize Adam-v_t
plasticity result across DATASETS and ARCHITECTURES.

EXPERIMENT.md round: scale_grid2 (full)
GRID (SMALL, designed to COMPLETE):
  Datasets     : CIFAR-100 (5 cls/task, 4 tasks) and CIFAR-10 (5 cls/task, 2 tasks)
  Architectures: (1) SmallConvNetGN, (2) MLP3GN  — NO ResNet (too slow)
  Arms         : A_floor (no reset), B (CBP heuristic), C (explicit Fisher),
                 D (Adam v_t, OURS), RANDOM (random-reset control)
  Seeds        : 6, PAIRED (same seed → same task split across reset arms)
  A_floor      : run ONCE per (dataset,arch) using seed=0

Total reset-runs: 2 × 2 × 4 × 6 = 96
Floor-runs     : 2 × 2 × 1 = 4
Grand total    : 100 runs × ~15–20s ≈ 30–40 min → must complete in < 90 min

Writes:
  results/scale_grid2/RESULTS.json   — rewritten after every (dataset,arch) block
  results/scale_grid2/rows.jsonl     — one row per (dataset,arch,arm,seed)
  results/scale_grid2/run.log        — captured externally via tee

Time guard: if elapsed ≥ 80 min, stop launching new seeds (floor is kept if done).
"""
import os
import sys
import json
import time
import random
import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats as spstats


# ─── JSON encoder for numpy types ─────────────────────────────────────────────
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.bool_): return bool(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super().default(obj)


# ─── Paths ────────────────────────────────────────────────────────────────────
RESULTS_DIR  = "/workspace/results/scale_grid2"
RESULTS_PATH = os.path.join(RESULTS_DIR, "RESULTS.json")
ROWS_PATH    = os.path.join(RESULTS_DIR, "rows.jsonl")
WEIGHTS_DIR  = "/workspace/_weights"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(WEIGHTS_DIR, exist_ok=True)

# ─── Hyperparameters ──────────────────────────────────────────────────────────
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE     = 128
LR             = 1e-3
RESET_EVERY    = 100      # steps between CBP sweeps
RESET_FRAC     = 0.10     # fraction of penultimate units reset
STEPS_PER_TASK = 800
DATASETS_DIR   = os.environ.get("SH_DATASETS_DIR", "/opt/datasets")
MAX_WALL_MIN   = 80.0     # abort new seeds after this wall time
MIN_SEEDS      = 4        # minimum seeds per cell before we can finalize

# Per-architecture penultimate size
N_PEN = {"convnet": 512, "mlp3": 512}

# Dataset configs — 5 cls/task per spec
DATASET_CFG = {
    "cifar100": {
        "n_tasks":     4,    # 4 × 5 cls = 20 classes (subset of 100)
        "cls_per_task": 5,
        "num_classes": 100,
        "mean": [0.5071, 0.4867, 0.4408],
        "std":  [0.2675, 0.2565, 0.2761],
    },
    "cifar10": {
        "n_tasks":     2,    # 2 × 5 cls = 10 classes (all of CIFAR-10)
        "cls_per_task": 5,
        "num_classes": 10,
        "mean": [0.4914, 0.4822, 0.4465],
        "std":  [0.2470, 0.2435, 0.2616],
    },
}

ARCH_NAMES  = ["convnet", "mlp3"]   # NO ResNet per spec
RESET_ARMS  = ["B", "C", "D", "RANDOM"]
ALL_ARMS    = ["A_floor"] + RESET_ARMS
SEEDS       = list(range(6))       # 6 seeds

total_runs = 2 * 2 * len(RESET_ARMS) * len(SEEDS) + 2 * 2  # 96 + 4 = 100

print(f"[init] Device: {DEVICE}", flush=True)
print(f"[init] Grid: {list(DATASET_CFG)} × {ARCH_NAMES} × {ALL_ARMS} × {SEEDS}", flush=True)
print(f"[init] steps/task={STEPS_PER_TASK}  reset_every={RESET_EVERY}  frac={RESET_FRAC}", flush=True)
print(f"[init] Total training runs: {total_runs}", flush=True)


# ─── Data loading ─────────────────────────────────────────────────────────────
def load_cifar100():
    """Load CIFAR-100 as numpy arrays (download=False, ABORT if missing)."""
    dcfg = DATASET_CFG["cifar100"]
    try:
        import torchvision.datasets as tvds
        tr_ds = tvds.CIFAR100(root=DATASETS_DIR, train=True,  download=False)
        te_ds = tvds.CIFAR100(root=DATASETS_DIR, train=False, download=False)
        X_tr = np.array(tr_ds.data,    dtype=np.float32).transpose(0, 3, 1, 2) / 255.0
        y_tr = np.array(tr_ds.targets, dtype=np.int64)
        X_te = np.array(te_ds.data,    dtype=np.float32).transpose(0, 3, 1, 2) / 255.0
        y_te = np.array(te_ds.targets, dtype=np.int64)
        print(f"[cifar100] torchvision: {X_tr.shape}", flush=True)
    except Exception:
        data_dir = os.path.join(DATASETS_DIR, "cifar-100-python")
        if not os.path.isdir(data_dir):
            sys.exit(f"ABORT: CIFAR-100 not found at {data_dir}")
        def unpickle(f):
            with open(f, "rb") as fo:
                return pickle.load(fo, encoding="bytes")
        tr = unpickle(os.path.join(data_dir, "train"))
        te = unpickle(os.path.join(data_dir, "test"))
        X_tr = tr[b"data"].reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
        y_tr = np.array(tr[b"fine_labels"], dtype=np.int64)
        X_te = te[b"data"].reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
        y_te = np.array(te[b"fine_labels"], dtype=np.int64)
        print(f"[cifar100] pickle: {X_tr.shape}", flush=True)
    m = np.array(dcfg["mean"], dtype=np.float32)[:, None, None]
    s = np.array(dcfg["std"],  dtype=np.float32)[:, None, None]
    return (X_tr - m) / s, y_tr, (X_te - m) / s, y_te


def load_cifar10():
    """Load CIFAR-10 as numpy arrays (download=False, ABORT if missing)."""
    dcfg = DATASET_CFG["cifar10"]
    try:
        import torchvision.datasets as tvds
        tr_ds = tvds.CIFAR10(root=DATASETS_DIR, train=True,  download=False)
        te_ds = tvds.CIFAR10(root=DATASETS_DIR, train=False, download=False)
        X_tr = np.array(tr_ds.data,    dtype=np.float32).transpose(0, 3, 1, 2) / 255.0
        y_tr = np.array(tr_ds.targets, dtype=np.int64)
        X_te = np.array(te_ds.data,    dtype=np.float32).transpose(0, 3, 1, 2) / 255.0
        y_te = np.array(te_ds.targets, dtype=np.int64)
        print(f"[cifar10] torchvision: {X_tr.shape}", flush=True)
    except Exception:
        data_dir = os.path.join(DATASETS_DIR, "cifar-10-batches-py")
        if not os.path.isdir(data_dir):
            sys.exit(f"ABORT: CIFAR-10 not found at {data_dir}")
        def unpickle(f):
            with open(f, "rb") as fo:
                return pickle.load(fo, encoding="bytes")
        batches = [unpickle(os.path.join(data_dir, f"data_batch_{i}")) for i in range(1, 6)]
        X_tr = np.concatenate([b[b"data"] for b in batches]).reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
        y_tr = np.concatenate([b[b"labels"] for b in batches]).astype(np.int64)
        te   = unpickle(os.path.join(data_dir, "test_batch"))
        X_te = te[b"data"].reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
        y_te = np.array(te[b"labels"], dtype=np.int64)
        print(f"[cifar10] pickle: {X_tr.shape}", flush=True)
    m = np.array(dcfg["mean"], dtype=np.float32)[:, None, None]
    s = np.array(dcfg["std"],  dtype=np.float32)[:, None, None]
    return (X_tr - m) / s, y_tr, (X_te - m) / s, y_te


# ─── Task splits — GPU-preloaded ─────────────────────────────────────────────
def make_task_splits(X_tr, y_tr, X_te, y_te, seed, n_tasks, cls_per_task, num_classes):
    """SAME seed → SAME splits for ALL arms (PAIRED).
    Returns list of (tr_X, tr_y, te_X, te_y, cls_list) tensors on DEVICE.
    """
    rng = random.Random(seed)
    all_cls = list(range(num_classes))
    rng.shuffle(all_cls)
    task_classes = [all_cls[i * cls_per_task:(i + 1) * cls_per_task] for i in range(n_tasks)]
    splits = []
    for cls_list in task_classes:
        cls_arr = np.array(cls_list)
        tr_m = np.isin(y_tr, cls_arr)
        te_m = np.isin(y_te, cls_arr)
        splits.append((
            torch.from_numpy(X_tr[tr_m]).to(DEVICE),
            torch.from_numpy(y_tr[tr_m]).long().to(DEVICE),
            torch.from_numpy(X_te[te_m]),           # CPU for eval
            torch.from_numpy(y_te[te_m]).long(),
            cls_list,
        ))
    return splits


# ─── GroupNorm 1-D helper ─────────────────────────────────────────────────────
def gn1d(gn_module, x):
    return gn_module(x.unsqueeze(-1)).squeeze(-1)


# ─── Architecture 1: SmallConvNetGN ──────────────────────────────────────────
class SmallConvNetGN(nn.Module):
    def __init__(self, num_output=100):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),  nn.GroupNorm(8, 64),  nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.GroupNorm(8, 64),  nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),  nn.GroupNorm(8, 128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.GroupNorm(8, 128), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.GroupNorm(8, 256), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.GroupNorm(8, 256), nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc1 = nn.Linear(256 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, num_output)
        self._penultimate  = None
        self._last_fc1_act = None

    def forward(self, x, store_pen=False):
        h   = self.features(x).view(x.size(0), -1)
        pre = self.fc1(h)
        act = F.relu(pre)
        self._last_fc1_act = act.detach()
        if store_pen:
            self._penultimate = act.detach()
        return self.fc2(act)

    def forward_with_act_grad(self, x):
        h   = self.features(x).view(x.size(0), -1)
        pre = self.fc1(h)
        act = F.relu(pre)
        act.retain_grad()
        return self.fc2(act), act


# ─── Architecture 2: MLP3GN ──────────────────────────────────────────────────
class MLP3GN(nn.Module):
    def __init__(self, num_output=100, hidden=512, groups=8):
        super().__init__()
        self.l1  = nn.Linear(3 * 32 * 32, hidden)
        self.gn1 = nn.GroupNorm(groups, hidden)
        self.l2  = nn.Linear(hidden, hidden)
        self.gn2 = nn.GroupNorm(groups, hidden)
        self.fc1 = nn.Linear(hidden, hidden)   # penultimate
        self.gn3 = nn.GroupNorm(groups, hidden)
        self.fc2 = nn.Linear(hidden, num_output)
        self._penultimate  = None
        self._last_fc1_act = None

    def forward(self, x, store_pen=False):
        x   = x.view(x.size(0), -1)
        h1  = F.relu(gn1d(self.gn1, self.l1(x)))
        h2  = F.relu(gn1d(self.gn2, self.l2(h1)))
        pre = self.fc1(h2)
        act = F.relu(gn1d(self.gn3, pre))
        self._last_fc1_act = act.detach()
        if store_pen:
            self._penultimate = act.detach()
        return self.fc2(act)

    def forward_with_act_grad(self, x):
        x   = x.view(x.size(0), -1)
        h1  = F.relu(gn1d(self.gn1, self.l1(x)))
        h2  = F.relu(gn1d(self.gn2, self.l2(h1)))
        pre = self.fc1(h2)
        act = F.relu(gn1d(self.gn3, pre))
        act.retain_grad()
        return self.fc2(act), act


def make_model(arch_name, num_classes):
    if arch_name == "convnet":
        return SmallConvNetGN(num_output=num_classes).to(DEVICE)
    elif arch_name == "mlp3":
        return MLP3GN(num_output=num_classes).to(DEVICE)
    raise ValueError(f"Unknown arch: {arch_name}")


# ─── GPU batch sampler ────────────────────────────────────────────────────────
def sample_batch(tr_X, tr_y):
    idx = torch.randint(0, tr_X.size(0), (BATCH_SIZE,), device=DEVICE)
    x   = tr_X[idx]
    # Horizontal flip augmentation
    mask = torch.rand(x.size(0), device=DEVICE) > 0.5
    if mask.any():
        x = x.clone()
        x[mask] = x[mask].flip(-1)
    return x, tr_y[idx]


# ─── Metrics ─────────────────────────────────────────────────────────────────
def compute_accuracy(model, te_X, te_y, cls_list):
    model.eval()
    cls_t = torch.tensor(cls_list, device=DEVICE)
    correct = total = 0
    with torch.no_grad():
        for i in range(0, te_X.size(0), 256):
            x = te_X[i:i + 256].to(DEVICE)
            y = te_y[i:i + 256].to(DEVICE)
            logits = model(x)
            mask   = torch.full_like(logits, float('-inf'))
            mask[:, cls_t] = logits[:, cls_t]
            correct += (mask.argmax(1) == y).sum().item()
            total   += y.size(0)
    return correct / total if total else 0.0


def dead_unit_fraction(model, te_X, max_samples=2048, thr=0.01):
    model.eval()
    acts = []
    with torch.no_grad():
        for i in range(0, min(max_samples, te_X.size(0)), 256):
            x = te_X[i:i + 256].to(DEVICE)
            model(x, store_pen=True)
            acts.append(model._penultimate.cpu())
    A = torch.cat(acts, 0)
    return (A.abs().mean(0) < thr).float().mean().item()


def effective_rank(model, te_X, max_samples=2000):
    model.eval()
    acts = []
    with torch.no_grad():
        for i in range(0, min(max_samples, te_X.size(0)), 256):
            x = te_X[i:i + 256].to(DEVICE)
            model(x, store_pen=True)
            acts.append(model._penultimate.cpu())
    A = torch.cat(acts, 0).float()
    if A.abs().max() < 1e-10:
        return 0.0
    A = A - A.mean(0, keepdim=True)
    if A.size(0) > 2000:
        A = A[torch.randperm(A.size(0))[:2000]]
    try:
        _, S, _ = torch.linalg.svd(A, full_matrices=False)
        S = S.clamp(min=1e-8)
        p = S / S.sum()
        return torch.exp(-(p * p.log()).sum()).item()
    except Exception:
        return float('nan')


# ─── Utility functions ────────────────────────────────────────────────────────
def utility_B(model, arm_state, n_pen):
    """CBP heuristic: |out_weight| × EMA mean activation."""
    out_w = model.fc2.weight.data.abs().mean(0).cpu()[:n_pen]
    rm    = arm_state.get("running_mean_act", torch.ones(n_pen))
    return out_w * rm


def utility_C(model, x_batch, y_batch, criterion, n_pen):
    """Explicit empirical Fisher — one extra forward+backward pass."""
    was_training = model.training
    model.train()
    model.zero_grad()
    logits, act = model.forward_with_act_grad(x_batch)
    loss = criterion(logits, y_batch)
    loss.backward()
    if act.grad is not None:
        u = (act.grad.detach() ** 2).mean(0).cpu()[:n_pen]
    else:
        u = torch.zeros(n_pen)
    model.zero_grad()
    if not was_training:
        model.eval()
    return u


def utility_D(model, optimizer, n_pen):
    """Adam exp_avg_sq proxy — zero extra compute."""
    param = model.fc1.weight
    state = optimizer.state.get(param, {})
    if "exp_avg_sq" not in state:
        return torch.zeros(n_pen)
    vt = state["exp_avg_sq"].detach().cpu()
    return vt.mean(dim=1)[:n_pen]


def utility_RANDOM(n_pen, rng):
    return torch.from_numpy(rng.rand(n_pen).astype(np.float32))


# ─── CBP reset ────────────────────────────────────────────────────────────────
def cbp_reset(model, optimizer, utility, n_pen):
    n_reset   = max(1, int(n_pen * RESET_FRAC))
    _, r_idx  = torch.topk(utility, n_reset, largest=False)
    with torch.no_grad():
        fan_in = model.fc1.weight.shape[1]
        bound  = float(np.sqrt(3.0) * np.sqrt(2.0 / fan_in))
        model.fc1.weight.data[r_idx] = torch.empty(
            n_reset, fan_in, device=DEVICE).uniform_(-bound, bound)
        if model.fc1.bias is not None:
            model.fc1.bias.data[r_idx] = 0.0
        model.fc2.weight.data[:, r_idx] = 0.0

    def _rst_rows(p, rows):
        s = optimizer.state.get(p, {})
        if "exp_avg" in s:
            s["exp_avg"][rows]    = 0.0
            s["exp_avg_sq"][rows] = 0.0

    def _rst_cols(p, cols):
        s = optimizer.state.get(p, {})
        if "exp_avg" in s:
            s["exp_avg"][:, cols]    = 0.0
            s["exp_avg_sq"][:, cols] = 0.0

    _rst_rows(model.fc1.weight, r_idx)
    if model.fc1.bias is not None:
        _rst_rows(model.fc1.bias, r_idx)
    _rst_cols(model.fc2.weight, r_idx)


# ─── Single (arm, seed) run ───────────────────────────────────────────────────
def run_arm_seed(arm, seed, splits, arch_name, num_classes, n_pen):
    t0 = time.time()

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

    model     = make_model(arch_name, num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    rand_rng  = np.random.RandomState(2000 + seed)   # RANDOM arm RNG
    arm_state = {}                                    # Arm B EMA

    n_tasks = len(splits)
    per_task_acc   = []
    per_task_dead  = []
    per_task_erank = []

    for tid, (tr_X, tr_y, te_X, te_y, cls_list) in enumerate(splits):
        t_task = time.time()
        model.train()
        losses = []

        for step in range(1, STEPS_PER_TASK + 1):
            x, y = sample_batch(tr_X, tr_y)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

            # Arm B: update EMA activation mean
            if arm == "B" and model._last_fc1_act is not None:
                ca = model._last_fc1_act.abs().mean(0).cpu()
                if "running_mean_act" not in arm_state:
                    arm_state["running_mean_act"] = ca.clone()
                else:
                    arm_state["running_mean_act"].mul_(0.99).add_(0.01 * ca)

            # CBP reset (reset arms only)
            if arm != "A_floor" and step % RESET_EVERY == 0:
                if arm == "B":
                    util = utility_B(model, arm_state, n_pen)
                elif arm == "C":
                    util = utility_C(model, x, y, criterion, n_pen)
                elif arm == "D":
                    util = utility_D(model, optimizer, n_pen)
                elif arm == "RANDOM":
                    util = utility_RANDOM(n_pen, rand_rng)
                else:
                    raise ValueError(f"Unknown arm: {arm}")
                cbp_reset(model, optimizer, util, n_pen)
                model.train()

        wall = time.time() - t_task
        acc  = compute_accuracy(model, te_X, te_y, cls_list)
        duf  = dead_unit_fraction(model, te_X)
        er   = effective_rank(model, te_X)

        per_task_acc.append(round(float(acc), 4))
        per_task_dead.append(round(float(duf), 4))
        per_task_erank.append(round(float(er), 4) if er == er else 0.0)

        print(
            f"  [{arm:8s}/s{seed}/{arch_name}] "
            f"T{tid+1}/{n_tasks} acc={acc:.3f} dead={duf:.3f} "
            f"erank={er:.2f} loss={np.mean(losses[-100:]):.4f} "
            f"wall={wall:.1f}s",
            flush=True,
        )

    total_wall = time.time() - t0
    # Primary metric: mean over tasks 2..end (skip warmup task 1)
    primary_acc = float(np.mean(per_task_acc[1:])) if len(per_task_acc) > 1 else per_task_acc[0]
    return {
        "primary_acc":    round(primary_acc, 4),
        "per_task_acc":   per_task_acc,
        "per_task_dead":  per_task_dead,
        "per_task_erank": per_task_erank,
        "dead_final":     per_task_dead[-1],
        "erank_final":    per_task_erank[-1],
        "wall_s":         round(total_wall, 1),
    }


# ─── Statistics ───────────────────────────────────────────────────────────────
def paired_stats(a_accs, b_accs):
    """Paired diff (a – b) in pp with 95% CI, one-sided and two-sided p."""
    diffs = [(av - bv) * 100.0 for av, bv in zip(a_accs, b_accs)]
    n = len(diffs)
    if n < 2:
        return {
            "mean_pp": round(float(np.mean(diffs)), 3),
            "ci95_pp": [None, None],
            "p_one_sided": None, "p_two_sided": None,
            "n": n, "per_seed": [round(d, 3) for d in diffs],
        }
    t_stat, p_two = spstats.ttest_rel(a_accs, b_accs)
    p_one = p_two / 2.0 if t_stat > 0 else 1.0 - p_two / 2.0
    se    = np.std(diffs, ddof=1) / np.sqrt(n)
    t_c   = spstats.t.ppf(0.975, df=n - 1)
    mu    = float(np.mean(diffs))
    return {
        "per_seed":    [round(d, 3) for d in diffs],
        "mean_pp":     round(mu, 3),
        "ci95_pp":     [round(mu - float(t_c) * float(se), 3),
                        round(mu + float(t_c) * float(se), 3)],
        "p_two_sided": round(float(p_two), 6),
        "p_one_sided": round(float(p_one), 6),
        "n":           n,
    }


# ─── Aggregate RESULTS.json ───────────────────────────────────────────────────
def aggregate_results(cell_accs, cell_meta, floor_accs, floor_meta, status="RUNNING"):
    cells = {}
    eq_count = mech_count = n_cells = 0

    for (dataset, arch) in [(d, a) for d in DATASET_CFG for a in ARCH_NAMES]:
        arm_accs = cell_accs.get((dataset, arch), {})
        arm_meta = cell_meta.get((dataset, arch), {})
        ckey = f"{dataset}_{arch}"

        # Need at least the floor to report a cell
        fl_list = floor_accs.get((dataset, arch), [])
        if not fl_list:
            continue
        n_cells += 1

        def arm_stats(arm):
            accs = arm_accs.get(arm, [])
            if not accs:
                return None
            metas = arm_meta.get(arm, [])
            deads  = [m["dead_final"]  for m in metas if m.get("dead_final")  is not None]
            eranks = [m["erank_final"] for m in metas if m.get("erank_final") is not None]
            return {
                "mean":             round(float(np.mean(accs)), 4),
                "std":              round(float(np.std(accs, ddof=1)) if len(accs) > 1 else 0.0, 4),
                "per_seed_acc":     [round(a, 4) for a in accs],
                "n_seeds":          len(accs),
                "dead_final_mean":  round(float(np.mean(deads)),  4) if deads  else None,
                "erank_final_mean": round(float(np.mean(eranks)), 4) if eranks else None,
            }

        floor_val = round(float(np.mean(fl_list)), 4)

        D_accs = arm_accs.get("D", [])
        C_accs = arm_accs.get("C", [])
        B_accs = arm_accs.get("B", [])
        R_accs = arm_accs.get("RANDOM", [])

        paired_DC = paired_DR = paired_DB = None

        if len(D_accs) >= 2 and len(C_accs) >= 2:
            n  = min(len(D_accs), len(C_accs))
            ps = paired_stats(D_accs[:n], C_accs[:n])
            ci = ps.get("ci95_pp", [None, None])
            ps["equivalent_within_3pp"] = bool(
                ci[0] is not None and ci[1] is not None
                and float(ci[0]) >= -3.0 and float(ci[1]) <= 3.0
            )
            paired_DC = ps
            if ps["equivalent_within_3pp"]:
                eq_count += 1

        if len(D_accs) >= 2 and len(R_accs) >= 2:
            n  = min(len(D_accs), len(R_accs))
            ps = paired_stats(D_accs[:n], R_accs[:n])
            ps["beats_random"] = bool(
                ps.get("p_one_sided") is not None and float(ps["p_one_sided"]) < 0.05
            )
            paired_DR = ps
            if ps["beats_random"]:
                mech_count += 1

        if len(D_accs) >= 2 and len(B_accs) >= 2:
            n  = min(len(D_accs), len(B_accs))
            paired_DB = paired_stats(D_accs[:n], B_accs[:n])

        cells[ckey] = {
            "floor":           floor_val,
            "B":               arm_stats("B"),
            "C":               arm_stats("C"),
            "D":               arm_stats("D"),
            "RANDOM":          arm_stats("RANDOM"),
            "paired_D_C":      paired_DC,
            "paired_D_RANDOM": paired_DR,
            "paired_D_B":      paired_DB,
        }

    # Seeds done = min across reset arms (ignoring A_floor)
    seed_counts = [
        len(cell_accs.get((d, a), {}).get(arm, []))
        for d in DATASET_CFG for a in ARCH_NAMES for arm in RESET_ARMS
    ]
    n_seeds_done = min(seed_counts) if seed_counts else 0

    return {
        "status":      status,
        "scale":       "full",
        "data_source": "cifar_real",
        "seeds_per_cell": n_seeds_done,
        "cells":       cells,
        "summary": {
            "equivalence_holds_in": f"{eq_count}/{n_cells} cells",
            "mechanism_holds_in":   f"{mech_count}/{n_cells} cells",
            "datasets":             list(DATASET_CFG.keys()),
            "architectures":        ARCH_NAMES,
            "n_cells_done":         n_cells,
            "n_cells_total":        len(DATASET_CFG) * len(ARCH_NAMES),
        },
        "subject_executed": (
            f"CIFAR-100 ({DATASET_CFG['cifar100']['n_tasks']}T×"
            f"{DATASET_CFG['cifar100']['cls_per_task']}cls) + "
            f"CIFAR-10 ({DATASET_CFG['cifar10']['n_tasks']}T×"
            f"{DATASET_CFG['cifar10']['cls_per_task']}cls) × "
            f"ConvNet+MLP3 × A_floor/B/C/D/RANDOM × "
            f"{STEPS_PER_TASK} steps/task, "
            f"reset_every={RESET_EVERY}, frac={RESET_FRAC}, "
            f"seeds={SEEDS[:n_seeds_done]}"
        ),
        "notes": (
            "Does Adam v_t zero-cost plasticity reset generalize across "
            "datasets (CIFAR-100 4T, CIFAR-10 2T) and architectures (ConvNet, MLP3)?"
        ),
    }


# ─── I/O helpers ─────────────────────────────────────────────────────────────
def write_results(payload):
    tmp = RESULTS_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2, cls=NumpyEncoder)
    os.replace(tmp, RESULTS_PATH)
    print(f"  [io] RESULTS.json written  status={payload['status']}", flush=True)


def append_row(row):
    with open(ROWS_PATH, "a") as f:
        f.write(json.dumps(row, cls=NumpyEncoder) + "\n")


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    t_global = time.time()

    print("\n[main] Loading datasets...", flush=True)
    raw_data = {
        "cifar100": load_cifar100(),
        "cifar10":  load_cifar10(),
    }
    print("[main] Datasets loaded.\n", flush=True)

    # Accumulators
    cell_accs  = {(d, a): {arm: [] for arm in RESET_ARMS} for d in DATASET_CFG for a in ARCH_NAMES}
    cell_meta  = {(d, a): {arm: [] for arm in RESET_ARMS} for d in DATASET_CFG for a in ARCH_NAMES}
    floor_accs = {(d, a): [] for d in DATASET_CFG for a in ARCH_NAMES}
    floor_meta = {(d, a): [] for d in DATASET_CFG for a in ARCH_NAMES}

    run_count = 0
    aborted   = False

    for dataset_name, dcfg in DATASET_CFG.items():
        X_tr, y_tr, X_te, y_te = raw_data[dataset_name]
        n_tasks      = dcfg["n_tasks"]
        cls_per_task = dcfg["cls_per_task"]
        num_classes  = dcfg["num_classes"]

        print(f"\n{'='*68}", flush=True)
        print(f"DATASET: {dataset_name.upper()}  "
              f"({n_tasks} tasks × {cls_per_task} cls = {n_tasks*cls_per_task} / {num_classes} classes)", flush=True)
        print(f"{'='*68}", flush=True)

        for arch_name in ARCH_NAMES:
            n_pen = N_PEN[arch_name]
            print(f"\n{'─'*56}", flush=True)
            print(f"ARCH: {arch_name}  (n_pen={n_pen})", flush=True)
            print(f"{'─'*56}", flush=True)

            # ── A_floor: run ONCE per (dataset, arch), seed=0 ─────────────────
            elapsed = (time.time() - t_global) / 60.0
            if elapsed >= MAX_WALL_MIN:
                print(f"[TIME GUARD] {elapsed:.1f} min elapsed; skipping floor", flush=True)
                aborted = True
                break

            print(f"\n  A_floor (seed=0)", flush=True)
            floor_splits = make_task_splits(
                X_tr, y_tr, X_te, y_te,
                seed=0, n_tasks=n_tasks,
                cls_per_task=cls_per_task, num_classes=num_classes,
            )
            run_count += 1
            print(f"\n  [run {run_count}] dataset={dataset_name} arch={arch_name} arm=A_floor seed=0", flush=True)
            result = run_arm_seed("A_floor", 0, floor_splits, arch_name, num_classes, n_pen)
            floor_accs[(dataset_name, arch_name)].append(result["primary_acc"])
            floor_meta[(dataset_name, arch_name)].append({
                "dead_final": result["dead_final"], "erank_final": result["erank_final"]
            })
            append_row({
                "dataset": dataset_name, "arch": arch_name, "arm": "A_floor", "seed": 0,
                **{k: result[k] for k in ("primary_acc", "per_task_acc", "dead_final", "erank_final", "wall_s")},
            })
            print(f"  → primary_acc={result['primary_acc']:.4f}  "
                  f"dead={result['dead_final']:.3f}  "
                  f"erank={result['erank_final']:.2f}  "
                  f"wall={result['wall_s']:.1f}s", flush=True)

            # ── Reset arms — 6 seeds, PAIRED ─────────────────────────────────
            for seed in SEEDS:
                elapsed = (time.time() - t_global) / 60.0
                if elapsed >= MAX_WALL_MIN:
                    print(f"[TIME GUARD] {elapsed:.1f} min ≥ {MAX_WALL_MIN} min; stopping", flush=True)
                    aborted = True
                    break

                print(f"\n  SEED {seed}", flush=True)
                splits = make_task_splits(
                    X_tr, y_tr, X_te, y_te,
                    seed=seed, n_tasks=n_tasks,
                    cls_per_task=cls_per_task, num_classes=num_classes,
                )

                for arm in RESET_ARMS:
                    run_count += 1
                    print(f"\n  [run {run_count}/{total_runs}] "
                          f"dataset={dataset_name} arch={arch_name} arm={arm} seed={seed}",
                          flush=True)

                    result = run_arm_seed(arm, seed, splits, arch_name, num_classes, n_pen)

                    cell_accs[(dataset_name, arch_name)][arm].append(result["primary_acc"])
                    cell_meta[(dataset_name, arch_name)][arm].append({
                        "dead_final": result["dead_final"], "erank_final": result["erank_final"]
                    })
                    append_row({
                        "dataset": dataset_name, "arch": arch_name, "arm": arm, "seed": seed,
                        **{k: result[k] for k in
                           ("primary_acc", "per_task_acc", "dead_final", "erank_final", "wall_s")},
                    })
                    print(f"  → primary_acc={result['primary_acc']:.4f}  "
                          f"dead={result['dead_final']:.3f}  "
                          f"erank={result['erank_final']:.2f}  "
                          f"wall={result['wall_s']:.1f}s", flush=True)

                # Incremental save after each seed block
                elapsed = (time.time() - t_global) / 60.0
                payload = aggregate_results(
                    cell_accs, cell_meta, floor_accs, floor_meta, status="RUNNING"
                )
                payload["elapsed_min"] = round(elapsed, 2)
                write_results(payload)
                print(f"  [seed done] elapsed={elapsed:.1f}min", flush=True)

                if aborted:
                    break

            # Write after finishing all seeds for this (dataset, arch) block
            elapsed = (time.time() - t_global) / 60.0
            payload = aggregate_results(
                cell_accs, cell_meta, floor_accs, floor_meta, status="RUNNING"
            )
            payload["elapsed_min"] = round(elapsed, 2)
            write_results(payload)
            print(f"  [arch done] {dataset_name}/{arch_name} elapsed={elapsed:.1f}min", flush=True)

            if aborted:
                break

        if aborted:
            break

    # ── Finalize ──────────────────────────────────────────────────────────────
    elapsed = (time.time() - t_global) / 60.0
    print(f"\n{'='*68}", flush=True)
    if aborted:
        print(f"EXPERIMENT ABORTED BY TIME GUARD after {elapsed:.1f} min", flush=True)
    else:
        print(f"ALL DONE — Total wall time: {elapsed:.1f} min", flush=True)

    final_status = "ABORTED" if aborted else "SUCCESS"

    # Check minimum seeds
    seed_counts = [
        len(cell_accs.get((d, a), {}).get(arm, []))
        for d in DATASET_CFG for a in ARCH_NAMES for arm in RESET_ARMS
    ]
    min_seeds = min(seed_counts) if seed_counts else 0
    if min_seeds < MIN_SEEDS:
        print(f"  WARNING: only {min_seeds} seeds completed (< {MIN_SEEDS} minimum)", flush=True)
        if final_status == "SUCCESS":
            final_status = "PARTIAL"

    final = aggregate_results(
        cell_accs, cell_meta, floor_accs, floor_meta, status=final_status
    )
    final["wall_time_min"] = round(elapsed, 2)
    final["run_count"]     = run_count

    # Build top-level metrics summary
    metrics = {}
    for ckey, cval in final["cells"].items():
        dc = cval.get("paired_D_C") or {}
        dr = cval.get("paired_D_RANDOM") or {}
        db = cval.get("paired_D_B") or {}
        metrics[ckey] = {
            "floor":                 cval.get("floor"),
            "D_mean_acc":            (cval["D"] or {}).get("mean"),
            "C_mean_acc":            (cval["C"] or {}).get("mean"),
            "B_mean_acc":            (cval["B"] or {}).get("mean"),
            "RANDOM_mean_acc":       (cval["RANDOM"] or {}).get("mean"),
            "D_C_mean_pp":           dc.get("mean_pp"),
            "D_C_ci95_pp":           dc.get("ci95_pp"),
            "equivalent_within_3pp": dc.get("equivalent_within_3pp"),
            "D_RANDOM_mean_pp":      dr.get("mean_pp"),
            "D_RANDOM_p_one_sided":  dr.get("p_one_sided"),
            "beats_random":          dr.get("beats_random"),
            "D_B_mean_pp":           db.get("mean_pp"),
        }
    final["metrics"] = metrics

    # Print summary table
    print("\n── RESULTS SUMMARY ──", flush=True)
    hdr = f"  {'Cell':30s}  floor  D      C      B      RAND   D-C(pp)  eq?  D>RND(p)"
    print(hdr, flush=True)
    for ckey, m in metrics.items():
        try:
            line = (
                f"  {ckey:30s}  "
                f"{m['floor']:.3f}  "
                f"{m['D_mean_acc']:.3f}  "
                f"{m['C_mean_acc']:.3f}  "
                f"{m['B_mean_acc']:.3f}  "
                f"{m['RANDOM_mean_acc']:.3f}  "
                f"{m['D_C_mean_pp']:+.2f}   "
                f"{'Y' if m['equivalent_within_3pp'] else 'N'}    "
                f"{m['D_RANDOM_p_one_sided']:.4f}"
            )
        except (TypeError, ValueError):
            line = f"  {ckey:30s}  (incomplete)"
        print(line, flush=True)

    print(f"\nEquivalence D≈C: {final['summary']['equivalence_holds_in']}", flush=True)
    print(f"Mechanism D>RND: {final['summary']['mechanism_holds_in']}", flush=True)

    write_results(final)
    print(f"\n[done] RESULTS.json written to {RESULTS_PATH}", flush=True)
    print(json.dumps(final, indent=2, cls=NumpyEncoder), flush=True)


if __name__ == "__main__":
    main()
