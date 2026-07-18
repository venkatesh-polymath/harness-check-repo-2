"""
run_grid.py — Scale-out experiment: generalize Adam-v_t plasticity result.

EXPERIMENT.md round: scale_grid (full)
GRID:
  Datasets : CIFAR-100 (5 cls/task × 10 tasks) and CIFAR-10 (2 cls/task × 5 tasks)
  Architectures: (1) SmallConvNetGN, (2) MLP3GN, (3) TinyResNetGN
  Arms: A_floor (no reset), B (heuristic |w|×act), C (explicit Fisher),
        D (Adam v_t), RANDOM
  Seeds: 8. PAIRED per seed — same task split for all arms.
  800 steps/task; Adam lr=1e-3; masked eval.

KEY OPTIMIZATION vs v1: data preloaded as GPU tensors → eliminates DataLoader
cycling overhead on small per-task datasets (18s→~2s per task).

Writes results/scale_grid/RESULTS.json INCREMENTALLY after every (dataset,arch,seed).
Also writes results/scale_grid/rows.jsonl (one row per (dataset,arch,arm,seed)).
"""
import os, sys, json, time, random, pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats as spstats


class NumpyEncoder(json.JSONEncoder):
    """Handle numpy scalars/bools that standard json can't serialize."""
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.bool_): return bool(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super().default(obj)

# ─── Output dirs ─────────────────────────────────────────────────────────────
RESULTS_DIR  = "/workspace/results/scale_grid"
RESULTS_PATH = os.path.join(RESULTS_DIR, "RESULTS.json")
ROWS_PATH    = os.path.join(RESULTS_DIR, "rows.jsonl")
WEIGHTS_DIR  = "/workspace/_weights"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(WEIGHTS_DIR, exist_ok=True)

# ─── Global hyperparameters ──────────────────────────────────────────────────
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE     = 128
LR             = 1e-3
RESET_EVERY    = 100      # steps between reset sweeps
RESET_FRAC     = 0.10     # fraction of neurons to reset
STEPS_PER_TASK = 800      # within 800-1000 per spec
DATASETS_DIR   = os.environ.get("SH_DATASETS_DIR", "/opt/datasets")

# Penultimate layer sizes per architecture
N_PEN = {"convnet": 512, "mlp3": 512, "resnet": 256}

# Dataset configs
DATASET_CFG = {
    "cifar100": {
        "n_tasks": 10, "cls_per_task": 5, "num_classes": 100,
        "mean": [0.5071, 0.4867, 0.4408], "std": [0.2675, 0.2565, 0.2761],
    },
    "cifar10": {
        "n_tasks": 5, "cls_per_task": 2, "num_classes": 10,
        "mean": [0.4914, 0.4822, 0.4465], "std": [0.2470, 0.2435, 0.2616],
    },
}

ARCH_NAMES = ["convnet", "mlp3", "resnet"]
ARMS       = ["A_floor", "B", "C", "D", "RANDOM"]
SEEDS      = list(range(8))

print(f"[init] Device: {DEVICE}", flush=True)
print(f"[init] Grid: {list(DATASET_CFG)} × {ARCH_NAMES} × {ARMS} × {len(SEEDS)} seeds", flush=True)
print(f"[init] steps/task={STEPS_PER_TASK}  reset_every={RESET_EVERY}  frac={RESET_FRAC}", flush=True)
total_runs = len(DATASET_CFG) * len(ARCH_NAMES) * len(ARMS) * len(SEEDS)
print(f"[init] Total training runs: {total_runs}", flush=True)


# ─── Data loading ─────────────────────────────────────────────────────────────
def load_cifar100():
    """Load CIFAR-100 as numpy arrays (download=False, ABORT if missing)."""
    dcfg = DATASET_CFG["cifar100"]
    try:
        import torchvision.datasets as tvds
        tr_ds = tvds.CIFAR100(root=DATASETS_DIR, train=True,  download=False)
        te_ds = tvds.CIFAR100(root=DATASETS_DIR, train=False, download=False)
        X_tr = np.array(tr_ds.data, dtype=np.float32).transpose(0, 3, 1, 2) / 255.0
        y_tr = np.array(tr_ds.targets, dtype=np.int64)
        X_te = np.array(te_ds.data, dtype=np.float32).transpose(0, 3, 1, 2) / 255.0
        y_te = np.array(te_ds.targets, dtype=np.int64)
        print(f"[cifar100] torchvision: {X_tr.shape}", flush=True)
    except Exception:
        data_dir = os.path.join(DATASETS_DIR, "cifar-100-python")
        if not os.path.isdir(data_dir):
            sys.exit(f"ABORT: CIFAR-100 not found at {data_dir}")
        def unpickle(f):
            with open(f, "rb") as fo: return pickle.load(fo, encoding="bytes")
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
        X_tr = np.array(tr_ds.data, dtype=np.float32).transpose(0, 3, 1, 2) / 255.0
        y_tr = np.array(tr_ds.targets, dtype=np.int64)
        X_te = np.array(te_ds.data, dtype=np.float32).transpose(0, 3, 1, 2) / 255.0
        y_te = np.array(te_ds.targets, dtype=np.int64)
        print(f"[cifar10] torchvision: {X_tr.shape}", flush=True)
    except Exception:
        data_dir = os.path.join(DATASETS_DIR, "cifar-10-batches-py")
        if not os.path.isdir(data_dir):
            sys.exit(f"ABORT: CIFAR-10 not found at {data_dir}")
        def unpickle(f):
            with open(f, "rb") as fo: return pickle.load(fo, encoding="bytes")
        batches = [unpickle(os.path.join(data_dir, f"data_batch_{i}")) for i in range(1, 6)]
        X_tr = np.concatenate([b[b"data"] for b in batches], 0).reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
        y_tr = np.concatenate([b[b"labels"] for b in batches]).astype(np.int64)
        te = unpickle(os.path.join(data_dir, "test_batch"))
        X_te = te[b"data"].reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
        y_te = np.array(te[b"labels"], dtype=np.int64)
        print(f"[cifar10] pickle: {X_tr.shape}", flush=True)
    m = np.array(dcfg["mean"], dtype=np.float32)[:, None, None]
    s = np.array(dcfg["std"],  dtype=np.float32)[:, None, None]
    return (X_tr - m) / s, y_tr, (X_te - m) / s, y_te


# ─── Task splits — GPU-preloaded tensors ──────────────────────────────────────
def make_task_splits(X_tr, y_tr, X_te, y_te, seed, n_tasks, cls_per_task, num_classes):
    """
    Split num_classes into n_tasks groups of cls_per_task.
    SAME seed → SAME splits for ALL arms — PAIRED design.
    Returns list of (tr_X_gpu, tr_y_gpu, te_X_gpu, te_y_gpu, cls_list).
    Data is loaded onto GPU for fast batch sampling.
    """
    rng = random.Random(seed)
    all_cls = list(range(num_classes))
    rng.shuffle(all_cls)
    task_classes = [all_cls[i*cls_per_task:(i+1)*cls_per_task] for i in range(n_tasks)]
    splits = []
    for cls_list in task_classes:
        cls_arr = np.array(cls_list)
        tr_m = np.isin(y_tr, cls_arr)
        te_m = np.isin(y_te, cls_arr)
        # Train: GPU tensor (for fast batch sampling)
        tr_X = torch.from_numpy(X_tr[tr_m]).to(DEVICE)
        tr_y = torch.from_numpy(y_tr[tr_m]).long().to(DEVICE)
        # Test: CPU tensor (less GPU memory pressure; evaluation is quick)
        te_X = torch.from_numpy(X_te[te_m])
        te_y = torch.from_numpy(y_te[te_m]).long()
        splits.append((tr_X, tr_y, te_X, te_y, cls_list))
    return splits


# ─── GroupNorm helper for MLP ─────────────────────────────────────────────────
def gn1d(gn_module, x):
    """Apply GroupNorm to 1D linear output: (N, C) → (N, C, 1) → gn → (N, C)."""
    return gn_module(x.unsqueeze(-1)).squeeze(-1)


# ─── Architecture 1: SmallConvNetGN ──────────────────────────────────────────
class SmallConvNetGN(nn.Module):
    """3 conv-blocks + 2 FC with GroupNorm. Validated in all prior rounds."""
    def __init__(self, num_output=100):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.GroupNorm(8, 64),  nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.GroupNorm(8, 64), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.GroupNorm(8, 128), nn.ReLU(),
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


# ─── Architecture 2: MLP3GN — 3-hidden-layer MLP with GroupNorm ──────────────
class MLP3GN(nn.Module):
    def __init__(self, num_output=100, hidden=512, groups=8):
        super().__init__()
        self.l1  = nn.Linear(3 * 32 * 32, hidden)
        self.gn1 = nn.GroupNorm(groups, hidden)
        self.l2  = nn.Linear(hidden, hidden)
        self.gn2 = nn.GroupNorm(groups, hidden)
        self.fc1 = nn.Linear(hidden, hidden)   # ← penultimate
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


# ─── Architecture 3: TinyResNetGN — 4 residual blocks with GroupNorm ─────────
class ResBlockGN(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, groups=8):
        super().__init__()
        g = min(groups, out_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.gn1   = nn.GroupNorm(g, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.gn2   = nn.GroupNorm(g, out_ch)
        self.skip  = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.skip = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.GroupNorm(g, out_ch)
            )

    def forward(self, x):
        out = F.relu(self.gn1(self.conv1(x)))
        out = self.gn2(self.conv2(out))
        return F.relu(out + self.skip(x))


class TinyResNetGN(nn.Module):
    """4 residual blocks (2 per stage 64ch→128ch), GroupNorm, FC-256 penultimate."""
    def __init__(self, num_output=100, pen=256):
        super().__init__()
        self.stem   = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1, bias=False),
            nn.GroupNorm(8, 64), nn.ReLU()
        )
        self.layer1 = nn.Sequential(ResBlockGN(64, 64), ResBlockGN(64, 64))
        self.layer2 = nn.Sequential(ResBlockGN(64, 128, stride=2), ResBlockGN(128, 128))
        self.pool   = nn.AdaptiveAvgPool2d(1)
        self.fc1    = nn.Linear(128, pen)   # ← penultimate
        self.fc2    = nn.Linear(pen, num_output)
        self._penultimate  = None
        self._last_fc1_act = None

    def forward(self, x, store_pen=False):
        h   = self.pool(self.layer2(self.layer1(self.stem(x)))).view(x.size(0), -1)
        pre = self.fc1(h)
        act = F.relu(pre)
        self._last_fc1_act = act.detach()
        if store_pen:
            self._penultimate = act.detach()
        return self.fc2(act)

    def forward_with_act_grad(self, x):
        h   = self.pool(self.layer2(self.layer1(self.stem(x)))).view(x.size(0), -1)
        pre = self.fc1(h)
        act = F.relu(pre)
        act.retain_grad()
        return self.fc2(act), act


def make_model(arch_name, num_classes):
    if arch_name == "convnet":
        return SmallConvNetGN(num_output=num_classes).to(DEVICE)
    elif arch_name == "mlp3":
        return MLP3GN(num_output=num_classes).to(DEVICE)
    elif arch_name == "resnet":
        return TinyResNetGN(num_output=num_classes).to(DEVICE)
    raise ValueError(f"Unknown arch: {arch_name}")


# ─── GPU batch sampler ────────────────────────────────────────────────────────
def sample_train_batch(tr_X, tr_y, augment=True):
    """
    Fast random batch from pre-loaded GPU tensors.
    No DataLoader overhead — just index into VRAM.
    """
    idx = torch.randint(0, tr_X.size(0), (BATCH_SIZE,), device=DEVICE)
    x   = tr_X[idx]
    if augment:
        mask = torch.rand(x.size(0), device=DEVICE) > 0.5
        if mask.any():
            x = x.clone()
            x[mask] = x[mask].flip(-1)
    return x, tr_y[idx]


# ─── Metrics (operate on pre-loaded tensors) ─────────────────────────────────
def compute_accuracy(model, te_X, te_y, cls_list):
    """Masked accuracy on test tensors."""
    model.eval()
    cls_t   = torch.tensor(cls_list, device=DEVICE)
    correct = total = 0
    with torch.no_grad():
        for i in range(0, te_X.size(0), 256):
            x = te_X[i:i+256].to(DEVICE)
            y = te_y[i:i+256].to(DEVICE)
            logits = model(x)
            mask   = torch.full_like(logits, float('-inf'))
            mask[:, cls_t] = logits[:, cls_t]
            correct += (mask.argmax(1) == y).sum().item()
            total   += y.size(0)
    return correct / total if total else 0.0


def dead_unit_fraction(model, te_X, te_y, thr=0.01, max_samples=2048):
    """Fraction of penultimate units with mean activation < thr."""
    model.eval()
    acts = []
    with torch.no_grad():
        for i in range(0, min(max_samples, te_X.size(0)), 256):
            x = te_X[i:i+256].to(DEVICE)
            model(x, store_pen=True)
            acts.append(model._penultimate.cpu())
    A = torch.cat(acts, 0)
    return (A.abs().mean(0) < thr).float().mean().item()


def effective_rank(model, te_X, te_y, max_samples=2000):
    """Effective rank of penultimate activations via spectral entropy."""
    model.eval()
    acts = []
    with torch.no_grad():
        for i in range(0, min(max_samples, te_X.size(0)), 256):
            x = te_X[i:i+256].to(DEVICE)
            model(x, store_pen=True)
            acts.append(model._penultimate.cpu())
    A = torch.cat(acts, 0).float()
    if A.abs().max() < 1e-10: return 0.0
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
    """CBP heuristic: |out_weight| × EMA mean_post_activation."""
    out_w = model.fc2.weight.data.abs().mean(0).cpu()[:n_pen]
    rm    = arm_state.get('running_mean_act', torch.ones(n_pen))
    return out_w * rm


def utility_C(model, x_batch, y_batch, criterion, n_pen):
    """Explicit empirical-Fisher — extra forward+backward pass."""
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
    """Adam exp_avg_sq proxy — ZERO extra compute."""
    param = model.fc1.weight
    state = optimizer.state.get(param, {})
    if 'exp_avg_sq' not in state:
        return torch.zeros(n_pen)
    vt = state['exp_avg_sq'].detach().cpu()
    return vt.mean(dim=1)[:n_pen]


def utility_RANDOM(n_pen, rand_state):
    """Random utility (for mechanistic control arm)."""
    return torch.from_numpy(rand_state.rand(n_pen).astype(np.float32))


# ─── CBP Reset ────────────────────────────────────────────────────────────────
def cbp_reset(model, optimizer, utility, n_pen):
    """
    Reset bottom RESET_FRAC of penultimate units:
    - incoming fc1 weights → Kaiming reinit
    - outgoing fc2 weights → 0 (output continuity)
    - Adam state for affected rows/cols → 0
    """
    n_reset = max(1, int(n_pen * RESET_FRAC))
    _, reset_idx = torch.topk(utility, n_reset, largest=False)

    with torch.no_grad():
        fan_in = model.fc1.weight.shape[1]
        bound  = float(np.sqrt(3.0) * np.sqrt(2.0 / fan_in))
        model.fc1.weight.data[reset_idx] = torch.empty(
            n_reset, fan_in, device=DEVICE).uniform_(-bound, bound)
        if model.fc1.bias is not None:
            model.fc1.bias.data[reset_idx] = 0.0
        model.fc2.weight.data[:, reset_idx] = 0.0

    def _rst_rows(param, rows):
        s = optimizer.state.get(param, {})
        if 'exp_avg' in s:
            s['exp_avg'][rows]    = 0.0
            s['exp_avg_sq'][rows] = 0.0

    def _rst_cols(param, cols):
        s = optimizer.state.get(param, {})
        if 'exp_avg' in s:
            s['exp_avg'][:, cols]    = 0.0
            s['exp_avg_sq'][:, cols] = 0.0

    _rst_rows(model.fc1.weight, reset_idx)
    if model.fc1.bias is not None:
        _rst_rows(model.fc1.bias, reset_idx)
    _rst_cols(model.fc2.weight, reset_idx)


# ─── Single (arm, seed) training run ─────────────────────────────────────────
def run_arm_seed(arm, seed, splits, arch_name, num_classes, n_pen):
    """
    Run one (arm, seed) combination on given task splits.
    Returns: primary_acc, per_task_acc, per_task_dead, per_task_erank,
             dead_final, erank_final, wall_s.
    """
    t_run_start = time.time()

    # Reproducible seeding per (arm, seed) — identical seeds → identical model init
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

    model     = make_model(arch_name, num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    # RANDOM arm: separate deterministic RNG (independent of model training RNG)
    rand_state = np.random.RandomState(1000 + seed)

    # Arm B: EMA running mean of activations
    arm_state = {}

    n_tasks = len(splits)
    per_task_acc   = []
    per_task_dead  = []
    per_task_erank = []

    for tid, (tr_X, tr_y, te_X, te_y, cls_list) in enumerate(splits):
        t_task = time.time()
        model.train()
        losses = []

        for step in range(1, STEPS_PER_TASK + 1):
            x, y = sample_train_batch(tr_X, tr_y, augment=True)

            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

            # Update running mean for Arm B
            if arm == 'B' and model._last_fc1_act is not None:
                ca = model._last_fc1_act.abs().mean(0).cpu()
                if 'running_mean_act' not in arm_state:
                    arm_state['running_mean_act'] = ca.clone()
                else:
                    arm_state['running_mean_act'].mul_(0.99).add_(0.01 * ca)

            # CBP reset every RESET_EVERY steps
            if arm != 'A_floor' and step % RESET_EVERY == 0:
                if arm == 'B':
                    util = utility_B(model, arm_state, n_pen)
                elif arm == 'C':
                    util = utility_C(model, x, y, criterion, n_pen)
                elif arm == 'D':
                    util = utility_D(model, optimizer, n_pen)
                elif arm == 'RANDOM':
                    util = utility_RANDOM(n_pen, rand_state)
                else:
                    raise ValueError(f"Unknown arm: {arm}")
                cbp_reset(model, optimizer, util, n_pen)
                model.train()

        wall = time.time() - t_task
        acc   = compute_accuracy(model, te_X, te_y, cls_list)
        duf   = dead_unit_fraction(model, te_X, te_y)
        er    = effective_rank(model, te_X, te_y)
        per_task_acc.append(round(float(acc),  4))
        per_task_dead.append(round(float(duf), 4))
        per_task_erank.append(round(float(er), 4) if er == er else 0.0)

        print(
            f"  [{arm:8s}/s{seed}/{arch_name:7s}] T{tid+1:2d}/{n_tasks} "
            f"acc={acc:.3f} dead={duf:.3f} erank={er:.2f} "
            f"loss={np.mean(losses[-100:]):.4f} wall={wall:.1f}s",
            flush=True,
        )

    total_wall    = time.time() - t_run_start
    # Primary metric: mean over tasks 2..end (exclude task-1 warmup)
    primary_acc   = float(np.mean(per_task_acc[1:]))
    return {
        "primary_acc":   round(primary_acc, 4),
        "per_task_acc":  per_task_acc,
        "per_task_dead": per_task_dead,
        "per_task_erank": per_task_erank,
        "dead_final":    per_task_dead[-1],
        "erank_final":   per_task_erank[-1],
        "wall_s":        round(total_wall, 1),
    }


# ─── Paired statistics ────────────────────────────────────────────────────────
def paired_stats(a_accs, b_accs):
    """Paired D-B diff in pp with 95% CI and one-sided p."""
    diffs = [(av - bv) * 100 for av, bv in zip(a_accs, b_accs)]
    n = len(diffs)
    if n < 2:
        return {"mean_pp": round(np.mean(diffs), 3), "ci95_pp": [None, None],
                "p_one_sided": None, "p_two_sided": None, "n": n,
                "per_seed": [round(d, 3) for d in diffs]}
    t_stat, p_two = spstats.ttest_rel(a_accs, b_accs)
    p_one = p_two / 2 if t_stat > 0 else 1 - p_two / 2
    se    = np.std(diffs, ddof=1) / np.sqrt(n)
    t_c   = spstats.t.ppf(0.975, df=n-1)
    mean_d = float(np.mean(diffs))
    se_f   = float(se)
    t_c_f  = float(t_c)
    return {
        "per_seed":     [round(d, 3) for d in diffs],
        "mean_pp":      round(mean_d, 3),
        "ci95_pp":      [round(mean_d - t_c_f * se_f, 3), round(mean_d + t_c_f * se_f, 3)],
        "p_two_sided":  round(float(p_two), 6),
        "p_one_sided":  round(float(p_one), 6),
        "n":            n,
    }


# ─── Aggregate results ────────────────────────────────────────────────────────
def aggregate_results(cell_accs, cell_meta, status="RUNNING"):
    """Build full RESULTS.json from accumulated per-seed data."""
    cells = {}
    eq_count = mech_count = n_cells = 0

    for (dataset, arch), arm_accs in cell_accs.items():
        if not arm_accs.get("A_floor"):
            continue
        n_cells += 1
        ckey = f"{dataset}_{arch}"

        def arm_stats(arm):
            accs = arm_accs.get(arm, [])
            if not accs: return None
            metas = cell_meta[(dataset, arch)].get(arm, [])
            deads  = [m["dead_final"]  for m in metas if m.get("dead_final")  is not None]
            eranks = [m["erank_final"] for m in metas if m.get("erank_final") is not None]
            return {
                "mean":           round(float(np.mean(accs)), 4),
                "std":            round(float(np.std(accs, ddof=1) if len(accs) > 1 else 0.0), 4),
                "per_seed_acc":   [round(a, 4) for a in accs],
                "n_seeds":        len(accs),
                "dead_final_mean":  round(float(np.mean(deads)),  4) if deads  else None,
                "erank_final_mean": round(float(np.mean(eranks)), 4) if eranks else None,
            }

        floor = round(float(np.mean(arm_accs["A_floor"])), 4) if arm_accs["A_floor"] else None

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
            ps["beats_random"] = bool(ps.get("p_one_sided") is not None and float(ps["p_one_sided"]) < 0.05)
            paired_DR = ps
            if ps["beats_random"]:
                mech_count += 1

        if len(D_accs) >= 2 and len(B_accs) >= 2:
            n  = min(len(D_accs), len(B_accs))
            paired_DB = paired_stats(D_accs[:n], B_accs[:n])

        cells[ckey] = {
            "floor":           floor,
            "B":               arm_stats("B"),
            "C":               arm_stats("C"),
            "D":               arm_stats("D"),
            "RANDOM":          arm_stats("RANDOM"),
            "paired_D_C":      paired_DC,
            "paired_D_RANDOM": paired_DR,
            "paired_D_B":      paired_DB,
        }

    n_done_per_arm = [len(v) for d in cell_accs.values() for v in d.values()]
    n_seeds_done   = max(n_done_per_arm) if n_done_per_arm else 0

    return {
        "status":       status,
        "scale":        "full",
        "data_source":  "cifar_real",
        "n_seeds":      n_seeds_done,
        "seeds":        SEEDS[:n_seeds_done],
        "cells":        cells,
        "summary": {
            "equivalence_holds_in": f"{eq_count}/{n_cells} cells",
            "mechanism_holds_in":   f"{mech_count}/{n_cells} cells",
            "datasets":             list(DATASET_CFG.keys()),
            "architectures":        ARCH_NAMES,
            "n_cells_done":         n_cells,
            "n_cells_total":        len(DATASET_CFG) * len(ARCH_NAMES),
        },
        "subject_executed": (
            f"CIFAR-100 (10T×5cls) + CIFAR-10 (5T×2cls) × "
            f"ConvNet+MLP3+ResNet × A_floor/B/C/D/RANDOM × "
            f"{STEPS_PER_TASK} steps/task, "
            f"reset_every={RESET_EVERY}, frac={RESET_FRAC}, seeds={SEEDS[:n_seeds_done]}"
        ),
        "notes": (
            "Does Adam's v_t zero-cost plasticity reset generalize across "
            "datasets (CIFAR-100, CIFAR-10) and architectures (ConvNet, MLP3, ResNet)?"
        ),
    }


# ─── I/O ─────────────────────────────────────────────────────────────────────
def write_results(payload):
    tmp = RESULTS_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2, cls=NumpyEncoder)
    os.replace(tmp, RESULTS_PATH)
    print(f"  [io] RESULTS.json  status={payload['status']}", flush=True)


def append_row(row):
    with open(ROWS_PATH, "a") as f:
        f.write(json.dumps(row) + "\n")


# ─── Main grid loop ───────────────────────────────────────────────────────────
def main():
    t0 = time.time()

    print("\n[main] Loading datasets...", flush=True)
    raw_data = {
        "cifar100": load_cifar100(),
        "cifar10":  load_cifar10(),
    }
    print("[main] Datasets loaded.\n", flush=True)

    # Accumulated per-cell data
    cell_accs = {}   # (dataset, arch)[arm] → [seed_acc, ...]
    cell_meta = {}   # same but {dead_final, erank_final}
    for dset in DATASET_CFG:
        for arch in ARCH_NAMES:
            cell_accs[(dset, arch)] = {arm: [] for arm in ARMS}
            cell_meta[(dset, arch)] = {arm: [] for arm in ARMS}

    run_count = 0

    for dataset_name, dcfg in DATASET_CFG.items():
        X_tr, y_tr, X_te, y_te = raw_data[dataset_name]
        n_tasks      = dcfg["n_tasks"]
        cls_per_task = dcfg["cls_per_task"]
        num_classes  = dcfg["num_classes"]

        print(f"\n{'='*68}", flush=True)
        print(f"DATASET: {dataset_name.upper()}  "
              f"({n_tasks} tasks × {cls_per_task} cls = {num_classes} classes)", flush=True)
        print(f"{'='*68}", flush=True)

        for arch_name in ARCH_NAMES:
            n_pen = N_PEN[arch_name]
            print(f"\n{'─'*56}", flush=True)
            print(f"ARCH: {arch_name}  (n_pen={n_pen})", flush=True)
            print(f"{'─'*56}", flush=True)

            for seed in SEEDS:
                print(f"\n  SEED {seed}", flush=True)

                # One call — same task splits for ALL arms (PAIRED)
                splits = make_task_splits(
                    X_tr, y_tr, X_te, y_te,
                    seed=seed, n_tasks=n_tasks,
                    cls_per_task=cls_per_task, num_classes=num_classes
                )

                for arm in ARMS:
                    run_count += 1
                    print(f"\n  [{run_count}/{total_runs}] "
                          f"dataset={dataset_name} arch={arch_name} arm={arm} seed={seed}",
                          flush=True)

                    result = run_arm_seed(arm, seed, splits, arch_name, num_classes, n_pen)

                    cell_accs[(dataset_name, arch_name)][arm].append(result["primary_acc"])
                    cell_meta[(dataset_name, arch_name)][arm].append({
                        "dead_final":  result["dead_final"],
                        "erank_final": result["erank_final"],
                    })

                    append_row({
                        "dataset": dataset_name, "arch": arch_name,
                        "arm": arm, "seed": seed,
                        "primary_acc":  result["primary_acc"],
                        "per_task_acc": result["per_task_acc"],
                        "dead_final":   result["dead_final"],
                        "erank_final":  result["erank_final"],
                        "wall_s":       result["wall_s"],
                    })

                    print(f"  → primary_acc={result['primary_acc']:.4f}  "
                          f"dead={result['dead_final']:.3f}  "
                          f"erank={result['erank_final']:.2f}  "
                          f"wall={result['wall_s']:.1f}s", flush=True)

                # Write incremental results after this (dataset, arch, seed)
                elapsed = time.time() - t0
                payload = aggregate_results(cell_accs, cell_meta, status="RUNNING")
                payload["elapsed_min"] = round(elapsed / 60, 2)
                write_results(payload)
                print(f"  [cell done] elapsed={elapsed/60:.1f}min", flush=True)

    # Finalize
    elapsed = time.time() - t0
    print(f"\n{'='*68}", flush=True)
    print(f"ALL DONE — Total wall time: {elapsed/60:.1f} min", flush=True)

    final = aggregate_results(cell_accs, cell_meta, status="SUCCESS")
    final["wall_time_min"] = round(elapsed / 60, 2)
    final["n_seeds"]       = len(SEEDS)

    # Top-level metrics summary
    metrics = {}
    for ckey, cval in final["cells"].items():
        metrics[ckey] = {
            "floor":                 cval.get("floor"),
            "D_mean_acc":            cval["D"]["mean"]      if cval.get("D")      else None,
            "C_mean_acc":            cval["C"]["mean"]      if cval.get("C")      else None,
            "B_mean_acc":            cval["B"]["mean"]      if cval.get("B")      else None,
            "RANDOM_mean_acc":       cval["RANDOM"]["mean"] if cval.get("RANDOM") else None,
            "D_C_mean_pp":           cval["paired_D_C"]["mean_pp"]           if cval.get("paired_D_C") else None,
            "D_C_ci95_pp":           cval["paired_D_C"]["ci95_pp"]           if cval.get("paired_D_C") else None,
            "equivalent_within_3pp": cval["paired_D_C"]["equivalent_within_3pp"] if cval.get("paired_D_C") else None,
            "D_RANDOM_mean_pp":      cval["paired_D_RANDOM"]["mean_pp"]      if cval.get("paired_D_RANDOM") else None,
            "D_RANDOM_p_one_sided":  cval["paired_D_RANDOM"]["p_one_sided"]  if cval.get("paired_D_RANDOM") else None,
            "beats_random":          cval["paired_D_RANDOM"]["beats_random"] if cval.get("paired_D_RANDOM") else None,
        }
    final["metrics"] = metrics

    # Print summary table
    print("\n── RESULTS SUMMARY ──", flush=True)
    print(f"  {'Cell':30s}  floor  D      C      B      RAND   D-C(pp)  eq?  D>RND(p)", flush=True)
    for ckey, m in metrics.items():
        print(
            f"  {ckey:30s}  "
            f"{m['floor']:.3f}  "
            f"{m['D_mean_acc']:.3f}  "
            f"{m['C_mean_acc']:.3f}  "
            f"{m['B_mean_acc']:.3f}  "
            f"{m['RANDOM_mean_acc']:.3f}  "
            f"{m['D_C_mean_pp']:+.2f}   "
            f"{'Y' if m['equivalent_within_3pp'] else 'N'}    "
            f"{m['D_RANDOM_p_one_sided']:.3f}",
            flush=True,
        )

    print(f"\nEquivalence D≈C holds in: {final['summary']['equivalence_holds_in']}", flush=True)
    print(f"Mechanism D>RANDOM holds: {final['summary']['mechanism_holds_in']}", flush=True)

    write_results(final)
    print(f"\n[done] RESULTS.json written to: {RESULTS_PATH}", flush=True)
    print(json.dumps(final, indent=2), flush=True)


if __name__ == "__main__":
    main()
