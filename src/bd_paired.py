"""
bd_paired.py — PAIRED B vs D run (EXPERIMENT.md round: bd_paired, full scale)

PURPOSE:
  Is Adam v_t (free) as good as the original CBP contribution heuristic?

TWO ARMS (only utility criterion differs; everything else is identical):
  B: CBP contribution heuristic = |outgoing weight| × mean post-activation
     (running average). Reset lowest-utility neurons.
  D: Adam exp_avg_sq (v_t) aggregated per neuron (mean over fc1.weight[i, :] fan).
     Reset lowest. Zero extra compute.

DESIGN:
  - 8 seeds {0..7}, PAIRED: same task split for B and D at each seed
  - 4 tasks × 5 classes/task = 20 classes (class-incremental)
  - ~1000 steps/task
  - GroupNorm ConvNet (same as all prior rounds)
  - Real CIFAR-100 from /opt/datasets (download=False; ABORT if missing)
  - Masked eval (current task classes only)
  - Adam lr=1e-3, reset_every=100 steps, reset_frac=0.10

HEADLINE METRICS:
  - Per-arm mean±std final-task acc
  - Paired D-B mean_pp, 95% CI, equivalent_within_3pp?
  - Is v_t a drop-in for the CBP heuristic?

OUTPUT: results/bd_paired/RESULTS.json (incremental, finalize at >=6 seeds)
        results/bd_paired/run.log (captured externally)
        results/bd_paired/LOG.md (decision log)
"""

import os, sys, json, time, random, pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats as spstats

RESULTS_DIR  = "/workspace/results/bd_paired"
RESULTS_PATH = os.path.join(RESULTS_DIR, "RESULTS.json")
WEIGHTS_DIR  = "/workspace/_weights"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(WEIGHTS_DIR, exist_ok=True)

# ── Hyper-parameters ──────────────────────────────────────────────────────────
NUM_CLASSES    = 100
TASKS          = 4            # 4 tasks per EXPERIMENT.md spec
CLS_PER_TASK   = 5            # 5 classes/task → 20 classes total
STEPS_PER_TASK = 1000         # ~1000 steps/task (~20 min per EXPERIMENT.md)
BATCH_SIZE     = 128
LR             = 1e-3
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEEDS          = list(range(8))   # {0..7}
ARMS           = ["B", "D"]       # ONLY B and D in this round

# CBP reset hyperparameters (SAME for both arms — only utility differs)
RESET_EVERY  = 100
RESET_FRAC   = 0.10
FC1_UNITS    = 512

DATASETS_DIR = os.environ.get("SH_DATASETS_DIR", "/opt/datasets")

print(f"Device: {DEVICE}", flush=True)
print(f"Datasets: {DATASETS_DIR}", flush=True)
print(f"Seeds: {SEEDS}  Tasks: {TASKS}  Steps/task: {STEPS_PER_TASK}", flush=True)
print(f"Arms: {ARMS}  (B=CBP-heuristic, D=Adam-v_t)", flush=True)


# ── Model: GroupNorm ConvNet (identical to all prior rounds) ──────────────────
class SmallConvNetGN(nn.Module):
    """3 conv-blocks + 2 FC with GroupNorm. No BN running stats → no collapse."""
    def __init__(self, num_output=100):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.GroupNorm(8, 64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.GroupNorm(8, 64), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.GroupNorm(8, 128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.GroupNorm(8, 128), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.GroupNorm(8, 256), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.GroupNorm(8, 256), nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc1 = nn.Linear(256 * 4 * 4, FC1_UNITS)
        self.fc2 = nn.Linear(FC1_UNITS, num_output)
        self._penultimate  = None
        self._last_fc1_act = None

    def forward(self, x, store_pen=False):
        h   = self.features(x).view(x.size(0), -1)
        pre = self.fc1(h)
        act = F.relu(pre)
        self._last_fc1_act = act.detach()   # stored for Arm B running mean
        if store_pen:
            self._penultimate = act.detach()
        return self.fc2(act)

    def count_params(self):
        return sum(p.numel() for p in self.parameters())


# ── Data loading (download=False; ABORT if missing) ───────────────────────────
def load_real_cifar100():
    """Load CIFAR-100 from /opt/datasets. Never download."""
    try:
        import torchvision.datasets as tvds
        tr_ds = tvds.CIFAR100(root=DATASETS_DIR, train=True,  download=False)
        te_ds = tvds.CIFAR100(root=DATASETS_DIR, train=False, download=False)
        X_tr = np.array(tr_ds.data,    dtype=np.float32).transpose(0, 3, 1, 2) / 255.0
        y_tr = np.array(tr_ds.targets, dtype=np.int64)
        X_te = np.array(te_ds.data,    dtype=np.float32).transpose(0, 3, 1, 2) / 255.0
        y_te = np.array(te_ds.targets, dtype=np.int64)
        print(f"[torchvision] Loaded CIFAR-100: train={X_tr.shape}, test={X_te.shape}", flush=True)
    except Exception as e:
        print(f"[torchvision] Failed: {e}. Trying pickle...", flush=True)
        data_dir = os.path.join(DATASETS_DIR, "cifar-100-python")
        if not os.path.isdir(data_dir):
            return None, None
        def unpickle(f):
            with open(f, "rb") as fo:
                return pickle.load(fo, encoding="bytes")
        try:
            tr = unpickle(os.path.join(data_dir, "train"))
            te = unpickle(os.path.join(data_dir, "test"))
            X_tr = tr[b"data"].reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
            y_tr = np.array(tr[b"fine_labels"], dtype=np.int64)
            X_te = te[b"data"].reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
            y_te = np.array(te[b"fine_labels"], dtype=np.int64)
            print(f"[pickle] Loaded CIFAR-100: train={X_tr.shape}", flush=True)
        except Exception as e2:
            print(f"[pickle] Failed: {e2}", flush=True)
            return None, None

    # Normalize with CIFAR-100 channel stats
    mean = np.array([0.5071, 0.4867, 0.4408], dtype=np.float32)[:, None, None]
    std  = np.array([0.2675, 0.2565, 0.2761], dtype=np.float32)[:, None, None]
    X_tr = (X_tr - mean) / std
    X_te = (X_te - mean) / std
    return (X_tr, y_tr), (X_te, y_te)


class ArrayDataset(torch.utils.data.Dataset):
    def __init__(self, X, y, augment=False):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y).long()
        self.augment = augment
    def __len__(self): return len(self.y)
    def __getitem__(self, i):
        x = self.X[i]
        if self.augment and random.random() > 0.5:
            x = x.flip(-1)
        return x, self.y[i]


def make_task_splits(X_tr, y_tr, X_te, y_te, seed=0, n_tasks=TASKS):
    """
    Split 100 classes into n_tasks groups of CLS_PER_TASK.
    SAME seed → SAME splits for B and D — CRITICAL for paired analysis.
    """
    rng = random.Random(seed)
    all_cls = list(range(NUM_CLASSES))
    rng.shuffle(all_cls)
    task_classes = [all_cls[i*CLS_PER_TASK:(i+1)*CLS_PER_TASK] for i in range(n_tasks)]
    splits = []
    for cls_list in task_classes:
        cls_arr = np.array(cls_list)
        tr_m  = np.isin(y_tr, cls_arr)
        te_m  = np.isin(y_te, cls_arr)
        tr_ds = ArrayDataset(X_tr[tr_m], y_tr[tr_m], augment=True)
        te_ds = ArrayDataset(X_te[te_m], y_te[te_m], augment=False)
        tr_ldr = torch.utils.data.DataLoader(
            tr_ds, BATCH_SIZE, shuffle=True, drop_last=True, num_workers=2, pin_memory=True)
        te_ldr = torch.utils.data.DataLoader(
            te_ds, 256, shuffle=False, num_workers=2, pin_memory=True)
        splits.append((tr_ldr, te_ldr, cls_list))
    return splits, task_classes


# ── Metrics ───────────────────────────────────────────────────────────────────
def compute_accuracy(model, loader, cls_list):
    """Masked evaluation: argmax restricted to current task's classes."""
    model.eval()
    correct = total = 0
    cls_t = torch.tensor(cls_list, device=DEVICE)
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x)
            mask = torch.full_like(logits, float('-inf'))
            mask[:, cls_t] = logits[:, cls_t]
            correct += (mask.argmax(1) == y).sum().item()
            total   += y.size(0)
    return correct / total if total else 0.0


def dead_unit_fraction(model, loader, thr=0.01, n=10):
    model.eval()
    acts = []
    with torch.no_grad():
        for i, (x, _) in enumerate(loader):
            if i >= n: break
            model(x.to(DEVICE), store_pen=True)
            acts.append(model._penultimate.cpu())
    if not acts: return 0.0
    A = torch.cat(acts, 0)
    return (A.abs().mean(0) < thr).float().mean().item()


def effective_rank(model, loader, n=20):
    model.eval()
    acts = []
    with torch.no_grad():
        for i, (x, _) in enumerate(loader):
            if i >= n: break
            model(x.to(DEVICE), store_pen=True)
            acts.append(model._penultimate.cpu())
    if not acts: return float('nan')
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


# ── CBP Utility Functions ─────────────────────────────────────────────────────
def utility_B(model, running_act_mean):
    """
    Arm B: |mean outgoing weight| × running mean activation.
    utility_i = mean_j(|fc2.weight[j, i]|) × running_act_mean[i]
    This is the original CBP contribution heuristic (Dohare et al. 2024).
    """
    with torch.no_grad():
        out_w_mag = model.fc2.weight.detach().abs().mean(0).cpu()   # [FC1_UNITS]
        utility   = out_w_mag * running_act_mean.cpu()               # elementwise
    return utility   # [FC1_UNITS]


def utility_D(model, optimizer):
    """
    Arm D: Adam exp_avg_sq proxy — ZERO extra compute.
    utility_i = mean(exp_avg_sq[i, :]) for fc1.weight[i, :]
    Just reads already-stored Adam state; no extra passes.
    """
    param = model.fc1.weight
    state = optimizer.state.get(param, {})
    if 'exp_avg_sq' not in state:
        # Adam not yet initialized (first step) → return zeros
        return torch.zeros(FC1_UNITS)
    vt = state['exp_avg_sq'].detach().cpu()   # [FC1_UNITS, fan_in]
    return vt.mean(dim=1)                      # [FC1_UNITS]


# ── CBP Reset Procedure (Dohare et al. 2024) ─────────────────────────────────
def cbp_reset(model, optimizer, utility, reset_frac=RESET_FRAC):
    """
    Identify bottom reset_frac of fc1 neurons by utility and reset them:
      - fc1.weight[i, :] → kaiming_uniform init
      - fc2.weight[:, i] → 0  (output continuity preservation)
      - fc1.bias[i]      → 0
      - Adam state for affected rows/cols → 0  (fresh optimizer start)
    """
    n_reset = max(1, int(FC1_UNITS * reset_frac))
    _, reset_idx = torch.topk(utility, n_reset, largest=False)

    with torch.no_grad():
        fan_in = model.fc1.weight.shape[1]
        bound  = float(np.sqrt(3.0) * np.sqrt(2.0 / fan_in))   # kaiming uniform
        new_w  = torch.empty(n_reset, fan_in).uniform_(-bound, bound)
        model.fc1.weight.data[reset_idx] = new_w.to(DEVICE)
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
    return reset_idx


# ── Result I/O ────────────────────────────────────────────────────────────────
def write_results(payload):
    tmp = RESULTS_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, RESULTS_PATH)


# ── Single (arm, seed) run ────────────────────────────────────────────────────
def run_arm_seed(arm, seed, X_tr, y_tr, X_te, y_te):
    """Run one (arm, seed) combination and return per-task metrics."""
    print(f"\n{'='*60}", flush=True)
    print(f"ARM {arm} | SEED {seed}", flush=True)
    print(f"{'='*60}", flush=True)

    # Deterministic RNG
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

    model     = SmallConvNetGN(num_output=NUM_CLASSES).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    # SAME seed → SAME task-split for B and D — CRITICAL for paired analysis
    splits, _ = make_task_splits(X_tr, y_tr, X_te, y_te, seed=seed, n_tasks=TASKS)

    per_task_acc   = []
    per_task_dead  = []
    per_task_erank = []
    t_start = time.time()

    # Arm B: running mean activation (EMA with α=0.01, persisted across tasks)
    running_act_mean = torch.zeros(FC1_UNITS)

    for tid, (tr_ldr, te_ldr, cls_list) in enumerate(splits):
        t_task = time.time()
        model.train()
        tr_it  = iter(tr_ldr)
        losses = []
        resets_this_task = 0

        for step in range(1, STEPS_PER_TASK + 1):
            try:
                x, y = next(tr_it)
            except StopIteration:
                tr_it = iter(tr_ldr)
                x, y = next(tr_it)
            x, y = x.to(DEVICE), y.to(DEVICE)

            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

            # Update running activation mean (Arm B only)
            if arm == 'B':
                with torch.no_grad():
                    alpha = 0.01
                    running_act_mean = (
                        (1 - alpha) * running_act_mean
                        + alpha * model._last_fc1_act.mean(0).cpu()
                    )

            # CBP reset check every RESET_EVERY steps
            if step % RESET_EVERY == 0:
                if arm == 'B':
                    util = utility_B(model, running_act_mean)
                elif arm == 'D':
                    util = utility_D(model, optimizer)
                else:
                    raise ValueError(f"Unknown arm: {arm}")

                cbp_reset(model, optimizer, util)
                resets_this_task += 1
                model.train()   # ensure train mode after reset

        wall = time.time() - t_task
        acc   = compute_accuracy(model, te_ldr, cls_list)
        duf   = dead_unit_fraction(model, te_ldr)
        erank = effective_rank(model, te_ldr)
        per_task_acc.append(acc)
        per_task_dead.append(duf)
        per_task_erank.append(erank)

        print(
            f"  [{arm}/s{seed}] Task {tid+1:2d}/{TASKS} | "
            f"acc={acc:.3f} dead={duf:.3f} erank={erank:.2f} "
            f"resets={resets_this_task} loss={np.mean(losses[-200:]):.4f} wall={wall:.1f}s",
            flush=True,
        )

    wall_total = time.time() - t_start
    # For 4-task runs:
    #   - "final_task_acc" = task-4 acc
    #   - per-arm summary uses mean over tasks 2-4 (tasks 1 excluded as warm-up)
    final_task_acc   = per_task_acc[-1]
    mean_tasks_2to4  = float(np.mean(per_task_acc[1:]))   # tasks 2,3,4 (0-indexed: [1:])

    return {
        "arm":             arm,
        "seed":            seed,
        "per_task_acc":    per_task_acc,
        "per_task_dead":   per_task_dead,
        "per_task_erank":  per_task_erank,
        "final_task_acc":  final_task_acc,
        "mean_tasks_2to4": mean_tasks_2to4,
        "dead_final":      per_task_dead[-1],
        "erank_final":     per_task_erank[-1],
        "wall_sec":        wall_total,
    }


# ── Paired statistical analysis ───────────────────────────────────────────────
def paired_analysis_db(b_results, d_results, equiv_margin_pp=3.0):
    """
    Compute paired difference D-B using mean_tasks_2to4 per seed.
    Returns per-seed diffs, mean_pp, 95% CI, t-test p-value, equivalent_within_3pp.
    """
    b_acc = np.array([r["mean_tasks_2to4"] for r in b_results]) * 100   # → pp
    d_acc = np.array([r["mean_tasks_2to4"] for r in d_results]) * 100   # → pp
    diff  = d_acc - b_acc   # D - B (positive = D better)

    n         = len(diff)
    mean_diff = float(np.mean(diff))
    std_diff  = float(np.std(diff, ddof=1)) if n > 1 else float('nan')
    se        = std_diff / np.sqrt(n) if n > 1 else float('nan')

    if n >= 2:
        t_stat, p_val = spstats.ttest_1samp(diff, 0.0)
        t_crit = float(spstats.t.ppf(0.975, df=n - 1))
        ci_lo  = mean_diff - t_crit * se
        ci_hi  = mean_diff + t_crit * se
    else:
        t_stat, p_val = float('nan'), float('nan')
        ci_lo = ci_hi = float('nan')

    equivalent = bool(ci_lo >= -equiv_margin_pp and ci_hi <= equiv_margin_pp)

    return {
        "per_seed":              [round(float(d), 3) for d in diff],
        "mean_pp":               round(mean_diff, 3),
        "std_pp":                round(std_diff, 3) if not np.isnan(std_diff) else None,
        "ci95_pp":               [round(ci_lo, 3), round(ci_hi, 3)],
        "t_stat":                round(float(t_stat), 4) if not np.isnan(t_stat) else None,
        "t_p_value":             round(float(p_val), 6) if not np.isnan(p_val) else None,
        "n_seeds":               n,
        "equivalent_within_3pp": equivalent,
    }


def build_arm_summary(arm_results):
    """Build per-arm summary dict from list of per-seed result dicts."""
    per_seed_acc  = [r["mean_tasks_2to4"] for r in arm_results]
    dead_vals     = [r["dead_final"]       for r in arm_results]
    erank_vals    = [r["erank_final"]      for r in arm_results]
    n = len(per_seed_acc)
    return {
        "per_seed_acc":        [round(v, 4) for v in per_seed_acc],
        "mean":                round(float(np.mean(per_seed_acc)), 4),
        "std":                 round(float(np.std(per_seed_acc, ddof=1)) if n > 1 else 0.0, 4),
        "dead_final":          round(float(np.mean(dead_vals)), 4),
        "erank_final":         round(float(np.nanmean(erank_vals)), 2),
        "per_task_acc_mean":   np.mean([r["per_task_acc"]   for r in arm_results], axis=0).tolist(),
        "per_task_dead_mean":  np.mean([r["per_task_dead"]  for r in arm_results], axis=0).tolist(),
        "per_task_erank_mean": np.nanmean([r["per_task_erank"] for r in arm_results], axis=0).tolist(),
        "seeds_done":          [r["seed"] for r in arm_results],
    }


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    t_start = time.time()

    # Stub result (written immediately)
    results = {
        "status":           "RUNNING",
        "scale":            "full",
        "data_source":      "cifar100_real",
        "n_seeds":          0,
        "seeds_complete":   [],
        "B":                {"per_seed_acc": [], "seeds_done": []},
        "D":                {"per_seed_acc": [], "seeds_done": []},
        "paired_D_minus_B": {},
        "notes":            "Run in progress...",
    }
    write_results(results)

    # ── Load real CIFAR-100 ───────────────────────────────────────────────────
    print(f"\nLoading real CIFAR-100 from {DATASETS_DIR}", flush=True)
    train_data, test_data = load_real_cifar100()
    if train_data is None:
        msg = f"ABORT: Real CIFAR-100 not found at {DATASETS_DIR}."
        print(msg, flush=True)
        results.update({"status": "FAILED", "notes": msg})
        write_results(results)
        sys.exit(1)

    X_tr, y_tr = train_data
    X_te, y_te = test_data
    print(f"Data loaded: train={X_tr.shape} test={X_te.shape}", flush=True)
    print(f"Model params: {SmallConvNetGN(100).count_params():,}", flush=True)

    # ── PAIRED: for each seed, run B then D on the SAME task-split ───────────
    b_results = []
    d_results = []

    for seed in SEEDS:
        elapsed_min = (time.time() - t_start) / 60
        print(f"\n[{elapsed_min:.1f}m elapsed] === SEED {seed} ===", flush=True)

        # Run arm B first
        print(f"\n[{elapsed_min:.1f}m] Running ARM B (CBP heuristic), seed {seed}", flush=True)
        sr_b = run_arm_seed("B", seed, X_tr, y_tr, X_te, y_te)
        b_results.append(sr_b)

        # Run arm D on the SAME seed (same task-split — PAIRED)
        elapsed_min = (time.time() - t_start) / 60
        print(f"\n[{elapsed_min:.1f}m] Running ARM D (Adam v_t), seed {seed}", flush=True)
        sr_d = run_arm_seed("D", seed, X_tr, y_tr, X_te, y_te)
        d_results.append(sr_d)

        # ── Incremental write after BOTH B and D complete for this seed ───────
        b_summary = build_arm_summary(b_results)
        d_summary = build_arm_summary(d_results)
        paired    = paired_analysis_db(b_results, d_results)

        n_complete = len(b_results)
        results.update({
            "status":           "RUNNING",
            "n_seeds":          n_complete,
            "seeds_complete":   [r["seed"] for r in b_results],
            "B":                b_summary,
            "D":                d_summary,
            "paired_D_minus_B": paired,
            "latest": {
                "seed":          seed,
                "B_acc":         sr_b["mean_tasks_2to4"],
                "D_acc":         sr_d["mean_tasks_2to4"],
                "D_minus_B_pp": (sr_d["mean_tasks_2to4"] - sr_b["mean_tasks_2to4"]) * 100,
            },
        })
        write_results(results)

        print(
            f"\n  >> Seed {seed} done: "
            f"B={sr_b['mean_tasks_2to4']:.3f} D={sr_d['mean_tasks_2to4']:.3f} "
            f"D-B={(sr_d['mean_tasks_2to4']-sr_b['mean_tasks_2to4'])*100:+.2f}pp",
            flush=True,
        )
        print(
            f"  >> Running paired D-B: mean={paired['mean_pp']:+.2f}pp  "
            f"CI=[{paired['ci95_pp'][0]:.2f},{paired['ci95_pp'][1]:.2f}]  "
            f"p={paired['t_p_value']}  n={n_complete}",
            flush=True,
        )

        # Early finalization note at >=6 seeds (per EXPERIMENT.md)
        if n_complete >= 6:
            elapsed_min = (time.time() - t_start) / 60
            print(
                f"\n[{elapsed_min:.1f}m] {n_complete} seeds complete — "
                f"meets >=6 seed threshold for finalization",
                flush=True,
            )

    # ── Final summary ─────────────────────────────────────────────────────────
    n_seeds   = len(b_results)
    b_summary = build_arm_summary(b_results)
    d_summary = build_arm_summary(d_results)
    paired    = paired_analysis_db(b_results, d_results)

    equiv_3pp = paired["equivalent_within_3pp"]
    d_b_diff  = paired["mean_pp"]
    d_b_ci    = paired["ci95_pp"]
    d_b_pval  = paired["t_p_value"]

    if equiv_3pp:
        verdict_note = (
            f"YES — v_t IS statistically equivalent to the CBP contribution heuristic within ±3pp. "
            f"Paired D-B diff = {d_b_diff:+.2f}pp (95% CI [{d_b_ci[0]:.2f},{d_b_ci[1]:.2f}]), "
            f"p={d_b_pval}. CI fully within ±3pp margin. "
            f"Adam's exp_avg_sq is a zero-cost drop-in for the CBP heuristic."
        )
    else:
        ci_width = d_b_ci[1] - d_b_ci[0]
        if ci_width > 6:
            verdict_note = (
                f"INCONCLUSIVE — CI too wide ({ci_width:.1f}pp) to declare equivalence. "
                f"Point estimate D-B = {d_b_diff:+.2f}pp (CI [{d_b_ci[0]:.2f},{d_b_ci[1]:.2f}]). "
                f"More seeds needed."
            )
        else:
            verdict_note = (
                f"NO — v_t is NOT statistically equivalent to the CBP heuristic within ±3pp. "
                f"Paired D-B diff = {d_b_diff:+.2f}pp (CI [{d_b_ci[0]:.2f},{d_b_ci[1]:.2f}]), "
                f"p={d_b_pval}. CI not fully within ±3pp."
            )

    total_wall_min = (time.time() - t_start) / 60
    notes = (
        f"bd_paired: {n_seeds} seeds, {TASKS} tasks×{CLS_PER_TASK} cls, "
        f"{STEPS_PER_TASK} steps/task, reset_every={RESET_EVERY}, reset_frac={RESET_FRAC}. "
        f"B=CBP-heuristic, D=Adam-v_t. "
        + verdict_note
        + f" Wall: {total_wall_min:.1f}m."
    )

    final = {
        "status":       "SUCCESS",
        "n_seeds":      n_seeds,
        "data_source":  "cifar100_real",
        "B": {
            "per_seed_acc": [round(v * 100, 4) for v in b_summary["per_seed_acc"]],
            "mean":         round(b_summary["mean"] * 100, 4),
            "std":          round(b_summary["std"]  * 100, 4),
            "dead_final":   b_summary["dead_final"],
            "erank_final":  b_summary["erank_final"],
            "per_task_acc_mean":   b_summary["per_task_acc_mean"],
        },
        "D": {
            "per_seed_acc": [round(v * 100, 4) for v in d_summary["per_seed_acc"]],
            "mean":         round(d_summary["mean"] * 100, 4),
            "std":          round(d_summary["std"]  * 100, 4),
            "dead_final":   d_summary["dead_final"],
            "erank_final":  d_summary["erank_final"],
            "per_task_acc_mean":   d_summary["per_task_acc_mean"],
        },
        "paired_D_minus_B": {
            "per_seed":              paired["per_seed"],
            "mean_pp":               paired["mean_pp"],
            "ci95_pp":               paired["ci95_pp"],
            "equivalent_within_3pp": paired["equivalent_within_3pp"],
            "t_stat":                paired["t_stat"],
            "t_p_value":             paired["t_p_value"],
            "n_seeds":               n_seeds,
        },
        "wall_time_min":    round(total_wall_min, 2),
        "subject_executed": (
            f"Arms B (CBP heuristic) and D (Adam v_t), PAIRED, "
            f"SmallConvNetGN, real CIFAR-100, "
            f"{TASKS}×{CLS_PER_TASK} classes, {STEPS_PER_TASK} steps/task, "
            f"reset_frac={RESET_FRAC}, reset_every={RESET_EVERY}, seeds={SEEDS}"
        ),
        "notes": notes,
    }

    write_results(final)

    print(f"\n{'='*60}", flush=True)
    print(f"DONE in {total_wall_min:.1f} min", flush=True)
    print(f"\nArm B (CBP-heuristic): mean={b_summary['mean']*100:.2f}% ± {b_summary['std']*100:.2f}%", flush=True)
    print(f"Arm D (Adam-v_t):      mean={d_summary['mean']*100:.2f}% ± {d_summary['std']*100:.2f}%", flush=True)
    print(f"Paired D-B: {paired['mean_pp']:+.2f}pp  CI=[{paired['ci95_pp'][0]:.2f},{paired['ci95_pp'][1]:.2f}]  p={paired['t_p_value']}", flush=True)
    print(f"Equivalent within 3pp: {equiv_3pp}", flush=True)
    print(f"\n{verdict_note}", flush=True)
    print(f"\nFull results at: {RESULTS_PATH}", flush=True)
    print(json.dumps(final, indent=2), flush=True)


if __name__ == "__main__":
    main()
