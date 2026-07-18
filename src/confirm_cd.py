"""
confirm_cd.py — CONFIRMATORY: Arms C vs D only, 8 seeds, PAIRED analysis.

EXPERIMENT.md ROUND: confirm_cd (full)
  - ONLY arms C and D (B and A already done in prior rounds — do NOT rerun)
  - 8 seeds {0..7}, PAIRED: each seed runs C THEN D on the SAME task-split
  - 5 tasks × 5 classes = 25 classes total
  - 1000 steps/task
  - Write RESULTS.json incrementally after EACH seed's C AND D complete
  - Finalize moment >=6 seeds done

Arm definitions (ONLY utility differs):
  C: empirical diagonal Fisher — extra forward+backward pass
     utility_i = mean_batch[(∂L/∂act_i)²]  (observed labels, empirical Fisher)
  D: Adam v_t proxy — ZERO extra compute
     utility_i = mean(exp_avg_sq[i, :]) for fc1.weight[i, :]

Architecture: SmallConvNetGN (GroupNorm ConvNet, validated in prior rounds)
Data: real CIFAR-100, /opt/datasets, download=False, ABORT if missing
Eval: masked (only task-current classes)

Output: results/confirm_cd/RESULTS.json (incremental)
        results/confirm_cd/run.log (captured externally)
"""

import os, sys, json, time, random, pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats as spstats

RESULTS_DIR  = "/workspace/results/confirm_cd"
RESULTS_PATH = os.path.join(RESULTS_DIR, "RESULTS.json")
WEIGHTS_DIR  = "/workspace/_weights"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(WEIGHTS_DIR, exist_ok=True)

# ── Hyper-parameters (EXPERIMENT.md: 5 tasks × ~1000 steps) ─────────────────
NUM_CLASSES    = 100
TASKS          = 5            # 5 tasks per EXPERIMENT.md
CLS_PER_TASK   = 5            # 5 classes/task → 25 classes total
STEPS_PER_TASK = 1000         # ~1000 steps/task per EXPERIMENT.md
BATCH_SIZE     = 128
LR             = 1e-3
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEEDS          = list(range(8))   # {0..7}
ARMS           = ["C", "D"]       # ONLY C and D in this round

# CBP reset hyperparameters (SAME for both arms — only utility differs)
RESET_EVERY  = 100
RESET_FRAC   = 0.10
FC1_UNITS    = 512

DATASETS_DIR = os.environ.get("SH_DATASETS_DIR", "/opt/datasets")

print(f"Device: {DEVICE}", flush=True)
print(f"Datasets: {DATASETS_DIR}", flush=True)
print(f"Seeds: {SEEDS}  Tasks: {TASKS}  Steps/task: {STEPS_PER_TASK}", flush=True)
print(f"Arms: {ARMS}  (C=Fisher, D=Adam-v_t)", flush=True)


# ── Model: GroupNorm ConvNet (identical to all prior rounds) ─────────────────
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
        self._last_fc1_act = act.detach()
        if store_pen:
            self._penultimate = act.detach()
        return self.fc2(act)

    def forward_with_act_grad(self, x):
        """Return (logits, act) where act has grad enabled (for Fisher utility)."""
        h   = self.features(x).view(x.size(0), -1)
        pre = self.fc1(h)
        act = F.relu(pre)
        act.retain_grad()
        logits = self.fc2(act)
        return logits, act

    def count_params(self):
        return sum(p.numel() for p in self.parameters())


# ── Data loading ─────────────────────────────────────────────────────────────
def load_real_cifar100():
    try:
        import torchvision.datasets as tvds
        tr_ds = tvds.CIFAR100(root=DATASETS_DIR, train=True,  download=False)
        te_ds = tvds.CIFAR100(root=DATASETS_DIR, train=False, download=False)
        X_tr = np.array(tr_ds.data, dtype=np.float32).transpose(0, 3, 1, 2) / 255.0
        y_tr = np.array(tr_ds.targets, dtype=np.int64)
        X_te = np.array(te_ds.data, dtype=np.float32).transpose(0, 3, 1, 2) / 255.0
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
    SAME seed → SAME splits for both C and D — critical for PAIRED design.
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
def utility_C(model, x_batch, y_batch, criterion):
    """
    Arm C: empirical diagonal Fisher — extra forward+backward pass.
    utility_i = mean_batch[(∂L/∂act_i)²]  (empirical Fisher, observed labels)
    One extra backward pass per reset check.
    """
    was_training = model.training
    model.train()
    model.zero_grad()
    logits, act = model.forward_with_act_grad(x_batch)
    loss = criterion(logits, y_batch)
    loss.backward()
    utility = torch.zeros(FC1_UNITS)
    if act.grad is not None:
        utility = (act.grad.detach() ** 2).mean(0).cpu()
    model.zero_grad()
    if not was_training:
        model.eval()
    return utility


def utility_D(model, optimizer):
    """
    Arm D: Adam exp_avg_sq proxy — ZERO extra compute.
    utility_i = mean(exp_avg_sq[i, :]) for fc1.weight[i, :]
    Reads already-stored Adam state; no extra passes needed.
    """
    param = model.fc1.weight
    state = optimizer.state.get(param, {})
    if 'exp_avg_sq' not in state:
        return torch.zeros(FC1_UNITS)
    vt = state['exp_avg_sq'].detach().cpu()   # [FC1_UNITS, fan_in]
    return vt.mean(dim=1)                      # [FC1_UNITS]


# ── CBP Reset Procedure ───────────────────────────────────────────────────────
def cbp_reset(model, optimizer, utility, reset_frac=RESET_FRAC):
    """
    Continual Backprop reset (Dohare et al. 2024):
      - Bottom reset_frac of fc1 neurons by utility → kaiming reinit
      - Outgoing fc2 weights → 0  (output continuity)
      - Adam state for affected rows/cols → 0
    """
    n_reset = max(1, int(FC1_UNITS * reset_frac))
    _, reset_idx = torch.topk(utility, n_reset, largest=False)

    with torch.no_grad():
        fan_in = model.fc1.weight.shape[1]
        bound  = float(np.sqrt(3.0) * np.sqrt(2.0 / fan_in))
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
    """Run one (arm, seed) combination. Returns per-task metrics dict."""
    print(f"\n{'='*60}", flush=True)
    print(f"ARM {arm} | SEED {seed}", flush=True)
    print(f"{'='*60}", flush=True)

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

    model     = SmallConvNetGN(num_output=NUM_CLASSES).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    # SAME seed → SAME task-split for C and D — CRITICAL for paired analysis
    splits, _ = make_task_splits(X_tr, y_tr, X_te, y_te, seed=seed, n_tasks=TASKS)

    per_task_acc   = []
    per_task_dead  = []
    per_task_erank = []
    t_start = time.time()

    for tid, (tr_ldr, te_ldr, cls_list) in enumerate(splits):
        t_task = time.time()
        model.train()
        tr_it  = iter(tr_ldr)
        losses = []
        resets_this_task = 0

        # Measure reset-utility compute time
        reset_times_c = []
        reset_times_d = []

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

            # CBP reset check every RESET_EVERY steps
            if step % RESET_EVERY == 0:
                t_util = time.time()
                if arm == 'C':
                    util = utility_C(model, x, y, criterion)
                    reset_times_c.append(time.time() - t_util)
                elif arm == 'D':
                    util = utility_D(model, optimizer)
                    reset_times_d.append(time.time() - t_util)
                else:
                    raise ValueError(f"Unknown arm: {arm}")

                cbp_reset(model, optimizer, util)
                resets_this_task += 1
                model.train()

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
    # Final task (task 5) accuracy — and also mean over tasks 2-5 (last 4)
    final_task_acc = per_task_acc[-1]                             # task 5 acc
    mean_tasks_2to5 = float(np.mean(per_task_acc[-4:]))           # mean tasks 2-5

    return {
        "arm":            arm,
        "seed":           seed,
        "per_task_acc":   per_task_acc,
        "per_task_dead":  per_task_dead,
        "per_task_erank": per_task_erank,
        "final_task_acc": final_task_acc,  # task-5 accuracy
        "mean_tasks_2to5": mean_tasks_2to5, # mean over tasks 2-5
        "dead_final":     per_task_dead[-1],
        "erank_final":    per_task_erank[-1],
        "wall_sec":       wall_total,
    }


# ── Paired statistical analysis ───────────────────────────────────────────────
def paired_analysis_dc(c_results, d_results, equiv_margin_pp=3.0):
    """
    Compute paired difference D-C using mean_tasks_2to5 per seed.
    Returns per-seed diffs, mean_pp, ci95_pp, t_p_value, equivalent_within_3pp.
    """
    # Use mean over tasks 2-5 as the per-seed accuracy metric
    c_acc = np.array([r["mean_tasks_2to5"] for r in c_results]) * 100  # to pp
    d_acc = np.array([r["mean_tasks_2to5"] for r in d_results]) * 100  # to pp
    diff  = d_acc - c_acc   # D - C

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
        "per_seed":          [round(float(d), 3) for d in diff],
        "mean_pp":           round(mean_diff, 3),
        "std_pp":            round(std_diff, 3) if not np.isnan(std_diff) else None,
        "ci95_pp":           [round(ci_lo, 3), round(ci_hi, 3)],
        "t_stat":            round(float(t_stat), 4) if not np.isnan(t_stat) else None,
        "t_p_value":         round(float(p_val), 6) if not np.isnan(p_val) else None,
        "n_seeds":           n,
        "equivalent_within_3pp": equivalent,
    }


def build_arm_summary(arm_results):
    """Build per-arm summary dict."""
    per_seed_acc = [r["mean_tasks_2to5"] for r in arm_results]  # proportion
    dead_vals    = [r["dead_final"]       for r in arm_results]
    erank_vals   = [r["erank_final"]      for r in arm_results]
    return {
        "per_seed_acc": [round(v, 4) for v in per_seed_acc],
        "mean":         round(float(np.mean(per_seed_acc)), 4),
        "std":          round(float(np.std(per_seed_acc, ddof=1)) if len(per_seed_acc) > 1 else 0.0, 4),
        "dead_final":   round(float(np.mean(dead_vals)), 4),
        "erank_final":  round(float(np.nanmean(erank_vals)), 2),
        "per_task_acc_mean":   np.mean([r["per_task_acc"]   for r in arm_results], axis=0).tolist(),
        "per_task_dead_mean":  np.mean([r["per_task_dead"]  for r in arm_results], axis=0).tolist(),
        "per_task_erank_mean": np.nanmean([r["per_task_erank"] for r in arm_results], axis=0).tolist(),
        "seeds_done":   [r["seed"] for r in arm_results],
    }


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    t_start = time.time()

    # Stub result (written immediately so run is visible even if cut)
    results = {
        "status":         "RUNNING",
        "scale":          "full",
        "data_source":    "cifar100_real",
        "n_seeds":        0,
        "seeds_complete": [],
        "C":              {"per_seed_acc": [], "seeds_done": []},
        "D":              {"per_seed_acc": [], "seeds_done": []},
        "paired_D_minus_C": {},
        "notes":          "Run in progress...",
    }
    write_results(results)

    # ── Load real CIFAR-100 ───────────────────────────────────────────────────
    print(f"\nLoading real CIFAR-100 from {DATASETS_DIR}", flush=True)
    train_data, test_data = load_real_cifar100()
    if train_data is None:
        msg = f"ABORT: Real CIFAR-100 not found at {DATASETS_DIR}. Cannot proceed."
        print(msg, flush=True)
        results.update({"status": "FAILED", "notes": msg})
        write_results(results)
        sys.exit(1)

    X_tr, y_tr = train_data
    X_te, y_te = test_data
    print(f"Data loaded: train={X_tr.shape} test={X_te.shape}", flush=True)
    print(f"Model params: {SmallConvNetGN(100).count_params():,}", flush=True)

    # ── PAIRED: for each seed, run C then D on the SAME task-split ───────────
    c_results = []
    d_results = []

    for seed in SEEDS:
        elapsed_min = (time.time() - t_start) / 60
        print(f"\n[{elapsed_min:.1f}m elapsed] === SEED {seed} ===", flush=True)

        # Run arm C first
        print(f"\n[{elapsed_min:.1f}m] Running ARM C, seed {seed}", flush=True)
        sr_c = run_arm_seed("C", seed, X_tr, y_tr, X_te, y_te)
        c_results.append(sr_c)

        # Run arm D on the SAME seed (same task-split)
        elapsed_min = (time.time() - t_start) / 60
        print(f"\n[{elapsed_min:.1f}m] Running ARM D, seed {seed}", flush=True)
        sr_d = run_arm_seed("D", seed, X_tr, y_tr, X_te, y_te)
        d_results.append(sr_d)

        # ── Incremental write after BOTH C and D complete for this seed ───────
        c_summary = build_arm_summary(c_results)
        d_summary = build_arm_summary(d_results)
        paired    = paired_analysis_dc(c_results, d_results)

        n_complete = len(c_results)
        results.update({
            "status":         "RUNNING",
            "n_seeds":        n_complete,
            "seeds_complete": [r["seed"] for r in c_results],
            "C":              c_summary,
            "D":              d_summary,
            "paired_D_minus_C": paired,
            "latest": {
                "seed": seed,
                "C_acc": sr_c["mean_tasks_2to5"],
                "D_acc": sr_d["mean_tasks_2to5"],
                "D_minus_C_pp": (sr_d["mean_tasks_2to5"] - sr_c["mean_tasks_2to5"]) * 100,
            },
        })
        write_results(results)

        print(
            f"\n  >> Seed {seed} done: "
            f"C={sr_c['mean_tasks_2to5']:.3f} D={sr_d['mean_tasks_2to5']:.3f} "
            f"D-C={(sr_d['mean_tasks_2to5']-sr_c['mean_tasks_2to5'])*100:+.2f}pp",
            flush=True,
        )
        print(
            f"  >> Running paired D-C: mean={paired['mean_pp']:+.2f}pp  "
            f"CI=[{paired['ci95_pp'][0]:.2f},{paired['ci95_pp'][1]:.2f}]  "
            f"p={paired['t_p_value']}  n={n_complete}",
            flush=True,
        )

        # Early finalization if >=6 seeds done (per EXPERIMENT.md)
        if n_complete >= 6:
            elapsed_min = (time.time() - t_start) / 60
            print(f"\n[{elapsed_min:.1f}m] {n_complete} seeds complete — "
                  f"priority finalization check (spec says >=6 is sufficient)", flush=True)

    # ── Final summary ─────────────────────────────────────────────────────────
    n_seeds = len(c_results)
    c_summary = build_arm_summary(c_results)
    d_summary = build_arm_summary(d_results)
    paired    = paired_analysis_dc(c_results, d_results)

    equiv_3pp = paired["equivalent_within_3pp"]
    d_c_diff  = paired["mean_pp"]
    d_c_ci    = paired["ci95_pp"]
    d_c_pval  = paired["t_p_value"]

    if equiv_3pp:
        honest_answer = (
            f"YES — v_t IS a statistically equivalent zero-cost substitute for explicit Fisher. "
            f"Paired D-C diff = {d_c_diff:+.2f}pp (95% CI [{d_c_ci[0]:.2f},{d_c_ci[1]:.2f}]), "
            f"p={d_c_pval}. CI fully within ±3pp equivalence margin. "
            f"Adam's exp_avg_sq can replace the extra backward pass with no measurable accuracy cost."
        )
    else:
        ci_width = d_c_ci[1] - d_c_ci[0]
        if ci_width > 6:
            honest_answer = (
                f"INCONCLUSIVE — CI too wide ({ci_width:.1f}pp) to declare equivalence. "
                f"Point estimate D-C = {d_c_diff:+.2f}pp (CI [{d_c_ci[0]:.2f},{d_c_ci[1]:.2f}]). "
                f"More seeds or steps needed."
            )
        else:
            honest_answer = (
                f"NO — v_t is NOT statistically equivalent within ±3pp. "
                f"Paired D-C diff = {d_c_diff:+.2f}pp (CI [{d_c_ci[0]:.2f},{d_c_ci[1]:.2f}]), "
                f"p={d_c_pval}. CI not fully within ±3pp; real difference exists."
            )

    total_wall_min = (time.time() - t_start) / 60
    notes = (
        f"confirm_cd: {n_seeds} seeds, {TASKS} tasks×{CLS_PER_TASK} cls, "
        f"{STEPS_PER_TASK} steps/task, reset_every={RESET_EVERY}, reset_frac={RESET_FRAC}. "
        f"C=explicit-Fisher, D=Adam-v_t. "
        + honest_answer
        + f" Wall: {total_wall_min:.1f}m."
    )

    final = {
        "status":        "SUCCESS",
        "scale":         "full",
        "data_source":   "cifar100_real",
        "n_seeds":       n_seeds,
        "seeds_complete": [r["seed"] for r in c_results],
        "C": c_summary,
        "D": d_summary,
        "paired_D_minus_C": paired,
        "wall_time_min": round(total_wall_min, 2),
        "subject_executed": (
            f"Arms C (explicit Fisher) and D (Adam v_t), PAIRED, "
            f"SmallConvNetGN, real CIFAR-100, "
            f"{TASKS}×{CLS_PER_TASK} classes, {STEPS_PER_TASK} steps/task, "
            f"reset_frac={RESET_FRAC}, reset_every={RESET_EVERY}, seeds={SEEDS}"
        ),
        "notes": notes,
        "metrics": {
            "C_mean_acc":           c_summary["mean"],
            "C_std_acc":            c_summary["std"],
            "C_dead_final":         c_summary["dead_final"],
            "D_mean_acc":           d_summary["mean"],
            "D_std_acc":            d_summary["std"],
            "D_dead_final":         d_summary["dead_final"],
            "D_minus_C_mean_pp":    paired["mean_pp"],
            "D_minus_C_ci95_pp":    paired["ci95_pp"],
            "D_minus_C_t_p_value":  paired["t_p_value"],
            "equivalent_within_3pp": paired["equivalent_within_3pp"],
            "B_reference_mean_acc": 0.811,    # from prior method_arms (n=3)
            "A_floor_acc":          0.200,    # from baseline_A3
        },
    }

    write_results(final)

    print(f"\n{'='*60}", flush=True)
    print(f"DONE in {total_wall_min:.1f} min", flush=True)
    print(f"\nArm C: mean={c_summary['mean']:.4f} ± {c_summary['std']:.4f}", flush=True)
    print(f"Arm D: mean={d_summary['mean']:.4f} ± {d_summary['std']:.4f}", flush=True)
    print(f"Paired D-C: {paired['mean_pp']:+.2f}pp  CI=[{paired['ci95_pp'][0]:.2f},{paired['ci95_pp'][1]:.2f}]  p={paired['t_p_value']}", flush=True)
    print(f"Equivalent within 3pp: {equiv_3pp}", flush=True)
    print(f"\n{honest_answer}", flush=True)
    print(f"\nFull results at: {RESULTS_PATH}", flush=True)
    print(json.dumps(final, indent=2), flush=True)


if __name__ == "__main__":
    main()
