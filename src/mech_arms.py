"""
mech_arms.py — MECHANISTIC + PAIRED: Four-arm CBP comparison (B/C/D/RANDOM).

EXPERIMENT.md ROUND: mech_arms (full)
  FOUR ARMS on IDENTICAL per-seed task splits (paired), differing ONLY in
  which neurons get reset each cycle:
    B:      CBP heuristic utility (|out weight| × mean post-activation)
    C:      explicit empirical-Fisher utility (mean (dL/da_i)^2 over held-out batch)
    D (OURS): Adam exp_avg_sq (v_t) aggregated per neuron — zero extra compute
    RANDOM: reset a RANDOM subset of units (same cadence + reset fraction as B/C/D;
            utility ignored — mechanistic control)

  6 seeds {0..5}, PAIRED: each seed runs ALL FOUR arms on the SAME task split
  4 tasks × 5 classes = 20 classes class-incremental
  ~1000 steps/task
  Write RESULTS.json INCREMENTALLY after each seed (all 4 arms done)
  Finalize when >=4 seeds done

Architecture: SmallConvNetGN (GroupNorm ConvNet, validated in all prior rounds)
Data: real CIFAR-100, /opt/datasets, download=False, ABORT if missing
Eval: masked (only task-current classes)

RESULTS.json format (per spec):
  {"status":"DONE","n_seeds":<>=4>,"data_source":"cifar100_real",
   "arms":{
     "B":{"per_seed_acc":[...],"mean":..,"std":..,"dead_final":..,"erank_final":..},
     "C":{...},"D":{...},"RANDOM":{...},"A_floor":0.20},
   "paired":{
     "D_minus_C":{"mean_pp":..,"ci95_pp":[..],"equivalent_within_3pp":bool},
     "B_minus_D":{"mean_pp":..,"ci95_pp":[..]},
     "D_minus_RANDOM":{"mean_pp":..,"ci95_pp":[..],"t_p_value":..,"v_t_ranking_beats_random":bool}
   },
   "notes":"..."}
"""

import os, sys, json, time, random, pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats as spstats

RESULTS_DIR  = "/workspace/results/mech_arms"
RESULTS_PATH = os.path.join(RESULTS_DIR, "RESULTS.json")
WEIGHTS_DIR  = "/workspace/_weights"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(WEIGHTS_DIR, exist_ok=True)

# ── Hyper-parameters (EXPERIMENT.md spec) ────────────────────────────────────
NUM_CLASSES    = 100
TASKS          = 4             # 4 tasks/run per spec ("~22s each → 35 min total")
CLS_PER_TASK   = 5             # 5 classes/task → 20 classes total
STEPS_PER_TASK = 1000          # ~1000 steps/task per spec
BATCH_SIZE     = 128
LR             = 1e-3          # Adam, same across all arms
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEEDS          = list(range(6))  # {0..5}
ARMS           = ["B", "C", "D", "RANDOM"]

# CBP reset hyperparameters (IDENTICAL for all four arms — only utility differs)
RESET_EVERY  = 100             # reset every K steps
RESET_FRAC   = 0.10            # reset bottom 10% of units
FC1_UNITS    = 512             # neurons in the penultimate layer

DATASETS_DIR = os.environ.get("SH_DATASETS_DIR", "/opt/datasets")

print(f"Device: {DEVICE}", flush=True)
print(f"Datasets: {DATASETS_DIR}", flush=True)
print(f"Seeds: {SEEDS}  Tasks: {TASKS}×{CLS_PER_TASK}cls  Steps/task: {STEPS_PER_TASK}", flush=True)
print(f"Arms: {ARMS}  reset_every={RESET_EVERY}  reset_frac={RESET_FRAC}", flush=True)
print(f"Estimated wall-time: ~4 arms × 6 seeds × 4 tasks × ~22s ≈ 35 min", flush=True)


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
        self._last_fc1_act = None   # post-ReLU activations (for Arm B running mean)

    def forward(self, x, store_pen=False):
        h   = self.features(x).view(x.size(0), -1)
        pre = self.fc1(h)
        act = F.relu(pre)
        self._last_fc1_act = act.detach()   # always stored (Arm B running mean)
        if store_pen:
            self._penultimate = act.detach()
        return self.fc2(act)

    def forward_with_act_grad(self, x):
        """Return (logits, act) where act has grad enabled (for Arm C Fisher)."""
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
    """Load real CIFAR-100. ABORT if not present (never download)."""
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
    SAME seed → SAME splits for all 4 arms — critical for PAIRED design.
    """
    rng = random.Random(seed)
    all_cls = list(range(NUM_CLASSES))
    rng.shuffle(all_cls)
    task_classes = [all_cls[i * CLS_PER_TASK:(i + 1) * CLS_PER_TASK] for i in range(n_tasks)]
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
    """Masked eval: predict only among current task's classes."""
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
    """Fraction of fc1 neurons with mean activation < thr."""
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
    """Effective rank of fc1 post-ReLU activations via entropy of singular values."""
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
    Arm B: CBP heuristic — |mean outgoing weight| × running mean activation.
    utility_i = mean_j(|fc2.weight[j, i]|) × running_act_mean[i]
    (Dohare et al. 2024 heuristic)
    """
    with torch.no_grad():
        out_w_mag = model.fc2.weight.detach().abs().mean(0).cpu()  # [FC1_UNITS]
        utility   = out_w_mag * running_act_mean.cpu()
    return utility   # [FC1_UNITS]


def utility_C(model, x_batch, y_batch, criterion):
    """
    Arm C: explicit empirical-Fisher — extra forward+backward pass.
    utility_i = mean_batch[(∂L/∂act_i)²]  (observed labels, empirical Fisher)
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
    return utility   # [FC1_UNITS]


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


def utility_RANDOM(rng_state):
    """
    Arm RANDOM: assign random utility values (uniform [0,1]).
    This is the mechanistic control: same reset cadence + fraction, but
    v_t's RANKING is replaced by random choice.
    rng_state: a numpy.random.RandomState seeded per (seed, task, step)
    """
    return torch.tensor(rng_state.rand(FC1_UNITS).astype(np.float32))


# ── CBP Reset Procedure ───────────────────────────────────────────────────────
def cbp_reset(model, optimizer, utility, reset_frac=RESET_FRAC):
    """
    Continual Backprop reset (Dohare et al. 2024):
      - Bottom reset_frac of fc1 neurons by utility → kaiming reinit
      - Outgoing fc2 weights → 0  (output continuity)
      - Adam state for affected rows/cols → 0
    """
    n_reset   = max(1, int(FC1_UNITS * reset_frac))
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
    """
    Run one (arm, seed) combination on the SAME task-split (same seed).
    Returns per-task metrics dict.
    """
    print(f"\n{'='*60}", flush=True)
    print(f"ARM {arm} | SEED {seed}", flush=True)
    print(f"{'='*60}", flush=True)

    # Reproducible seed (same for all arms with same seed → same data)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

    model     = SmallConvNetGN(num_output=NUM_CLASSES).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    # SAME seed → SAME task-split for all arms — CRITICAL for paired design
    splits, _ = make_task_splits(X_tr, y_tr, X_te, y_te, seed=seed, n_tasks=TASKS)

    # Arm B running activation mean (EMA, α=0.01)
    running_act_mean = torch.zeros(FC1_UNITS)

    # RANDOM arm: separate RNG seeded deterministically from (arm, seed)
    # so RANDOM arm is reproducible but independent of the model's random state
    rand_rng = np.random.RandomState(seed=1000 + seed)

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

            # ── Update Arm B running activation mean (EMA) ────────────────
            if arm == 'B':
                with torch.no_grad():
                    alpha = 0.01
                    running_act_mean = ((1 - alpha) * running_act_mean
                                       + alpha * model._last_fc1_act.mean(0).cpu())

            # ── CBP reset check every RESET_EVERY steps ───────────────────
            if step % RESET_EVERY == 0:
                if arm == 'B':
                    util = utility_B(model, running_act_mean)
                elif arm == 'C':
                    util = utility_C(model, x, y, criterion)
                elif arm == 'D':
                    util = utility_D(model, optimizer)
                elif arm == 'RANDOM':
                    util = utility_RANDOM(rand_rng)
                else:
                    raise ValueError(f"Unknown arm: {arm}")

                cbp_reset(model, optimizer, util)
                resets_this_task += 1
                model.train()   # stay in train mode after Fisher pass

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

    # Per-seed accuracy metric: mean over tasks 2–4 (exclude task-1 warmup)
    # With 4 tasks: tasks 2,3,4 = indices 1,2,3
    tasks_after_warmup = per_task_acc[1:]   # tasks 2-4 (indices 1,2,3)
    mean_tasks_2to4    = float(np.mean(tasks_after_warmup))

    return {
        "arm":              arm,
        "seed":             seed,
        "per_task_acc":     per_task_acc,
        "per_task_dead":    per_task_dead,
        "per_task_erank":   per_task_erank,
        "final_task_acc":   per_task_acc[-1],      # task-4 accuracy
        "mean_tasks_2to4":  mean_tasks_2to4,       # mean over tasks 2-4 (primary metric)
        "dead_final":       per_task_dead[-1],
        "erank_final":      per_task_erank[-1],
        "wall_sec":         wall_total,
    }


# ── Statistical Analysis ──────────────────────────────────────────────────────
def paired_analysis(a_results, b_results, label="A_minus_B", equiv_margin_pp=3.0):
    """
    Compute paired difference (A - B) using mean_tasks_2to4 per seed.
    Returns per-seed diffs, mean_pp, ci95_pp, t_p_value, equivalent_within_3pp.
    """
    a_acc = np.array([r["mean_tasks_2to4"] for r in a_results]) * 100
    b_acc = np.array([r["mean_tasks_2to4"] for r in b_results]) * 100
    diff  = a_acc - b_acc   # A - B

    n         = len(diff)
    mean_diff = float(np.mean(diff))
    std_diff  = float(np.std(diff, ddof=1)) if n > 1 else float('nan')
    se        = std_diff / np.sqrt(n) if n > 1 else float('nan')

    if n >= 2:
        t_stat, p_val_2sided = spstats.ttest_1samp(diff, 0.0)
        t_crit = float(spstats.t.ppf(0.975, df=n - 1))
        ci_lo  = mean_diff - t_crit * se
        ci_hi  = mean_diff + t_crit * se
    else:
        t_stat, p_val_2sided = float('nan'), float('nan')
        ci_lo = ci_hi = float('nan')

    equivalent = bool(ci_lo >= -equiv_margin_pp and ci_hi <= equiv_margin_pp)

    return {
        "label":     label,
        "per_seed":  [round(float(d), 3) for d in diff],
        "mean_pp":   round(mean_diff, 3),
        "std_pp":    round(std_diff, 3) if not np.isnan(std_diff) else None,
        "ci95_pp":   [round(ci_lo, 3), round(ci_hi, 3)],
        "t_stat":    round(float(t_stat), 4) if not np.isnan(t_stat) else None,
        "t_p_value": round(float(p_val_2sided), 6) if not np.isnan(p_val_2sided) else None,
        "n_seeds":   n,
        "equivalent_within_3pp": equivalent,
    }


def build_arm_summary(arm_results):
    """Build per-arm summary dict."""
    per_seed_acc = [r["mean_tasks_2to4"] for r in arm_results]
    dead_vals    = [r["dead_final"]       for r in arm_results]
    erank_vals   = [r["erank_final"]      for r in arm_results]
    return {
        "per_seed_acc": [round(v, 4) for v in per_seed_acc],
        "mean":         round(float(np.mean(per_seed_acc)), 4),
        "std":          round(float(np.std(per_seed_acc, ddof=1)) if len(per_seed_acc) > 1 else 0.0, 4),
        "dead_final":   round(float(np.mean(dead_vals)), 4),
        "erank_final":  round(float(np.nanmean(erank_vals)), 2),
        "per_task_acc_mean":   list(np.mean([r["per_task_acc"] for r in arm_results], axis=0)),
        "seeds_done":   [r["seed"] for r in arm_results],
    }


def compile_results(b_results, c_results, d_results, rand_results,
                    status="RUNNING", wall_min=0.0):
    """Compile the RESULTS.json payload (called incrementally and at end)."""
    n_seeds = len(b_results)
    assert len(c_results) == len(d_results) == len(rand_results) == n_seeds

    b_sum    = build_arm_summary(b_results)
    c_sum    = build_arm_summary(c_results)
    d_sum    = build_arm_summary(d_results)
    rand_sum = build_arm_summary(rand_results)

    # Paired D-C: does v_t match explicit Fisher?
    dc = paired_analysis(d_results, c_results, label="D_minus_C")
    dc_out = {
        "mean_pp":              dc["mean_pp"],
        "ci95_pp":              dc["ci95_pp"],
        "t_stat":               dc["t_stat"],
        "t_p_value":            dc["t_p_value"],
        "per_seed":             dc["per_seed"],
        "equivalent_within_3pp": dc["equivalent_within_3pp"],
        "interpretation": (
            "v_t IS statistically equivalent to Fisher within 3pp"
            if dc["equivalent_within_3pp"]
            else f"CI=[{dc['ci95_pp'][0]:.2f},{dc['ci95_pp'][1]:.2f}] not fully within ±3pp"
        ),
    }

    # Paired B-D: does v_t beat/match the CBP heuristic?
    bd = paired_analysis(b_results, d_results, label="B_minus_D")
    bd_out = {
        "mean_pp":  bd["mean_pp"],   # positive = B better, negative = D better
        "ci95_pp":  bd["ci95_pp"],
        "t_stat":   bd["t_stat"],
        "t_p_value": bd["t_p_value"],
        "per_seed": bd["per_seed"],
        "interpretation": (
            f"B better than D by {bd['mean_pp']:+.2f}pp (CI=[{bd['ci95_pp'][0]:.2f},{bd['ci95_pp'][1]:.2f}])"
            if bd["mean_pp"] > 0
            else f"D better than B by {-bd['mean_pp']:+.2f}pp (CI=[{bd['ci95_pp'][0]:.2f},{bd['ci95_pp'][1]:.2f}])"
        ),
    }

    # Paired D-RANDOM: is v_t's RANKING better than random reset?
    dr = paired_analysis(d_results, rand_results, label="D_minus_RANDOM")
    # One-sided p-value: H0 = D <= RANDOM, H1 = D > RANDOM
    # one_sided_p = p_twosided / 2 if D > RANDOM, else 1 - p_twosided / 2
    if dr["t_stat"] is not None and not np.isnan(dr["t_stat"]):
        one_sided_p = float(spstats.t.sf(dr["t_stat"], df=n_seeds - 1)) if n_seeds >= 2 else float('nan')
    else:
        one_sided_p = float('nan')
    ranking_beats_random = (
        dr["mean_pp"] > 0
        and (not np.isnan(one_sided_p))
        and one_sided_p < 0.05
        and n_seeds >= 4
    )
    dr_out = {
        "mean_pp":               dr["mean_pp"],
        "ci95_pp":               dr["ci95_pp"],
        "t_stat":                dr["t_stat"],
        "t_p_value":             dr["t_p_value"],           # two-sided
        "t_p_value_one_sided":   round(one_sided_p, 6) if not np.isnan(one_sided_p) else None,
        "per_seed":              dr["per_seed"],
        "v_t_ranking_beats_random": ranking_beats_random,
        "interpretation": (
            f"D >> RANDOM: v_t ranking carries real information "
            f"(D-RANDOM={dr['mean_pp']:+.2f}pp, p_one_sided={one_sided_p:.4f})"
            if ranking_beats_random
            else f"D-RANDOM={dr['mean_pp']:+.2f}pp CI=[{dr['ci95_pp'][0]:.2f},{dr['ci95_pp'][1]:.2f}], "
                 f"p_one_sided={one_sided_p:.4f} (need p<0.05 and D>RANDOM)"
        ),
    }

    # Compose RESULTS.json
    return {
        "status":      status,
        "scale":       "full",
        "n_seeds":     n_seeds,
        "data_source": "cifar100_real",
        "seeds_complete": [r["seed"] for r in b_results],
        "arms": {
            "B":      b_sum,
            "C":      c_sum,
            "D":      d_sum,
            "RANDOM": rand_sum,
            "A_floor": 0.20,
        },
        "paired": {
            "D_minus_C":      dc_out,
            "B_minus_D":      bd_out,
            "D_minus_RANDOM": dr_out,
        },
        "wall_time_min": round(wall_min, 2),
        "subject_executed": (
            f"Four-arm CBP (B=heuristic, C=Fisher, D=Adam-v_t, RANDOM=control), "
            f"PAIRED on SAME task splits, SmallConvNetGN, real CIFAR-100, "
            f"{TASKS}×{CLS_PER_TASK} classes, {STEPS_PER_TASK} steps/task, "
            f"reset_every={RESET_EVERY}, reset_frac={RESET_FRAC}, seeds={list(range(n_seeds))}"
        ),
        "notes": (
            f"n={n_seeds} seeds, {TASKS} tasks×{CLS_PER_TASK}cls, {STEPS_PER_TASK} steps/task. "
            f"D~C (v_t ≈ Fisher)? equiv_within_3pp={dc_out['equivalent_within_3pp']} "
            f"(D-C={dc_out['mean_pp']:+.2f}pp CI={dc_out['ci95_pp']}). "
            f"B vs D (heuristic vs v_t)? B-D={bd_out['mean_pp']:+.2f}pp CI={bd_out['ci95_pp']}. "
            f"D>>RANDOM (ranking carries info)? {dr_out['v_t_ranking_beats_random']} "
            f"(D-RANDOM={dr_out['mean_pp']:+.2f}pp p_1sided={dr_out['t_p_value_one_sided']})."
        ),
    }


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    t_start = time.time()

    # Stub (written immediately for visibility)
    stub = {
        "status":  "RUNNING",
        "scale":   "full",
        "n_seeds": 0,
        "data_source": "cifar100_real",
        "notes":   "Run in progress...",
    }
    write_results(stub)

    # ── Load real CIFAR-100 ───────────────────────────────────────────────────
    print(f"\nLoading real CIFAR-100 from {DATASETS_DIR}", flush=True)
    train_data, test_data = load_real_cifar100()
    if train_data is None:
        msg = f"ABORT: Real CIFAR-100 not found at {DATASETS_DIR}. Cannot proceed."
        print(msg, flush=True)
        write_results({"status": "FAILED", "scale": "full", "n_seeds": 0,
                       "data_source": "missing", "notes": msg})
        sys.exit(1)

    X_tr, y_tr = train_data
    X_te, y_te = test_data
    print(f"Data loaded: train={X_tr.shape} test={X_te.shape}", flush=True)
    print(f"Model params: {SmallConvNetGN(100).count_params():,}", flush=True)

    # ── PAIRED LOOP: for each seed, run ALL FOUR arms on the SAME task-split ──
    b_results    = []
    c_results    = []
    d_results    = []
    rand_results = []

    for seed in SEEDS:
        elapsed_min = (time.time() - t_start) / 60
        print(f"\n[{elapsed_min:.1f}m elapsed] === SEED {seed} ===", flush=True)

        # Run all four arms (same task-split = same seed)
        for arm, result_list in [("B", b_results), ("C", c_results),
                                   ("D", d_results), ("RANDOM", rand_results)]:
            elapsed_min = (time.time() - t_start) / 60
            print(f"\n[{elapsed_min:.1f}m] Running ARM {arm}, seed {seed}", flush=True)
            r = run_arm_seed(arm, seed, X_tr, y_tr, X_te, y_te)
            result_list.append(r)

        # ── Incremental write after ALL FOUR arms complete for this seed ──────
        n_complete  = len(b_results)
        wall_min    = (time.time() - t_start) / 60
        payload     = compile_results(b_results, c_results, d_results, rand_results,
                                      status="RUNNING", wall_min=wall_min)
        write_results(payload)

        # Print seed summary
        b_acc    = b_results[-1]["mean_tasks_2to4"]
        c_acc    = c_results[-1]["mean_tasks_2to4"]
        d_acc    = d_results[-1]["mean_tasks_2to4"]
        r_acc    = rand_results[-1]["mean_tasks_2to4"]
        print(f"\n  >> Seed {seed} done:", flush=True)
        print(f"     B={b_acc:.3f}  C={c_acc:.3f}  D={d_acc:.3f}  RANDOM={r_acc:.3f}", flush=True)
        print(f"     D-C={(d_acc - c_acc)*100:+.2f}pp  "
              f"B-D={(b_acc - d_acc)*100:+.2f}pp  "
              f"D-RANDOM={(d_acc - r_acc)*100:+.2f}pp",
              flush=True)
        # Print running paired stats
        paired = payload["paired"]
        print(f"  >> Running D-C: mean={paired['D_minus_C']['mean_pp']:+.2f}pp  "
              f"CI={paired['D_minus_C']['ci95_pp']}  n={n_complete}", flush=True)
        print(f"  >> Running D-RANDOM: mean={paired['D_minus_RANDOM']['mean_pp']:+.2f}pp  "
              f"CI={paired['D_minus_RANDOM']['ci95_pp']}  "
              f"p_1sided={paired['D_minus_RANDOM']['t_p_value_one_sided']}", flush=True)

        # ── Finalization trigger: >=4 seeds done ──────────────────────────────
        if n_complete >= 4:
            elapsed_min = (time.time() - t_start) / 60
            print(f"\n[{elapsed_min:.1f}m] {n_complete} seeds complete — "
                  f"FINALIZATION TRIGGERED (spec: >=4 sufficient)", flush=True)
            # Continue to 6 seeds but mark early finalization possible

    # ── Final summary ─────────────────────────────────────────────────────────
    wall_min = (time.time() - t_start) / 60
    final    = compile_results(b_results, c_results, d_results, rand_results,
                               status="DONE", wall_min=wall_min)

    # Add metrics block for RESULTS.json compatibility
    final["metrics"] = {
        "B_mean_acc":    final["arms"]["B"]["mean"],
        "B_std_acc":     final["arms"]["B"]["std"],
        "B_dead_final":  final["arms"]["B"]["dead_final"],
        "B_erank_final": final["arms"]["B"]["erank_final"],
        "C_mean_acc":    final["arms"]["C"]["mean"],
        "C_std_acc":     final["arms"]["C"]["std"],
        "C_dead_final":  final["arms"]["C"]["dead_final"],
        "C_erank_final": final["arms"]["C"]["erank_final"],
        "D_mean_acc":    final["arms"]["D"]["mean"],
        "D_std_acc":     final["arms"]["D"]["std"],
        "D_dead_final":  final["arms"]["D"]["dead_final"],
        "D_erank_final": final["arms"]["D"]["erank_final"],
        "RANDOM_mean_acc":    final["arms"]["RANDOM"]["mean"],
        "RANDOM_std_acc":     final["arms"]["RANDOM"]["std"],
        "RANDOM_dead_final":  final["arms"]["RANDOM"]["dead_final"],
        "RANDOM_erank_final": final["arms"]["RANDOM"]["erank_final"],
        "A_floor_acc":   0.20,
        "D_minus_C_mean_pp":      final["paired"]["D_minus_C"]["mean_pp"],
        "D_minus_C_ci95_pp":      final["paired"]["D_minus_C"]["ci95_pp"],
        "D_minus_C_equiv_3pp":    final["paired"]["D_minus_C"]["equivalent_within_3pp"],
        "B_minus_D_mean_pp":      final["paired"]["B_minus_D"]["mean_pp"],
        "B_minus_D_ci95_pp":      final["paired"]["B_minus_D"]["ci95_pp"],
        "D_minus_RANDOM_mean_pp": final["paired"]["D_minus_RANDOM"]["mean_pp"],
        "D_minus_RANDOM_ci95_pp": final["paired"]["D_minus_RANDOM"]["ci95_pp"],
        "D_minus_RANDOM_p1sided": final["paired"]["D_minus_RANDOM"]["t_p_value_one_sided"],
        "v_t_ranking_beats_random": final["paired"]["D_minus_RANDOM"]["v_t_ranking_beats_random"],
    }

    write_results(final)

    # ── Print summary ─────────────────────────────────────────────────────────
    b_s = final["arms"]["B"]
    c_s = final["arms"]["C"]
    d_s = final["arms"]["D"]
    r_s = final["arms"]["RANDOM"]
    dc  = final["paired"]["D_minus_C"]
    bd  = final["paired"]["B_minus_D"]
    dr  = final["paired"]["D_minus_RANDOM"]

    print(f"\n{'='*70}", flush=True)
    print(f"MECH_ARMS FINAL RESULTS ({len(b_results)} seeds, {wall_min:.1f} min)", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"  B (heuristic): mean={b_s['mean']:.4f}±{b_s['std']:.4f}  dead={b_s['dead_final']:.3f}  erank={b_s['erank_final']:.2f}", flush=True)
    print(f"  C (Fisher):    mean={c_s['mean']:.4f}±{c_s['std']:.4f}  dead={c_s['dead_final']:.3f}  erank={c_s['erank_final']:.2f}", flush=True)
    print(f"  D (Adam-v_t):  mean={d_s['mean']:.4f}±{d_s['std']:.4f}  dead={d_s['dead_final']:.3f}  erank={d_s['erank_final']:.2f}", flush=True)
    print(f"  RANDOM:        mean={r_s['mean']:.4f}±{r_s['std']:.4f}  dead={r_s['dead_final']:.3f}  erank={r_s['erank_final']:.2f}", flush=True)
    print(f"  A floor:       0.200 (from prior baseline_A3)", flush=True)
    print(f"\nPaired D-C: {dc['mean_pp']:+.2f}pp  CI={dc['ci95_pp']}  equiv_3pp={dc['equivalent_within_3pp']}", flush=True)
    print(f"Paired B-D: {bd['mean_pp']:+.2f}pp  CI={bd['ci95_pp']}", flush=True)
    print(f"Paired D-RANDOM: {dr['mean_pp']:+.2f}pp  CI={dr['ci95_pp']}  "
          f"p_1sided={dr['t_p_value_one_sided']}  beats_random={dr['v_t_ranking_beats_random']}", flush=True)
    print(f"\nFull results at: {RESULTS_PATH}", flush=True)
    print(json.dumps(final, indent=2), flush=True)


if __name__ == "__main__":
    main()
