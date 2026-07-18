"""
mech_rand.py — MECHANISTIC CONTROL: Does v_t's RANKING carry information,
               or does ANY reset help equally?

EXPERIMENT.md ROUND: mech_rand (full)
  TWO ARMS on IDENTICAL per-seed task splits (paired), differing ONLY in
  which neurons get reset each cycle:
    D (OURS):  reset the lowest-utility units by Adam exp_avg_sq (v_t) per-neuron score.
    RANDOM:    reset a RANDOM subset of units of the same size (ignores utility).

  If D significantly beats RANDOM, v_t's ranking carries real signal.

  PAIRED: 8 seeds {0..7}. Per seed, run D and RANDOM on the SAME task split.
  Per-seed diff d_s = acc_D(s) - acc_RANDOM(s).
  4 tasks × 5 classes = 20 classes class-incremental.
  ~1000 steps/task.

  Headline (finalize when >=6 seeds done):
    per-arm mean±std final-task acc;
    paired D-RANDOM mean_pp, 95% CI, paired t-test p (one-sided H1: D>RANDOM).
    Does v_t ranking significantly beat random?

Architecture: SmallConvNetGN (GroupNorm ConvNet, validated in all prior rounds)
Data: real CIFAR-100, /opt/datasets, download=False, ABORT if missing
Eval: masked (only task-current classes)

RESULTS.json format (per spec):
  {"status":"DONE","n_seeds":<>=6>,"data_source":"cifar100_real",
   "D":{"per_seed_acc":[...],"mean":..,"std":..},
   "RANDOM":{"per_seed_acc":[...],"mean":..,"std":..},
   "paired_D_minus_RANDOM":{"per_seed":[...],"mean_pp":..,"ci95_pp":[lo,hi],
                             "t_p_value_one_sided":..,"v_t_ranking_beats_random":bool},
   "notes":"does v_t's ranking carry information vs random reset? honest answer."}
"""

import os, sys, json, time, random, pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats as spstats

RESULTS_DIR  = "/workspace/results/mech_rand"
RESULTS_PATH = os.path.join(RESULTS_DIR, "RESULTS.json")
WEIGHTS_DIR  = "/workspace/_weights"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(WEIGHTS_DIR, exist_ok=True)

# ── Hyper-parameters (per EXPERIMENT.md spec) ────────────────────────────────
NUM_CLASSES    = 100
TASKS          = 4              # 4 tasks/run per spec
CLS_PER_TASK   = 5              # 5 classes/task → 20 classes total
STEPS_PER_TASK = 1000           # ~1000 steps/task per spec
BATCH_SIZE     = 128
LR             = 1e-3           # Adam, same across both arms
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEEDS          = list(range(8)) # {0..7} — 8 seeds per spec
ARMS           = ["D", "RANDOM"]

# CBP reset hyperparameters (IDENTICAL for BOTH arms — only utility differs)
RESET_EVERY  = 100              # reset every K steps
RESET_FRAC   = 0.10             # reset bottom 10% of units
FC1_UNITS    = 512              # neurons in the penultimate layer

DATASETS_DIR = os.environ.get("SH_DATASETS_DIR", "/opt/datasets")

print(f"Device: {DEVICE}", flush=True)
print(f"Datasets: {DATASETS_DIR}", flush=True)
print(f"Seeds: {SEEDS}  Tasks: {TASKS}×{CLS_PER_TASK}cls  Steps/task: {STEPS_PER_TASK}", flush=True)
print(f"Arms: {ARMS}  reset_every={RESET_EVERY}  reset_frac={RESET_FRAC}", flush=True)
print(f"Estimated wall-time: 2 arms × 8 seeds × 4 tasks × ~22s ≈ 23 min", flush=True)


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
    SAME seed → SAME splits for both arms — CRITICAL for PAIRED design.
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
    return splits


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

def utility_D(model, optimizer):
    """
    Arm D: Adam exp_avg_sq proxy — ZERO extra compute.
    utility_i = mean(exp_avg_sq[i, :]) for fc1.weight[i, :]
    Reads already-stored Adam state.
    """
    param = model.fc1.weight
    state = optimizer.state.get(param, {})
    if 'exp_avg_sq' not in state:
        return torch.zeros(FC1_UNITS)
    vt = state['exp_avg_sq'].detach().cpu()   # [FC1_UNITS, fan_in]
    return vt.mean(dim=1)                      # [FC1_UNITS]


def utility_RANDOM(rand_rng):
    """
    RANDOM arm: assign uniform random utility values — mechanistic control.
    Same reset cadence + fraction, but v_t's RANKING is replaced by random choice.
    rand_rng: a numpy.random.RandomState seeded per (seed) to be reproducible.
    """
    return torch.tensor(rand_rng.rand(FC1_UNITS).astype(np.float32))


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
    Run one (arm, seed) combination.
    SAME seed → same task-split and same model init for D and RANDOM — PAIRED.
    Returns per-task metrics dict.
    """
    print(f"\n{'='*60}", flush=True)
    print(f"ARM {arm} | SEED {seed}", flush=True)
    print(f"{'='*60}", flush=True)

    # Reproducible seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

    model     = SmallConvNetGN(num_output=NUM_CLASSES).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    # SAME seed → SAME task-split for both arms — CRITICAL for paired design
    splits = make_task_splits(X_tr, y_tr, X_te, y_te, seed=seed, n_tasks=TASKS)

    # RANDOM arm: separate RNG seeded deterministically from seed
    # ensures reproducibility without interfering with model RNG
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

            # ── CBP reset check every RESET_EVERY steps ───────────────────
            if step % RESET_EVERY == 0:
                if arm == 'D':
                    util = utility_D(model, optimizer)
                elif arm == 'RANDOM':
                    util = utility_RANDOM(rand_rng)
                else:
                    raise ValueError(f"Unknown arm: {arm}")

                cbp_reset(model, optimizer, util)
                resets_this_task += 1

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
    tasks_after_warmup = per_task_acc[1:]
    mean_tasks_2to4    = float(np.mean(tasks_after_warmup))

    return {
        "arm":              arm,
        "seed":             seed,
        "per_task_acc":     per_task_acc,
        "per_task_dead":    per_task_dead,
        "per_task_erank":   per_task_erank,
        "final_task_acc":   per_task_acc[-1],
        "mean_tasks_2to4":  mean_tasks_2to4,       # primary metric for paired test
        "dead_final":       per_task_dead[-1],
        "erank_final":      per_task_erank[-1],
        "wall_sec":         wall_total,
    }


# ── Statistical Analysis ──────────────────────────────────────────────────────
def paired_D_vs_RANDOM(d_results, rand_results):
    """
    Paired one-sided t-test: H1 = D > RANDOM.
    Returns mean_pp, ci95_pp, t_p_value_one_sided, v_t_ranking_beats_random.
    """
    d_acc   = np.array([r["mean_tasks_2to4"] for r in d_results]) * 100
    r_acc   = np.array([r["mean_tasks_2to4"] for r in rand_results]) * 100
    diff    = d_acc - r_acc   # D - RANDOM (positive = D better)

    n         = len(diff)
    mean_diff = float(np.mean(diff))
    std_diff  = float(np.std(diff, ddof=1)) if n > 1 else float('nan')
    se        = std_diff / np.sqrt(n) if n > 1 else float('nan')

    if n >= 2:
        t_stat, p_val_2sided = spstats.ttest_1samp(diff, 0.0)
        t_crit = float(spstats.t.ppf(0.975, df=n - 1))
        ci_lo  = mean_diff - t_crit * se
        ci_hi  = mean_diff + t_crit * se
        # One-sided p-value: H0 = D <= RANDOM, H1 = D > RANDOM
        one_sided_p = float(spstats.t.sf(float(t_stat), df=n - 1))
    else:
        t_stat, p_val_2sided = float('nan'), float('nan')
        ci_lo = ci_hi = one_sided_p = float('nan')

    ranking_beats_random = bool(
        mean_diff > 0
        and (not np.isnan(one_sided_p))
        and one_sided_p < 0.05
        and n >= 6   # need >=6 seeds to claim
    )

    return {
        "per_seed":              [round(float(d), 3) for d in diff],
        "mean_pp":               round(mean_diff, 3),
        "std_pp":                round(std_diff, 3) if not np.isnan(std_diff) else None,
        "ci95_pp":               [round(ci_lo, 3), round(ci_hi, 3)],
        "t_stat":                round(float(t_stat), 4) if not np.isnan(t_stat) else None,
        "t_p_value_two_sided":   round(float(p_val_2sided), 6) if not np.isnan(p_val_2sided) else None,
        "t_p_value_one_sided":   round(one_sided_p, 6) if not np.isnan(one_sided_p) else None,
        "n_seeds":               n,
        "v_t_ranking_beats_random": ranking_beats_random,
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
        "per_task_acc_mean": list(np.mean([r["per_task_acc"] for r in arm_results], axis=0)),
        "seeds_done":   [r["seed"] for r in arm_results],
    }


def compile_results(d_results, rand_results, status="RUNNING", wall_min=0.0):
    """Compile the RESULTS.json payload (called incrementally and at end)."""
    n_seeds  = len(d_results)
    assert len(rand_results) == n_seeds

    d_sum    = build_arm_summary(d_results)
    rand_sum = build_arm_summary(rand_results)

    paired   = paired_D_vs_RANDOM(d_results, rand_results)

    # Honest notes
    if paired["v_t_ranking_beats_random"]:
        verdict_note = (
            f"YES: v_t ranking carries real information beyond 'any reset helps'. "
            f"D beats RANDOM by {paired['mean_pp']:+.2f}pp "
            f"(p_one_sided={paired['t_p_value_one_sided']:.4f}, CI={paired['ci95_pp']})."
        )
    else:
        verdict_note = (
            f"NO (or insufficient evidence): D-RANDOM={paired['mean_pp']:+.2f}pp "
            f"CI={paired['ci95_pp']}, p_one_sided={paired['t_p_value_one_sided']} "
            f"— v_t's ranking does not significantly outperform random reset at p<0.05."
        )

    return {
        "status":      status,
        "scale":       "full",
        "n_seeds":     n_seeds,
        "data_source": "cifar100_real",
        "seeds_complete": [r["seed"] for r in d_results],
        "D":      d_sum,
        "RANDOM": rand_sum,
        "paired_D_minus_RANDOM": paired,
        "wall_time_min": round(wall_min, 2),
        "subject_executed": (
            f"Two-arm CBP (D=Adam-v_t utility, RANDOM=random-reset control), "
            f"PAIRED on same task splits, SmallConvNetGN, real CIFAR-100, "
            f"{TASKS}×{CLS_PER_TASK} classes, {STEPS_PER_TASK} steps/task, "
            f"reset_every={RESET_EVERY}, reset_frac={RESET_FRAC}, seeds={list(range(n_seeds))}"
        ),
        "notes": verdict_note,
    }


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    t_start = time.time()

    # Initial stub
    write_results({
        "status":  "RUNNING",
        "scale":   "full",
        "n_seeds": 0,
        "data_source": "cifar100_real",
        "notes":   "Run in progress — mech_rand two-arm paired experiment.",
    })

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

    # ── PAIRED LOOP: for each seed, run D then RANDOM on the SAME task-split ─
    d_results    = []
    rand_results = []

    for seed in SEEDS:
        elapsed_min = (time.time() - t_start) / 60
        print(f"\n[{elapsed_min:.1f}m elapsed] === SEED {seed} ===", flush=True)

        for arm, result_list in [("D", d_results), ("RANDOM", rand_results)]:
            elapsed_min = (time.time() - t_start) / 60
            print(f"\n[{elapsed_min:.1f}m] Running ARM {arm}, seed {seed}", flush=True)
            r = run_arm_seed(arm, seed, X_tr, y_tr, X_te, y_te)
            result_list.append(r)

        # ── Incremental write after BOTH arms complete for this seed ──────────
        n_complete = len(d_results)
        wall_min   = (time.time() - t_start) / 60
        payload    = compile_results(d_results, rand_results,
                                     status="RUNNING", wall_min=wall_min)
        write_results(payload)

        # Print seed summary
        d_acc   = d_results[-1]["mean_tasks_2to4"]
        r_acc   = rand_results[-1]["mean_tasks_2to4"]
        print(f"\n  >> Seed {seed} done:", flush=True)
        print(f"     D={d_acc:.4f}  RANDOM={r_acc:.4f}  diff={d_acc-r_acc:+.4f} ({(d_acc-r_acc)*100:+.2f}pp)", flush=True)

        paired = payload["paired_D_minus_RANDOM"]
        print(f"  >> Running D-RANDOM: mean={paired['mean_pp']:+.2f}pp  "
              f"CI={paired['ci95_pp']}  "
              f"p_one_sided={paired['t_p_value_one_sided']}  "
              f"n={n_complete}  beats_random={paired['v_t_ranking_beats_random']}", flush=True)

        # ── Finalization trigger: >=6 seeds done ──────────────────────────────
        if n_complete >= 6:
            elapsed_min = (time.time() - t_start) / 60
            print(f"\n[{elapsed_min:.1f}m] {n_complete} seeds complete — "
                  f"SPEC THRESHOLD MET (>=6). Continuing to all 8...", flush=True)

    # ── Final summary ─────────────────────────────────────────────────────────
    wall_min = (time.time() - t_start) / 60
    final    = compile_results(d_results, rand_results, status="DONE", wall_min=wall_min)

    # Add metrics block (for reviewer convenience)
    final["metrics"] = {
        "D_mean_acc":    final["D"]["mean"],
        "D_std_acc":     final["D"]["std"],
        "D_per_seed_acc": final["D"]["per_seed_acc"],
        "RANDOM_mean_acc":    final["RANDOM"]["mean"],
        "RANDOM_std_acc":     final["RANDOM"]["std"],
        "RANDOM_per_seed_acc": final["RANDOM"]["per_seed_acc"],
        "D_minus_RANDOM_mean_pp":       final["paired_D_minus_RANDOM"]["mean_pp"],
        "D_minus_RANDOM_ci95_pp":       final["paired_D_minus_RANDOM"]["ci95_pp"],
        "D_minus_RANDOM_p_one_sided":   final["paired_D_minus_RANDOM"]["t_p_value_one_sided"],
        "v_t_ranking_beats_random":     final["paired_D_minus_RANDOM"]["v_t_ranking_beats_random"],
    }

    write_results(final)

    # ── Print summary ─────────────────────────────────────────────────────────
    d_s  = final["D"]
    r_s  = final["RANDOM"]
    dr   = final["paired_D_minus_RANDOM"]

    print(f"\n{'='*70}", flush=True)
    print(f"MECH_RAND FINAL RESULTS ({len(d_results)} seeds, {wall_min:.1f} min)", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"  D (Adam-v_t):  mean={d_s['mean']:.4f}±{d_s['std']:.4f}  dead={d_s['dead_final']:.3f}  erank={d_s['erank_final']:.2f}", flush=True)
    print(f"  RANDOM:        mean={r_s['mean']:.4f}±{r_s['std']:.4f}  dead={r_s['dead_final']:.3f}  erank={r_s['erank_final']:.2f}", flush=True)
    print(f"\nPaired D-RANDOM: {dr['mean_pp']:+.2f}pp  CI={dr['ci95_pp']}  "
          f"p_one_sided={dr['t_p_value_one_sided']}  "
          f"v_t_ranking_beats_random={dr['v_t_ranking_beats_random']}", flush=True)
    print(f"\nConclusion: {final['notes']}", flush=True)
    print(f"\nFull results at: {RESULTS_PATH}", flush=True)
    print(json.dumps(final, indent=2), flush=True)


if __name__ == "__main__":
    main()
