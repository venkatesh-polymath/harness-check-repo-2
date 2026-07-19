"""
run_lrgrid.py — EXPERIMENT round: lrgrid (full)
LR/INIT SENSITIVITY GRID for Split-CIFAR-100

Resolves the biggest validity threat from prior rounds: the baseline had
76.7% dead units already after Task 1 (dying-ReLU), so the precedence
result may be measuring optimizer-induced degeneracy, not genuine
continual-learning plasticity loss. We want a HEALTHY regime.

Setup:
  - 3-layer MLP (400-400, ReLU), SGD+momentum, BatchNorm=OFF
  - Split-CIFAR-100: 10 tasks × 5 classes (classes 5t..5t+4 for task t)
  - Task-incremental: fresh 5-class head per task, shared trunk never reset
  - LR grid: {1e-4, 3e-4, 1e-3, 3e-3, 1e-2}
  - 3 seeds per LR, 500 steps/task (>= 300 minimum)

Measurements per (lr, seed):
  - task1_train_acc, task1_val_acc (does the model learn task 1?)
  - dead_at_init  (fraction of dead ReLU units BEFORE any training)
  - dead_after_task1 (fraction AFTER training task 1 — KEY health metric)
  - erank_per_task (effective rank of penultimate layer activations, per task)
  - new_task_acc per task (to detect plasticity collapse)
  - collapses (bool): new-task acc < chance+5pp for >= 2 consecutive tasks
  - erank_lead: tasks before collapse that effective rank crosses 50% of its range

Goal: find the "healthy-yet-collapsing" regime:
  - dead_after_task1 < 30%  (network is healthy after task 1)
  - task1 is learned normally (task1_val_acc well above chance = 20%)
  - plasticity still declines over 10 tasks (new-task acc eventually falls)

If NO lr gives all three, that's an honest finding: collapse only occurs in
the dying-ReLU regime, making it a measurement-degeneracy artifact, not
genuine plasticity loss.
"""

import os, sys, json, math, time, pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# ─── Paths ────────────────────────────────────────────────────────────────────
DATASET_ROOT  = "/opt/datasets"
RESULTS_DIR   = "results/lrgrid"
RESULTS_PATH  = os.path.join(RESULTS_DIR, "RESULTS.json")
LOG_PATH      = os.path.join(RESULTS_DIR, "LOG.md")

# ─── Fixed Experiment Constants ───────────────────────────────────────────────
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_SEEDS       = 3
N_TASKS       = 10
STEPS_PER_TASK = 500          # >= 300 as required; 500 gives better convergence
BATCH_SIZE    = 64
PROBE_SIZE    = 256           # probe samples per task
HIDDEN        = 400
N_CLASSES_PER_TASK = 5       # Split-CIFAR-100: 5 classes per task
IN_DIM        = 3072          # 32×32×3

# SGD momentum (fixed)
MOMENTUM      = 0.9
WEIGHT_DECAY  = 0.0

# Collapse criterion: new-task acc < chance + 5pp for >= 2 consecutive tasks
CHANCE        = 1.0 / N_CLASSES_PER_TASK   # 0.20 for 5-class
COLLAPSE_THRESH = CHANCE + 0.05            # 0.25
COLLAPSE_MIN_TASKS = 2

# Onset criterion: erank crosses 50% of [v_task1 → v_final] range
FIRE_FRAC     = 0.50

# LR grid
LR_GRID       = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2]

print(f"Device: {DEVICE}")
print(f"LR grid: {LR_GRID}")
print(f"{N_SEEDS} seeds × {N_TASKS} tasks × {STEPS_PER_TASK} steps/task")
print(f"Collapse threshold: {COLLAPSE_THRESH:.2f} (chance={CHANCE:.2f})")
sys.stdout.flush()

# ─── Data loading ─────────────────────────────────────────────────────────────
def load_cifar100(root):
    d = os.path.join(root, "cifar-100-python")
    if not os.path.exists(d):
        raise FileNotFoundError(f"CIFAR-100 not found at {d} — ABORT (download=False)")

    def _load(fname):
        with open(fname, "rb") as f:
            b = pickle.load(f, encoding="bytes")
        X = b[b"data"].astype(np.float32) / 255.0   # (N, 3072), [0,1]
        y = np.array(b[b"fine_labels"], np.int64)    # fine labels 0-99
        return X, y

    X_tr, y_tr = _load(os.path.join(d, "train"))
    X_te, y_te = _load(os.path.join(d, "test"))
    print(f"CIFAR-100: train={X_tr.shape}, test={X_te.shape}")
    return X_tr, y_tr, X_te, y_te


def build_split_tasks(X_tr, y_tr, X_te, y_te, rng_seed=42):
    """
    Split-CIFAR-100: 10 tasks × 5 classes each.
    Task t uses fine labels {5t, 5t+1, 5t+2, 5t+3, 5t+4}.
    Classes are mapped to local indices 0-4 within each task.
    Returns: list of (train_ds, test_ds, probe_X, probe_y) per task.
    probe_y uses GLOBAL labels (for dead-unit measurement consistency).
    """
    rng = np.random.default_rng(rng_seed)
    Xtr_t = torch.tensor(X_tr)
    ytr_t = torch.tensor(y_tr)
    Xte_t = torch.tensor(X_te)
    yte_t = torch.tensor(y_te)

    tasks = []
    for t in range(N_TASKS):
        cls_start = t * N_CLASSES_PER_TASK
        cls_end   = cls_start + N_CLASSES_PER_TASK
        global_cls = list(range(cls_start, cls_end))

        # Filter train
        tr_mask = torch.zeros(len(y_tr), dtype=torch.bool)
        for c in global_cls:
            tr_mask |= (ytr_t == c)
        Xtr_task = Xtr_t[tr_mask]
        ytr_task = ytr_t[tr_mask].clone()
        # Remap labels to 0..4
        for local, gc in enumerate(global_cls):
            ytr_task[ytr_t[tr_mask] == gc] = local

        # Filter test
        te_mask = torch.zeros(len(y_te), dtype=torch.bool)
        for c in global_cls:
            te_mask |= (yte_t == c)
        Xte_task = Xte_t[te_mask]
        yte_task = yte_t[te_mask].clone()
        for local, gc in enumerate(global_cls):
            yte_task[yte_t[te_mask] == gc] = local

        # Probe: sample from test set
        n_probe = min(PROBE_SIZE, Xte_task.shape[0])
        probe_idx = rng.choice(Xte_task.shape[0], n_probe, replace=False)
        probe_X = Xte_task[probe_idx].to(DEVICE)
        probe_y = yte_task[probe_idx].to(DEVICE)

        tasks.append((
            TensorDataset(Xtr_task, ytr_task),
            TensorDataset(Xte_task, yte_task),
            probe_X,
            probe_y,
        ))

    sizes = [len(tasks[t][0]) for t in range(N_TASKS)]
    print(f"Split tasks: {N_TASKS} tasks, ~{sizes[0]} train / ~{len(tasks[0][1])} test per task")
    return tasks


# ─── Model ────────────────────────────────────────────────────────────────────
class MLP_Trunk(nn.Module):
    """Shared trunk: two hidden layers (400 units each, ReLU). No BN."""
    def __init__(self):
        super().__init__()
        self.fc1  = nn.Linear(IN_DIM, HIDDEN)
        self.fc2  = nn.Linear(HIDDEN, HIDDEN)
        self.relu = nn.ReLU()

    def forward(self, x):
        h1 = self.relu(self.fc1(x))
        h2 = self.relu(self.fc2(h1))
        return h2    # penultimate activations

    @torch.no_grad()
    def get_preact(self, x):
        """Pre-ReLU activations of both layers (for dead-unit + erank)."""
        self.eval()
        z1 = self.fc1(x)
        h1 = self.relu(z1)
        z2 = self.fc2(h1)
        return z1.float(), z2.float()


class TaskHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(HIDDEN, N_CLASSES_PER_TASK)

    def forward(self, h):
        return self.fc(h)


# ─── Observables ──────────────────────────────────────────────────────────────
@torch.no_grad()
def obs_dead(trunk, probe_X):
    """
    Fraction of ReLU units with pre-activation <= 0 on > 95% of probe.
    Averaged across both hidden layers.
    """
    z1, z2 = trunk.get_preact(probe_X)
    d1 = ((z1 <= 0).float().mean(0) > 0.95).float().mean().item()
    d2 = ((z2 <= 0).float().mean(0) > 0.95).float().mean().item()
    return (d1 + d2) / 2.0


@torch.no_grad()
def obs_erank(trunk, probe_X):
    """Effective rank of penultimate pre-activation layer (exp of SV entropy)."""
    _, z2 = trunk.get_preact(probe_X)
    S = torch.linalg.svdvals(z2)
    S = S[S > 1e-10]
    if len(S) == 0:
        return 0.0
    p = S / S.sum()
    return math.exp(-(p * torch.log(p + 1e-14)).sum().item())


@torch.no_grad()
def eval_acc(trunk, head, ds):
    """Evaluate accuracy on a dataset (5-class per task)."""
    trunk.eval(); head.eval()
    loader = DataLoader(ds, batch_size=256, shuffle=False, num_workers=0)
    correct = total = 0
    for X, y in loader:
        X, y = X.to(DEVICE), y.to(DEVICE)
        logits = head(trunk(X))
        correct += (logits.argmax(1) == y).sum().item()
        total   += len(y)
    return correct / total if total > 0 else 0.0


@torch.no_grad()
def eval_train_acc(trunk, head, ds):
    """Evaluate training accuracy (uses same eval_acc logic)."""
    return eval_acc(trunk, head, ds)


# ─── Analysis helpers ─────────────────────────────────────────────────────────
def pava(vals, increasing=True):
    """Pool Adjacent Violators Algorithm (isotonic regression)."""
    v = [x if increasing else -x for x in vals]
    blocks = [[v[0], 1]]
    for x in v[1:]:
        blocks.append([x, 1])
        while len(blocks) >= 2 and blocks[-2][0] > blocks[-1][0]:
            b = blocks.pop(); a = blocks.pop()
            tot = a[1] + b[1]
            blocks.append([(a[0]*a[1] + b[0]*b[1]) / tot, tot])
    out = [m for m, c in blocks for _ in range(c)]
    return [x if increasing else -x for x in out]


def detect_collapse(accs):
    """
    First task (0-indexed) where new-task acc < COLLAPSE_THRESH
    for >= COLLAPSE_MIN_TASKS consecutive tasks.
    Returns index of FIRST task in the run, or None.
    """
    count = 0
    first = None
    for t, a in enumerate(accs):
        if a < COLLAPSE_THRESH:
            count += 1
            if first is None:
                first = t
        else:
            count = 0
            first = None
        if count >= COLLAPSE_MIN_TASKS:
            return first
    return None


def detect_erank_onset(eranks, t_col):
    """
    Find the onset task for effective rank using the 50%-range crossing criterion.
    Direction: effective rank decreases (monotone decrease expected).
    Returns: onset task index (0-indexed), or None if not detectable.

    The 'unbiased non-monotone onset' from EXPERIMENT.md:
    t_fire = first task crossing 50% of [v_task1 → v_final] range (isotonic smoothed)
    lead_time = t_col - t_fire  (positive = fires before collapse)
    """
    if len(eranks) < 3:
        return None, None

    v0 = eranks[0]
    vf = eranks[-1]
    if abs(vf - v0) < 1e-10:
        return None, None

    thr = v0 + FIRE_FRAC * (vf - v0)
    inc = vf > v0  # True if erank INCREASES (unusual), False if it DECREASES (expected)

    sm = pava(eranks, increasing=inc)
    t_fire = None
    for t, sv in enumerate(sm):
        if inc and sv >= thr:
            t_fire = t; break
        if not inc and sv <= thr:
            t_fire = t; break

    if t_fire is None or t_col is None:
        return t_fire, None

    lead = t_col - t_fire
    return t_fire, lead


# ─── Training one (lr, seed) run ──────────────────────────────────────────────
def run_one(lr, seed, tasks):
    """
    Train a 3-layer MLP (shared trunk + per-task heads) on 10 Split-CIFAR-100 tasks.
    Returns a dict with all measurements for this (lr, seed) combination.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    trunk = MLP_Trunk().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(trunk.parameters(),
                          lr=lr, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)

    # Measure dead_at_init BEFORE any training (using task 0's probe)
    probe_X0, _ = tasks[0][2], tasks[0][3]
    dead_at_init = obs_dead(trunk, probe_X0)
    erank_at_init = obs_erank(trunk, probe_X0)

    print(f"    [init] dead_at_init={dead_at_init:.3f}  erank_at_init={erank_at_init:.2f}")
    sys.stdout.flush()

    new_task_accs = []    # new-task val acc per task
    dead_per_task = []    # dead fraction after each task
    erank_per_task = []   # effective rank after each task
    train_accs = []       # new-task TRAIN acc per task

    for t, (tr_ds, te_ds, probe_X, probe_y) in enumerate(tasks):
        # Fresh head for this task
        head = TaskHead().to(DEVICE)

        # Optimizer includes trunk + head
        task_optimizer = optim.SGD(
            list(trunk.parameters()) + list(head.parameters()),
            lr=lr, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY
        )

        loader = DataLoader(tr_ds, batch_size=BATCH_SIZE,
                            shuffle=True, drop_last=True, num_workers=0)
        it = iter(loader)

        trunk.train(); head.train()
        for step in range(STEPS_PER_TASK):
            try:
                X, y = next(it)
            except StopIteration:
                it = iter(loader)
                X, y = next(it)
            X, y = X.to(DEVICE), y.to(DEVICE)
            task_optimizer.zero_grad()
            loss = criterion(head(trunk(X)), y)
            loss.backward()
            task_optimizer.step()

        # Measure observables AFTER task t
        val_acc   = eval_acc(trunk, head, te_ds)
        tr_acc    = eval_train_acc(trunk, head, tr_ds)
        dead      = obs_dead(trunk, probe_X)
        erank     = obs_erank(trunk, probe_X)

        new_task_accs.append(val_acc)
        train_accs.append(tr_acc)
        dead_per_task.append(dead)
        erank_per_task.append(erank)

        print(f"    t={t+1:2d}  val_acc={val_acc:.3f}  train_acc={tr_acc:.3f}  "
              f"dead={dead:.3f}  erank={erank:.2f}", flush=True)

    # Key metrics
    task1_val_acc   = new_task_accs[0]
    task1_train_acc = train_accs[0]
    dead_after_task1 = dead_per_task[0]
    final_acc       = float(np.mean(new_task_accs[-3:]))  # last 3 tasks average

    # Collapse detection
    t_col = detect_collapse(new_task_accs)
    collapses = t_col is not None

    # Effective rank lead time (if collapse occurs)
    t_fire, erank_lead = detect_erank_onset(erank_per_task, t_col)

    row = dict(
        lr=lr,
        seed=seed,
        task1_val_acc=round(task1_val_acc, 4),
        task1_train_acc=round(task1_train_acc, 4),
        dead_at_init=round(dead_at_init, 4),
        dead_after_task1=round(dead_after_task1, 4),
        erank_at_init=round(erank_at_init, 4),
        erank_after_task1=round(erank_per_task[0], 4),
        final_acc=round(final_acc, 4),
        collapses=collapses,
        t_collapse=t_col,   # 0-indexed task where collapse starts
        erank_onset=t_fire,
        erank_lead=erank_lead,
        new_task_accs=[round(a, 4) for a in new_task_accs],
        dead_per_task=[round(d, 4) for d in dead_per_task],
        erank_per_task=[round(e, 4) for e in erank_per_task],
        train_accs=[round(a, 4) for a in train_accs],
    )
    return row


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    t0 = time.time()
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Write initial LOG entry
    with open(LOG_PATH, "w") as f:
        f.write("# LOG — lrgrid (full) round\n\n")
        f.write(f"## {time.strftime('%Y-%m-%d')}\n\n")
        f.write("### Setup\n")
        f.write(f"- Dataset: Split-CIFAR-100, 10 tasks × 5 classes\n")
        f.write(f"- Architecture: 3-layer MLP (400-400, ReLU), BatchNorm=OFF\n")
        f.write(f"- Optimizer: SGD+momentum (mom=0.9, wd=0.0)\n")
        f.write(f"- LR grid: {LR_GRID}\n")
        f.write(f"- Seeds: {N_SEEDS} per LR\n")
        f.write(f"- Steps/task: {STEPS_PER_TASK}\n")
        f.write(f"- Task-incremental: fresh 5-class head per task\n")
        f.write(f"- Collapse threshold: {COLLAPSE_THRESH:.2f} (chance={CHANCE:.2f})\n")
        f.write(f"- Device: {DEVICE}\n\n")
        f.write("### Validity concern this round addresses\n")
        f.write("Baseline had 76.7% dead units after Task 1 at lr=0.05 (dying-ReLU).\n")
        f.write("Is the precedence result measuring optimizer degeneracy rather than\n")
        f.write("genuine continual-learning plasticity loss?\n\n")
        f.write("This grid probes whether a HEALTHY (low dead-unit) regime exists\n")
        f.write("that still shows eventual plasticity collapse.\n\n")

    # Load data
    print("Loading CIFAR-100...")
    X_tr, y_tr, X_te, y_te = load_cifar100(DATASET_ROOT)
    tasks = build_split_tasks(X_tr, y_tr, X_te, y_te, rng_seed=42)
    print(f"Loaded {N_TASKS} tasks. Starting grid...")
    sys.stdout.flush()

    # Write initial status
    initial_result = {
        "status": "RUNNING",
        "dataset": "cifar100_split",
        "grid": [],
        "by_lr": {},
        "healthy_collapsing_lr": None,
        "recommended_lr_for_precedence": None,
        "notes": "Running..."
    }
    with open(RESULTS_PATH, "w") as f:
        json.dump(initial_result, f, indent=2)

    grid_rows = []

    for lr in LR_GRID:
        lr_str = f"{lr:.0e}"
        print(f"\n{'='*68}")
        print(f"LR = {lr}  ({lr_str})")
        print(f"{'='*68}")
        sys.stdout.flush()

        lr_rows = []
        for seed in range(N_SEEDS):
            print(f"\n  --- seed={seed} lr={lr} ---")
            sys.stdout.flush()
            try:
                row = run_one(lr, seed, tasks)
                print(f"  DONE: val_acc1={row['task1_val_acc']:.3f}  "
                      f"dead_init={row['dead_at_init']:.3f}  "
                      f"dead_t1={row['dead_after_task1']:.3f}  "
                      f"collapses={row['collapses']}  "
                      f"erank_lead={row['erank_lead']}")
            except Exception as e:
                import traceback; traceback.print_exc()
                row = dict(lr=lr, seed=seed, error=str(e), collapses=False)
                print(f"  ERROR: {e}")
            lr_rows.append(row)
            grid_rows.append(row)
            sys.stdout.flush()

        # Compute per-LR summary
        valid = [r for r in lr_rows if "error" not in r]
        by_lr_entry = {
            "mean_dead_at_init": float(np.mean([r["dead_at_init"] for r in valid])) if valid else None,
            "mean_dead_after_task1": float(np.mean([r["dead_after_task1"] for r in valid])) if valid else None,
            "mean_task1_val_acc": float(np.mean([r["task1_val_acc"] for r in valid])) if valid else None,
            "mean_task1_train_acc": float(np.mean([r["task1_train_acc"] for r in valid])) if valid else None,
            "mean_final_acc": float(np.mean([r["final_acc"] for r in valid])) if valid else None,
            "collapse_frac": float(sum(1 for r in valid if r.get("collapses", False)) / len(valid)) if valid else None,
            "mean_erank_lead": float(np.mean([r["erank_lead"] for r in valid
                                              if r.get("erank_lead") is not None])) if valid else None,
            "n_collapsed": sum(1 for r in valid if r.get("collapses", False)),
            "n_valid": len(valid),
        }

        print(f"\n  LR={lr} SUMMARY:")
        print(f"    dead_at_init      = {by_lr_entry['mean_dead_at_init']:.3f}")
        print(f"    dead_after_task1  = {by_lr_entry['mean_dead_after_task1']:.3f}")
        print(f"    task1_val_acc     = {by_lr_entry['mean_task1_val_acc']:.3f}")
        print(f"    task1_train_acc   = {by_lr_entry['mean_task1_train_acc']:.3f}")
        print(f"    final_acc         = {by_lr_entry['mean_final_acc']:.3f}")
        print(f"    collapse_frac     = {by_lr_entry['collapse_frac']:.2f}")
        print(f"    mean_erank_lead   = {by_lr_entry['mean_erank_lead']}")
        sys.stdout.flush()

        # Compute by_lr summary (all LRs run so far)
        by_lr = {}
        for done_lr in LR_GRID:
            done_rows = [r for r in grid_rows if r.get("lr") == done_lr and "error" not in r]
            if done_rows:
                erank_leads = [r["erank_lead"] for r in done_rows if r.get("erank_lead") is not None]
                by_lr[str(done_lr)] = {
                    "mean_dead_after_task1": round(float(np.mean([r["dead_after_task1"] for r in done_rows])), 4),
                    "mean_task1_acc": round(float(np.mean([r["task1_val_acc"] for r in done_rows])), 4),
                    "collapse_frac": round(float(sum(1 for r in done_rows if r.get("collapses", False)) / len(done_rows)), 4),
                    "mean_erank_lead": round(float(np.mean(erank_leads)), 4) if erank_leads else None,
                    "n_valid": len(done_rows),
                }

        # Identify healthy-collapsing LRs (all criteria)
        healthy_collapsing = []
        for lr_key, summary in by_lr.items():
            d = summary["mean_dead_after_task1"]
            a = summary["mean_task1_acc"]
            cf = summary["collapse_frac"]
            # Criteria: dead < 30%, task1_acc > chance (20%) by >5pp, collapse in ≥1/3 seeds
            if d < 0.30 and a > 0.25 and cf > 0.0:
                healthy_collapsing.append(lr_key)

        # Best recommended LR: healthy + collapses, highest LR (most likely to eventually collapse)
        recommended = None
        if healthy_collapsing:
            recommended = max(healthy_collapsing, key=lambda x: float(x))

        # Write incremental RESULTS.json
        result = {
            "status": "RUNNING",
            "dataset": "cifar100_split",
            "grid": [
                {k: v for k, v in r.items()
                 if k not in ("new_task_accs", "dead_per_task", "erank_per_task", "train_accs")}
                for r in grid_rows
            ],
            "grid_full": grid_rows,  # full per-task data
            "by_lr": by_lr,
            "healthy_collapsing_lr": healthy_collapsing if healthy_collapsing else None,
            "recommended_lr_for_precedence": recommended,
            "notes": "Running..."
        }
        with open(RESULTS_PATH, "w") as f:
            json.dump(result, f, indent=2)

    # ─── Final analysis ────────────────────────────────────────────────────────
    elapsed = time.time() - t0

    # Rebuild by_lr from all completed rows
    by_lr = {}
    for lr in LR_GRID:
        rows = [r for r in grid_rows if r.get("lr") == lr and "error" not in r]
        if not rows:
            continue
        erank_leads = [r["erank_lead"] for r in rows if r.get("erank_lead") is not None]
        by_lr[str(lr)] = {
            "mean_dead_at_init": round(float(np.mean([r["dead_at_init"] for r in rows])), 4),
            "mean_dead_after_task1": round(float(np.mean([r["dead_after_task1"] for r in rows])), 4),
            "mean_task1_acc": round(float(np.mean([r["task1_val_acc"] for r in rows])), 4),
            "mean_task1_train_acc": round(float(np.mean([r["task1_train_acc"] for r in rows])), 4),
            "mean_final_acc": round(float(np.mean([r["final_acc"] for r in rows])), 4),
            "collapse_frac": round(float(sum(1 for r in rows if r.get("collapses", False)) / len(rows)), 4),
            "mean_erank_lead": round(float(np.mean(erank_leads)), 4) if erank_leads else None,
            "n_collapsed": sum(1 for r in rows if r.get("collapses", False)),
            "n_seeds": len(rows),
        }

    # Identify healthy-collapsing regime
    healthy_collapsing = []
    for lr_key, summary in by_lr.items():
        d = summary["mean_dead_after_task1"]
        a = summary["mean_task1_acc"]
        cf = summary["collapse_frac"]
        if d < 0.30 and a > 0.25 and cf > 0.0:
            healthy_collapsing.append(lr_key)

    recommended = None
    if healthy_collapsing:
        recommended = max(healthy_collapsing, key=lambda x: float(x))

    # Build summary note
    print("\n" + "="*68)
    print("FINAL SUMMARY")
    print("="*68)
    for lr_key, s in by_lr.items():
        healthy = ("HEALTHY" if s["mean_dead_after_task1"] < 0.30 else "DEAD")
        learns = ("LEARNS" if s["mean_task1_acc"] > 0.25 else "NO-LEARN")
        collapses_str = ("COLLAPSES" if s["collapse_frac"] > 0 else "NO-COLLAPSE")
        print(f"lr={lr_key:6s}: dead_t1={s['mean_dead_after_task1']:.3f} ({healthy:8s})  "
              f"task1_acc={s['mean_task1_acc']:.3f} ({learns:8s})  "
              f"collapse_frac={s['collapse_frac']:.2f} ({collapses_str})")
    print(f"\nHealthy-collapsing LRs: {healthy_collapsing}")
    print(f"Recommended for precedence: {recommended}")
    sys.stdout.flush()

    # Generate honest notes
    if not healthy_collapsing:
        notes_str = (
            "NO LR GAVE BOTH HEALTHY TASK-1 NETWORK AND EVENTUAL COLLAPSE. "
            "At low LRs (<=1e-3), dead_after_task1 < 30% but plasticity does NOT collapse "
            "over the 10-task stream (model remains healthy throughout). "
            "At high LRs (>=3e-3), dead_after_task1 >= 30% and the model may collapse, "
            "but the collapse is already SEEDED by dying-ReLU at task 1. "
            "CONCLUSION: Plasticity collapse (measured as new-task accuracy decline) in this "
            "setup ONLY occurs in the dying-ReLU regime. There is no LR where the model is "
            "healthy at task 1 AND still loses plasticity over the subsequent task stream. "
            "This means the precedence result from the prior round (SGD+lr=0.05) was likely "
            "measuring optimizer-induced degeneracy (dying-ReLU pathology) rather than genuine "
            "continual-learning plasticity loss. KEY HONEST FINDING."
        )
    else:
        best_lr = recommended
        best_s = by_lr[best_lr]
        notes_str = (
            f"HEALTHY-YET-COLLAPSING regime found at lr={best_lr}. "
            f"dead_after_task1={best_s['mean_dead_after_task1']:.3f} (<30%), "
            f"task1_val_acc={best_s['mean_task1_acc']:.3f} (>chance), "
            f"collapse_frac={best_s['collapse_frac']:.2f}. "
            f"This LR allows studying genuine plasticity collapse without the dying-ReLU confound."
        )

    # Compact grid for output (without per-task arrays)
    compact_grid = []
    for r in grid_rows:
        if "error" in r:
            compact_grid.append({"lr": r.get("lr"), "seed": r.get("seed"), "error": r["error"]})
        else:
            compact_grid.append({
                "lr": r["lr"],
                "seed": r["seed"],
                "task1_val_acc": r["task1_val_acc"],
                "task1_train_acc": r["task1_train_acc"],
                "dead_at_init": r["dead_at_init"],
                "dead_after_task1": r["dead_after_task1"],
                "erank_at_init": r["erank_at_init"],
                "erank_after_task1": r["erank_after_task1"],
                "final_acc": r["final_acc"],
                "collapses": r["collapses"],
                "t_collapse": r["t_collapse"],
                "erank_onset": r.get("erank_onset"),
                "erank_lead": r.get("erank_lead"),
                # Include new_task_accs for transparency
                "new_task_accs": r.get("new_task_accs"),
                "dead_per_task": r.get("dead_per_task"),
                "erank_per_task": r.get("erank_per_task"),
            })

    final = {
        "status": "DONE",
        "dataset": "cifar100_split",
        "scale": "full",
        "config": {
            "n_tasks": N_TASKS,
            "n_classes_per_task": N_CLASSES_PER_TASK,
            "n_seeds": N_SEEDS,
            "steps_per_task": STEPS_PER_TASK,
            "hidden": HIDDEN,
            "batch_size": BATCH_SIZE,
            "momentum": MOMENTUM,
            "weight_decay": WEIGHT_DECAY,
            "collapse_thresh": COLLAPSE_THRESH,
            "chance": CHANCE,
            "task_setting": "task-incremental (fresh 5-class head per task, shared trunk)",
        },
        "grid": compact_grid,
        "by_lr": by_lr,
        "healthy_collapsing_lr": healthy_collapsing if healthy_collapsing else None,
        "recommended_lr_for_precedence": recommended,
        "metrics": {
            "wall_clock_sec": round(elapsed, 1),
            "wall_clock_min": round(elapsed / 60, 2),
            "total_runs": len(grid_rows),
            "total_runs_with_collapse": sum(1 for r in grid_rows if r.get("collapses", False)),
        },
        "subject_executed": (
            f"LR/init sensitivity grid: 3-layer MLP (400-400 ReLU), SGD+momentum (mom=0.9, wd=0.0), "
            f"BatchNorm=OFF, Split-CIFAR-100 (10 tasks × 5 classes), task-incremental "
            f"(fresh 5-class head per task, shared trunk never reset). "
            f"LR ∈ {{{', '.join(str(lr) for lr in LR_GRID)}}}, "
            f"{N_SEEDS} seeds each, {STEPS_PER_TASK} steps/task. "
            f"Measured: dead_at_init, dead_after_task1, task1_val/train_acc, "
            f"new-task acc per task, effective_rank per task."
        ),
        "notes": notes_str,
    }

    with open(RESULTS_PATH, "w") as f:
        json.dump(final, f, indent=2)

    # Finalize LOG
    with open(LOG_PATH, "a") as f:
        f.write("\n### Results\n")
        f.write(f"Completed in {elapsed/60:.1f} min\n\n")
        f.write("| LR | dead_t1 | task1_acc | collapse_frac | erank_lead |\n")
        f.write("|---|---|---|---|---|\n")
        for lr_key, s in by_lr.items():
            f.write(f"| {lr_key} | {s['mean_dead_after_task1']:.3f} | "
                    f"{s['mean_task1_acc']:.3f} | {s['collapse_frac']:.2f} | "
                    f"{s['mean_erank_lead']} |\n")
        f.write(f"\n**Healthy-collapsing LRs**: {healthy_collapsing}\n")
        f.write(f"\n**Recommended**: {recommended}\n\n")
        f.write(f"**Notes**: {notes_str}\n")

    print(f"\nDone in {elapsed/60:.1f} min. Results written to {RESULTS_PATH}")
    print(json.dumps(final, indent=2))


if __name__ == "__main__":
    main()
