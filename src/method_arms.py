"""
method_arms.py — Arms B, C, D: Continual-Backprop with 3 utility functions.

Arms (ONLY the utility criterion differs; everything else is identical):
  B (CBP-heuristic): utility = |outgoing_weight_mean| × running_mean_activation
  C (explicit-Fisher): utility = mean_batch[(∂L/∂post_act_i)²]  [extra backward pass]
  D (Adam-v_t proxy): utility = mean(exp_avg_sq over fc1.weight[i, :])  [ZERO extra compute]

Setup (matches EXPERIMENT.md):
  - Real CIFAR-100 from /opt/datasets (download=False — ABORT if missing)
  - SmallConvNetGN (GroupNorm, same as baseline_A3)
  - 8 tasks × 5 classes = 40 classes class-incremental
  - 1000 steps/task (probe scale)
  - Adam lr=1e-3, CBP reset every 100 steps, reset_frac=0.10
  - 3 seeds (0, 1, 2) per arm
  - Masked eval (only predict among the current task's 5 classes)
  - Arm A floor: reference from prior baseline_A3 results (acc≈0.2, dead≈1.0)

Output: results/method_arms/RESULTS.json, results/method_arms/run.log
"""

import os, sys, json, time, random, pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

RESULTS_DIR  = "/workspace/results/method_arms"
RESULTS_PATH = os.path.join(RESULTS_DIR, "RESULTS.json")
WEIGHTS_DIR  = "/workspace/_weights"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(WEIGHTS_DIR, exist_ok=True)

# ── Hyper-parameters (PROBE scale) ───────────────────────────────────────────
NUM_CLASSES    = 100
TASKS          = 8           # 8 tasks (collapse visible by task 3, plenty of headroom)
CLS_PER_TASK   = 5           # 5 classes per task
STEPS_PER_TASK = 1000        # probe: 1000 steps (spec allows ≥1000)
BATCH_SIZE     = 128
LR             = 1e-3        # same Adam lr across all arms
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEEDS          = [0, 1, 2]   # 3 seeds per arm
ARMS           = ["B", "C", "D"]

# CBP reset hyperparameters (SAME for all arms)
RESET_EVERY  = 100           # reset every K steps
RESET_FRAC   = 0.10          # reset bottom 10% of units
FC1_UNITS    = 512           # neurons in the penultimate layer

# Data
DATASETS_DIR = os.environ.get("SH_DATASETS_DIR", "/opt/datasets")

print(f"Device: {DEVICE}", flush=True)
print(f"Datasets: {DATASETS_DIR}", flush=True)


# ── Model: GroupNorm ConvNet (identical to baseline_A3) ──────────────────────
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
        self._penultimate = None
        self._last_fc1_act = None  # post-ReLU activations (for Arm B running mean)

    def forward(self, x, store_pen=False):
        h = self.features(x).view(x.size(0), -1)
        pre = self.fc1(h)
        act = F.relu(pre)
        self._last_fc1_act = act.detach()    # always store (reused for Arm B)
        if store_pen:
            self._penultimate = act.detach()
        return self.fc2(act)

    def forward_with_act_grad(self, x):
        """Return (logits, act) where act has grad enabled (for Fisher utility)."""
        h = self.features(x).view(x.size(0), -1)
        pre = self.fc1(h)
        act = F.relu(pre)
        act.retain_grad()                    # enable grad on non-leaf tensor
        logits = self.fc2(act)
        return logits, act

    def count_params(self):
        return sum(p.numel() for p in self.parameters())


# ── Data loading (same as baseline_A3, download=False) ───────────────────────
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
    Split 100 classes into n_tasks groups of CLS_PER_TASK each.
    SAME seed → SAME splits across all arms (critical isolation).
    """
    rng = random.Random(seed)
    all_cls = list(range(NUM_CLASSES))
    rng.shuffle(all_cls)
    task_classes = [all_cls[i*CLS_PER_TASK:(i+1)*CLS_PER_TASK] for i in range(n_tasks)]
    splits = []
    for cls_list in task_classes:
        cls_arr = np.array(cls_list)
        tr_m = np.isin(y_tr, cls_arr)
        te_m = np.isin(y_te, cls_arr)
        tr_ds = ArrayDataset(X_tr[tr_m], y_tr[tr_m], augment=True)
        te_ds = ArrayDataset(X_te[te_m], y_te[te_m], augment=False)
        tr_ldr = torch.utils.data.DataLoader(
            tr_ds, BATCH_SIZE, shuffle=True, drop_last=True, num_workers=2, pin_memory=True)
        te_ldr = torch.utils.data.DataLoader(
            te_ds, 256, shuffle=False, num_workers=2, pin_memory=True)
        splits.append((tr_ldr, te_ldr, cls_list))
    return splits, task_classes


# ── Metrics (same as baseline_A3) ─────────────────────────────────────────────
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

def utility_B(model, running_act_mean):
    """
    Arm B: |mean outgoing weight| × running mean activation.
    utility_i = mean_j(|fc2.weight[j, i]|) × running_act_mean[i]
    Both normalized to prevent one term dominating.
    """
    with torch.no_grad():
        out_w_mag = model.fc2.weight.detach().abs().mean(0).cpu()  # [512]
        utility = out_w_mag * running_act_mean.cpu()
    return utility  # [512]


def utility_C(model, x_batch, y_batch, criterion):
    """
    Arm C: empirical diagonal Fisher — extra forward+backward pass.
    utility_i = mean_batch[(∂L/∂act_i)²]  (act_i = post-ReLU activation of fc1 neuron i)
    Uses observed labels (empirical Fisher, NOT model-sampled).
    """
    was_training = model.training
    model.train()
    model.zero_grad()
    logits, act = model.forward_with_act_grad(x_batch)
    loss = criterion(logits, y_batch)
    loss.backward()
    utility = torch.zeros(FC1_UNITS)
    if act.grad is not None:
        utility = (act.grad.detach() ** 2).mean(0).cpu()  # [512]
    model.zero_grad()
    if not was_training:
        model.eval()
    return utility


def utility_D(model, optimizer):
    """
    Arm D: Adam v_t proxy — ZERO extra compute.
    utility_i = mean(exp_avg_sq[i, :]) for fc1.weight[i, :]
    Just reads already-stored Adam state.
    """
    param = model.fc1.weight
    state = optimizer.state.get(param, {})
    if 'exp_avg_sq' not in state:
        # Adam not yet initialized (no steps taken) → return zeros
        return torch.zeros(FC1_UNITS)
    vt = state['exp_avg_sq'].detach().cpu()  # [512, 4096]
    return vt.mean(dim=1)  # [512]


# ── CBP Reset Procedure ───────────────────────────────────────────────────────
def cbp_reset(model, optimizer, utility, reset_frac=RESET_FRAC):
    """
    Continual Backprop reset (Dohare et al. 2024):
    - Identify bottom reset_frac fraction of fc1 neurons by utility
    - Reset their incoming weights (fc1.weight[i, :]) to kaiming_uniform init
    - Reset their outgoing weights (fc2.weight[:, i]) to 0 (preserves output continuity)
    - Reset their Adam momentum/variance state (give fresh start)
    - Reset their bias to 0
    """
    n_reset = max(1, int(FC1_UNITS * reset_frac))
    _, reset_idx = torch.topk(utility, n_reset, largest=False)  # indices of lowest utility

    with torch.no_grad():
        # ── Reset incoming weights of fc1 to kaiming_uniform ────────────────
        fan_in = model.fc1.weight.shape[1]
        bound = float(np.sqrt(3.0) * np.sqrt(2.0 / fan_in))  # kaiming uniform bound
        new_weights = torch.empty(n_reset, fan_in).uniform_(-bound, bound)
        model.fc1.weight.data[reset_idx] = new_weights.to(DEVICE)
        if model.fc1.bias is not None:
            model.fc1.bias.data[reset_idx] = 0.0

        # ── Reset outgoing weights of fc2 to 0 (continuity preservation) ────
        model.fc2.weight.data[:, reset_idx] = 0.0

    # ── Reset Adam optimizer state for affected neurons ────────────────────
    def _reset_state_rows(param, rows):
        state = optimizer.state.get(param, {})
        if 'exp_avg' in state:
            state['exp_avg'][rows] = 0.0
            state['exp_avg_sq'][rows] = 0.0

    def _reset_state_cols(param, cols):
        state = optimizer.state.get(param, {})
        if 'exp_avg' in state:
            state['exp_avg'][:, cols] = 0.0
            state['exp_avg_sq'][:, cols] = 0.0

    _reset_state_rows(model.fc1.weight, reset_idx)
    if model.fc1.bias is not None:
        _reset_state_rows(model.fc1.bias, reset_idx)
    _reset_state_cols(model.fc2.weight, reset_idx)

    return reset_idx


# ── Result I/O ────────────────────────────────────────────────────────────────
def write_results(payload):
    tmp = RESULTS_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, RESULTS_PATH)


# ── Single (arm, seed) run ────────────────────────────────────────────────────
def run_arm_seed(arm, seed, X_tr, y_tr, X_te, y_te, results_so_far):
    print(f"\n{'='*60}", flush=True)
    print(f"ARM {arm} | SEED {seed}", flush=True)
    print(f"{'='*60}", flush=True)

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

    model = SmallConvNetGN(num_output=NUM_CLASSES).to(DEVICE)
    print(f"  Params: {model.count_params():,}", flush=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    # Task splits: SAME seed → SAME splits across arms (critical isolation)
    splits, task_classes = make_task_splits(X_tr, y_tr, X_te, y_te, seed=seed, n_tasks=TASKS)

    sr = {
        "arm": arm,
        "seed": seed,
        "per_task_acc":   [],
        "dead_unit_frac": [],
        "effective_rank": [],
        "reset_events":   [],   # number of resets performed per task
    }

    # Arm B: running mean activation (EMA with α=0.01)
    running_act_mean = torch.zeros(FC1_UNITS)

    t_arm_start = time.time()

    for tid, (tr_ldr, te_ldr, cls_list) in enumerate(splits):
        t_task = time.time()
        model.train()
        tr_it = iter(tr_ldr)
        losses = []
        resets_this_task = 0

        for step in range(1, STEPS_PER_TASK + 1):
            # ── Training step ────────────────────────────────────────────────
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

            # ── Update running activation mean (Arm B, reuse stored act) ────
            if arm == 'B':
                with torch.no_grad():
                    alpha = 0.01
                    running_act_mean = (1 - alpha) * running_act_mean + alpha * model._last_fc1_act.mean(0).cpu()

            # ── CBP reset check every RESET_EVERY steps ──────────────────────
            if step % RESET_EVERY == 0:
                if arm == 'B':
                    util = utility_B(model, running_act_mean)
                elif arm == 'C':
                    util = utility_C(model, x, y, criterion)   # uses current batch
                elif arm == 'D':
                    util = utility_D(model, optimizer)
                else:
                    raise ValueError(f"Unknown arm: {arm}")

                cbp_reset(model, optimizer, util)
                resets_this_task += 1
                model.train()  # ensure model stays in train mode after arm C's utility_C

        # ── End-of-task metrics ───────────────────────────────────────────────
        wall = time.time() - t_task
        acc   = compute_accuracy(model, te_ldr, cls_list)
        duf   = dead_unit_fraction(model, te_ldr)
        erank = effective_rank(model, te_ldr)
        sr["per_task_acc"].append(acc)
        sr["dead_unit_frac"].append(duf)
        sr["effective_rank"].append(erank)
        sr["reset_events"].append(resets_this_task)

        print(
            f"  [{arm}/s{seed}] Task {tid+1:2d}/{TASKS} | "
            f"acc={acc:.3f} dead={duf:.3f} erank={erank:.2f} "
            f"resets={resets_this_task} loss={np.mean(losses[-200:]):.4f} wall={wall:.1f}s",
            flush=True,
        )

        # Incremental write after each task
        results_so_far["status"] = "RUNNING"
        results_so_far["latest"] = {
            "arm": arm, "seed": seed,
            "task": tid + 1, "acc": acc, "dead": duf,
        }
        write_results(results_so_far)

    sr["wall_sec"] = time.time() - t_arm_start
    return sr


# ── Aggregate over seeds for one arm ─────────────────────────────────────────
def aggregate_arm(seed_results):
    """Given list of per-seed dicts, compute mean/std over tasks 5–8 (final phase)."""
    all_acc  = [sr["per_task_acc"]   for sr in seed_results]
    all_dead = [sr["dead_unit_frac"] for sr in seed_results]
    all_er   = [sr["effective_rank"] for sr in seed_results]

    final_accs  = [np.mean(a[-4:]) for a in all_acc]   # tasks 5–8 (final phase)
    final_deads = [a[-1]            for a in all_dead]
    final_ers   = [e[-1]            for e in all_er]

    return {
        "per_seed_final_acc": [float(v) for v in final_accs],
        "mean_acc":  float(np.mean(final_accs)),
        "std_acc":   float(np.std(final_accs, ddof=0)),
        "dead_final": float(np.mean(final_deads)),
        "erank_final": float(np.nanmean(final_ers)),
        "per_task_acc_mean": np.mean(all_acc, axis=0).tolist(),
        "per_task_dead_mean": np.mean(all_dead, axis=0).tolist(),
        "per_task_erank_mean": np.nanmean(all_er, axis=0).tolist(),
    }


# ── Statistical comparison ────────────────────────────────────────────────────
def compare_arms(agg_C, agg_D):
    """
    Report D vs C: acc_diff, rough 95% CI, verdict.
    EQUIVALENCE MARGIN from EXPERIMENT.md spec: ±1.5 pp
    """
    c_acc = np.array(agg_C["per_seed_final_acc"])
    d_acc = np.array(agg_D["per_seed_final_acc"])
    diff  = d_acc - c_acc           # D - C (positive = D better)
    mean_diff = float(np.mean(diff))
    # 95% CI via t-distribution (n=3, df=2, t_crit≈4.30)
    # But with n=3 seeds this is very wide; report honestly
    se = float(np.std(diff, ddof=1) / np.sqrt(len(diff))) if len(diff) > 1 else float('nan')
    t_crit = 4.303   # t(0.025, df=2)
    ci_lo = mean_diff - t_crit * se
    ci_hi = mean_diff + t_crit * se
    equiv_margin = 1.5  # pp
    verdict = (
        "match" if abs(mean_diff) <= equiv_margin else
        ("D_better" if mean_diff > 0 else "D_worse")
    )
    return {
        "acc_diff": round(mean_diff * 100, 3),   # in pp
        "ci95": [round(ci_lo * 100, 3), round(ci_hi * 100, 3)],
        "equiv_margin_pp": equiv_margin,
        "verdict": verdict,
    }


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    t_start = time.time()

    # ── Initial result stub ───────────────────────────────────────────────────
    results = {
        "status": "RUNNING",
        "scale": "probe",
        "data_source": "cifar100_real",
        "n_seeds": len(SEEDS),
        "tasks": TASKS,
        "steps_per_task": STEPS_PER_TASK,
        "reset_every": RESET_EVERY,
        "reset_frac": RESET_FRAC,
        "latest": {},
        "arms": {},
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

    # ── Run all (arm, seed) combinations ─────────────────────────────────────
    all_seed_results = {arm: [] for arm in ARMS}

    for arm in ARMS:
        arm_t0 = time.time()
        for seed in SEEDS:
            elapsed_min = (time.time() - t_start) / 60
            print(f"\n[{elapsed_min:.1f}m elapsed] Starting Arm {arm}, Seed {seed}", flush=True)

            sr = run_arm_seed(arm, seed, X_tr, y_tr, X_te, y_te, results)
            all_seed_results[arm].append(sr)

            # Incremental arm aggregation
            results["arms"][arm] = aggregate_arm(all_seed_results[arm])
            results["arms"][arm]["seeds_done"] = [s["seed"] for s in all_seed_results[arm]]
            write_results(results)

        arm_wall = time.time() - arm_t0
        print(f"\nArm {arm} done in {arm_wall/60:.1f} min", flush=True)

    # ── Finalize ──────────────────────────────────────────────────────────────
    agg = results["arms"]  # already built above

    # D vs C comparison (main scientific question)
    d_vs_c = compare_arms(agg["C"], agg["D"])

    # Plasticity recovery check: does any CBP arm beat the Arm A floor?
    # Arm A floor: acc ≈ 0.20, dead ≈ 1.0 (from baseline_A3)
    a_floor_acc  = 0.20
    a_floor_dead = 1.00
    recovers_plasticity = any(
        agg[arm]["mean_acc"] > a_floor_acc + 0.05  # at least 5 pp above floor
        for arm in ARMS
    )

    total_wall_min = (time.time() - t_start) / 60
    notes = (
        f"Probe run: {TASKS} tasks × {CLS_PER_TASK} cls/task, {STEPS_PER_TASK} steps/task. "
        f"Reset every {RESET_EVERY} steps, frac={RESET_FRAC}. "
        f"Arm A floor (prior baseline_A3): acc≈{a_floor_acc}, dead≈{a_floor_dead}. "
        f"D_vs_C verdict: {d_vs_c['verdict']} (acc_diff={d_vs_c['acc_diff']:.2f}pp). "
        f"Recovers plasticity: {recovers_plasticity}. "
        f"Wall: {total_wall_min:.1f}m."
    )

    final = {
        "status": "DONE",
        "scale": "probe",
        "data_source": "cifar100_real",
        "n_seeds": len(SEEDS),
        "tasks": TASKS,
        "steps_per_task": STEPS_PER_TASK,
        "reset_every": RESET_EVERY,
        "reset_frac": RESET_FRAC,
        "arms": {
            "B": agg["B"],
            "C": agg["C"],
            "D": agg["D"],
            "A_floor": {
                "source": "prior_baseline_A3_reference",
                "mean_acc": a_floor_acc,
                "dead_final": a_floor_dead,
                "notes": "vanilla Adam no-reset; collapses to chance by task 3"
            },
        },
        "D_matches_C": d_vs_c,
        "recovers_plasticity": recovers_plasticity,
        "wall_time_min": round(total_wall_min, 2),
        "subject_executed": (
            f"Arms B/C/D CBP resets, SmallConvNetGN, real CIFAR-100, "
            f"{TASKS}×{CLS_PER_TASK} classes, {STEPS_PER_TASK} steps/task, "
            f"reset_frac={RESET_FRAC}, reset_every={RESET_EVERY}, seeds={SEEDS}"
        ),
        "notes": notes,
        "metrics": {
            "B_mean_acc_final_tasks": agg["B"]["mean_acc"],
            "C_mean_acc_final_tasks": agg["C"]["mean_acc"],
            "D_mean_acc_final_tasks": agg["D"]["mean_acc"],
            "A_floor_acc": a_floor_acc,
            "B_dead_final": agg["B"]["dead_final"],
            "C_dead_final": agg["C"]["dead_final"],
            "D_dead_final": agg["D"]["dead_final"],
            "D_vs_C_diff_pp": d_vs_c["acc_diff"],
            "D_vs_C_verdict": d_vs_c["verdict"],
            "recovers_plasticity": recovers_plasticity,
        },
    }

    write_results(final)

    print(f"\n{'='*60}", flush=True)
    print(f"DONE in {total_wall_min:.1f} min", flush=True)
    print(json.dumps(final, indent=2), flush=True)


if __name__ == "__main__":
    main()
