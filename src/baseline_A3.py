"""
baseline_A3 — Arm A: Adam + no resets + GroupNorm, REAL CIFAR-100.

This is the clean reference run for the study:
  • Real CIFAR-100 loaded from /opt/datasets (pre-baked into image).
  • download=False — data is already present.
  • If data cannot be loaded → ABORT with RESULTS.json status:"FAILED". NO synthetic fallback.
  • 20 tasks × 5 classes/task = 100 classes, class-incremental.
  • 2 seeds to completion.
  • Metrics: per-task accuracy, dead-unit fraction (ReLU units ~0), effective rank.
  • Output: results/baseline_A3/

Inherits model from src/baseline_A_v2.py (SmallConvNetGN, 3 conv-blocks + 2 FC, GroupNorm).
"""

import os, sys, json, time, random, pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

RESULTS_DIR  = "/workspace/results/baseline_A3"
RESULTS_PATH = os.path.join(RESULTS_DIR, "RESULTS.json")
WEIGHTS_DIR  = "/workspace/_weights"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(WEIGHTS_DIR, exist_ok=True)

# ── Hyper-parameters ─────────────────────────────────────────────────────────
NUM_CLASSES    = 100
TASKS          = 20          # 20 tasks × 5 classes = 100 classes
CLS_PER_TASK   = NUM_CLASSES // TASKS   # 5
STEPS_PER_TASK = 2000
BATCH_SIZE     = 128
LR             = 1e-3
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEEDS          = [0, 1]
WALL_LIMIT_MIN = 33          # stop after this many minutes (budget = 35 min)

# Dataset location (pre-baked in image)
DATASETS_DIR = os.environ.get("SH_DATASETS_DIR", "/opt/datasets")

print(f"Device: {DEVICE}", flush=True)
print(f"Datasets dir: {DATASETS_DIR}", flush=True)


# ── Model: GroupNorm ConvNet ──────────────────────────────────────────────────
class SmallConvNetGN(nn.Module):
    """3 conv-blocks + 2 FC with GroupNorm. No running statistics → no BN collapse."""
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
        self.fc1 = nn.Linear(256 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, num_output)
        self._penultimate = None

    def forward(self, x, store_pen=False):
        h = self.features(x).view(x.size(0), -1)
        h = F.relu(self.fc1(h))
        if store_pen:
            self._penultimate = h.detach()
        return self.fc2(h)

    def count_params(self):
        return sum(p.numel() for p in self.parameters())


# ── Data loading ──────────────────────────────────────────────────────────────
def load_real_cifar100():
    """
    Load CIFAR-100 from /opt/datasets. Tries torchvision first, then direct pickle.
    Returns ((X_tr, y_tr), (X_te, y_te)) on success or aborts on failure.
    NO fallback to synthetic data.
    """
    # Method 1: torchvision (cleanest)
    try:
        import torchvision.datasets as tvds
        tr_ds = tvds.CIFAR100(root=DATASETS_DIR, train=True,  download=False)
        te_ds = tvds.CIFAR100(root=DATASETS_DIR, train=False, download=False)
        X_tr = np.array(tr_ds.data, dtype=np.float32).transpose(0, 3, 1, 2) / 255.0
        y_tr = np.array(tr_ds.targets, dtype=np.int64)
        X_te = np.array(te_ds.data, dtype=np.float32).transpose(0, 3, 1, 2) / 255.0
        y_te = np.array(te_ds.targets, dtype=np.int64)
        print(f"[torchvision] Loaded CIFAR-100: train={X_tr.shape}, test={X_te.shape}", flush=True)
        return _normalize(X_tr, y_tr, X_te, y_te)
    except Exception as e:
        print(f"[torchvision] Failed: {e}. Trying direct pickle...", flush=True)

    # Method 2: direct pickle
    data_dir = os.path.join(DATASETS_DIR, "cifar-100-python")
    if not os.path.isdir(data_dir):
        return None, None   # caller will abort

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
        print(f"[pickle] Loaded CIFAR-100: train={X_tr.shape}, test={X_te.shape}", flush=True)
        return _normalize(X_tr, y_tr, X_te, y_te)
    except Exception as e:
        print(f"[pickle] Failed: {e}", flush=True)
        return None, None


def _normalize(X_tr, y_tr, X_te, y_te):
    mean = np.array([0.5071, 0.4867, 0.4408], dtype=np.float32)[:, None, None]
    std  = np.array([0.2675, 0.2565, 0.2761], dtype=np.float32)[:, None, None]
    X_tr = (X_tr - mean) / std
    X_te = (X_te - mean) / std
    print(f"  Normalized: X_tr range [{X_tr.min():.2f}, {X_tr.max():.2f}]", flush=True)
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


def make_task_splits(X_tr, y_tr, X_te, y_te, seed=0):
    """Split 100 classes into TASKS groups of CLS_PER_TASK each."""
    rng = random.Random(seed)
    all_cls = list(range(NUM_CLASSES))
    rng.shuffle(all_cls)
    task_classes = [all_cls[i*CLS_PER_TASK:(i+1)*CLS_PER_TASK] for i in range(TASKS)]
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


# ── Metrics ───────────────────────────────────────────────────────────────────
def compute_accuracy(model, loader, cls_list):
    model.eval()
    correct = total = 0
    cls_t = torch.tensor(cls_list, device=DEVICE)
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x)
            # Mask to only the classes in this task
            mask = torch.full_like(logits, float('-inf'))
            mask[:, cls_t] = logits[:, cls_t]
            correct += (mask.argmax(1) == y).sum().item()
            total   += y.size(0)
    return correct / total if total else 0.0


def dead_unit_fraction(model, loader, thr=0.01, n=10):
    """Fraction of penultimate-layer units with mean |activation| < thr."""
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
    """
    Spectral entropy / effective rank of penultimate-layer activations.
    Returns 0.0 if all units are dead (max activation < 1e-10).
    Returns NaN on SVD failure.
    """
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


# ── Result I/O ────────────────────────────────────────────────────────────────
def write_results(payload):
    tmp = RESULTS_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, RESULTS_PATH)


# ── Single seed ───────────────────────────────────────────────────────────────
def run_seed(seed, X_tr, y_tr, X_te, y_te, results_accum):
    print(f"\n{'='*60}\nSEED {seed}\n{'='*60}", flush=True)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

    model = SmallConvNetGN(num_output=NUM_CLASSES).to(DEVICE)
    print(f"Params: {model.count_params():,}", flush=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    splits, task_classes = make_task_splits(X_tr, y_tr, X_te, y_te, seed=seed)

    sr = {
        "seed": seed,
        "per_task_acc": [],
        "dead_unit_frac": [],
        "effective_rank": [],
        "task_classes": task_classes,
        "data_source": "cifar100_real",
    }

    for tid, (tr_ldr, te_ldr, cls_list) in enumerate(splits):
        t0 = time.time()
        model.train()
        step = 0
        tr_it = iter(tr_ldr)
        losses = []

        while step < STEPS_PER_TASK:
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
            step += 1
            if step % 500 == 0:
                print(f"  Task {tid+1:2d}/{TASKS} step {step:4d} loss={loss.item():.4f}", flush=True)

        wall = time.time() - t0
        acc   = compute_accuracy(model, te_ldr, cls_list)
        duf   = dead_unit_fraction(model, te_ldr)
        erank = effective_rank(model, te_ldr)
        sr["per_task_acc"].append(acc)
        sr["dead_unit_frac"].append(duf)
        sr["effective_rank"].append(erank)
        print(
            f"  Task {tid+1:2d} | acc={acc:.3f} dead={duf:.3f} erank={erank:.2f} "
            f"loss={np.mean(losses[-200:]):.4f} wall={wall:.1f}s",
            flush=True,
        )

        # Incremental result write after each task
        tmp_payload = {
            **results_accum,
            "latest_seed": seed,
            "latest_task": tid + 1,
            "latest_per_task_acc": sr["per_task_acc"],
        }
        write_results(tmp_payload)

    return sr


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    t_start = time.time()

    results_accum = {
        "arm": "A_adam_noreset_groupnorm_realcifar100",
        "status": "RUNNING",
        "data_source": "cifar100_real",
        "per_seed": [],
        "per_task_acc": [],
        "mean_acc": None,
        "plasticity_trend": None,
        "dead_unit_frac_final": None,
        "effective_rank_trend": None,
        "notes": "",
    }
    write_results(results_accum)

    # ── Load real CIFAR-100 — ABORT if unavailable ────────────────────────────
    print("\nLoading real CIFAR-100 from", DATASETS_DIR, flush=True)
    train_data, test_data = load_real_cifar100()
    if train_data is None:
        msg = (
            f"ABORT: Real CIFAR-100 not found at {DATASETS_DIR}. "
            "Synthetic fallback is not permitted in this round."
        )
        print(msg, flush=True)
        results_accum["status"] = "FAILED"
        results_accum["notes"] = msg
        write_results(results_accum)
        sys.exit(1)

    X_tr, y_tr = train_data
    X_te, y_te = test_data
    print(f"Data loaded. train={X_tr.shape} test={X_te.shape}", flush=True)

    all_accs   = []
    all_dufs   = []
    all_eranks = []
    seeds_done = []

    for seed in SEEDS:
        elapsed = (time.time() - t_start) / 60
        if elapsed > WALL_LIMIT_MIN:
            note = f" Time limit {WALL_LIMIT_MIN}m reached at {elapsed:.1f}m."
            print(note, flush=True)
            results_accum["notes"] += note
            break

        sr = run_seed(seed, X_tr, y_tr, X_te, y_te, results_accum)
        all_accs.append(sr["per_task_acc"])
        all_dufs.append(sr["dead_unit_frac"])
        all_eranks.append(sr["effective_rank"])
        seeds_done.append(seed)
        results_accum["per_seed"].append(
            {k: v for k, v in sr.items() if k != "task_classes"}
        )

    if not all_accs:
        results_accum.update({"status": "FAILED", "notes": "No seeds completed."})
        write_results(results_accum)
        print(json.dumps(results_accum, indent=2), flush=True)
        sys.exit(1)

    # ── Aggregate across seeds ────────────────────────────────────────────────
    mpt   = np.mean(all_accs,   axis=0).tolist()
    mduf  = np.mean(all_dufs,   axis=0).tolist()
    mer   = np.mean(all_eranks, axis=0).tolist()

    mean_acc = float(np.mean(mpt))
    h1 = float(np.mean(mpt[:TASKS // 2]))
    h2 = float(np.mean(mpt[TASKS // 2:]))
    trend = (
        "declining" if h2 < h1 - 0.02
        else ("flat" if abs(h2 - h1) <= 0.02 else "improving")
    )

    er1  = mer[0]  if mer else float('nan')
    er_l = mer[-1] if mer else float('nan')
    er_ch = (er_l - er1) / max(abs(er1), 1e-6)

    # Effective-rank trend (monotone decay? compute slope)
    erank_trend = "declining" if er_l < er1 * 0.8 else ("flat" if abs(er_ch) < 0.2 else "improving")

    results_accum.update({
        "status": "DONE",
        "scale": "probe",
        "data_source": "cifar100_real",
        "per_task_acc": mpt,
        "mean_acc": mean_acc,
        "plasticity_trend": trend,
        "first_half_mean_acc": h1,
        "second_half_mean_acc": h2,
        "dead_unit_frac": mduf,
        "dead_unit_frac_final": mduf[-1] if mduf else None,
        "effective_rank": mer,
        "erank_task1": er1,
        "erank_last_task": er_l,
        "erank_relative_change": er_ch,
        "effective_rank_trend": erank_trend,
        "num_seeds_run": len(seeds_done),
        "seeds": seeds_done,
        "wall_time_min": (time.time() - t_start) / 60,
        "metrics": {
            "mean_acc_all_tasks": mean_acc,
            "mean_acc_first_half": h1,
            "mean_acc_second_half": h2,
            "plasticity_trend": trend,
            "dead_unit_frac_task1": mduf[0] if mduf else None,
            "dead_unit_frac_final": mduf[-1] if mduf else None,
            "erank_task1": er1,
            "erank_last_task": er_l,
            "erank_relative_change_pct": er_ch * 100,
            "effective_rank_trend": erank_trend,
        },
        "subject_executed": (
            f"Arm A: SmallConvNetGN (GroupNorm), Adam lr=1e-3, no resets, "
            f"real CIFAR-100 ({TASKS} tasks × {CLS_PER_TASK} cls/task, "
            f"{STEPS_PER_TASK} steps/task, batch={BATCH_SIZE}), "
            f"seeds={seeds_done}"
        ),
        "notes": (
            f"Arm A baseline: Adam no-reset GroupNorm on REAL CIFAR-100. "
            f"data_source=cifar100_real. "
            f"seeds={seeds_done}. "
            f"Acc first-half={h1:.3f}, second-half={h2:.3f} → {trend}. "
            f"Erank: task1={er1:.2f} → last={er_l:.2f} (Δ={er_ch:.1%}) → {erank_trend}. "
            f"Dead units final={mduf[-1] if mduf else 'N/A':.3f}. "
            f"Wall={results_accum['wall_time_min']:.1f}m."
        ),
    })

    write_results(results_accum)

    # Print the final RESULTS.json as per experiment spec
    final = {
        "status": results_accum["status"],
        "scale": "probe",
        "metrics": results_accum["metrics"],
        "subject_executed": results_accum["subject_executed"],
        "notes": results_accum["notes"],
        # Study-level fields
        "arm": results_accum["arm"],
        "data_source": "cifar100_real",
        "per_task_acc": mpt,
        "mean_acc": mean_acc,
        "plasticity_trend": trend,
        "dead_unit_frac_final": mduf[-1] if mduf else None,
        "effective_rank_trend": erank_trend,
        "per_seed": results_accum["per_seed"],
        "wall_time_min": results_accum["wall_time_min"],
    }
    write_results(final)

    print(f"\n{'='*60}", flush=True)
    print(f"Done in {results_accum['wall_time_min']:.1f}m", flush=True)
    print(json.dumps(final, indent=2), flush=True)


if __name__ == "__main__":
    main()
