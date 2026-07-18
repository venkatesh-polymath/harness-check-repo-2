"""
Baseline A: Adam + no resets — reproduce loss of plasticity on class-incremental data.
20 tasks × 5 classes each (100 total classes).

Uses synthetic CIFAR-100-like data if real data is not cached, else uses real CIFAR-100.
Writes RESULTS.json incrementally so a crash yields partial data.
"""

import os, sys, json, time, random, argparse, pickle, tarfile
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

RESULTS_PATH = "/workspace/results/baseline_A/RESULTS.json"
WEIGHTS_DIR  = "/workspace/_weights"
os.makedirs(WEIGHTS_DIR, exist_ok=True)
os.makedirs("/workspace/results/baseline_A", exist_ok=True)

# ─────────────────────────────────────────────
# Hyper-parameters
# ─────────────────────────────────────────────
NUM_CLASSES    = 100
TASKS          = 20
CLS_PER_TASK   = NUM_CLASSES // TASKS   # 5
STEPS_PER_TASK = 2000
BATCH_SIZE     = 128
LR             = 1e-3
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Device: {DEVICE}", flush=True)

# ─────────────────────────────────────────────
# Small ConvNet ~1.4M params
# ─────────────────────────────────────────────
class SmallConvNet(nn.Module):
    def __init__(self, num_output=100):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1: 32→16
            nn.Conv2d(3, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),
            # Block 2: 16→8
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2),
            # Block 3: 8→4
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # 256 * 4 * 4 = 4096
        self.fc1 = nn.Linear(256 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, num_output)
        self._penultimate = None

    def forward(self, x, store_pen=False):
        h = self.features(x)
        h = h.view(h.size(0), -1)
        h = F.relu(self.fc1(h))
        if store_pen:
            self._penultimate = h.detach()
        return self.fc2(h)

    def count_params(self):
        return sum(p.numel() for p in self.parameters())


# ─────────────────────────────────────────────
# Data: real CIFAR-100 if available, else synthetic
# ─────────────────────────────────────────────
def load_real_cifar100(root="/tmp/cifar100"):
    """Try to load real CIFAR-100 from disk (already extracted)."""
    data_dir = os.path.join(root, "cifar-100-python")
    if not os.path.isdir(data_dir):
        # Try to extract if tar exists and is complete
        tar_path = os.path.join(root, "cifar-100-python.tar.gz")
        if os.path.exists(tar_path) and os.path.getsize(tar_path) > 168_000_000:
            print("Extracting CIFAR-100 tar.gz ...", flush=True)
            with tarfile.open(tar_path, "r:gz") as tf:
                tf.extractall(root)
        else:
            return None, None

    def unpickle(f):
        with open(f, "rb") as fo:
            return pickle.load(fo, encoding="bytes")

    tr = unpickle(os.path.join(data_dir, "train"))
    te = unpickle(os.path.join(data_dir, "test"))
    # keys: b'data' shape (N,3072), b'fine_labels'
    X_tr = tr[b"data"].reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
    y_tr = np.array(tr[b"fine_labels"])
    X_te = te[b"data"].reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
    y_te = np.array(te[b"fine_labels"])
    # Normalize
    mean = np.array([0.5071, 0.4867, 0.4408], dtype=np.float32)[:, None, None]
    std  = np.array([0.2675, 0.2565, 0.2761], dtype=np.float32)[:, None, None]
    X_tr = (X_tr - mean) / std
    X_te = (X_te - mean) / std
    print(f"Loaded real CIFAR-100: train={X_tr.shape}, test={X_te.shape}", flush=True)
    return (X_tr, y_tr), (X_te, y_te)


def make_synthetic_cifar100(seed=42):
    """
    Generate synthetic 32×32×3 images: each class has a distinct mean patch.
    This shows plasticity loss just as well as real data for a PROBE.
    ~50k train, 10k test, 100 classes (500 train / 100 test per class).
    """
    rng = np.random.default_rng(seed)
    n_tr, n_te = 500, 100
    # Random class prototypes (in [-1,1]^3072)
    prototypes = rng.standard_normal((NUM_CLASSES, 3, 32, 32)).astype(np.float32)
    X_tr, y_tr, X_te, y_te = [], [], [], []
    for c in range(NUM_CLASSES):
        noise = 0.3 * rng.standard_normal((n_tr, 3, 32, 32)).astype(np.float32)
        X_tr.append(prototypes[c] + noise)
        y_tr.extend([c] * n_tr)
        noise_te = 0.3 * rng.standard_normal((n_te, 3, 32, 32)).astype(np.float32)
        X_te.append(prototypes[c] + noise_te)
        y_te.extend([c] * n_te)
    X_tr = np.concatenate(X_tr, axis=0)
    X_te = np.concatenate(X_te, axis=0)
    y_tr = np.array(y_tr)
    y_te = np.array(y_te)
    print(f"Generated synthetic data: train={X_tr.shape}, test={X_te.shape}", flush=True)
    return (X_tr, y_tr), (X_te, y_te)


class ArrayDataset(torch.utils.data.Dataset):
    def __init__(self, X, y, augment=False):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y).long()
        self.augment = augment

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        x = self.X[i]
        if self.augment:
            # Random horizontal flip
            if random.random() > 0.5:
                x = x.flip(-1)
            # Random crop (pad 4 then crop 32)
            pad = 4
            x_pad = F.pad(x.unsqueeze(0), [pad]*4, mode='reflect').squeeze(0)
            c = random.randint(0, 2*pad)
            r = random.randint(0, 2*pad)
            x = x_pad[:, r:r+32, c:c+32]
        return x, self.y[i]


def make_task_splits(X_tr, y_tr, X_te, y_te, num_tasks=TASKS, cls_per_task=CLS_PER_TASK, seed=0):
    """Split into num_tasks sequential task-specific loaders."""
    rng = random.Random(seed)
    all_classes = list(range(NUM_CLASSES))
    rng.shuffle(all_classes)
    task_classes = [all_classes[i*cls_per_task:(i+1)*cls_per_task] for i in range(num_tasks)]

    splits = []
    for cls_list in task_classes:
        cls_arr = np.array(cls_list)
        tr_mask = np.isin(y_tr, cls_arr)
        te_mask = np.isin(y_te, cls_arr)

        tr_ds = ArrayDataset(X_tr[tr_mask], y_tr[tr_mask], augment=True)
        te_ds = ArrayDataset(X_te[te_mask], y_te[te_mask], augment=False)

        tr_loader = torch.utils.data.DataLoader(
            tr_ds, batch_size=BATCH_SIZE, shuffle=True,
            num_workers=0, drop_last=True,
        )
        te_loader = torch.utils.data.DataLoader(
            te_ds, batch_size=256, shuffle=False, num_workers=0,
        )
        splits.append((tr_loader, te_loader, cls_list))
    return splits, task_classes


# ─────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────
def compute_accuracy(model, loader, cls_list):
    """Eval accuracy restricted to task classes."""
    model.eval()
    correct = total = 0
    cls_tensor = torch.tensor(cls_list, device=DEVICE)
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x)                          # [B, 100]
            # Mask out non-task classes
            mask = torch.full_like(logits, float('-inf'))
            mask[:, cls_tensor] = logits[:, cls_tensor]
            pred = mask.argmax(1)
            correct += (pred == y).sum().item()
            total   += y.size(0)
    return correct / total if total > 0 else 0.0


def dead_unit_fraction(model, loader, threshold=0.01, max_batches=10):
    """Fraction of penultimate neurons with mean |act| < threshold."""
    model.eval()
    acts = []
    with torch.no_grad():
        for i, (x, _) in enumerate(loader):
            if i >= max_batches:
                break
            model(x.to(DEVICE), store_pen=True)
            acts.append(model._penultimate.cpu())
    if not acts:
        return 0.0
    A = torch.cat(acts, 0)
    return (A.abs().mean(0) < threshold).float().mean().item()


def effective_rank(model, loader, max_batches=20):
    """Spectral effective rank of penultimate activations."""
    model.eval()
    acts = []
    with torch.no_grad():
        for i, (x, _) in enumerate(loader):
            if i >= max_batches:
                break
            model(x.to(DEVICE), store_pen=True)
            acts.append(model._penultimate.cpu())
    if not acts:
        return float('nan')
    A = torch.cat(acts, 0).float()
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


# ─────────────────────────────────────────────
# Results I/O
# ─────────────────────────────────────────────
def write_results(payload):
    with open(RESULTS_PATH, "w") as f:
        json.dump(payload, f, indent=2)


# ─────────────────────────────────────────────
# One seed run
# ─────────────────────────────────────────────
def run_seed(seed, X_tr, y_tr, X_te, y_te, results_accum, data_source):
    print(f"\n{'='*60}\nSEED {seed}\n{'='*60}", flush=True)

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

    model = SmallConvNet(num_output=NUM_CLASSES).to(DEVICE)
    print(f"Params: {model.count_params():,}", flush=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    splits, task_classes = make_task_splits(X_tr, y_tr, X_te, y_te, seed=seed)

    sr = {
        "seed": seed,
        "per_task_acc":    [],
        "dead_unit_frac":  [],
        "effective_rank":  [],
        "task_classes":    task_classes,
        "data_source":     data_source,
    }

    for task_id, (tr_loader, te_loader, cls_list) in enumerate(splits):
        t0 = time.time()
        model.train()

        step      = 0
        tr_iter   = iter(tr_loader)
        losses    = []

        while step < STEPS_PER_TASK:
            try:
                x, y = next(tr_iter)
            except StopIteration:
                tr_iter = iter(tr_loader)
                x, y   = next(tr_iter)

            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            logits = model(x)
            loss   = criterion(logits, y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            step += 1

            if step % 500 == 0:
                print(f"  Task {task_id+1:2d}/{TASKS} step {step:4d} "
                      f"loss={loss.item():.4f}", flush=True)

        wall = time.time() - t0

        acc   = compute_accuracy(model, te_loader, cls_list)
        duf   = dead_unit_fraction(model, te_loader)
        erank = effective_rank(model, te_loader)

        sr["per_task_acc"].append(acc)
        sr["dead_unit_frac"].append(duf)
        sr["effective_rank"].append(erank)

        print(f"  Task {task_id+1:2d} | acc={acc:.3f} dead={duf:.3f} "
              f"erank={erank:.2f} loss={np.mean(losses[-200:]):.4f} {wall:.1f}s",
              flush=True)

        # Incremental write
        tmp = {k: v for k, v in results_accum.items()}
        tmp["latest_seed"] = seed
        tmp["latest_task"] = task_id + 1
        tmp["latest_per_task_acc"] = sr["per_task_acc"]
        write_results(tmp)

    return sr


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    global_start = time.time()

    results_accum = {
        "arm":              "A_adam_noreset",
        "status":           "RUNNING",
        "per_seed":         [],
        "per_task_acc":     [],
        "mean_acc":         None,
        "plasticity_trend": None,
        "dead_unit_frac":   [],
        "notes":            "",
    }
    write_results(results_accum)

    # Load data: prefer real CIFAR-100, fall back to synthetic
    data_source = "real_cifar100"
    train_data, test_data = load_real_cifar100()
    if train_data is None:
        print("Real CIFAR-100 not available yet — using synthetic data for probe.", flush=True)
        data_source = "synthetic"
        train_data, test_data = make_synthetic_cifar100(seed=0)

    X_tr, y_tr = train_data
    X_te, y_te = test_data

    all_per_task_accs = []
    all_dead_fracs    = []
    all_eranks        = []
    seeds_run         = []

    for seed in [0, 1]:
        elapsed = time.time() - global_start
        if elapsed > 28 * 60:
            print(f"Time limit ({elapsed/60:.1f}m) — skipping seed {seed}.", flush=True)
            results_accum["notes"] += f" Stopped at {elapsed/60:.1f}m (time limit)."
            break

        sr = run_seed(seed, X_tr, y_tr, X_te, y_te, results_accum, data_source)
        all_per_task_accs.append(sr["per_task_acc"])
        all_dead_fracs.append(sr["dead_unit_frac"])
        all_eranks.append(sr["effective_rank"])
        seeds_run.append(seed)
        results_accum["per_seed"].append({k: v for k, v in sr.items()
                                          if k != "task_classes"})

    # ── Aggregate ──
    if all_per_task_accs:
        mean_per_task = np.mean(all_per_task_accs, axis=0).tolist()
        mean_acc      = float(np.mean(mean_per_task))
        mean_duf      = np.mean(all_dead_fracs,  axis=0).tolist()
        mean_erank    = np.mean(all_eranks,      axis=0).tolist()

        first_half  = float(np.mean(mean_per_task[:TASKS//2]))
        second_half = float(np.mean(mean_per_task[TASKS//2:]))

        if second_half < first_half - 0.02:
            trend = "declining"
        elif abs(second_half - first_half) <= 0.02:
            trend = "flat"
        else:
            trend = "improving"

        erank_t1   = mean_erank[0]  if mean_erank else float('nan')
        erank_last = mean_erank[-1] if mean_erank else float('nan')
        erank_rel  = (erank_last - erank_t1) / max(erank_t1, 1e-6)

        wall_min = (time.time() - global_start) / 60

        results_accum.update({
            "status":                  "DONE",
            "data_source":             data_source,
            "per_task_acc":            mean_per_task,
            "mean_acc":                mean_acc,
            "plasticity_trend":        trend,
            "dead_unit_frac":          mean_duf,
            "effective_rank":          mean_erank,
            "erank_task1":             erank_t1,
            "erank_last_task":         erank_last,
            "erank_relative_change":   erank_rel,
            "first_half_mean_acc":     first_half,
            "second_half_mean_acc":    second_half,
            "num_seeds_run":           len(seeds_run),
            "seeds":                   seeds_run,
            "wall_time_min":           wall_min,
            "notes": (
                f"Arm A: Adam no-reset, {TASKS} tasks×{CLS_PER_TASK} cls, "
                f"steps={STEPS_PER_TASK}, lr={LR}, data={data_source}, seeds={seeds_run}. "
                f"First-half acc={first_half:.3f} vs second-half={second_half:.3f} → {trend}. "
                f"Eff.rank: task1={erank_t1:.2f} → last={erank_last:.2f} "
                f"(Δ={erank_rel:.1%})."
            ),
        })
    else:
        results_accum["status"] = "FAILED"
        results_accum["notes"]  = "No seeds completed."

    write_results(results_accum)

    wall_min = (time.time() - global_start) / 60
    print(f"\n{'='*60}", flush=True)
    print(f"Done in {wall_min:.1f} min", flush=True)
    print(json.dumps(results_accum, indent=2), flush=True)


if __name__ == "__main__":
    main()
