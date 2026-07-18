"""
Baseline A v2 — CLEAN RUN for baseline_A2 results directory.

Key changes vs baseline_A (v1):
  - GroupNorm instead of BatchNorm (avoids BN running-stat collapse with small batches)
  - Output goes to results/baseline_A2/ (not baseline_A/)
  - CIFAR-100 loaded via torchvision (preferred) or manual pickle, fallback to synthetic
  - Effective rank: guards against zero-matrix → reports 0.0, never a spurious 500
  - Results written incrementally after every task

Scientific question: Does average accuracy on NEW tasks decline across the sequence
(genuine loss of plasticity), with GroupNorm removing the BN artifact?
"""

import os, sys, json, time, random, pickle, tarfile
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

RESULTS_DIR  = "/workspace/results/baseline_A2"
RESULTS_PATH = os.path.join(RESULTS_DIR, "RESULTS.json")
WEIGHTS_DIR  = "/workspace/_weights"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(WEIGHTS_DIR, exist_ok=True)

# ── Hyper-parameters ─────────────────────────────────────────────────────────
NUM_CLASSES    = 100
TASKS          = 20          # 20 tasks × 5 classes = 100 classes
CLS_PER_TASK   = NUM_CLASSES // TASKS
STEPS_PER_TASK = 2000
BATCH_SIZE     = 128
LR             = 1e-3
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEEDS          = [0, 1]
WALL_LIMIT_MIN = 33          # stop after this many minutes to stay under 35 min budget

print(f"Device: {DEVICE}", flush=True)


# ── Model: GroupNorm ConvNet (3 conv-blocks + 2 FC) ──────────────────────────
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
def load_cifar100_torchvision(root="/tmp/cifar100"):
    """
    Try to load CIFAR-100 via torchvision WITHOUT downloading.
    Returns None,None if the data isn't already on disk (avoids slow download).
    """
    try:
        import torchvision
        # Only load if already extracted on disk (download=False avoids a slow network fetch)
        tr_ds = torchvision.datasets.CIFAR100(root=root, train=True, download=False)
        te_ds = torchvision.datasets.CIFAR100(root=root, train=False, download=False)
        mean = np.array([0.5071, 0.4867, 0.4408], dtype=np.float32)[:, None, None]
        std  = np.array([0.2675, 0.2565, 0.2761], dtype=np.float32)[:, None, None]
        X_tr = (tr_ds.data.transpose(0, 3, 1, 2).astype(np.float32) / 255.0 - mean) / std
        y_tr = np.array(tr_ds.targets, dtype=np.int64)
        X_te = (te_ds.data.transpose(0, 3, 1, 2).astype(np.float32) / 255.0 - mean) / std
        y_te = np.array(te_ds.targets, dtype=np.int64)
        print(f"Loaded real CIFAR-100 (torchvision): train={X_tr.shape} test={X_te.shape}", flush=True)
        return (X_tr, y_tr), (X_te, y_te)
    except Exception as e:
        print(f"torchvision CIFAR-100 not available without download: {e}", flush=True)
        return None, None


def load_cifar100_pickle(root="/tmp/cifar100"):
    """Try to load CIFAR-100 from manual pickle (extracted tar)."""
    data_dir = os.path.join(root, "cifar-100-python")
    if not os.path.isdir(data_dir):
        tar_path = os.path.join(root, "cifar-100-python.tar.gz")
        if os.path.exists(tar_path) and os.path.getsize(tar_path) > 168_000_000:
            print("Extracting CIFAR-100 ...", flush=True)
            with tarfile.open(tar_path, "r:gz") as tf:
                tf.extractall(root)
        else:
            size = os.path.getsize(tar_path) if os.path.exists(tar_path) else 0
            print(f"CIFAR-100 tar incomplete ({size/1e6:.1f}MB < 169MB). Skipping pickle.", flush=True)
            return None, None

    def unpickle(f):
        with open(f, "rb") as fo:
            return pickle.load(fo, encoding="bytes")
    try:
        tr = unpickle(os.path.join(data_dir, "train"))
        te = unpickle(os.path.join(data_dir, "test"))
        X_tr = tr[b"data"].reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
        y_tr = np.array(tr[b"fine_labels"])
        X_te = te[b"data"].reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
        y_te = np.array(te[b"fine_labels"])
        mean = np.array([0.5071, 0.4867, 0.4408], dtype=np.float32)[:, None, None]
        std  = np.array([0.2675, 0.2565, 0.2761], dtype=np.float32)[:, None, None]
        X_tr = (X_tr - mean) / std
        X_te = (X_te - mean) / std
        print(f"Loaded real CIFAR-100 (pickle): {X_tr.shape}", flush=True)
        return (X_tr, y_tr), (X_te, y_te)
    except Exception as e:
        print(f"Pickle load failed: {e}", flush=True)
        return None, None


def make_synthetic_cifar100(seed=42, noise_scale=1.0, proto_scale=0.5):
    """
    Synthetic data: 100 classes, 500 train / 100 test per class.
    proto_scale=0.5 + noise_scale=1.0 → SNR≈0.5 (hard enough to show learning).
    GroupNorm normalizes each batch independently, so no BN-collapse artifact.
    """
    rng = np.random.default_rng(seed)
    n_tr, n_te = 500, 100
    prototypes = proto_scale * rng.standard_normal((NUM_CLASSES, 3, 32, 32)).astype(np.float32)
    X_tr, y_tr, X_te, y_te = [], [], [], []
    for c in range(NUM_CLASSES):
        noise = noise_scale * rng.standard_normal((n_tr, 3, 32, 32)).astype(np.float32)
        X_tr.append(prototypes[c] + noise)
        y_tr.extend([c] * n_tr)
        noise_te = noise_scale * rng.standard_normal((n_te, 3, 32, 32)).astype(np.float32)
        X_te.append(prototypes[c] + noise_te)
        y_te.extend([c] * n_te)
    X_tr = np.concatenate(X_tr)
    X_te = np.concatenate(X_te)
    y_tr = np.array(y_tr, dtype=np.int64)
    y_te = np.array(y_te, dtype=np.int64)
    print(f"Synthetic data: {X_tr.shape}  SNR≈{proto_scale/noise_scale:.2f}", flush=True)
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
            x = x.flip(-1)   # horizontal flip only
        return x, self.y[i]


def make_task_splits(X_tr, y_tr, X_te, y_te, seed=0):
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
        tr_ldr = torch.utils.data.DataLoader(tr_ds, BATCH_SIZE, shuffle=True, drop_last=True, num_workers=0)
        te_ldr = torch.utils.data.DataLoader(te_ds, 256, shuffle=False, num_workers=0)
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
            # Restrict to task classes (task-incremental oracle mask)
            mask = torch.full_like(logits, float('-inf'))
            mask[:, cls_t] = logits[:, cls_t]
            correct += (mask.argmax(1) == y).sum().item()
            total   += y.size(0)
    return correct / total if total else 0.0


def dead_unit_fraction(model, loader, thr=0.01, n_batches=10):
    """Fraction of penultimate-layer ReLU units with mean |activation| < thr."""
    model.eval()
    acts = []
    with torch.no_grad():
        for i, (x, _) in enumerate(loader):
            if i >= n_batches: break
            model(x.to(DEVICE), store_pen=True)
            acts.append(model._penultimate.cpu())
    if not acts: return 0.0
    A = torch.cat(acts, 0)  # (N, 512)
    return (A.abs().mean(0) < thr).float().mean().item()


def effective_rank(model, loader, n_batches=20):
    """
    Effective rank via spectral entropy of penultimate-layer activations.
    Returns 0.0 if activation matrix is zero (all units dead).
    Returns NaN if SVD fails.
    Never returns the spurious artifact value ~500 from a zero-matrix SVD.
    """
    model.eval()
    acts = []
    with torch.no_grad():
        for i, (x, _) in enumerate(loader):
            if i >= n_batches: break
            model(x.to(DEVICE), store_pen=True)
            acts.append(model._penultimate.cpu())
    if not acts: return float('nan')
    A = torch.cat(acts, 0).float()  # (N, 512)

    # Guard: if all activations near zero, rank is 0 (not the SVD artifact)
    if A.abs().max() < 1e-10:
        return 0.0

    A = A - A.mean(0, keepdim=True)  # center columns
    if A.size(0) > 2000:
        idx = torch.randperm(A.size(0))[:2000]
        A = A[idx]
    try:
        _, S, _ = torch.linalg.svd(A, full_matrices=False)
        S = S.clamp(min=1e-8)
        p = S / S.sum()
        return torch.exp(-(p * p.log()).sum()).item()
    except Exception as e:
        print(f"  SVD failed: {e}", flush=True)
        return float('nan')


# ── Results I/O ───────────────────────────────────────────────────────────────
def write_results(payload):
    tmp = RESULTS_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, RESULTS_PATH)


# ── Single seed run ───────────────────────────────────────────────────────────
def run_seed(seed, X_tr, y_tr, X_te, y_te, results_accum, data_source, t_global_start):
    print(f"\n{'='*60}\nSEED {seed}\n{'='*60}", flush=True)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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
        "data_source": data_source,
    }

    for tid, (tr_ldr, te_ldr, cls_list) in enumerate(splits):
        # Time budget check
        elapsed = (time.time() - t_global_start) / 60
        if elapsed > WALL_LIMIT_MIN:
            print(f"  [TIME LIMIT] {elapsed:.1f}m elapsed, stopping at task {tid+1}", flush=True)
            results_accum["notes"] += f" Time limit hit at task {tid+1} (seed {seed})."
            break

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
                print(f"  Task {tid+1:2d}/{TASKS}  step {step:4d}  loss={loss.item():.4f}", flush=True)

        wall = time.time() - t0
        acc   = compute_accuracy(model, te_ldr, cls_list)
        duf   = dead_unit_fraction(model, te_ldr)
        erank = effective_rank(model, te_ldr)
        sr["per_task_acc"].append(acc)
        sr["dead_unit_frac"].append(duf)
        sr["effective_rank"].append(erank)
        print(
            f"  Task {tid+1:2d}  acc={acc:.3f}  dead={duf:.3f}  "
            f"erank={erank:.2f}  loss={np.mean(losses[-200:]):.4f}  {wall:.1f}s",
            flush=True
        )

        # Incremental write after every task
        results_accum["latest_seed"]         = seed
        results_accum["latest_task"]         = tid + 1
        results_accum["latest_per_task_acc"] = sr["per_task_acc"]
        write_results(results_accum)

    return sr


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    t0 = time.time()

    results_accum = {
        "arm": "A_adam_noreset_groupnorm",
        "status": "RUNNING",
        "per_seed": [],
        "per_task_acc": [],
        "mean_acc": None,
        "plasticity_trend": None,
        "dead_unit_frac_final": None,
        "effective_rank_trend": None,
        "notes": "",
    }
    write_results(results_accum)

    # ── Load data (torchvision → pickle → synthetic) ──
    data_source = "real_cifar100"
    train_data, test_data = load_cifar100_torchvision("/tmp/cifar100")
    if train_data is None:
        train_data, test_data = load_cifar100_pickle("/tmp/cifar100")
    if train_data is None:
        data_source = "synthetic_groupnorm"
        train_data, test_data = make_synthetic_cifar100(seed=0, noise_scale=1.0, proto_scale=0.5)

    X_tr, y_tr = train_data
    X_te, y_te = test_data

    all_accs, all_dufs, all_eranks, seeds_done = [], [], [], []

    for seed in SEEDS:
        elapsed = (time.time() - t0) / 60
        if elapsed > WALL_LIMIT_MIN:
            results_accum["notes"] += f" Time limit before seed {seed}."
            break

        sr = run_seed(seed, X_tr, y_tr, X_te, y_te, results_accum, data_source, t0)
        all_accs.append(sr["per_task_acc"])
        all_dufs.append(sr["dead_unit_frac"])
        all_eranks.append(sr["effective_rank"])
        seeds_done.append(seed)

        # Save seed result (without task_classes to keep JSON light)
        sr_out = {k: v for k, v in sr.items() if k != "task_classes"}
        results_accum["per_seed"].append(sr_out)

    # ── Aggregate ──
    if all_accs:
        # Pad short runs to the same length with NaN
        maxlen = max(len(a) for a in all_accs)
        def pad(lst, val=float('nan')):
            return lst + [val] * (maxlen - len(lst))

        mpt  = np.nanmean([pad(a) for a in all_accs], axis=0).tolist()
        mduf = np.nanmean([pad(d) for d in all_dufs], axis=0).tolist()
        mer  = np.nanmean([pad(e) for e in all_eranks], axis=0).tolist()

        mean_acc = float(np.nanmean(mpt))
        n_done = len(mpt)
        h1 = float(np.nanmean(mpt[:n_done//2]))
        h2 = float(np.nanmean(mpt[n_done//2:]))
        trend = (
            "declining" if h2 < h1 - 0.02
            else ("flat" if abs(h2 - h1) <= 0.02 else "improving")
        )

        er1   = mer[0]   if mer else float('nan')
        er_l  = mer[-1]  if mer else float('nan')
        er_ch = (er_l - er1) / max(abs(er1), 1e-6) if not (np.isnan(er1) or np.isnan(er_l)) else float('nan')
        duf_l = mduf[-1] if mduf else float('nan')

        er_trend = (
            "declining" if (not np.isnan(er_ch) and er_ch < -0.10)
            else ("flat"    if (not np.isnan(er_ch) and abs(er_ch) <= 0.10)
            else ("improving" if not np.isnan(er_ch) else "unknown"))
        )

        notes = (
            f"Arm A: Adam no-reset GroupNorm, {TASKS}×{CLS_PER_TASK} cls, "
            f"data={data_source}, seeds={seeds_done}. "
            f"acc h1={h1:.3f} h2={h2:.3f} → {trend}. "
            f"erank: t1={er1:.2f} → last={er_l:.2f} (Δ={er_ch:.1%} if not nan). "
            f"dead_frac_final={duf_l:.3f}. "
        )
        if data_source == "real_cifar100":
            notes += "GroupNorm with real CIFAR-100: plasticity loss should be gradual (no BN artifact). "
        else:
            notes += (
                "Synthetic data with GroupNorm: each batch normalized independently, "
                "no BN running-stat collapse. Plasticity loss driven purely by weight drift. "
            )

        results_accum.update({
            "status": "DONE",
            "data_source": data_source,
            "per_task_acc": mpt,
            "mean_acc": mean_acc,
            "plasticity_trend": trend,
            "dead_unit_frac": mduf,
            "dead_unit_frac_final": duf_l,
            "effective_rank": mer,
            "erank_task1": er1,
            "erank_last_task": er_l,
            "erank_relative_change": er_ch,
            "effective_rank_trend": er_trend,
            "first_half_mean_acc": h1,
            "second_half_mean_acc": h2,
            "num_seeds_run": len(seeds_done),
            "seeds": seeds_done,
            "wall_time_min": (time.time() - t0) / 60,
            "notes": notes,
        })
    else:
        results_accum.update({
            "status": "FAILED",
            "notes": "No seeds completed.",
        })

    write_results(results_accum)
    total_min = (time.time() - t0) / 60
    print(f"\nDone in {total_min:.1f}m", flush=True)
    print(json.dumps(results_accum, indent=2), flush=True)


if __name__ == "__main__":
    main()
