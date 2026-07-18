"""
Baseline A v2: Adam + no resets — gradual loss of plasticity.
Switches from BatchNorm to GroupNorm to avoid BN running-stat collapse
with synthetic data (all class prototypes share zero-mean statistics,
but GN doesn't track running stats so cross-task normalization works).
Also uses a slightly harder synthetic distribution (smaller SNR).

If real CIFAR-100 is extracted, uses that instead.
"""

import os, sys, json, time, random, pickle, tarfile
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

RESULTS_PATH = "/workspace/results/baseline_A/RESULTS.json"
WEIGHTS_DIR  = "/workspace/_weights"
os.makedirs(WEIGHTS_DIR, exist_ok=True)
os.makedirs("/workspace/results/baseline_A", exist_ok=True)

NUM_CLASSES    = 100
TASKS          = 20
CLS_PER_TASK   = NUM_CLASSES // TASKS   # 5
STEPS_PER_TASK = 2000
BATCH_SIZE     = 128
LR             = 1e-3
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Device: {DEVICE}", flush=True)

# ─────────────────────────────────────────────
# Small ConvNet with GroupNorm (avoids running-stat collapse)
# ─────────────────────────────────────────────
class SmallConvNetGN(nn.Module):
    """3 conv-blocks + 2 FC, GroupNorm instead of BatchNorm."""
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


# ─────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────
def load_real_cifar100(root="/tmp/cifar100"):
    data_dir = os.path.join(root, "cifar-100-python")
    if not os.path.isdir(data_dir):
        tar_path = os.path.join(root, "cifar-100-python.tar.gz")
        if os.path.exists(tar_path) and os.path.getsize(tar_path) > 168_000_000:
            print("Extracting CIFAR-100 ...", flush=True)
            with tarfile.open(tar_path, "r:gz") as tf:
                tf.extractall(root)
        else:
            size = os.path.getsize(tar_path) if os.path.exists(tar_path) else 0
            print(f"CIFAR-100 tar incomplete ({size/1e6:.1f}MB / 169MB). Using synthetic.", flush=True)
            return None, None

    def unpickle(f):
        with open(f, "rb") as fo:
            return pickle.load(fo, encoding="bytes")

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
    print(f"Loaded real CIFAR-100: {X_tr.shape}", flush=True)
    return (X_tr, y_tr), (X_te, y_te)


def make_synthetic_cifar100(seed=42, noise_scale=1.0, proto_scale=0.5):
    """
    Synthetic data: 100 classes, 500 train / 100 test per class.
    proto_scale=0.5 + noise_scale=1.0 gives SNR≈0.5 (reasonably hard).
    All prototypes are small perturbations of a common zero background,
    so BatchNorm/GroupNorm statistics are stable across tasks.
    """
    rng = np.random.default_rng(seed)
    n_tr, n_te = 500, 100
    # Small prototypes + larger noise → harder to memorize
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
    print(f"Synthetic data: {X_tr.shape} SNR≈{proto_scale/noise_scale:.2f}", flush=True)
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


# ─────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────
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
    # If all activations are zero (all units dead), effective rank is 0
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


def write_results(payload):
    with open(RESULTS_PATH, "w") as f:
        json.dump(payload, f, indent=2)


# ─────────────────────────────────────────────
# Single seed run
# ─────────────────────────────────────────────
def run_seed(seed, X_tr, y_tr, X_te, y_te, results_accum, data_source):
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

    sr = {"seed": seed, "per_task_acc": [], "dead_unit_frac": [],
          "effective_rank": [], "task_classes": task_classes, "data_source": data_source}

    for tid, (tr_ldr, te_ldr, cls_list) in enumerate(splits):
        t0 = time.time()
        model.train()
        step = 0; tr_it = iter(tr_ldr); losses = []

        while step < STEPS_PER_TASK:
            try:    x, y = next(tr_it)
            except: tr_it = iter(tr_ldr); x, y = next(tr_it)
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward(); optimizer.step()
            losses.append(loss.item()); step += 1
            if step % 500 == 0:
                print(f"  Task {tid+1:2d}/{TASKS} step {step:4d} loss={loss.item():.4f}", flush=True)

        wall = time.time() - t0
        acc   = compute_accuracy(model, te_ldr, cls_list)
        duf   = dead_unit_fraction(model, te_ldr)
        erank = effective_rank(model, te_ldr)
        sr["per_task_acc"].append(acc); sr["dead_unit_frac"].append(duf); sr["effective_rank"].append(erank)
        print(f"  Task {tid+1:2d} | acc={acc:.3f} dead={duf:.3f} erank={erank:.2f} "
              f"loss={np.mean(losses[-200:]):.4f} {wall:.1f}s", flush=True)

        # Incremental write
        tmp = {**results_accum, "latest_seed": seed, "latest_task": tid+1,
               "latest_per_task_acc": sr["per_task_acc"]}
        write_results(tmp)

    return sr


# ─────────────────────────────────────────────
def main():
    t0 = time.time()
    results_accum = {"arm": "A_adam_noreset", "status": "RUNNING",
                     "per_seed": [], "per_task_acc": [], "mean_acc": None,
                     "plasticity_trend": None, "dead_unit_frac": [], "notes": ""}
    write_results(results_accum)

    # Prefer real CIFAR-100
    data_source = "real_cifar100"
    train_data, test_data = load_real_cifar100()
    if train_data is None:
        data_source = "synthetic_gn"
        train_data, test_data = make_synthetic_cifar100(seed=0, noise_scale=1.0, proto_scale=0.5)

    X_tr, y_tr = train_data
    X_te, y_te = test_data

    all_accs = []; all_dufs = []; all_eranks = []; seeds_done = []

    for seed in [0, 1]:
        if (time.time() - t0) > 28*60:
            results_accum["notes"] += f" Time limit at {(time.time()-t0)/60:.1f}m."
            break
        sr = run_seed(seed, X_tr, y_tr, X_te, y_te, results_accum, data_source)
        all_accs.append(sr["per_task_acc"]); all_dufs.append(sr["dead_unit_frac"])
        all_eranks.append(sr["effective_rank"]); seeds_done.append(seed)
        results_accum["per_seed"].append({k: v for k, v in sr.items() if k != "task_classes"})

    if all_accs:
        mpt  = np.mean(all_accs,  axis=0).tolist()
        mduf = np.mean(all_dufs,  axis=0).tolist()
        mer  = np.mean(all_eranks,axis=0).tolist()
        mean_acc = float(np.mean(mpt))
        h1 = float(np.mean(mpt[:TASKS//2]))
        h2 = float(np.mean(mpt[TASKS//2:]))
        trend = "declining" if h2 < h1 - 0.02 else ("flat" if abs(h2-h1) <= 0.02 else "improving")
        er1   = mer[0]  if mer else float('nan')
        er_l  = mer[-1] if mer else float('nan')
        er_ch = (er_l - er1) / max(er1, 1e-6)
        results_accum.update({
            "status": "DONE", "data_source": data_source,
            "per_task_acc": mpt, "mean_acc": mean_acc,
            "plasticity_trend": trend, "dead_unit_frac": mduf, "effective_rank": mer,
            "erank_task1": er1, "erank_last_task": er_l, "erank_relative_change": er_ch,
            "first_half_mean_acc": h1, "second_half_mean_acc": h2,
            "num_seeds_run": len(seeds_done), "seeds": seeds_done,
            "wall_time_min": (time.time()-t0)/60,
            "notes": (f"Arm A: Adam no-reset, GroupNorm, {TASKS}×{CLS_PER_TASK} cls, "
                      f"data={data_source}, seeds={seeds_done}. "
                      f"h1={h1:.3f} h2={h2:.3f} → {trend}. "
                      f"erank: t1={er1:.2f} → last={er_l:.2f} (Δ={er_ch:.1%})."),
        })
    else:
        results_accum.update({"status": "FAILED", "notes": "No seeds completed."})

    write_results(results_accum)
    print(f"\nDone in {(time.time()-t0)/60:.1f}m", flush=True)
    print(json.dumps(results_accum, indent=2), flush=True)

if __name__ == "__main__":
    main()
