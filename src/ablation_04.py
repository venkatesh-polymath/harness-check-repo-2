"""
ablation-04 (full): Multi-seed statistical confirmation of ablation-03 findings.

Runs the same three-arm experiment as ablation-03 across 5 independent seeds:
  A — Vanilla baseline: no re-init
  B — SNRI: null-space re-init (non-disruptive)
  C — Naive ReDo: dead-channel detection + kaiming re-init (disruptive)

Per-seed metrics: effective-rank AUC, final dead-neuron%, rank_drop_pct,
                  disruption mean/max (B and C only).
Cross-seed stats: mean ± std for each metric; sign-consistency check for
                  B-vs-A and C-vs-B differences across all 5 seeds.

Config: 20 tasks × 5 classes × 10 epochs, ResNet-18, SGD + cosine LR.
Seeds:  [0, 1, 2, 3, 4]
Output: results/ablation-04/
Weights: _weights/ (git-ignored)
"""

import os, sys, json, copy, time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader, Subset
from scipy import stats as scipy_stats

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── config ─────────────────────────────────────────────────────────────────────
SEEDS            = [0, 1, 2, 3, 4]
NUM_TASKS        = 20
CLASSES_PER_TASK = 5
EPOCHS_PER_TASK  = 10
BATCH_SIZE       = 128
LR_INIT          = 0.1
MOMENTUM         = 0.9
WEIGHT_DECAY     = 5e-4
DEAD_THRESHOLD   = 0.01
DEAD_PROBE_BATCHES = 20
SNRI_NULL_EPSILON  = 1e-6

OUT_DIR     = "/workspace/results/ablation-04"
WEIGHTS_DIR = "/workspace/_weights"
LOG_FILE    = os.path.join(OUT_DIR, "run.log")

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(WEIGHTS_DIR, exist_ok=True)

_log_fh = open(LOG_FILE, "w", buffering=1)

def log(msg):
    ts   = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    _log_fh.write(line + "\n")

log(f"Python {sys.version}")
log(f"PyTorch {torch.__version__}")
log(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
log(f"Config: {NUM_TASKS} tasks × {CLASSES_PER_TASK} classes × {EPOCHS_PER_TASK} epochs")
log(f"Seeds: {SEEDS}")
log(f"Arms: A=vanilla  B=SNRI  C=naive-random-reinit(ReDo)")

# ── CIFAR-100 ──────────────────────────────────────────────────────────────────
CIFAR_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR_STD  = (0.2675, 0.2565, 0.2761)

train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
])
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
])

DATA_ROOT  = "/tmp/cifar100"
full_train = torchvision.datasets.CIFAR100(DATA_ROOT, train=True,  download=True,
                                            transform=train_transform)
full_test  = torchvision.datasets.CIFAR100(DATA_ROOT, train=False, download=True,
                                            transform=test_transform)
log(f"CIFAR-100: {len(full_train)} train / {len(full_test)} test")

# ── data helpers ───────────────────────────────────────────────────────────────
def task_subset(dataset, classes):
    targets = np.array(dataset.targets)
    idx     = np.where(np.isin(targets, classes))[0]
    return Subset(dataset, idx)

def make_task_loaders(task_id):
    classes   = list(range(task_id * CLASSES_PER_TASK, (task_id + 1) * CLASSES_PER_TASK))
    tr_sub    = task_subset(full_train, classes)
    te_sub    = task_subset(full_test,  classes)
    tr_loader = DataLoader(tr_sub, batch_size=BATCH_SIZE, shuffle=True,
                           num_workers=4, pin_memory=True, drop_last=False)
    te_loader = DataLoader(te_sub, batch_size=BATCH_SIZE, shuffle=False,
                           num_workers=4, pin_memory=True)
    return tr_loader, te_loader, classes

# Shared full-test loader (no shuffle, used for dead-neuron probing)
te_loader_full = DataLoader(full_test, batch_size=256, shuffle=False,
                             num_workers=4, pin_memory=True)

# ── model factory ──────────────────────────────────────────────────────────────
def make_model():
    m = models.resnet18(weights=None)
    m.fc = nn.Linear(512, 100)
    return m.to(DEVICE)

# ── effective rank (stable rank = nuclear-norm / Frobenius-norm) ───────────────
def effective_rank(weight: torch.Tensor) -> float:
    w = weight.detach().float()
    if w.dim() > 2:
        w = w.view(w.size(0), -1)
    sv   = torch.linalg.svdvals(w)
    nuc  = sv.sum().item()
    frob = sv.norm().item()
    return (nuc / frob) if frob > 1e-12 else 0.0

def mean_effective_rank(model) -> float:
    vals = [effective_rank(m.weight)
            for _, m in model.named_modules()
            if isinstance(m, (nn.Conv2d, nn.Linear))]
    return float(np.mean(vals))

# ── dead-neuron fraction (over ReLU outputs) ───────────────────────────────────
def compute_dead_neuron_fraction(model, loader, n_batches=DEAD_PROBE_BATCHES):
    activations, hooks = {}, []

    def make_hook(name):
        def hook(module, inp, out):
            a = out.detach().cpu()
            if name not in activations:
                activations[name] = a
            else:
                activations[name] = torch.cat([activations[name], a], dim=0)
        return hook

    for name, module in model.named_modules():
        if isinstance(module, nn.ReLU):
            hooks.append(module.register_forward_hook(make_hook(name)))

    model.eval()
    with torch.no_grad():
        for i, (x, _) in enumerate(loader):
            if i >= n_batches:
                break
            model(x.to(DEVICE))

    for h in hooks:
        h.remove()

    total, dead = 0, 0
    for act in activations.values():
        flat     = act.view(act.size(0), -1)
        mean_abs = flat.abs().mean(0)
        dead  += (mean_abs < DEAD_THRESHOLD).sum().item()
        total += flat.size(1)

    return dead / total if total > 0 else 0.0

# ── BasicBlock layer pairs for SNRI/ReDo ──────────────────────────────────────
def get_block_layer_pairs(model):
    pairs = []
    for bname, bmod in model.named_modules():
        if (hasattr(bmod, 'conv1') and hasattr(bmod, 'conv2')
                and isinstance(bmod.conv1, nn.Conv2d)
                and isinstance(bmod.conv2, nn.Conv2d)
                and bmod.conv1.out_channels == bmod.conv2.in_channels):
            pairs.append((f"{bname}.conv1", bmod.conv1,
                           f"{bname}.conv2", bmod.conv2))
    return pairs

# ── dead channel detection (hooks on conv2 INPUT = conv1 post-BN+ReLU) ────────
def detect_dead_channels(model, layer_pairs, loader,
                          n_batches=DEAD_PROBE_BATCHES, threshold=DEAD_THRESHOLD):
    accum = {}
    hooks = []

    def make_hook(name1):
        def hook(module, inp, out):
            with torch.no_grad():
                mean_abs = inp[0].detach().abs().mean(dim=(0, 2, 3)).cpu()
                if name1 not in accum:
                    accum[name1] = [mean_abs, 1]
                else:
                    accum[name1][0] += mean_abs
                    accum[name1][1] += 1
        return hook

    for name1, conv1, name2, conv2 in layer_pairs:
        hooks.append(conv2.register_forward_hook(make_hook(name1)))

    model.eval()
    with torch.no_grad():
        for i, (x, _) in enumerate(loader):
            if i >= n_batches:
                break
            model(x.to(DEVICE))

    for h in hooks:
        h.remove()

    dead_masks = {}
    for name1, conv1, _, _ in layer_pairs:
        if name1 not in accum:
            continue
        mean_act = accum[name1][0] / accum[name1][1]
        dead_masks[name1] = mean_act < threshold

    return dead_masks

# ── SNRI: null-space re-init (non-disruptive) ─────────────────────────────────
def svd_null_and_small(W_2d, epsilon=SNRI_NULL_EPSILON):
    U, S, Vh = torch.linalg.svd(W_2d.float(), full_matrices=True)
    rank = int((S > epsilon).sum().item())
    return Vh, S, rank

def apply_snri(model, layer_pairs, dead_masks, epsilon=SNRI_NULL_EPSILON):
    """SNRI: null-space re-init + zero outgoing (non-disruptive)."""
    stats = {}
    for name1, conv1, name2, conv2 in layer_pairs:
        if name1 not in dead_masks:
            continue
        dead_mask = dead_masks[name1]
        n_dead    = int(dead_mask.sum().item())
        if n_dead == 0:
            stats[name1] = {"n_dead": 0, "n_reinit": 0}
            continue

        W1     = conv1.weight.data
        C_out, C_in, kH, kW = W1.shape
        W1_2d  = W1.view(C_out, -1)
        W2     = conv2.weight.data

        try:
            Vh1, S1, rank1 = svd_null_and_small(W1_2d, epsilon)
        except Exception as ex:
            stats[name1] = {"n_dead": n_dead, "n_reinit": 0, "error": str(ex)}
            continue

        alive_mask = ~dead_mask
        scale = (W1_2d[alive_mask].norm(dim=-1).mean().item()
                 if alive_mask.sum() > 0 else W1_2d.norm(dim=-1).mean().item())
        scale = max(float(scale), 1e-4)

        dead_indices = dead_mask.nonzero(as_tuple=True)[0]
        n_reinit = 0

        for j, idx in enumerate(dead_indices):
            idx = int(idx.item())
            vec_row = Vh1.shape[0] - 1 - (j % min(n_dead, Vh1.shape[0]))
            new_dir = Vh1[vec_row].to(W1.device)
            new_dir = (new_dir / (new_dir.norm() + 1e-12)) * scale

            W1[idx] = new_dir.view(C_in, kH, kW).to(W1.dtype)
            W2[:, idx] = 0.0   # zero outgoing → non-disruptive
            if conv1.bias is not None:
                conv1.bias.data[idx] = 0.0
            n_reinit += 1

        stats[name1] = {"n_dead": n_dead, "n_reinit": n_reinit, "rank_W1": rank1}
    return stats

# ── Naive ReDo-style re-init (disruptive) ─────────────────────────────────────
def apply_naive_reinit(model, layer_pairs, dead_masks):
    """
    ReDo-style: re-init dead channels' incoming weights with kaiming_normal.
    Do NOT zero outgoing weights → disruptive.
    """
    stats = {}
    for name1, conv1, name2, conv2 in layer_pairs:
        if name1 not in dead_masks:
            continue
        dead_mask = dead_masks[name1]
        n_dead    = int(dead_mask.sum().item())
        if n_dead == 0:
            stats[name1] = {"n_dead": 0, "n_reinit": 0}
            continue

        dead_indices = dead_mask.nonzero(as_tuple=True)[0]
        n_reinit = 0

        for idx in dead_indices:
            idx = int(idx.item())
            tmp = torch.empty_like(conv1.weight.data[idx])
            nn.init.kaiming_normal_(tmp.unsqueeze(0), mode='fan_in',
                                    nonlinearity='relu')
            conv1.weight.data[idx] = tmp.squeeze(0)
            # Outgoing weights: NOT zeroed (disruptive!)
            if conv1.bias is not None:
                conv1.bias.data[idx] = 0.0
            n_reinit += 1

        stats[name1] = {"n_dead": n_dead, "n_reinit": n_reinit}
    return stats

# ── disruption measurement ─────────────────────────────────────────────────────
def measure_disruption_snri(model, probe_x, layer_pairs, dead_masks):
    """Measure max |logit_diff| for SNRI re-init (uses deepcopy)."""
    model.eval()
    with torch.no_grad():
        before = model(probe_x).cpu()
    mc = copy.deepcopy(model)
    apply_snri(mc, get_block_layer_pairs(mc), dead_masks)
    mc.eval()
    with torch.no_grad():
        after = mc(probe_x).cpu()
    diff = (before - after).abs()
    return float(diff.max().item()), float(diff.mean().item())

def measure_disruption_naive(model, probe_x, layer_pairs, dead_masks, seed_offset):
    """Measure max |logit_diff| for naive random re-init (uses deepcopy)."""
    model.eval()
    with torch.no_grad():
        before = model(probe_x).cpu()
    mc = copy.deepcopy(model)
    torch.manual_seed(seed_offset)  # reproducible kaiming for copy
    apply_naive_reinit(mc, get_block_layer_pairs(mc), dead_masks)
    mc.eval()
    with torch.no_grad():
        after = mc(probe_x).cpu()
    diff = (before - after).abs()
    return float(diff.max().item()), float(diff.mean().item())

# ── training helpers ───────────────────────────────────────────────────────────
criterion = nn.CrossEntropyLoss()

def train_epoch(model, loader, optimizer):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        out  = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * y.size(0)
        correct    += (out.argmax(1) == y).sum().item()
        total      += y.size(0)
    return total_loss / total, correct / total

@torch.no_grad()
def eval_epoch(model, loader):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        out  = model(x)
        loss = criterion(out, y)
        total_loss += loss.item() * y.size(0)
        correct    += (out.argmax(1) == y).sum().item()
        total      += y.size(0)
    return total_loss / total, correct / total

# ── probe batch (fixed across arms within a seed, refreshed per seed) ──────────
def make_probe_batch(seed):
    """Return a fixed 256-image probe batch from the test set."""
    g = torch.Generator()
    g.manual_seed(seed)
    loader = DataLoader(full_test, batch_size=256, shuffle=True,
                        generator=g, num_workers=0)
    x, _ = next(iter(loader))
    return x.to(DEVICE)

# =============================================================================
# RUN ONE ARM FOR ONE SEED
# =============================================================================
def run_arm(arm_name, reinit_fn, seed):
    """
    Train 20 sequential tasks; apply re-init after each task (arms B & C only).
    Returns arm-level metrics dict.
    """
    # Reset ALL random state for this (arm, seed) combination
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

    model  = make_model()
    pairs  = get_block_layer_pairs(model)
    probe_x = make_probe_batch(seed)

    init_rank = mean_effective_rank(model)
    init_dead = compute_dead_neuron_fraction(model, te_loader_full)
    log(f"  [{arm_name} seed={seed}] Init rank={init_rank:.4f}  init_dead={init_dead*100:.2f}%")

    per_task   = []
    reinit_log = []
    t0         = time.time()

    for task_id in range(NUM_TASKS):
        tr_loader, te_loader, task_classes = make_task_loaders(task_id)

        optimizer = optim.SGD(model.parameters(), lr=LR_INIT, momentum=MOMENTUM,
                              weight_decay=WEIGHT_DECAY, nesterov=True)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                          T_max=EPOCHS_PER_TASK,
                                                          eta_min=0.0)
        final_tr_acc = final_te_acc = None

        for epoch in range(1, EPOCHS_PER_TASK + 1):
            tr_loss, tr_acc = train_epoch(model, tr_loader, optimizer)
            te_loss, te_acc = eval_epoch(model, te_loader)
            scheduler.step()
            if epoch == EPOCHS_PER_TASK:
                final_tr_acc = tr_acc * 100
                final_te_acc = te_acc * 100

        task_rank = mean_effective_rank(model)
        task_dead = compute_dead_neuron_fraction(model, te_loader_full)

        # ── Re-init (B and C only) ────────────────────────────────────────────
        disruption_max = disruption_mean = None
        n_reinit = 0

        if reinit_fn is not None:
            dead_masks = detect_dead_channels(model, pairs, te_loader_full,
                                              n_batches=DEAD_PROBE_BATCHES,
                                              threshold=DEAD_THRESHOLD)
            # Measure disruption on deepcopy BEFORE applying to real model
            if arm_name == "B_SNRI":
                disruption_max, disruption_mean = measure_disruption_snri(
                    model, probe_x, pairs, dead_masks)
            elif arm_name == "C_naive":
                disruption_max, disruption_mean = measure_disruption_naive(
                    model, probe_x, pairs, dead_masks,
                    seed_offset=seed * 1000 + task_id)

            # Apply re-init to the actual model
            torch.manual_seed(seed * 1000 + task_id)
            stats  = reinit_fn(model, pairs, dead_masks)
            n_reinit = sum(s.get("n_reinit", 0) for s in stats.values())

            reinit_log.append({
                "task_id":        task_id,
                "n_reinit":       n_reinit,
                "disruption_max": disruption_max,
                "disruption_mean": disruption_mean,
            })

        rank_drop_pct = (init_rank - task_rank) / init_rank * 100

        per_task.append({
            "task_id":             task_id,
            "classes":             task_classes,
            "final_train_acc_pct": round(final_tr_acc, 2),
            "final_test_acc_pct":  round(final_te_acc, 2),
            "mean_effective_rank": round(task_rank, 4),
            "rank_drop_from_init_pct": round(rank_drop_pct, 2),
            "dead_neuron_pct":     round(task_dead * 100, 2),
            "n_reinit":            n_reinit,
            "disruption_max":      disruption_max,
            "disruption_mean":     disruption_mean,
        })

        elapsed = time.time() - t0
        dis_str = f"{disruption_max:.2e}" if disruption_max is not None else "N/A"
        log(f"  [{arm_name} s={seed}] T{task_id:02d}: tr={final_tr_acc:.1f}% "
            f"rank={task_rank:.3f} drop={rank_drop_pct:.1f}% "
            f"dead={task_dead*100:.2f}% reinit={n_reinit} dis={dis_str} "
            f"t={elapsed:.0f}s")

    total_time = time.time() - t0

    # ── Summary stats ─────────────────────────────────────────────────────────
    ranks     = [m["mean_effective_rank"] for m in per_task]
    dead_pcts = [m["dead_neuron_pct"]     for m in per_task]
    x_arr     = np.arange(NUM_TASKS, dtype=float)
    slope_r, _, r_r, _, _ = scipy_stats.linregress(x_arr, ranks)
    slope_d, _, r_d, _, _ = scipy_stats.linregress(x_arr, dead_pcts)

    rank_auc   = float(np.mean(ranks))
    final_rank = ranks[-1]
    final_dead = dead_pcts[-1]
    rank_drop  = (init_rank - final_rank) / init_rank * 100

    # Disruption summary for arms B/C
    dis_maxs  = [e["disruption_max"]  for e in reinit_log if e["disruption_max"] is not None]
    dis_means = [e["disruption_mean"] for e in reinit_log if e["disruption_mean"] is not None]
    dis_mean_val  = float(np.mean(dis_means)) if dis_means else None
    dis_max_val   = float(np.max(dis_maxs))   if dis_maxs  else None
    dis_min_val   = float(np.min(dis_maxs))   if dis_maxs  else None
    total_reinit_arm = sum(e["n_reinit"] for e in reinit_log)

    log(f"  [{arm_name} s={seed}] SUMMARY: "
        f"rank_auc={rank_auc:.4f} rank_drop={rank_drop:.2f}% "
        f"dead_final={final_dead:.2f}% reinit={total_reinit_arm} "
        f"time={total_time/60:.1f}min")
    if dis_mean_val is not None:
        log(f"  [{arm_name} s={seed}] disruption: mean={dis_mean_val:.3e} max={dis_max_val:.3e}")

    return {
        "arm":              arm_name,
        "seed":             seed,
        "init_rank":        round(init_rank, 4),
        "final_rank":       round(final_rank, 4),
        "rank_auc":         round(rank_auc, 4),
        "rank_drop_pct":    round(rank_drop, 2),
        "rank_trend_slope": round(slope_r, 5),
        "rank_trend_r":     round(r_r, 4),
        "init_dead_pct":    round(init_dead * 100, 2),
        "final_dead_pct":   round(final_dead, 2),
        "dead_trend_slope": round(slope_d, 5),
        "dead_trend_r":     round(r_d, 4),
        "disruption_mean":  round(dis_mean_val, 6) if dis_mean_val is not None else None,
        "disruption_max":   round(dis_max_val,  6) if dis_max_val  is not None else None,
        "disruption_min":   round(dis_min_val,  6) if dis_min_val  is not None else None,
        "total_reinit":     total_reinit_arm,
        "training_time_min": round(total_time / 60, 2),
        "per_task":         per_task,
        "reinit_log":       reinit_log,
    }

# =============================================================================
# MAIN LOOP: run all seeds × all arms
# =============================================================================
log("\n" + "="*70)
log("ABLATION-04: Full-scale multi-seed run (5 seeds × 3 arms)")
log("="*70)

# Structure: results_by_seed[seed][arm_name] = arm_result_dict
results_by_seed = {}

t_global = time.time()

for seed in SEEDS:
    log(f"\n{'='*60}")
    log(f"SEED {seed}")
    log(f"{'='*60}")

    results_by_seed[seed] = {}
    results_by_seed[seed]["A_vanilla"] = run_arm("A_vanilla", reinit_fn=None,              seed=seed)
    results_by_seed[seed]["B_SNRI"]    = run_arm("B_SNRI",    reinit_fn=apply_snri,         seed=seed)
    results_by_seed[seed]["C_naive"]   = run_arm("C_naive",   reinit_fn=apply_naive_reinit, seed=seed)

    elapsed_global = time.time() - t_global
    log(f"Seed {seed} done. Total elapsed: {elapsed_global/60:.1f}min")

log(f"\nAll seeds done. Total time: {(time.time()-t_global)/60:.1f}min")

# =============================================================================
# AGGREGATE STATISTICS
# =============================================================================
log("\n" + "="*70)
log("AGGREGATE STATISTICS")
log("="*70)

def collect_metric(arm_name, metric):
    """Collect per-seed values for one arm/metric."""
    return [results_by_seed[s][arm_name][metric] for s in SEEDS
            if results_by_seed[s][arm_name][metric] is not None]

def summarize(vals):
    arr = np.array(vals, dtype=float)
    return {"mean": round(float(arr.mean()), 4),
            "std":  round(float(arr.std(ddof=1)), 4),
            "min":  round(float(arr.min()), 4),
            "max":  round(float(arr.max()), 4),
            "values": [round(float(v), 4) for v in arr]}

metrics_agg = {}
for arm in ["A_vanilla", "B_SNRI", "C_naive"]:
    metrics_agg[arm] = {
        "rank_auc":      summarize(collect_metric(arm, "rank_auc")),
        "final_dead_pct": summarize(collect_metric(arm, "final_dead_pct")),
        "rank_drop_pct": summarize(collect_metric(arm, "rank_drop_pct")),
    }
    if arm in ("B_SNRI", "C_naive"):
        dis_means = collect_metric(arm, "disruption_mean")
        dis_maxs  = collect_metric(arm, "disruption_max")
        if dis_means:
            metrics_agg[arm]["disruption_mean"] = summarize(dis_means)
        if dis_maxs:
            metrics_agg[arm]["disruption_max"]  = summarize(dis_maxs)

for arm, d in metrics_agg.items():
    log(f"\n  {arm}:")
    for k, v in d.items():
        log(f"    {k}: mean={v['mean']:.4f} std={v['std']:.4f} "
            f"[{v['min']:.4f}, {v['max']:.4f}]")

# ── Sign consistency ───────────────────────────────────────────────────────────
log("\n--- Sign consistency across seeds ---")

# B-vs-A: rank_auc difference (positive = B better than A)
rank_auc_B_minus_A = [results_by_seed[s]["B_SNRI"]["rank_auc"]
                       - results_by_seed[s]["A_vanilla"]["rank_auc"]
                       for s in SEEDS]
# C-vs-B: rank_auc difference (positive = C better than B)
rank_auc_C_minus_B = [results_by_seed[s]["C_naive"]["rank_auc"]
                       - results_by_seed[s]["B_SNRI"]["rank_auc"]
                       for s in SEEDS]
# B-vs-A: dead_pct difference (positive = B has more dead than A)
dead_B_minus_A     = [results_by_seed[s]["B_SNRI"]["final_dead_pct"]
                       - results_by_seed[s]["A_vanilla"]["final_dead_pct"]
                       for s in SEEDS]
# C-vs-B: dead_pct difference (negative = C has fewer dead than B)
dead_C_minus_B     = [results_by_seed[s]["C_naive"]["final_dead_pct"]
                       - results_by_seed[s]["B_SNRI"]["final_dead_pct"]
                       for s in SEEDS]
# Disruption: C-vs-B (positive = C more disruptive)
dis_C_minus_B      = [results_by_seed[s]["C_naive"]["disruption_mean"]
                       - results_by_seed[s]["B_SNRI"]["disruption_mean"]
                       for s in SEEDS
                       if (results_by_seed[s]["C_naive"]["disruption_mean"] is not None
                           and results_by_seed[s]["B_SNRI"]["disruption_mean"] is not None)]

# Sign counts
def sign_consistency(diffs, positive_expected=True):
    expected_sign = 1 if positive_expected else -1
    consistent    = sum(1 for d in diffs if np.sign(d) == expected_sign)
    return consistent, len(diffs)

sc_rank_C_gt_B = sign_consistency(rank_auc_C_minus_B, positive_expected=True)
sc_dead_B_gt_A = sign_consistency(dead_B_minus_A,     positive_expected=True)
sc_dead_C_lt_B = sign_consistency(dead_C_minus_B,     positive_expected=False)
sc_dis_C_gt_B  = sign_consistency(dis_C_minus_B,      positive_expected=True)

log(f"  rank_auc: C > B in {sc_rank_C_gt_B[0]}/{sc_rank_C_gt_B[1]} seeds "
    f"(diffs: {[round(d,4) for d in rank_auc_C_minus_B]})")
log(f"  dead%:    B > A in {sc_dead_B_gt_A[0]}/{sc_dead_B_gt_A[1]} seeds "
    f"(diffs: {[round(d,2) for d in dead_B_minus_A]})")
log(f"  dead%:    C < B in {sc_dead_C_lt_B[0]}/{sc_dead_C_lt_B[1]} seeds "
    f"(diffs: {[round(d,2) for d in dead_C_minus_B]})")
log(f"  disruption: C > B in {sc_dis_C_gt_B[0]}/{sc_dis_C_gt_B[1]} seeds "
    f"(diffs: {[round(d,4) for d in dis_C_minus_B]})")

# ── Wilcoxon signed-rank tests ─────────────────────────────────────────────────
log("\n--- Wilcoxon signed-rank tests (H0: median difference = 0) ---")

def wilcoxon_or_na(a, b, label):
    arr = np.array(a) - np.array(b)
    if len(arr) < 2:
        log(f"  {label}: N/A (n={len(arr)})")
        return None, None
    # With only 5 samples, exact test
    try:
        stat, pval = scipy_stats.wilcoxon(a, b, alternative='two-sided')
        # rank-biserial r = 1 - 2*stat / (n*(n+1)/2)  [for Wilcoxon T statistic]
        n = len(a)
        r = 1 - 2*stat / (n * (n + 1) / 2)
        log(f"  {label}: stat={stat:.2f} p={pval:.4f} r={r:.3f}  "
            f"(mean diff = {float(arr.mean()):.4f} ± {float(arr.std(ddof=1)):.4f})")
        return float(pval), float(r)
    except Exception as e:
        log(f"  {label}: error {e}")
        return None, None

rank_auc_A = [results_by_seed[s]["A_vanilla"]["rank_auc"] for s in SEEDS]
rank_auc_B = [results_by_seed[s]["B_SNRI"]["rank_auc"]    for s in SEEDS]
rank_auc_C = [results_by_seed[s]["C_naive"]["rank_auc"]   for s in SEEDS]
dead_A     = [results_by_seed[s]["A_vanilla"]["final_dead_pct"] for s in SEEDS]
dead_B     = [results_by_seed[s]["B_SNRI"]["final_dead_pct"]    for s in SEEDS]
dead_C     = [results_by_seed[s]["C_naive"]["final_dead_pct"]   for s in SEEDS]

p_rank_B_vs_A, r_rank_B_vs_A = wilcoxon_or_na(rank_auc_B, rank_auc_A, "rank_auc B-vs-A")
p_rank_C_vs_B, r_rank_C_vs_B = wilcoxon_or_na(rank_auc_C, rank_auc_B, "rank_auc C-vs-B")
p_dead_B_vs_A, r_dead_B_vs_A = wilcoxon_or_na(dead_B,     dead_A,     "dead_pct B-vs-A")
p_dead_C_vs_B, r_dead_C_vs_B = wilcoxon_or_na(dead_C,     dead_B,     "dead_pct C-vs-B")

# Disruption C vs B
dis_B_vals = [results_by_seed[s]["B_SNRI"]["disruption_mean"] for s in SEEDS
              if results_by_seed[s]["B_SNRI"]["disruption_mean"] is not None]
dis_C_vals = [results_by_seed[s]["C_naive"]["disruption_mean"] for s in SEEDS
              if results_by_seed[s]["C_naive"]["disruption_mean"] is not None]
p_dis_C_vs_B, r_dis_C_vs_B = wilcoxon_or_na(dis_C_vals, dis_B_vals, "disruption C-vs-B")

# Bonferroni-corrected α (4 tests)
ALPHA_CORRECTED = 0.05 / 4
log(f"\n  Bonferroni-corrected α = {ALPHA_CORRECTED:.4f} (4 primary tests)")

# ── Hypothesis verification ────────────────────────────────────────────────────
log("\n--- Primary hypotheses ---")

# H1: C more disruptive than B (mean disruption C > B in every seed)
h1_consistent = sc_dis_C_gt_B[0] == sc_dis_C_gt_B[1]  # all seeds
log(f"  H1 (C more disruptive than B): consistent in {sc_dis_C_gt_B[0]}/{sc_dis_C_gt_B[1]} seeds → {h1_consistent}")

# H2: C has better rank AUC than B (C > B in ≥ 4/5 seeds)
h2_consistent = sc_rank_C_gt_B[0] >= 4
log(f"  H2 (C higher rank AUC than B): consistent in {sc_rank_C_gt_B[0]}/{sc_rank_C_gt_B[1]} seeds → {h2_consistent}")

# H3: B ends with more dead neurons than A in every seed
h3_consistent = sc_dead_B_gt_A[0] == sc_dead_B_gt_A[1]
log(f"  H3 (B more dead than A): consistent in {sc_dead_B_gt_A[0]}/{sc_dead_B_gt_A[1]} seeds → {h3_consistent}")

# H4: C has fewer dead neurons than B in every seed
h4_consistent = sc_dead_C_lt_B[0] == sc_dead_C_lt_B[1]
log(f"  H4 (C fewer dead than B): consistent in {sc_dead_C_lt_B[0]}/{sc_dead_C_lt_B[1]} seeds → {h4_consistent}")

h_all = h1_consistent and h2_consistent and h3_consistent and h4_consistent
log(f"\n  ALL HYPOTHESES CONSISTENT: {h_all}")

# =============================================================================
# BUILD RESULTS.JSON
# =============================================================================
log("\n" + "="*70)
log("Writing RESULTS.json")
log("="*70)

# Compact per-seed summary (no per_task / reinit_log)
def compact_arm(d):
    return {k: v for k, v in d.items() if k not in ("per_task", "reinit_log")}

per_seed_compact = {
    str(s): {arm: compact_arm(results_by_seed[s][arm])
             for arm in ("A_vanilla", "B_SNRI", "C_naive")}
    for s in SEEDS
}

results = {
    "status": "SUCCESS",
    "scale":  "full",
    "metrics": {
        # ── Aggregated means ──
        "arm_A_rank_auc_mean":         metrics_agg["A_vanilla"]["rank_auc"]["mean"],
        "arm_A_rank_auc_std":          metrics_agg["A_vanilla"]["rank_auc"]["std"],
        "arm_B_rank_auc_mean":         metrics_agg["B_SNRI"]["rank_auc"]["mean"],
        "arm_B_rank_auc_std":          metrics_agg["B_SNRI"]["rank_auc"]["std"],
        "arm_C_rank_auc_mean":         metrics_agg["C_naive"]["rank_auc"]["mean"],
        "arm_C_rank_auc_std":          metrics_agg["C_naive"]["rank_auc"]["std"],
        "arm_A_final_dead_pct_mean":   metrics_agg["A_vanilla"]["final_dead_pct"]["mean"],
        "arm_A_final_dead_pct_std":    metrics_agg["A_vanilla"]["final_dead_pct"]["std"],
        "arm_B_final_dead_pct_mean":   metrics_agg["B_SNRI"]["final_dead_pct"]["mean"],
        "arm_B_final_dead_pct_std":    metrics_agg["B_SNRI"]["final_dead_pct"]["std"],
        "arm_C_final_dead_pct_mean":   metrics_agg["C_naive"]["final_dead_pct"]["mean"],
        "arm_C_final_dead_pct_std":    metrics_agg["C_naive"]["final_dead_pct"]["std"],
        "arm_A_rank_drop_pct_mean":    metrics_agg["A_vanilla"]["rank_drop_pct"]["mean"],
        "arm_B_rank_drop_pct_mean":    metrics_agg["B_SNRI"]["rank_drop_pct"]["mean"],
        "arm_C_rank_drop_pct_mean":    metrics_agg["C_naive"]["rank_drop_pct"]["mean"],
        # ── Disruption ──
        "arm_B_disruption_mean_mean":  metrics_agg["B_SNRI"].get("disruption_mean", {}).get("mean"),
        "arm_B_disruption_mean_std":   metrics_agg["B_SNRI"].get("disruption_mean", {}).get("std"),
        "arm_B_disruption_max_mean":   metrics_agg["B_SNRI"].get("disruption_max", {}).get("mean"),
        "arm_C_disruption_mean_mean":  metrics_agg["C_naive"].get("disruption_mean", {}).get("mean"),
        "arm_C_disruption_mean_std":   metrics_agg["C_naive"].get("disruption_mean", {}).get("std"),
        "arm_C_disruption_max_mean":   metrics_agg["C_naive"].get("disruption_max", {}).get("mean"),
        # ── Per-seed values ──
        "arm_A_rank_auc_per_seed":     metrics_agg["A_vanilla"]["rank_auc"]["values"],
        "arm_B_rank_auc_per_seed":     metrics_agg["B_SNRI"]["rank_auc"]["values"],
        "arm_C_rank_auc_per_seed":     metrics_agg["C_naive"]["rank_auc"]["values"],
        "arm_A_dead_pct_per_seed":     metrics_agg["A_vanilla"]["final_dead_pct"]["values"],
        "arm_B_dead_pct_per_seed":     metrics_agg["B_SNRI"]["final_dead_pct"]["values"],
        "arm_C_dead_pct_per_seed":     metrics_agg["C_naive"]["final_dead_pct"]["values"],
        # ── Sign consistency ──
        "sign_C_gt_B_rank_auc":       f"{sc_rank_C_gt_B[0]}/{sc_rank_C_gt_B[1]}",
        "sign_B_gt_A_dead_pct":       f"{sc_dead_B_gt_A[0]}/{sc_dead_B_gt_A[1]}",
        "sign_C_lt_B_dead_pct":       f"{sc_dead_C_lt_B[0]}/{sc_dead_C_lt_B[1]}",
        "sign_C_gt_B_disruption":     f"{sc_dis_C_gt_B[0]}/{sc_dis_C_gt_B[1]}",
        # ── Wilcoxon tests ──
        "wilcoxon_p_rank_auc_B_vs_A": p_rank_B_vs_A,
        "wilcoxon_p_rank_auc_C_vs_B": p_rank_C_vs_B,
        "wilcoxon_p_dead_B_vs_A":     p_dead_B_vs_A,
        "wilcoxon_p_dead_C_vs_B":     p_dead_C_vs_B,
        "wilcoxon_p_dis_C_vs_B":      p_dis_C_vs_B,
        "wilcoxon_r_rank_auc_B_vs_A": r_rank_B_vs_A,
        "wilcoxon_r_rank_auc_C_vs_B": r_rank_C_vs_B,
        "wilcoxon_r_dead_B_vs_A":     r_dead_B_vs_A,
        "wilcoxon_r_dead_C_vs_B":     r_dead_C_vs_B,
        "bonferroni_alpha":           ALPHA_CORRECTED,
        # ── Hypothesis verdicts ──
        "hyp_H1_C_more_disruptive":   h1_consistent,
        "hyp_H2_C_higher_rank":        h2_consistent,
        "hyp_H3_B_more_dead_than_A":  h3_consistent,
        "hyp_H4_C_fewer_dead_than_B": h4_consistent,
        "hyp_all_consistent":          h_all,
        # ── Full per-seed data ──
        "per_seed": per_seed_compact,
        "seeds": SEEDS,
        # ── Sanity gates ──
        "sanity_gates_1_4":      "PASS (cited from baseline-00)",
        "sanity_gates_G7_G8_G9": "PASS (cited from increase_complexity-02)",
    },
    "subject_executed": (
        f"ResNet-18 + SGD (lr={LR_INIT}, mom={MOMENTUM}, wd={WEIGHT_DECAY}, nesterov=True) "
        f"+ CosineAnnealingLR/task, CIFAR-100 class-incremental, "
        f"{NUM_TASKS} tasks × {CLASSES_PER_TASK} classes × {EPOCHS_PER_TASK} epochs/task, "
        f"batch={BATCH_SIZE}, seeds={SEEDS}. "
        f"Three arms: A=vanilla, B=SNRI null-space re-init, C=naive kaiming re-init (ReDo-style). "
        f"Full-scale: 5 independent seeds."
    ),
    "notes": (
        f"Ablation-04 full: 5 seeds × 3 arms. "
        f"Rank AUC: A={metrics_agg['A_vanilla']['rank_auc']['mean']:.4f}±{metrics_agg['A_vanilla']['rank_auc']['std']:.4f}  "
        f"B={metrics_agg['B_SNRI']['rank_auc']['mean']:.4f}±{metrics_agg['B_SNRI']['rank_auc']['std']:.4f}  "
        f"C={metrics_agg['C_naive']['rank_auc']['mean']:.4f}±{metrics_agg['C_naive']['rank_auc']['std']:.4f}. "
        f"Final dead%: A={metrics_agg['A_vanilla']['final_dead_pct']['mean']:.2f}±{metrics_agg['A_vanilla']['final_dead_pct']['std']:.2f}  "
        f"B={metrics_agg['B_SNRI']['final_dead_pct']['mean']:.2f}±{metrics_agg['B_SNRI']['final_dead_pct']['std']:.2f}  "
        f"C={metrics_agg['C_naive']['final_dead_pct']['mean']:.2f}±{metrics_agg['C_naive']['final_dead_pct']['std']:.2f}. "
        f"Sign consistency C>B rank_auc: {sc_rank_C_gt_B[0]}/{sc_rank_C_gt_B[1]}, "
        f"B>A dead%: {sc_dead_B_gt_A[0]}/{sc_dead_B_gt_A[1]}, "
        f"C<B dead%: {sc_dead_C_lt_B[0]}/{sc_dead_C_lt_B[1]}, "
        f"C>B disruption: {sc_dis_C_gt_B[0]}/{sc_dis_C_gt_B[1]}. "
        f"All hypotheses consistent: {h_all}. "
        f"Sanity gates 1-4 from baseline-00, G7/G8/G9 from increase_complexity-02."
    ),
}

results_path = os.path.join(OUT_DIR, "RESULTS.json")
with open(results_path, "w") as f:
    json.dump(results, f, indent=2)
log(f"\nResults written to {results_path}")

# Print compact view
compact_top = {
    "status": results["status"],
    "scale":  results["scale"],
    "subject_executed": results["subject_executed"],
    "notes":  results["notes"],
    "metrics": {k: v for k, v in results["metrics"].items()
                if k not in ("per_seed",)},
}
log("\n" + "="*70)
log("FINAL RESULTS (compact, no per_seed):")
log(json.dumps(compact_top, indent=2))
log("="*70)

_log_fh.close()
print(f"\n✓ Done. {results_path}", flush=True)
