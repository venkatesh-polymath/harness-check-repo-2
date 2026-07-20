"""
ablation-03 (probe): Three-arm mechanism isolation ablation on 20-task CIFAR-100.

Arms (run sequentially, identical seeds / schedule):
  A — Vanilla baseline: no re-init (refine-01 design)
  B — SNRI: null-space re-init of dead channels (increase_complexity-02 design)
  C — Naive ReDo: same dead-channel detection, but re-init with standard
       kaiming-normal (He init), outgoing weights NOT zeroed → disruptive

Hypothesis:
  SNRI (B) is provably non-disruptive (logit change ≈ 0) but injects directions
  orthogonal to learned subspace → no function-aligned gradient → poor rank/dead
  restoration. Naive random re-init (C) is disruptive (logit change > 0) but
  injects directions in the full weight space → optimizer can build on them →
  better effective-rank AUC and lower dead-neuron fraction than SNRI.

Primary metrics: effective-rank AUC, final dead-neuron%, disruption (B & C).
Sanity gates: G7/G8/G9 cited from increase_complexity-02 (all PASS).
              Gates 1-4 cited from baseline-00 (all PASS).

Config: 20 tasks × 5 classes × 10 epochs, ResNet-18, SGD + cosine LR, seed=42.
Output: results/ablation-03/
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

# ── reproducibility ────────────────────────────────────────────────────────────
SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── config ─────────────────────────────────────────────────────────────────────
NUM_TASKS        = 20
CLASSES_PER_TASK = 5
EPOCHS_PER_TASK  = 10
BATCH_SIZE       = 128
LR_INIT          = 0.1
MOMENTUM         = 0.9
WEIGHT_DECAY     = 5e-4
DEAD_THRESHOLD   = 0.01   # mean abs post-activation < this → dead
DEAD_PROBE_BATCHES = 20
SNRI_NULL_EPSILON  = 1e-6

OUT_DIR     = "/workspace/results/ablation-03"
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


def task_subset(dataset, classes):
    targets = np.array(dataset.targets)
    idx     = np.where(np.isin(targets, classes))[0]
    return Subset(dataset, idx)


def make_task_loaders(task_id):
    classes   = list(range(task_id * CLASSES_PER_TASK, (task_id + 1) * CLASSES_PER_TASK))
    tr_sub    = task_subset(full_train, classes)
    te_sub    = task_subset(full_test,  classes)
    tr_loader = DataLoader(tr_sub, batch_size=BATCH_SIZE, shuffle=True,
                           num_workers=4, pin_memory=True)
    te_loader = DataLoader(te_sub, batch_size=BATCH_SIZE, shuffle=False,
                           num_workers=4, pin_memory=True)
    return tr_loader, te_loader, classes


# ── model factory ──────────────────────────────────────────────────────────────
def make_model():
    m = models.resnet18(weights=None)
    m.fc = nn.Linear(512, 100)
    return m.to(DEVICE)


# ── effective rank (weight-matrix stable rank) ─────────────────────────────────
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


# ── dead-neuron fraction (whole model, over ReLU outputs) ─────────────────────
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


# ── BasicBlock layer pairs ────────────────────────────────────────────────────
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
def detect_dead_channels(model, layer_pairs, loader, n_batches=DEAD_PROBE_BATCHES,
                          threshold=DEAD_THRESHOLD):
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


# ── SNRI: null-space re-init ──────────────────────────────────────────────────
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
    ReDo-style: re-init dead channels' incoming weights with kaiming_normal
    (standard He init for ReLU). Do NOT zero outgoing weights → disruptive.
    Reset bias to 0.
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
            # Standard He (kaiming) init for one filter
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


# ── disruption measurement ────────────────────────────────────────────────────
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
    return float((before - after).abs().max().item())


def measure_disruption_naive(model, probe_x, layer_pairs, dead_masks):
    """Measure max |logit_diff| for naive random re-init (uses deepcopy)."""
    model.eval()
    with torch.no_grad():
        before = model(probe_x).cpu()
    mc = copy.deepcopy(model)
    # Important: fix seed so the random re-init on the copy is reproducible
    torch.manual_seed(SEED)
    apply_naive_reinit(mc, get_block_layer_pairs(mc), dead_masks)
    mc.eval()
    with torch.no_grad():
        after = mc(probe_x).cpu()
    return float((before - after).abs().max().item())


# ── training helpers ──────────────────────────────────────────────────────────
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


# ── full-test loader (shared across arms) ─────────────────────────────────────
te_loader_full = DataLoader(full_test, batch_size=128, shuffle=False,
                             num_workers=4, pin_memory=True)
_probe_batch_x = next(iter(
    DataLoader(full_test, batch_size=256, shuffle=False, num_workers=4)
))[0].to(DEVICE)


# =============================================================================
# RUN ONE ARM
# =============================================================================
def run_arm(arm_name, reinit_fn):
    """
    Train 20 sequential tasks with the given re-init function (called after each task).
    reinit_fn(model, layer_pairs, dead_masks) → stats_dict  |  None (if arm A)

    Returns dict of arm-level metrics.
    """
    log(f"\n{'='*60}")
    log(f"ARM {arm_name}: starting")
    log(f"{'='*60}")

    # Reset seed for each arm so they start from the same init weights
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)

    model  = make_model()
    pairs  = get_block_layer_pairs(model)

    init_rank = mean_effective_rank(model)
    init_dead = compute_dead_neuron_fraction(model, te_loader_full)
    log(f"  Init rank={init_rank:.4f}  init_dead={init_dead*100:.2f}%")

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
        disruption = None
        n_reinit   = 0

        if reinit_fn is not None:
            dead_masks = detect_dead_channels(model, pairs, te_loader_full,
                                              n_batches=DEAD_PROBE_BATCHES,
                                              threshold=DEAD_THRESHOLD)

            # Measure disruption BEFORE applying to real model
            if arm_name == "B_SNRI":
                disruption = measure_disruption_snri(model, _probe_batch_x,
                                                     pairs, dead_masks)
            elif arm_name == "C_naive":
                disruption = measure_disruption_naive(model, _probe_batch_x,
                                                      pairs, dead_masks)

            # Apply re-init
            torch.manual_seed(SEED + task_id)   # reproducible kaiming for arm C
            stats = reinit_fn(model, pairs, dead_masks)
            n_reinit = sum(s.get("n_reinit", 0) for s in stats.values())

            reinit_log.append({
                "task_id":   task_id,
                "n_reinit":  n_reinit,
                "disruption_max_logit_diff": disruption,
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
            "disruption":          disruption,
        })

        elapsed = time.time() - t0
        dis_str = f"{disruption:.2e}" if disruption is not None else "N/A"
        log(f"  Arm {arm_name} T{task_id:02d}: tr={final_tr_acc:.1f}% "
            f"rank={task_rank:.3f} (drop={rank_drop_pct:.1f}%) "
            f"dead={task_dead*100:.2f}% reinit={n_reinit} "
            f"dis={dis_str} "
            f"elapsed={elapsed:.0f}s")

    total_time = time.time() - t0

    # ── Summary stats ─────────────────────────────────────────────────────────
    ranks     = [m["mean_effective_rank"] for m in per_task]
    dead_pcts = [m["dead_neuron_pct"]     for m in per_task]
    x         = np.arange(NUM_TASKS, dtype=float)
    slope_r, _, r_r, _, _ = scipy_stats.linregress(x, ranks)
    slope_d, _, r_d, _, _ = scipy_stats.linregress(x, dead_pcts)

    rank_auc  = float(np.mean(ranks))
    final_rank = ranks[-1]
    final_dead = dead_pcts[-1]
    rank_drop  = (init_rank - final_rank) / init_rank * 100

    # Disruption summary for arms B/C
    dispts = [e["disruption_max_logit_diff"] for e in reinit_log
              if e["disruption_max_logit_diff"] is not None]
    dis_mean = float(np.mean(dispts)) if dispts else None
    dis_max  = float(np.max(dispts))  if dispts else None
    dis_min  = float(np.min(dispts))  if dispts else None
    total_reinit_arm = sum(e["n_reinit"] for e in reinit_log)

    log(f"\n  Arm {arm_name} summary:")
    log(f"    rank: init={init_rank:.4f} final={final_rank:.4f} "
        f"AUC={rank_auc:.4f} drop={rank_drop:.2f}% "
        f"slope={slope_r:.5f}/task r={r_r:.3f}")
    log(f"    dead: init={init_dead*100:.2f}% final={final_dead:.2f}% "
        f"slope={slope_d:.5f}/task r={r_d:.3f}")
    if dis_mean is not None:
        log(f"    disruption: mean={dis_mean:.3e} max={dis_max:.3e} min={dis_min:.3e}")
    log(f"    total_reinit={total_reinit_arm}  time={total_time/60:.1f}min")

    return {
        "arm":               arm_name,
        "init_rank":         round(init_rank, 4),
        "final_rank":        round(final_rank, 4),
        "rank_auc":          round(rank_auc, 4),
        "rank_drop_pct":     round(rank_drop, 2),
        "rank_trend_slope":  round(slope_r, 5),
        "rank_trend_r":      round(r_r, 4),
        "init_dead_pct":     round(init_dead * 100, 2),
        "final_dead_pct":    round(final_dead, 2),
        "dead_trend_slope":  round(slope_d, 5),
        "dead_trend_r":      round(r_d, 4),
        "disruption_mean":   round(dis_mean, 6) if dis_mean is not None else None,
        "disruption_max":    round(dis_max,  6) if dis_max  is not None else None,
        "disruption_min":    round(dis_min,  6) if dis_min  is not None else None,
        "total_reinit":      total_reinit_arm,
        "training_time_min": round(total_time / 60, 2),
        "per_task":          per_task,
        "reinit_log":        reinit_log,
    }


# =============================================================================
# RUN ALL THREE ARMS
# =============================================================================
log("\n" + "="*60)
log("ABLATION-03: Running arms A, B, C")
log("="*60)

arm_results = {}

# Arm A — Vanilla (no re-init)
arm_results["A_vanilla"] = run_arm("A_vanilla", reinit_fn=None)

# Arm B — SNRI (null-space, non-disruptive)
arm_results["B_SNRI"] = run_arm("B_SNRI", reinit_fn=apply_snri)

# Arm C — Naive random re-init (ReDo-style, disruptive)
arm_results["C_naive"] = run_arm("C_naive", reinit_fn=apply_naive_reinit)


# =============================================================================
# COMPARISON
# =============================================================================
log("\n" + "="*60)
log("COMPARISON: A vs B vs C")
log("="*60)

A = arm_results["A_vanilla"]
B = arm_results["B_SNRI"]
C = arm_results["C_naive"]

def pct_diff(x, ref):
    return (x - ref) / ref * 100 if ref != 0 else 0.0

rank_auc_B_vs_A = pct_diff(B["rank_auc"], A["rank_auc"])
rank_auc_C_vs_A = pct_diff(C["rank_auc"], A["rank_auc"])
rank_auc_C_vs_B = pct_diff(C["rank_auc"], B["rank_auc"])

dead_diff_B_vs_A = B["final_dead_pct"] - A["final_dead_pct"]
dead_diff_C_vs_A = C["final_dead_pct"] - A["final_dead_pct"]
dead_diff_C_vs_B = C["final_dead_pct"] - B["final_dead_pct"]

log(f"Rank AUC:   A={A['rank_auc']:.4f}  B={B['rank_auc']:.4f}  C={C['rank_auc']:.4f}")
log(f"  B vs A: {rank_auc_B_vs_A:+.2f}%   C vs A: {rank_auc_C_vs_A:+.2f}%   C vs B: {rank_auc_C_vs_B:+.2f}%")
log(f"Final dead%: A={A['final_dead_pct']:.2f}%  B={B['final_dead_pct']:.2f}%  C={C['final_dead_pct']:.2f}%")
log(f"  B-A={dead_diff_B_vs_A:+.2f}pp   C-A={dead_diff_C_vs_A:+.2f}pp   C-B={dead_diff_C_vs_B:+.2f}pp")
log(f"Disruption (max logit diff):")
log(f"  A=0.0 (by definition, no re-init)")
log(f"  B={B['disruption_mean']:.3e}/{B['disruption_max']:.3e} (mean/max over tasks)")
log(f"  C={C['disruption_mean']:.3e}/{C['disruption_max']:.3e} (mean/max over tasks)")

# Hypothesis check: C should have higher disruption AND better rank/dead than B
hyp_disruption_C_gt_B = (C["disruption_mean"] is not None
                          and B["disruption_mean"] is not None
                          and C["disruption_mean"] > B["disruption_mean"])
hyp_rank_C_better_B   = C["rank_auc"] > B["rank_auc"]
hyp_dead_C_better_B   = C["final_dead_pct"] < B["final_dead_pct"]

log(f"\nHypothesis checks:")
log(f"  C more disruptive than B: {hyp_disruption_C_gt_B}")
log(f"  C higher rank AUC than B: {hyp_rank_C_better_B} "
    f"({rank_auc_C_vs_B:+.2f}%)")
log(f"  C lower dead% than B:     {hyp_dead_C_better_B} "
    f"({dead_diff_C_vs_B:+.2f}pp)")
hyp_all = hyp_disruption_C_gt_B and hyp_rank_C_better_B and hyp_dead_C_better_B
log(f"  All three conditions met: {hyp_all}")

# =============================================================================
# WRITE RESULTS.JSON
# =============================================================================
results = {
    "status": "SUCCESS",
    "scale":  "probe",
    "metrics": {
        # Per-arm scalars (top-level for easy reading)
        "arm_A_rank_auc":          A["rank_auc"],
        "arm_B_rank_auc":          B["rank_auc"],
        "arm_C_rank_auc":          C["rank_auc"],
        "arm_A_final_dead_pct":    A["final_dead_pct"],
        "arm_B_final_dead_pct":    B["final_dead_pct"],
        "arm_C_final_dead_pct":    C["final_dead_pct"],
        "arm_A_rank_drop_pct":     A["rank_drop_pct"],
        "arm_B_rank_drop_pct":     B["rank_drop_pct"],
        "arm_C_rank_drop_pct":     C["rank_drop_pct"],
        # Disruption
        "arm_B_disruption_mean":   B["disruption_mean"],
        "arm_B_disruption_max":    B["disruption_max"],
        "arm_C_disruption_mean":   C["disruption_mean"],
        "arm_C_disruption_max":    C["disruption_max"],
        # Comparisons
        "rank_auc_B_vs_A_pct":     round(rank_auc_B_vs_A, 2),
        "rank_auc_C_vs_A_pct":     round(rank_auc_C_vs_A, 2),
        "rank_auc_C_vs_B_pct":     round(rank_auc_C_vs_B, 2),
        "dead_pct_B_minus_A_pp":   round(dead_diff_B_vs_A, 2),
        "dead_pct_C_minus_A_pp":   round(dead_diff_C_vs_A, 2),
        "dead_pct_C_minus_B_pp":   round(dead_diff_C_vs_B, 2),
        # Hypothesis result
        "hyp_C_more_disruptive_than_B": hyp_disruption_C_gt_B,
        "hyp_C_higher_rank_than_B":     hyp_rank_C_better_B,
        "hyp_C_lower_dead_than_B":      hyp_dead_C_better_B,
        "hyp_all_conditions_met":       hyp_all,
        # Trend metrics
        "arm_A_rank_trend_slope": A["rank_trend_slope"],
        "arm_B_rank_trend_slope": B["rank_trend_slope"],
        "arm_C_rank_trend_slope": C["rank_trend_slope"],
        "arm_A_dead_trend_slope": A["dead_trend_slope"],
        "arm_B_dead_trend_slope": B["dead_trend_slope"],
        "arm_C_dead_trend_slope": C["dead_trend_slope"],
        # Re-init counts
        "arm_B_total_reinit": B["total_reinit"],
        "arm_C_total_reinit": C["total_reinit"],
        # Sanity gates (cited from prior rounds)
        "sanity_gates_1_4":  "PASS (cited from baseline-00)",
        "sanity_gates_G7_G8_G9": "PASS (cited from increase_complexity-02)",
        # Full arm data
        "arms": {
            "A_vanilla": arm_results["A_vanilla"],
            "B_SNRI":    arm_results["B_SNRI"],
            "C_naive":   arm_results["C_naive"],
        },
    },
    "subject_executed": (
        f"ResNet-18 + SGD (lr={LR_INIT}, mom={MOMENTUM}, wd={WEIGHT_DECAY}) "
        f"+ CosineAnnealingLR/task, CIFAR-100 class-incremental, "
        f"{NUM_TASKS} tasks × {CLASSES_PER_TASK} classes × {EPOCHS_PER_TASK} epochs/task, "
        f"batch={BATCH_SIZE}, seed={SEED}. "
        f"Three arms: A=vanilla, B=SNRI null-space re-init, C=naive kaiming re-init (ReDo-style)."
    ),
    "notes": (
        f"Ablation-03: A vs B vs C head-to-head with identical seed/schedule. "
        f"Rank AUC: A={A['rank_auc']:.4f} B={B['rank_auc']:.4f} C={C['rank_auc']:.4f}. "
        f"Final dead: A={A['final_dead_pct']:.2f}% B={B['final_dead_pct']:.2f}% "
        f"C={C['final_dead_pct']:.2f}%. "
        f"Disruption (mean/max): B={B['disruption_mean']:.2e}/{B['disruption_max']:.2e} "
        f"C={C['disruption_mean']:.2e}/{C['disruption_max']:.2e}. "
        f"Hypothesis (C more disruptive AND better rank AND lower dead than B): {hyp_all}. "
        f"Sanity gates 1-4 from baseline-00, G7/G8/G9 from increase_complexity-02."
    ),
}

results_path = os.path.join(OUT_DIR, "RESULTS.json")
with open(results_path, "w") as f:
    json.dump(results, f, indent=2)
log(f"\nResults written to {results_path}")

# Print compact view (no per_task / reinit_log)
def compact_arm(d):
    return {k: v for k, v in d.items() if k not in ("per_task", "reinit_log")}

compact = {
    "status": results["status"],
    "scale":  results["scale"],
    "metrics": {k: v for k, v in results["metrics"].items()
                if k != "arms"},
    "subject_executed": results["subject_executed"],
    "notes": results["notes"],
}
log("\n" + "="*60)
log("FINAL RESULTS (compact):")
log(json.dumps(compact, indent=2))
log("="*60)

_log_fh.close()
print("\n✓ Done.", results_path, flush=True)
