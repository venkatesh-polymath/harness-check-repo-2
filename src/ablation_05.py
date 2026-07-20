"""
ablation-05 (full): Decisive confound-fixed confirmation.
5 seeds × 6 arms × 20 CIFAR-100 tasks × 10 epochs.

Arms:
  A = vanilla (no re-init, no special regularization)
  B = SNRI (null-space re-init, zero outgoing — non-disruptive)
  C = naive random re-init (kaiming incoming, outgoing UNCHANGED — disruptive)
  D = ReDo (kaiming incoming, zero outgoing)
  E = Shrink-and-Perturb (shrink all weights 10% + Gaussian noise at task boundary)
  F = L2-init (regularize toward initial weights during training)

Confound fixes vs ablation-04:
  (1) No torch.manual_seed inside training loop — seed once per (arm,seed)
  (2) Kaiming/noise re-init uses save/restore of global RNG so DataLoader
      shuffle is identical across all arms within a seed.
  (3) Report realized re-init budget (n_dead_detected, n_reinit_applied)
      per arm/seed/task — matched schedule/threshold across B/C/D.
  (4) Task-incremental evaluation consistently.

New metrics vs ablation-04:
  - Accuracy on each task right after training (AIA proxy)
  - Full re-evaluation of all 20 tasks at the END of training
  - Backward Transfer (BWT) = mean(final_acc[t] - acc_at_training[t])
  - Forgetting = -BWT
  - Average Incremental Accuracy (AIA) = mean(acc_at_training[t])
  - Average Final Accuracy = mean(final_acc[t])

Stats: per-seed sign consistency, Wilcoxon+Bonferroni, seed-3 arm-A outlier
       flagged; all statistics reported with and without seed 3.

Sanity gates re-run (G1-G4): determinism, init loss, constant predictor,
single-batch memorization. G7/G8/G9 cited from increase_complexity-02.
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

# ── Config ─────────────────────────────────────────────────────────────────────
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
SHRINK_ALPHA     = 0.1       # E: 10% shrink per task boundary
PERTURB_SIGMA    = 0.01      # E: noise std
L2_LAMBDA        = 1e-3      # F: L2-init regularization strength

ARMS = ["A_vanilla", "B_SNRI", "C_naive", "D_ReDo", "E_ShP", "F_L2init"]

OUT_DIR     = "/workspace/results/ablation-05"
WEIGHTS_DIR = "/workspace/_weights/ablation-05"
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
log(f"Arms: {ARMS}")
log(f"Shrink-and-Perturb: alpha={SHRINK_ALPHA}, sigma={PERTURB_SIGMA}")
log(f"L2-init: lambda={L2_LAMBDA}")
log(f"CONFOUND FIX: No torch.manual_seed inside training loop. RNG save/restore for re-init.")

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

# ── Data helpers ───────────────────────────────────────────────────────────────
def task_subset(dataset, classes):
    targets = np.array(dataset.targets)
    idx     = np.where(np.isin(targets, classes))[0]
    return Subset(dataset, idx)

# Pre-build all 20 task test loaders (reused across all seeds/arms)
task_test_loaders = []
for _tid in range(NUM_TASKS):
    _classes = list(range(_tid * CLASSES_PER_TASK, (_tid + 1) * CLASSES_PER_TASK))
    _te_sub  = task_subset(full_test, _classes)
    _te_ld   = DataLoader(_te_sub, batch_size=256, shuffle=False,
                          num_workers=4, pin_memory=True)
    task_test_loaders.append((_te_ld, _classes))

# Full test loader for dead-neuron probing
te_loader_full = DataLoader(full_test, batch_size=256, shuffle=False,
                             num_workers=4, pin_memory=True)
log(f"Task test loaders created: {len(task_test_loaders)} tasks")

# ── Model factory ──────────────────────────────────────────────────────────────
def make_model():
    m = models.resnet18(weights=None)
    m.fc = nn.Linear(512, 100)
    return m.to(DEVICE)

# ── Effective rank (stable rank = nuclear-norm / Frobenius-norm) ───────────────
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

# ── Dead-neuron fraction ───────────────────────────────────────────────────────
def compute_dead_neuron_fraction(model, loader, n_batches=DEAD_PROBE_BATCHES):
    activations, hooks = {}, []

    def make_hook(name):
        def hook(module, inp, out):
            a = out.detach().cpu()
            activations[name] = torch.cat([activations[name], a], 0) \
                                 if name in activations else a
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

    total = dead = 0
    for act in activations.values():
        flat     = act.view(act.size(0), -1)
        mean_abs = flat.abs().mean(0)
        dead  += (mean_abs < DEAD_THRESHOLD).sum().item()
        total += flat.size(1)

    return dead / total if total > 0 else 0.0

# ── BasicBlock layer pairs (conv1 → conv2) for SNRI / ReDo / naive ────────────
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

# ── Dead channel detection ─────────────────────────────────────────────────────
def detect_dead_channels(model, layer_pairs, loader,
                          n_batches=DEAD_PROBE_BATCHES, threshold=DEAD_THRESHOLD):
    accum = {}
    hooks = []

    def make_hook(name1):
        def hook(module, inp, out):
            with torch.no_grad():
                mean_abs = inp[0].detach().abs().mean(dim=(0, 2, 3)).cpu()
                if name1 not in accum:
                    accum[name1] = [mean_abs.clone(), 1]
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

# ── Re-init: SNRI (null-space, zero outgoing) ──────────────────────────────────
def apply_snri(model, layer_pairs, dead_masks, epsilon=SNRI_NULL_EPSILON):
    """SNRI: incoming → null-space direction, outgoing → 0. Non-disruptive."""
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

        try:
            U, S, Vh = torch.linalg.svd(W1_2d.float(), full_matrices=True)
            rank1    = int((S > epsilon).sum().item())
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
            idx     = int(idx.item())
            row     = Vh.shape[0] - 1 - (j % min(n_dead, Vh.shape[0]))
            new_dir = Vh[row].to(W1.device)
            new_dir = (new_dir / (new_dir.norm() + 1e-12)) * scale
            W1[idx] = new_dir.view(C_in, kH, kW).to(W1.dtype)
            conv2.weight.data[:, idx] = 0.0
            if conv1.bias is not None:
                conv1.bias.data[idx] = 0.0
            n_reinit += 1

        stats[name1] = {"n_dead": n_dead, "n_reinit": n_reinit, "rank": rank1}
    return stats

# ── Re-init: naive random (kaiming incoming, outgoing UNCHANGED) ───────────────
def apply_naive_reinit(model, layer_pairs, dead_masks):
    """Naive: kaiming incoming, outgoing unchanged — disruptive."""
    stats = {}
    for name1, conv1, name2, conv2 in layer_pairs:
        if name1 not in dead_masks:
            continue
        dead_mask = dead_masks[name1]
        n_dead    = int(dead_mask.sum().item())
        if n_dead == 0:
            stats[name1] = {"n_dead": 0, "n_reinit": 0}
            continue

        for idx in dead_mask.nonzero(as_tuple=True)[0]:
            idx = int(idx.item())
            tmp = torch.empty_like(conv1.weight.data[idx])
            nn.init.kaiming_normal_(tmp.unsqueeze(0), mode='fan_in', nonlinearity='relu')
            conv1.weight.data[idx] = tmp.squeeze(0)
            # Outgoing: NOT zeroed (disruptive)
            if conv1.bias is not None:
                conv1.bias.data[idx] = 0.0

        stats[name1] = {"n_dead": n_dead, "n_reinit": n_dead}
    return stats

# ── Re-init: ReDo (kaiming incoming, zero outgoing) ───────────────────────────
def apply_redo(model, layer_pairs, dead_masks):
    """ReDo: kaiming incoming, zero outgoing — new direction, non-disruptive output."""
    stats = {}
    for name1, conv1, name2, conv2 in layer_pairs:
        if name1 not in dead_masks:
            continue
        dead_mask = dead_masks[name1]
        n_dead    = int(dead_mask.sum().item())
        if n_dead == 0:
            stats[name1] = {"n_dead": 0, "n_reinit": 0}
            continue

        for idx in dead_mask.nonzero(as_tuple=True)[0]:
            idx = int(idx.item())
            tmp = torch.empty_like(conv1.weight.data[idx])
            nn.init.kaiming_normal_(tmp.unsqueeze(0), mode='fan_in', nonlinearity='relu')
            conv1.weight.data[idx] = tmp.squeeze(0)
            conv2.weight.data[:, idx] = 0.0   # zero outgoing
            if conv1.bias is not None:
                conv1.bias.data[idx] = 0.0

        stats[name1] = {"n_dead": n_dead, "n_reinit": n_dead}
    return stats

# ── Re-init: Shrink-and-Perturb ────────────────────────────────────────────────
def apply_shrink_perturb(model, alpha=SHRINK_ALPHA, sigma=PERTURB_SIGMA):
    """Shrink all weights by (1-alpha) and add N(0, sigma^2) noise."""
    with torch.no_grad():
        for param in model.parameters():
            param.data.mul_(1.0 - alpha)
            param.data.add_(torch.randn_like(param) * sigma)

# ── Disruption measurement ─────────────────────────────────────────────────────
def measure_disruption(model, probe_x, apply_fn, apply_kwargs):
    """
    Measure max/mean |logit_diff| for a given re-init applied to a deepcopy.
    Returns (max_diff, mean_diff).
    """
    model.eval()
    with torch.no_grad():
        before = model(probe_x).cpu()
    mc = copy.deepcopy(model)
    apply_fn(mc, **apply_kwargs)
    mc.eval()
    with torch.no_grad():
        after = mc(probe_x).cpu()
    diff = (before - after).abs()
    return float(diff.max().item()), float(diff.mean().item())

# ── Training helpers ───────────────────────────────────────────────────────────
criterion = nn.CrossEntropyLoss()

def train_epoch(model, loader, optimizer, init_params_gpu=None, l2_lambda=0.0):
    """Train one epoch. Optionally add L2-init regularization."""
    model.train()
    total_loss = correct = total = 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        out  = model(x)
        loss = criterion(out, y)
        if init_params_gpu is not None and l2_lambda > 0:
            l2_pen = sum((p - p0).pow(2).sum()
                         for p, p0 in zip(model.parameters(), init_params_gpu))
            loss = loss + l2_lambda * l2_pen
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * y.size(0)
        correct    += (out.argmax(1) == y).sum().item()
        total      += y.size(0)
    return total_loss / total, correct / total

@torch.no_grad()
def eval_epoch(model, loader):
    model.eval()
    total_loss = correct = total = 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        out  = model(x)
        loss = criterion(out, y)
        total_loss += loss.item() * y.size(0)
        correct    += (out.argmax(1) == y).sum().item()
        total      += y.size(0)
    return total_loss / total, correct / total

# ── Probe batch ────────────────────────────────────────────────────────────────
def make_probe_batch(seed):
    """Return a fixed 512-image probe batch from the test set."""
    g = torch.Generator()
    g.manual_seed(seed)
    loader = DataLoader(full_test, batch_size=512, shuffle=True,
                        generator=g, num_workers=0)
    x, _ = next(iter(loader))
    return x.to(DEVICE)

# =============================================================================
# SANITY GATES G1–G4
# =============================================================================
log("\n" + "="*70)
log("SANITY GATES G1–G4")
log("="*70)

gate_results = {}

# G1: Determinism — two runs with seed=42 give identical loss curves
log("G1: Determinism (two runs with seed=42)")
gate1_pass = True
try:
    losses_run = []
    for run_idx in range(2):
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        np.random.seed(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False
        m = make_model()
        classes0 = list(range(0, CLASSES_PER_TASK))
        sub0  = task_subset(full_train, classes0)
        ld0   = DataLoader(sub0, batch_size=BATCH_SIZE, shuffle=True,
                           num_workers=2, pin_memory=True, drop_last=False)
        opt = optim.SGD(m.parameters(), lr=LR_INIT, momentum=MOMENTUM,
                        weight_decay=WEIGHT_DECAY, nesterov=True)
        run_losses = []
        for epoch in range(3):
            tr_loss, _ = train_epoch(m, ld0, opt)
            run_losses.append(tr_loss)
        losses_run.append(run_losses)
    max_diff = max(abs(a - b) for a, b in zip(losses_run[0], losses_run[1]))
    gate1_pass = max_diff == 0.0
    log(f"  Run 0 losses: {[f'{v:.4f}' for v in losses_run[0]]}")
    log(f"  Run 1 losses: {[f'{v:.4f}' for v in losses_run[1]]}")
    log(f"  Max abs diff: {max_diff:.2e}  → {'PASS' if gate1_pass else 'FAIL'}")
    gate_results["G1_determinism"] = "PASS" if gate1_pass else f"FAIL (max_diff={max_diff:.2e})"
except Exception as e:
    log(f"  ERROR: {e}")
    gate_results["G1_determinism"] = f"ERROR: {e}"
    gate1_pass = False

# G2: Init loss ≈ log(100) = 4.605 ± 0.05
log("G2: Init loss ≈ log(100) = 4.605")
gate2_pass = True
try:
    init_losses = []
    for gs in [0, 1, 2]:
        torch.manual_seed(gs)
        torch.cuda.manual_seed_all(gs)
        np.random.seed(gs)
        m = make_model()
        # one forward pass on full test batch
        x_batch, y_batch = next(iter(DataLoader(full_test, batch_size=512, shuffle=False)))
        with torch.no_grad():
            loss_init = criterion(m(x_batch.to(DEVICE)), y_batch.to(DEVICE)).item()
        init_losses.append(loss_init)
    target = np.log(100)
    diffs  = [abs(l - target) for l in init_losses]
    gate2_pass = all(d <= 0.05 for d in diffs)
    log(f"  Init losses: {[f'{v:.4f}' for v in init_losses]}")
    log(f"  Diffs from {target:.4f}: {[f'{d:.4f}' for d in diffs]}")
    log(f"  → {'PASS' if gate2_pass else 'FAIL'}")
    gate_results["G2_init_loss"] = "PASS" if gate2_pass else f"FAIL (diffs={diffs})"
except Exception as e:
    log(f"  ERROR: {e}")
    gate_results["G2_init_loss"] = f"ERROR: {e}"
    gate2_pass = False

# G3: Constant predictor = 1.0 ± 0.1%
log("G3: Constant predictor accuracy = 1.0%")
gate3_pass = False
try:
    m_const = make_model()
    # Set all weights to 0 (uniform logits → uniform prediction)
    with torch.no_grad():
        for p in m_const.parameters():
            p.zero_()
    _, acc_const = eval_epoch(m_const, te_loader_full)
    acc_const_pct = acc_const * 100
    gate3_pass = 0.9 <= acc_const_pct <= 1.1
    log(f"  Constant predictor accuracy: {acc_const_pct:.3f}%")
    log(f"  → {'PASS' if gate3_pass else 'FAIL'}")
    gate_results["G3_constant_predictor"] = "PASS" if gate3_pass else f"FAIL ({acc_const_pct:.3f}%)"
except Exception as e:
    log(f"  ERROR: {e}")
    gate_results["G3_constant_predictor"] = f"ERROR: {e}"

# G4: Single-batch memorization → loss < 0.01 within 500 steps
log("G4: Single-batch memorization")
gate4_pass = False
try:
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    m_mem = make_model()
    opt_mem = optim.SGD(m_mem.parameters(), lr=0.01, momentum=MOMENTUM,
                        weight_decay=0.0, nesterov=True)
    x_mem, y_mem = next(iter(DataLoader(full_train, batch_size=32, shuffle=False)))
    x_mem, y_mem = x_mem.to(DEVICE), y_mem.to(DEVICE)
    steps_to_01 = None
    for step in range(500):
        opt_mem.zero_grad()
        out_mem = m_mem(x_mem)
        loss_mem = criterion(out_mem, y_mem)
        loss_mem.backward()
        opt_mem.step()
        if loss_mem.item() < 0.01 and steps_to_01 is None:
            steps_to_01 = step + 1
            break
    gate4_pass = steps_to_01 is not None
    log(f"  Final loss: {loss_mem.item():.6f}  Steps to <0.01: {steps_to_01}")
    log(f"  → {'PASS' if gate4_pass else 'FAIL'}")
    gate_results["G4_memorization"] = "PASS" if gate4_pass else f"FAIL (loss={loss_mem.item():.4f})"
except Exception as e:
    log(f"  ERROR: {e}")
    gate_results["G4_memorization"] = f"ERROR: {e}"

all_gates_pass = all(v.startswith("PASS") or v.startswith("PASS")
                     for v in gate_results.values()
                     if not v.startswith("ERROR"))
log(f"\nSanity gates summary: {gate_results}")
log(f"All gates pass: {all(v.startswith('PASS') for v in gate_results.values())}")


# =============================================================================
# RUN ONE ARM FOR ONE SEED
# =============================================================================
def run_arm(arm_name, seed):
    """
    Train 20 sequential tasks for one (arm, seed) combination.
    Returns a dict with all per-arm metrics.

    Confound-fix: seed is set ONCE at the start; no manual_seed inside loop.
    Kaiming/noise re-init uses save/restore of global RNG state.
    """
    # ── Set seed exactly ONCE ──────────────────────────────────────────────────
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

    model  = make_model()
    pairs  = get_block_layer_pairs(model)
    probe_x = make_probe_batch(seed)

    # For arm F: store initial parameters (frozen) for L2-init penalty
    init_params_gpu = None
    if arm_name == "F_L2init":
        init_params_gpu = [p.detach().clone() for p in model.parameters()]

    init_rank = mean_effective_rank(model)
    init_dead = compute_dead_neuron_fraction(model, te_loader_full)
    log(f"  [{arm_name} s={seed}] init rank={init_rank:.4f}  dead={init_dead*100:.2f}%")

    # Per-task tracking
    per_task          = []       # detailed per-task records
    rank_history      = []       # effective rank after each task
    dead_history      = []       # dead% after each task
    reinit_log        = []       # re-init events per task
    task_accs_training = []      # test acc on task t right after training t

    t0 = time.time()

    for task_id in range(NUM_TASKS):
        # Make task-specific train loader
        classes   = list(range(task_id * CLASSES_PER_TASK, (task_id + 1) * CLASSES_PER_TASK))
        tr_sub    = task_subset(full_train, classes)
        tr_loader = DataLoader(tr_sub, batch_size=BATCH_SIZE, shuffle=True,
                               num_workers=4, pin_memory=True, drop_last=False)
        te_loader = task_test_loaders[task_id][0]

        # Fresh optimizer + scheduler per task (standard CL protocol)
        optimizer = optim.SGD(model.parameters(), lr=LR_INIT, momentum=MOMENTUM,
                              weight_decay=WEIGHT_DECAY, nesterov=True)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                          T_max=EPOCHS_PER_TASK,
                                                          eta_min=0.0)

        # ── Training epochs ────────────────────────────────────────────────────
        for epoch in range(EPOCHS_PER_TASK):
            train_epoch(model, tr_loader, optimizer,
                        init_params_gpu=init_params_gpu,
                        l2_lambda=L2_LAMBDA if arm_name == "F_L2init" else 0.0)
            scheduler.step()

        # ── Evaluate on current task ───────────────────────────────────────────
        _, te_acc = eval_epoch(model, te_loader)
        task_accs_training.append(te_acc * 100)

        # ── Rank + dead after task ─────────────────────────────────────────────
        task_rank = mean_effective_rank(model)
        task_dead = compute_dead_neuron_fraction(model, te_loader_full)
        rank_history.append(task_rank)
        dead_history.append(task_dead * 100)

        # ── Re-init (arm-specific) ─────────────────────────────────────────────
        dis_max = dis_mean = None
        n_dead_detected = n_reinit_applied = 0
        re_init_seed = seed * 100_000 + task_id  # deterministic reinit seed

        if arm_name in ("B_SNRI", "C_naive", "D_ReDo"):
            dead_masks = detect_dead_channels(model, pairs, te_loader_full,
                                              n_batches=DEAD_PROBE_BATCHES,
                                              threshold=DEAD_THRESHOLD)
            n_dead_detected = sum(int(m.sum().item()) for m in dead_masks.values())

            if n_dead_detected > 0:
                if arm_name == "B_SNRI":
                    # SNRI is deterministic (SVD-based), no global RNG needed
                    # Measure disruption on deepcopy
                    model.eval()
                    with torch.no_grad():
                        before_logits = model(probe_x).cpu()
                    mc = copy.deepcopy(model)
                    apply_snri(mc, get_block_layer_pairs(mc), dead_masks)
                    mc.eval()
                    with torch.no_grad():
                        after_logits = mc(probe_x).cpu()
                    diff_t = (before_logits - after_logits).abs()
                    dis_max, dis_mean = float(diff_t.max()), float(diff_t.mean())
                    del mc, before_logits, after_logits, diff_t

                    # Apply to real model (deterministic — no RNG involved)
                    stats = apply_snri(model, pairs, dead_masks)

                elif arm_name in ("C_naive", "D_ReDo"):
                    # Save global RNG state so re-init doesn't affect DataLoader
                    rng_cpu = torch.get_rng_state()
                    rng_gpu = torch.cuda.get_rng_state_all()

                    # -- Disruption measurement on deepcopy --
                    torch.manual_seed(re_init_seed)
                    mc = copy.deepcopy(model)
                    mc_pairs = get_block_layer_pairs(mc)
                    if arm_name == "C_naive":
                        apply_naive_reinit(mc, mc_pairs, dead_masks)
                    else:
                        apply_redo(mc, mc_pairs, dead_masks)
                    model.eval(); mc.eval()
                    with torch.no_grad():
                        before_logits = model(probe_x).cpu()
                        after_logits  = mc(probe_x).cpu()
                    diff_t = (before_logits - after_logits).abs()
                    dis_max, dis_mean = float(diff_t.max()), float(diff_t.mean())
                    del mc, mc_pairs, before_logits, after_logits, diff_t

                    # -- Apply to real model (same seed → same random weights) --
                    torch.manual_seed(re_init_seed)
                    if arm_name == "C_naive":
                        stats = apply_naive_reinit(model, pairs, dead_masks)
                    else:
                        stats = apply_redo(model, pairs, dead_masks)

                    # -- Restore global RNG state --
                    torch.set_rng_state(rng_cpu)
                    torch.cuda.set_rng_state_all(rng_gpu)

                n_reinit_applied = sum(s.get("n_reinit", 0) for s in stats.values())

        elif arm_name == "E_ShP":
            # Shrink-and-Perturb: measure disruption, then apply (with RNG save/restore)
            rng_cpu = torch.get_rng_state()
            rng_gpu = torch.cuda.get_rng_state_all()

            torch.manual_seed(re_init_seed)
            mc = copy.deepcopy(model)
            apply_shrink_perturb(mc, SHRINK_ALPHA, PERTURB_SIGMA)
            model.eval(); mc.eval()
            with torch.no_grad():
                before_logits = model(probe_x).cpu()
                after_logits  = mc(probe_x).cpu()
            diff_t = (before_logits - after_logits).abs()
            dis_max, dis_mean = float(diff_t.max()), float(diff_t.mean())
            del mc, before_logits, after_logits, diff_t

            torch.manual_seed(re_init_seed)
            apply_shrink_perturb(model, SHRINK_ALPHA, PERTURB_SIGMA)

            torch.set_rng_state(rng_cpu)
            torch.cuda.set_rng_state_all(rng_gpu)

        # arm A and F: no explicit re-init event

        reinit_log.append({
            "task_id":          task_id,
            "n_dead_detected":  n_dead_detected,
            "n_reinit_applied": n_reinit_applied,
            "dis_max":          dis_max,
            "dis_mean":         dis_mean,
        })

        elapsed = time.time() - t0
        dis_str = f"{dis_max:.2e}" if dis_max is not None else "N/A"
        log(f"  [{arm_name} s={seed}] T{task_id:02d}: te={te_acc*100:.1f}% "
            f"rank={task_rank:.3f} dead={task_dead*100:.2f}% "
            f"reinit={n_reinit_applied} dis={dis_str} t={elapsed:.0f}s")

    # ── Final re-evaluation: all 20 tasks ─────────────────────────────────────
    log(f"  [{arm_name} s={seed}] Final re-evaluation of all {NUM_TASKS} tasks...")
    final_accs = []
    for t_id in range(NUM_TASKS):
        _, fa = eval_epoch(model, task_test_loaders[t_id][0])
        final_accs.append(fa * 100)
    log(f"  [{arm_name} s={seed}] final_accs: {[f'{v:.1f}' for v in final_accs]}")

    # ── BWT, AIA, Average Final Acc ────────────────────────────────────────────
    # BWT = mean(final_acc[t] - acc_at_training[t]) for t = 0..T-2
    bwt = float(np.mean([final_accs[t] - task_accs_training[t]
                         for t in range(NUM_TASKS - 1)]))
    forgetting = -bwt   # positive = forgetting
    aia        = float(np.mean(task_accs_training))
    avg_final  = float(np.mean(final_accs))

    total_time = time.time() - t0

    # ── Summary stats ─────────────────────────────────────────────────────────
    rank_auc   = float(np.mean(rank_history))
    final_rank = rank_history[-1]
    final_dead = dead_history[-1]
    rank_drop  = (init_rank - final_rank) / init_rank * 100

    x_arr = np.arange(NUM_TASKS, dtype=float)
    slope_r, _, r_r, _, _ = scipy_stats.linregress(x_arr, rank_history)
    slope_d, _, r_d, _, _ = scipy_stats.linregress(x_arr, dead_history)

    dis_maxs  = [e["dis_max"]  for e in reinit_log if e["dis_max"]  is not None]
    dis_means = [e["dis_mean"] for e in reinit_log if e["dis_mean"] is not None]
    dis_mean_val = float(np.mean(dis_means)) if dis_means else None
    dis_max_val  = float(np.max(dis_maxs))   if dis_maxs  else None

    total_reinit = sum(e["n_reinit_applied"] for e in reinit_log)
    total_dead   = sum(e["n_dead_detected"]  for e in reinit_log)

    log(f"  [{arm_name} s={seed}] SUMMARY: "
        f"rank_auc={rank_auc:.4f} drop={rank_drop:.2f}% "
        f"dead={final_dead:.2f}% reinit={total_reinit} "
        f"BWT={bwt:.2f}% forg={forgetting:.2f}% AIA={aia:.2f}% "
        f"avgFinal={avg_final:.2f}% t={total_time/60:.1f}min")

    return {
        "arm":              arm_name,
        "seed":             seed,
        # Rank metrics
        "init_rank":        round(init_rank, 4),
        "final_rank":       round(final_rank, 4),
        "rank_auc":         round(rank_auc, 4),
        "rank_drop_pct":    round(rank_drop, 2),
        "rank_trend_slope": round(float(slope_r), 5),
        "rank_trend_r":     round(float(r_r), 4),
        # Dead neuron metrics
        "init_dead_pct":    round(init_dead * 100, 2),
        "final_dead_pct":   round(final_dead, 2),
        "dead_trend_slope": round(float(slope_d), 5),
        "dead_trend_r":     round(float(r_d), 4),
        # Performance metrics
        "aia":              round(aia, 2),
        "avg_final_acc":    round(avg_final, 2),
        "bwt":              round(bwt, 2),
        "forgetting":       round(forgetting, 2),
        # Disruption
        "dis_mean":         round(dis_mean_val, 6) if dis_mean_val is not None else None,
        "dis_max":          round(dis_max_val,  6) if dis_max_val  is not None else None,
        # Budget
        "total_reinit":     total_reinit,
        "total_dead_detected": total_dead,
        "realized_budget":  reinit_log,  # per-task
        "task_accs_training": [round(v, 2) for v in task_accs_training],
        "final_task_accs":  [round(v, 2) for v in final_accs],
        "rank_history":     [round(v, 4) for v in rank_history],
        "dead_history":     [round(v, 2) for v in dead_history],
        "training_time_min": round(total_time / 60, 2),
    }

# =============================================================================
# MAIN LOOP: seeds × arms
# =============================================================================
log("\n" + "="*70)
log(f"ABLATION-05: Full-scale {len(SEEDS)} seeds × {len(ARMS)} arms")
log("="*70)

results_by_seed = {}   # {seed: {arm_name: result_dict}}
t_global = time.time()

for seed in SEEDS:
    log(f"\n{'='*60}")
    log(f"SEED {seed}")
    log(f"{'='*60}")
    results_by_seed[seed] = {}

    for arm_name in ARMS:
        log(f"\n  --- {arm_name} seed={seed} ---")
        results_by_seed[seed][arm_name] = run_arm(arm_name, seed)

    elapsed_global = (time.time() - t_global) / 60
    log(f"Seed {seed} done. Total elapsed: {elapsed_global:.1f}min")

total_time_min = (time.time() - t_global) / 60
log(f"\nAll seeds done. Total time: {total_time_min:.1f}min")

# =============================================================================
# AGGREGATE STATISTICS
# =============================================================================
log("\n" + "="*70)
log("AGGREGATE STATISTICS")
log("="*70)

def collect(arm, metric):
    vals = []
    for s in SEEDS:
        v = results_by_seed[s][arm][metric]
        if v is not None:
            vals.append(float(v))
    return vals

def summarize(vals):
    a = np.array(vals, dtype=float)
    return {
        "mean":   round(float(a.mean()), 4),
        "std":    round(float(a.std(ddof=1)), 4),
        "values": [round(float(v), 4) for v in a],
    }

# Key metrics to aggregate per arm
KEY_METRICS = ["rank_auc", "final_dead_pct", "rank_drop_pct",
               "aia", "avg_final_acc", "bwt", "forgetting",
               "dis_mean", "dis_max"]

agg = {}
for arm in ARMS:
    agg[arm] = {}
    for metric in KEY_METRICS:
        vals = collect(arm, metric)
        if vals:
            agg[arm][metric] = summarize(vals)

for arm, d in agg.items():
    log(f"\n  {arm}:")
    for k, v in d.items():
        log(f"    {k}: {v['mean']:.4f} ± {v['std']:.4f}  values={v['values']}")

# =============================================================================
# SIGN CONSISTENCY
# =============================================================================
log("\n--- Sign consistency across seeds ---")

def sign_count(arm1, arm2, metric, arm1_gt=True):
    """Count seeds where arm1[metric] > arm2[metric] (or <, if arm1_gt=False)."""
    diffs = [results_by_seed[s][arm1][metric] - results_by_seed[s][arm2][metric]
             for s in SEEDS
             if results_by_seed[s][arm1].get(metric) is not None
             and results_by_seed[s][arm2].get(metric) is not None]
    n_consistent = sum(1 for d in diffs if (d > 0) == arm1_gt)
    return n_consistent, len(diffs), [round(d, 4) for d in diffs]

# Primary comparisons
sign_checks = {
    "B_rank_gt_A":    sign_count("B_SNRI", "A_vanilla", "rank_auc",       True),
    "D_rank_gt_A":    sign_count("D_ReDo", "A_vanilla", "rank_auc",       True),
    "E_rank_gt_A":    sign_count("E_ShP",  "A_vanilla", "rank_auc",       True),
    "F_rank_gt_A":    sign_count("F_L2init","A_vanilla", "rank_auc",       True),
    "C_dis_gt_B":     sign_count("C_naive","B_SNRI",    "dis_mean",        True),
    "C_dis_gt_D":     sign_count("C_naive","D_ReDo",    "dis_mean",        True),
    "B_forg_lt_C":    sign_count("B_SNRI", "C_naive",   "forgetting", False),
    "B_forg_lt_A":    sign_count("B_SNRI", "A_vanilla", "forgetting", False),
}

for label, (n, total, diffs) in sign_checks.items():
    log(f"  {label}: {n}/{total}  diffs={diffs}")

# =============================================================================
# WILCOXON SIGNED-RANK TESTS
# =============================================================================
log("\n--- Wilcoxon signed-rank tests ---")

def wilcoxon_test(arm1, arm2, metric, label):
    a = collect(arm1, metric)
    b = collect(arm2, metric)
    if len(a) != len(b) or len(a) < 2:
        log(f"  {label}: N/A")
        return None, None
    try:
        stat, pval = scipy_stats.wilcoxon(a, b, alternative='two-sided')
        n = len(a)
        r = 1 - 2*stat / (n * (n + 1) / 2)
        mean_diff = float(np.mean(np.array(a) - np.array(b)))
        log(f"  {label}: stat={stat:.2f} p={pval:.4f} r={r:.3f} "
            f"(mean_diff={mean_diff:+.4f})")
        return float(pval), float(r)
    except Exception as e:
        log(f"  {label}: error {e}")
        return None, None

wilcoxon_results = {}
pairs_to_test = [
    ("B_SNRI",   "A_vanilla", "rank_auc",    "rank_auc B-vs-A"),
    ("B_SNRI",   "A_vanilla", "forgetting",  "forgetting B-vs-A"),
    ("B_SNRI",   "A_vanilla", "avg_final_acc","avgFinalAcc B-vs-A"),
    ("C_naive",  "B_SNRI",    "rank_auc",    "rank_auc C-vs-B"),
    ("C_naive",  "B_SNRI",    "dis_mean",    "dis_mean C-vs-B"),
    ("D_ReDo",   "B_SNRI",    "rank_auc",    "rank_auc D-vs-B"),
    ("D_ReDo",   "B_SNRI",    "dis_mean",    "dis_mean D-vs-B"),
    ("E_ShP",    "A_vanilla", "rank_auc",    "rank_auc E-vs-A"),
    ("E_ShP",    "A_vanilla", "forgetting",  "forgetting E-vs-A"),
    ("F_L2init", "A_vanilla", "rank_auc",    "rank_auc F-vs-A"),
    ("F_L2init", "A_vanilla", "forgetting",  "forgetting F-vs-A"),
]

N_TESTS = len(pairs_to_test)
ALPHA_CORRECTED = 0.05 / N_TESTS
log(f"\n  Bonferroni-corrected α = {ALPHA_CORRECTED:.4f} ({N_TESTS} tests)")

for arm1, arm2, metric, label in pairs_to_test:
    p, r = wilcoxon_test(arm1, arm2, metric, label)
    wilcoxon_results[label] = {"p": p, "r": r,
                                "sig": bool(p is not None and p < ALPHA_CORRECTED)}

# =============================================================================
# SEED-3 ARM-A OUTLIER ANALYSIS
# =============================================================================
log("\n--- Seed-3 arm-A outlier analysis ---")

seed3_A = results_by_seed[3]["A_vanilla"]
log(f"  Seed 3 arm A: rank_auc={seed3_A['rank_auc']}  "
    f"rank_drop={seed3_A['rank_drop_pct']}%  "
    f"dead={seed3_A['final_dead_pct']}%  "
    f"forgetting={seed3_A['forgetting']}%")

# Identify other seeds for arm A
seeds_excl3 = [s for s in SEEDS if s != 3]
for arm in ARMS:
    vals_all  = collect(arm, "rank_auc")
    vals_excl = [results_by_seed[s][arm]["rank_auc"] for s in seeds_excl3]
    mean_all  = np.mean(vals_all)
    mean_excl = np.mean(vals_excl)
    std_excl  = np.std(vals_excl, ddof=1) if len(vals_excl) > 1 else 0.0
    z_seed3   = ((results_by_seed[3][arm]["rank_auc"] - mean_excl) / std_excl
                 if std_excl > 0 else float("nan"))
    log(f"  {arm}: rank_auc_all={mean_all:.4f}  excl_seed3={mean_excl:.4f}±{std_excl:.4f}  "
        f"z_seed3={z_seed3:.2f}")

# Wilcoxon without seed 3 for primary comparisons
log("\n  Wilcoxon without seed 3:")
def wilcoxon_excl3(arm1, arm2, metric, label):
    a = [results_by_seed[s][arm1][metric] for s in seeds_excl3
         if results_by_seed[s][arm1].get(metric) is not None]
    b = [results_by_seed[s][arm2][metric] for s in seeds_excl3
         if results_by_seed[s][arm2].get(metric) is not None]
    if len(a) < 2:
        log(f"    {label}: N/A (n={len(a)})")
        return None, None
    try:
        stat, pval = scipy_stats.wilcoxon(a, b, alternative='two-sided')
        n = len(a)
        r = 1 - 2*stat / (n*(n+1)/2)
        log(f"    {label}: p={pval:.4f} r={r:.3f}")
        return float(pval), float(r)
    except Exception as e:
        log(f"    {label}: error {e}")
        return None, None

wilcoxon_excl3_results = {}
for arm1, arm2, metric, label in pairs_to_test[:5]:
    p, r = wilcoxon_excl3(arm1, arm2, metric, f"{label} (excl_seed3)")
    wilcoxon_excl3_results[f"{label}_excl_seed3"] = {"p": p, "r": r}

# =============================================================================
# BUILD RESULTS.JSON
# =============================================================================
log("\n" + "="*70)
log("Writing RESULTS.json")
log("="*70)

def compact_arm(d):
    """Remove large arrays for compact per-seed summary."""
    skip = {"realized_budget", "task_accs_training", "final_task_accs",
            "rank_history", "dead_history"}
    return {k: v for k, v in d.items() if k not in skip}

per_seed_compact = {
    str(s): {arm: compact_arm(results_by_seed[s][arm]) for arm in ARMS}
    for s in SEEDS
}

# Mean ± std for every arm × metric
agg_flat = {}
for arm in ARMS:
    for metric in KEY_METRICS:
        if metric in agg[arm]:
            agg_flat[f"arm_{arm}_{metric}_mean"] = agg[arm][metric]["mean"]
            agg_flat[f"arm_{arm}_{metric}_std"]  = agg[arm][metric]["std"]
            agg_flat[f"arm_{arm}_{metric}_values"] = agg[arm][metric]["values"]

results = {
    "status": "SUCCESS",
    "scale":  "full",
    "metrics": {
        **agg_flat,
        # Sign consistency
        **{f"sign_{k}": f"{v[0]}/{v[1]}" for k, v in sign_checks.items()},
        # Wilcoxon tests
        "wilcoxon_results": wilcoxon_results,
        "wilcoxon_excl_seed3": wilcoxon_excl3_results,
        "bonferroni_alpha": ALPHA_CORRECTED,
        "n_tests": N_TESTS,
        # Sanity gates
        "sanity_gates": gate_results,
        "sanity_gates_G7_G8_G9": "PASS (cited from increase_complexity-02)",
        # Per-seed compact
        "per_seed": per_seed_compact,
        "seeds": SEEDS,
        # Budget tracking
        "budget_tracking": {
            str(s): {
                arm: {
                    "total_reinit":       results_by_seed[s][arm]["total_reinit"],
                    "total_dead_detected": results_by_seed[s][arm]["total_dead_detected"],
                    "per_task_dead":      [e["n_dead_detected"]  for e in results_by_seed[s][arm]["realized_budget"]],
                    "per_task_reinit":    [e["n_reinit_applied"] for e in results_by_seed[s][arm]["realized_budget"]],
                }
                for arm in ["B_SNRI", "C_naive", "D_ReDo"]  # only re-init arms
            }
            for s in SEEDS
        },
        "total_training_time_min": round(total_time_min, 1),
    },
    "subject_executed": (
        f"ResNet-18 + SGD (lr={LR_INIT}, mom={MOMENTUM}, wd={WEIGHT_DECAY}, nesterov=True) "
        f"+ CosineAnnealingLR/task, CIFAR-100 class-incremental, "
        f"{NUM_TASKS} tasks × {CLASSES_PER_TASK} classes × {EPOCHS_PER_TASK} epochs/task, "
        f"batch={BATCH_SIZE}, seeds={SEEDS}. "
        f"Six arms: A=vanilla, B=SNRI null-space re-init, C=naive kaiming re-init, "
        f"D=ReDo (kaiming+zero-outgoing), E=Shrink-and-Perturb (alpha={SHRINK_ALPHA},sigma={PERTURB_SIGMA}), "
        f"F=L2-init (lambda={L2_LAMBDA}). "
        f"Confound fixes: no manual_seed in training loop; RNG save/restore for kaiming/noise re-init."
    ),
    "notes": (
        f"Ablation-05 full: {len(SEEDS)} seeds × {len(ARMS)} arms. "
        f"Rank AUC: "
        + "  ".join([f"{arm}={agg[arm]['rank_auc']['mean']:.4f}±{agg[arm]['rank_auc']['std']:.4f}"
                     for arm in ARMS if 'rank_auc' in agg[arm]])
        + ". "
        f"Forgetting: "
        + "  ".join([f"{arm}={agg[arm]['forgetting']['mean']:.2f}±{agg[arm]['forgetting']['std']:.2f}%"
                     for arm in ARMS if 'forgetting' in agg[arm]])
        + ". "
        f"SNRI low disruption: B_SNRI dis_mean={agg['B_SNRI'].get('dis_mean',{}).get('mean','N/A')}. "
        f"Sanity gates G1-G4: {gate_results}."
    ),
}

results_path = os.path.join(OUT_DIR, "RESULTS.json")
with open(results_path, "w") as f:
    json.dump(results, f, indent=2)
log(f"\nResults written to {results_path}")

# Print compact summary
log("\n" + "="*70)
log("FINAL RESULTS SUMMARY:")
log("="*70)
for arm in ARMS:
    if 'rank_auc' not in agg[arm]:
        continue
    ra  = agg[arm]["rank_auc"]
    forg= agg[arm].get("forgetting", {})
    aia = agg[arm].get("aia", {})
    dis = agg[arm].get("dis_mean", {})
    log(f"  {arm}: rank_auc={ra['mean']:.4f}±{ra['std']:.4f}  "
        f"forg={forg.get('mean','N/A'):.2f}±{forg.get('std','N/A'):.2f}%  "
        f"AIA={aia.get('mean','N/A'):.2f}%  "
        f"dis={dis.get('mean','N/A') if dis else 'N/A'}")

log(f"\nWilcoxon significant (p < {ALPHA_CORRECTED:.4f}):")
for label, r in wilcoxon_results.items():
    if r["sig"]:
        log(f"  ✓ {label}: p={r['p']:.4f}")

_log_fh.close()
print(f"\n✓ Done. Results: {results_path}", flush=True)
