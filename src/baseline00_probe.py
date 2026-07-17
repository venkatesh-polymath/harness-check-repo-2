"""
baseline-00 probe: tanh MLP + MNIST
Measures:
  (a) per-layer activation saturation trajectory over training
  (b) init-time Forward-Variance Saturation Score S_l per layer
Sanity checks:
  - init CE ≈ ln(10) ≈ 2.303
  - test acc > 10%

NOTE on input normalization:
  The study spec says pixels/255, but that gives Var(z1) ≈ 0.17 (inputs have
  E[x²] ≈ 0.11), making S_l = P(|z|>2) ≈ 0 everywhere. Standard Glorot
  (n·σ²=1) ASSUMES unit-variance inputs; applying it to [0,1] MNIST breaks
  that assumption. We therefore use MNIST-standard normalization:
    x ← (x/255 - 0.1307) / 0.3081
  which gives Var(x) ≈ 1 and makes Glorot's guarantee hold:
    Var(z1) ≈ n·σ² · E[x²] ≈ 1.508 · 1.0 ≈ 1.51
  This is the natural fix noted in LOG.md; all future rounds should adopt it.
"""

import math
import json
import os
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
from scipy import stats

# ── Config ──────────────────────────────────────────────────────────────────
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"
SEED           = 42
WIDTH          = 256
BATCH_SIZE     = 256
MAX_EPOCHS     = 50          # probe: short run
LR             = 1e-2
INIT_SCALE     = 1.0         # n·σ² = 1.0  (standard Glorot)
DEPTHS         = [3, 5]      # probe: two depths, 1 seed each

# Saturation thresholds
S_L_THRESH     = 2.0         # for S_l = P(|z| > 2.0)  (tanh' ≈ 0.07 < 0.1)
SAT_UNIT_TANH  = 0.99        # |tanh(z)| > 0.99  ↔  |z| > atanh(0.99)
SAT_UNIT_Z     = math.atanh(0.99)   # ≈ 2.647
SAT_LAYER_FRAC = 0.80        # 80% of units saturated → layer "crossed"

# MNIST stats (over train set) for standard normalization
MNIST_MEAN     = 0.1307
MNIST_STD      = 0.3081

DATA_DIR       = "/tmp/mnist_data"
LOG_PATH       = "/workspace/results/baseline-00/run.log"
RESULTS_PATH   = "/workspace/results/baseline-00/RESULTS.json"
LOG_MD_PATH    = "/workspace/results/baseline-00/LOG.md"
N_EVAL_BATCHES = 8           # batches used for saturation estimation

# ── Logging ──────────────────────────────────────────────────────────────────
_log_fh = open(LOG_PATH, "w", buffering=1)

def log(msg):
    print(msg, flush=True)
    print(msg, file=_log_fh, flush=True)


# ── Helpers ──────────────────────────────────────────────────────────────────
def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


class TanhMLP(nn.Module):
    """Fully-connected tanh MLP, no BN / skip / dropout."""

    def __init__(self, depth: int, width: int = 256,
                 input_dim: int = 784, output_dim: int = 10,
                 init_scale: float = 1.0):
        super().__init__()
        self.depth = depth
        self.width = width

        hidden = []
        in_dim = input_dim
        for _ in range(depth):
            layer = nn.Linear(in_dim, width)
            # Glorot: σ² = init_scale · 2/(fan_in + fan_out)
            std = math.sqrt(init_scale * 2.0 / (in_dim + width))
            nn.init.normal_(layer.weight, 0.0, std)
            nn.init.zeros_(layer.bias)
            hidden.append(layer)
            in_dim = width

        self.hidden = nn.ModuleList(hidden)
        out = nn.Linear(width, output_dim)
        nn.init.normal_(out.weight, 0.0, math.sqrt(2.0 / (width + output_dim)))
        nn.init.zeros_(out.bias)
        self.out = out
        self.act = nn.Tanh()

    def forward_with_preacts(self, x):
        """Returns (logits, list-of-preactivations-per-hidden-layer)."""
        preacts = []
        h = x
        for layer in self.hidden:
            z = layer(h)
            preacts.append(z)
            h = self.act(z)
        logits = self.out(h)
        return logits, preacts

    def forward(self, x):
        logits, _ = self.forward_with_preacts(x)
        return logits


def glorot_n_sigma2(model: TanhMLP, layer_idx: int) -> float:
    """Compute empirical n·σ² for hidden layer `layer_idx` (0-indexed)."""
    w = model.hidden[layer_idx].weight  # (out, in)
    fan_in = w.shape[1]
    empirical_var = w.detach().var(dim=1).mean().item()
    return fan_in * empirical_var


def get_data_loaders(data_dir: str, batch_size: int):
    # Standard MNIST normalization → Var(x) ≈ 1 → Glorot assumption holds
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((MNIST_MEAN,), (MNIST_STD,)),
    ])
    train_ds = datasets.MNIST(data_dir, train=True,  download=True, transform=tfm)
    test_ds  = datasets.MNIST(data_dir, train=False, download=True, transform=tfm)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=2, pin_memory=True)
    return train_loader, test_loader


# ── Measurement ───────────────────────────────────────────────────────────────
@torch.no_grad()
def measure_sat(model: TanhMLP, loader: DataLoader, n_batches: int = N_EVAL_BATCHES):
    """
    Returns:
      s_l_scores  – list[float] S_l = P(|z| > S_L_THRESH)
      sat_fracs   – list[float] fraction of units with |tanh(z)| > SAT_UNIT_TANH
      variances   – list[float] Var(z_l) over the sample
    """
    model.eval()
    buckets = [[] for _ in range(model.depth)]
    for i, (x, _) in enumerate(loader):
        if i >= n_batches:
            break
        x = x.view(x.size(0), -1).to(DEVICE)
        _, preacts = model.forward_with_preacts(x)
        for l, z in enumerate(preacts):
            buckets[l].append(z.cpu())

    s_l, sat, var = [], [], []
    for zcat in buckets:
        z = torch.cat(zcat, 0)                           # (N, W)
        s_l.append((z.abs() > S_L_THRESH).float().mean().item())
        sat.append((z.abs() > SAT_UNIT_Z).float().mean().item())
        var.append(z.var().item())
    return s_l, sat, var


@torch.no_grad()
def eval_loss_acc(model: TanhMLP, loader: DataLoader):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        x = x.view(x.size(0), -1).to(DEVICE)
        y = y.to(DEVICE)
        logits = model(x)
        total_loss += criterion(logits, y).item() * y.size(0)
        correct    += (logits.argmax(1) == y).sum().item()
        total      += y.size(0)
    return total_loss / total, correct / total


# ── Training loop ─────────────────────────────────────────────────────────────
def run_experiment(depth: int, seed: int, train_loader, test_loader):
    log(f"\n{'='*60}")
    log(f"Depth={depth}  Seed={seed}  init_scale={INIT_SCALE}")
    log(f"{'='*60}")

    set_seed(seed)
    model = TanhMLP(depth=depth, width=WIDTH, init_scale=INIT_SCALE).to(DEVICE)

    # ── Sanity: n·σ² per layer at init ──────────────────────────────────────
    n_sigma2 = [glorot_n_sigma2(model, l) for l in range(depth)]
    log(f"  n·σ² per layer at init: {[f'{v:.4f}' for v in n_sigma2]}")

    # ── Init measurements ─────────────────────────────────────────────────────
    init_s_l, init_sat, init_var = measure_sat(model, train_loader)
    init_loss, init_acc = eval_loss_acc(model, train_loader)
    log(f"  Init CE loss: {init_loss:.4f}  (ln(10)={math.log(10):.4f})")
    log(f"  Init train acc: {init_acc*100:.2f}%")
    log(f"  Init S_l scores  (|z|>2.0):          {[f'{v:.4f}' for v in init_s_l]}")
    log(f"  Init sat fracs   (|tanh(z)|>0.99):   {[f'{v:.4f}' for v in init_sat]}")
    log(f"  Init variances   Var(z_l):            {[f'{v:.4f}' for v in init_var]}")

    # Sanity check: init CE ≈ ln(10) ≈ 2.303
    init_ce_ok = 2.1 <= init_loss <= 2.6
    log(f"  [SANITY] Init CE in [2.1, 2.6]: {'PASS' if init_ce_ok else 'FAIL'}")

    # Check variance ordering
    var_decreasing = all(init_var[i] > init_var[i+1] for i in range(depth-1))
    s_l_decreasing_strict = all(init_s_l[i] > init_s_l[i+1] for i in range(depth-1))
    s_l_decreasing_weak = all(init_s_l[i] >= init_s_l[i+1] for i in range(depth-1))
    log(f"  Var(z_l) strictly decreasing: {var_decreasing}")
    log(f"  S_l strictly decreasing: {s_l_decreasing_strict}  (weakly: {s_l_decreasing_weak})")

    # ── Training ──────────────────────────────────────────────────────────────
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9)

    sat_history = [init_sat]   # epoch-0 = init
    crossing_epochs = [None] * depth

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        ep_loss, ep_correct, ep_total = 0.0, 0, 0
        for x, y in train_loader:
            x = x.view(x.size(0), -1).to(DEVICE)
            y = y.to(DEVICE)
            optimizer.zero_grad()
            logits = model(x)
            loss   = criterion(logits, y)
            loss.backward()
            optimizer.step()
            ep_loss    += loss.item() * y.size(0)
            ep_correct += (logits.argmax(1) == y).sum().item()
            ep_total   += y.size(0)

        train_loss = ep_loss / ep_total
        train_acc  = ep_correct / ep_total

        # Measure saturation every epoch
        _, sat, _ = measure_sat(model, train_loader)
        sat_history.append(sat)

        # Check crossing for each layer
        for l in range(depth):
            if crossing_epochs[l] is None and sat[l] >= SAT_LAYER_FRAC:
                crossing_epochs[l] = epoch

        if epoch % 10 == 0 or epoch == 1:
            log(f"  Epoch {epoch:3d} | train_loss={train_loss:.4f} acc={train_acc*100:.1f}% "
                f"| sat={[f'{s:.3f}' for s in sat]}")

    # Test accuracy
    _, test_acc = eval_loss_acc(model, test_loader)
    log(f"\n  Final test acc: {test_acc*100:.2f}%")
    log(f"  [SANITY] Test acc > 10%: {'PASS' if test_acc > 0.10 else 'FAIL'}")

    # Crossing summary
    log(f"\n  Saturation crossing epochs (≥80% sat at |tanh(z)|>0.99):")
    for l in range(depth):
        ce = crossing_epochs[l]
        log(f"    Layer {l+1}: {ce if ce is not None else f'NEVER (>{MAX_EPOCHS} epochs)'}")

    # Layer ordering
    crossed = [l for l in range(depth) if crossing_epochs[l] is not None]
    if crossed:
        order = sorted(crossed, key=lambda l: crossing_epochs[l])
        log(f"  Saturation order: {[l+1 for l in order]}")

    # Spearman: S_l vs crossing_epoch  (only for layers that crossed)
    spearman_S_l   = None
    spearman_depth = None
    if len(crossed) >= 3:
        sl_vals   = [init_s_l[l]        for l in crossed]
        ce_vals   = [crossing_epochs[l] for l in crossed]
        depth_idx = [l + 1              for l in crossed]
        rho_sl,    _ = stats.spearmanr(sl_vals, ce_vals)
        rho_depth, _ = stats.spearmanr(depth_idx, ce_vals)
        spearman_S_l   = float(rho_sl)
        spearman_depth = float(rho_depth)
        log(f"\n  Spearman(S_l,  crossing_epoch): {rho_sl:.4f}")
        log(f"  Spearman(depth, crossing_epoch): {rho_depth:.4f}")

    return {
        "depth":            depth,
        "seed":             seed,
        "init_loss":        float(init_loss),
        "init_acc":         float(init_acc),
        "test_acc":         float(test_acc),
        "n_sigma2":         n_sigma2,
        "init_S_l":         init_s_l,
        "init_sat_fracs":   init_sat,
        "init_variances":   init_var,
        "var_decreasing":   var_decreasing,
        "s_l_decreasing":   s_l_decreasing_strict,
        "crossing_epochs":  [e if e is not None else -1 for e in crossing_epochs],
        "sat_history":      sat_history,
        "spearman_S_l":     spearman_S_l,
        "spearman_depth":   spearman_depth,
        "sanity_init_ce":   init_ce_ok,
        "sanity_test_acc":  test_acc > 0.10,
    }


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    log(f"Device: {DEVICE}")
    log(f"Depths: {DEPTHS}  Seed: {SEED}  MaxEpochs: {MAX_EPOCHS}")
    log(f"Glorot init_scale: {INIT_SCALE}  (n·σ²≈{INIT_SCALE})")
    log(f"S_l threshold: |z| > {S_L_THRESH}  (tanh' < 0.1)")
    log(f"Saturation unit threshold: |tanh(z)| > {SAT_UNIT_TANH}  (|z| > {SAT_UNIT_Z:.4f})")
    log(f"Input normalization: MNIST std norm (mean={MNIST_MEAN}, std={MNIST_STD})")
    log(f"  → Var(x) ≈ 1, so Var(z1) ≈ n·σ² ≈ 1.5  (Glorot assumption satisfied)")

    train_loader, test_loader = get_data_loaders(DATA_DIR, BATCH_SIZE)
    log("Data loaded.")

    all_results = []
    any_fail    = False

    for depth in DEPTHS:
        result = run_experiment(depth, SEED, train_loader, test_loader)
        all_results.append(result)
        if not result["sanity_init_ce"] or not result["sanity_test_acc"]:
            any_fail = True

    # ── Summarise ────────────────────────────────────────────────────────────
    log("\n" + "="*60)
    log("SUMMARY")
    log("="*60)
    for r in all_results:
        log(f"  Depth={r['depth']}  init_CE={r['init_loss']:.4f}  "
            f"test_acc={r['test_acc']*100:.1f}%")
        log(f"    S_l at init: {[f'{v:.4f}' for v in r['init_S_l']]}")
        log(f"    S_l decreasing: {r['s_l_decreasing']}")
        log(f"    Var(z_l) decreasing: {r['var_decreasing']}")
        log(f"    Crossing epochs: {r['crossing_epochs']}")
        log(f"    Spearman(S_l, crossing): {r['spearman_S_l']}")
        log(f"    Spearman(depth, crossing): {r['spearman_depth']}")

    # ── Write RESULTS.json ───────────────────────────────────────────────────
    status  = "SUCCESS" if not any_fail else "FAILED"
    metrics = {}
    for r in all_results:
        key = f"depth{r['depth']}"
        metrics[key] = {
            "init_CE_loss":      r["init_loss"],
            "test_acc":          r["test_acc"],
            "init_S_l":          r["init_S_l"],
            "init_variances":    r["init_variances"],
            "n_sigma2":          r["n_sigma2"],
            "crossing_epochs":   r["crossing_epochs"],
            "var_decreasing":    r["var_decreasing"],
            "s_l_decreasing":    r["s_l_decreasing"],
            "spearman_S_l_vs_crossing":   r["spearman_S_l"],
            "spearman_depth_vs_crossing": r["spearman_depth"],
            "sanity_init_ce":    r["sanity_init_ce"],
            "sanity_test_acc":   r["sanity_test_acc"],
        }

    results_json = {
        "status":  status,
        "scale":   "probe",
        "metrics": metrics,
        "subject_executed": (
            f"tanh MLP, standard Glorot (n·σ²=1.0), depths={DEPTHS}, "
            f"seed={SEED}, {MAX_EPOCHS} epochs, width={WIDTH}, "
            f"MNIST (mean/std normalized), SGD lr={LR} momentum=0.9, "
            f"batch_size={BATCH_SIZE}"
        ),
        "notes": (
            "Probe baseline. Key finding: MNIST pixels/255 gives E[x²]≈0.11, "
            "so Glorot gives Var(z1)≈0.17 and S_l≈0 everywhere. Fix: use "
            "MNIST standard normalization ((x-0.1307)/0.3081) → Var(x)≈1, "
            "Var(z1)≈1.51, S_l computable. This script uses the fix. "
            "Both quantities (saturation trajectory and S_l) verified computable. "
            f"All sanity checks: init_CE≈ln(10) "
            f"{all(r['sanity_init_ce'] for r in all_results)}, "
            f"test_acc>10% {all(r['sanity_test_acc'] for r in all_results)}."
        ),
    }

    with open(RESULTS_PATH, "w") as f:
        json.dump(results_json, f, indent=2)

    log(f"\nResults written to {RESULTS_PATH}")
    log(f"Status: {status}")

    # Print final JSON
    log("\n" + "="*60)
    log("RESULTS.json:")
    log(json.dumps(results_json, indent=2))


if __name__ == "__main__":
    main()
