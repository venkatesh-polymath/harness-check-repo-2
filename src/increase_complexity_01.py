"""
increase_complexity_01.py  (v2 – preloaded GPU data for speed)
==============================================================
Build on baseline-00 (Glorot, n·σ²≈1, no crossings in 50 epochs).
Test two SATURATING init regimes across depths {3,5,7} × seeds {0,1,2}.

  STANDARD:  uniform[-1/sqrt(fan_in), 1/sqrt(fan_in)]  n·σ² = 1/3 (sub-unit)
  SUPER_2x:  Normal(0, sqrt(2/fan_in))                 n·σ² = 2
  SUPER_4x:  Normal(0, sqrt(4/fan_in))                 n·σ² = 4

Key optimisation: ALL MNIST data preloaded into GPU tensors at startup.
No DataLoader overhead → expect ~10× speedup vs DataLoader approach.

Prediction (registered before run):
  - Under STANDARD: S_l decreases with depth; shallow layers saturate first (if ever)
    → Spearman(S_l, crossing_epoch) < 0 (high S_l at layer 1 = early saturation)
  - Under SUPER-STANDARD: gradient vanishing keeps shallow layers frozen; deep
    layers get stronger gradient signal and saturate first
    → Spearman(S_l, crossing_epoch) > 0 (high S_l at layer 1 = LATE saturation)
  Sign reversal between standard and super-standard is the primary test.
"""

import math, json, os, time
import numpy as np
from scipy import stats
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

# ─────────────────────────────── Config ──────────────────────────────────────
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
WIDTH         = 256
BATCH_SIZE    = 256
MAX_EPOCHS    = 30
SEEDS         = [0, 1, 2]
DEPTHS        = [3, 5, 7]

S_L_THRESH    = 2.0
SAT_UNIT_Z    = math.atanh(0.99)   # ≈ 2.647
SAT_LAYER_FRAC= 0.80
CENSOR_EPOCH  = MAX_EPOCHS + 1

MNIST_MEAN    = 0.1307
MNIST_STD     = 0.3081
DATA_DIR      = "/tmp/mnist_data"

RESULTS_DIR   = "/workspace/results/increase_complexity-01"
LOG_PATH      = f"{RESULTS_DIR}/run.log"
RESULTS_PATH  = f"{RESULTS_DIR}/RESULTS.json"

EVAL_N        = 2048  # samples for saturation measurement (fast, GPU-resident)
BOOTSTRAP_N   = 5000

REGIMES = {
    "standard": {"init": "uniform_fan_in", "scale": 1.0/3, "lr": 1e-2},
    "super_2x": {"init": "normal_fan_in",  "scale": 2.0,   "lr": 1e-2},
    "super_4x": {"init": "normal_fan_in",  "scale": 4.0,   "lr": 1e-2},
}

# ─────────────────────────────── Logging ─────────────────────────────────────
os.makedirs(RESULTS_DIR, exist_ok=True)
_log_fh = open(LOG_PATH, "w", buffering=1)
def log(msg: str):
    print(msg, flush=True)
    print(msg, file=_log_fh, flush=True)


# ─────────────────────────────── Data (preloaded) ────────────────────────────
def load_mnist_to_gpu(data_dir: str):
    """Load entire MNIST train+test into GPU tensors (one-time cost)."""
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((MNIST_MEAN,), (MNIST_STD,)),
    ])
    train_ds = datasets.MNIST(data_dir, train=True,  download=True, transform=tfm)
    test_ds  = datasets.MNIST(data_dir, train=False, download=True, transform=tfm)

    def ds_to_gpu(ds):
        from torch.utils.data import DataLoader
        loader = DataLoader(ds, batch_size=len(ds), shuffle=False, num_workers=0)
        X, Y = next(iter(loader))
        X = X.view(X.size(0), -1).to(DEVICE, dtype=torch.float32)
        Y = Y.to(DEVICE)
        return X, Y

    log("Loading MNIST to GPU…")
    X_train, Y_train = ds_to_gpu(train_ds)
    X_test,  Y_test  = ds_to_gpu(test_ds)
    log(f"  Train: {X_train.shape}  Test: {X_test.shape}")
    return X_train, Y_train, X_test, Y_test


def make_batches(X: torch.Tensor, Y: torch.Tensor,
                 batch_size: int, shuffle: bool = True):
    """Yield (x_batch, y_batch) without any DataLoader overhead."""
    N = X.size(0)
    idx = torch.randperm(N, device=DEVICE) if shuffle else torch.arange(N, device=DEVICE)
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        sl = idx[start:end]
        yield X[sl], Y[sl]


# ─────────────────────────────── Model ───────────────────────────────────────
class TanhMLP(nn.Module):
    def __init__(self, depth: int, width: int = WIDTH, input_dim: int = 784,
                 output_dim: int = 10):
        super().__init__()
        self.depth = depth
        self.width = width
        hidden = []
        in_dim = input_dim
        for _ in range(depth):
            hidden.append(nn.Linear(in_dim, width))
            in_dim = width
        self.hidden = nn.ModuleList(hidden)
        self.out = nn.Linear(width, output_dim)
        self.act = nn.Tanh()

    def forward_with_preacts(self, x):
        preacts, h = [], x
        for layer in self.hidden:
            z = layer(h); preacts.append(z); h = self.act(z)
        return self.out(h), preacts

    def forward(self, x):
        return self.forward_with_preacts(x)[0]


def apply_init(model: TanhMLP, regime: str):
    cfg = REGIMES[regime]
    for layer in model.hidden:
        fan_in = layer.weight.shape[1]
        if cfg["init"] == "uniform_fan_in":
            bound = 1.0 / math.sqrt(fan_in)
            nn.init.uniform_(layer.weight, -bound, bound)
        else:  # normal_fan_in
            nn.init.normal_(layer.weight, 0.0, math.sqrt(cfg["scale"] / fan_in))
        nn.init.zeros_(layer.bias)
    nn.init.xavier_normal_(model.out.weight)
    nn.init.zeros_(model.out.bias)


def empirical_n_sigma2(layer: nn.Linear) -> float:
    w = layer.weight.detach()
    return (w.shape[1] * w.var(dim=1).mean()).item()


# ─────────────────────────────── Measurement ─────────────────────────────────
@torch.no_grad()
def measure_sat(model: TanhMLP, X_eval: torch.Tensor):
    """Fast: uses fixed preloaded GPU tensor."""
    model.eval()
    _, preacts = model.forward_with_preacts(X_eval)
    s_l, sat, var = [], [], []
    for z in preacts:
        s_l.append((z.abs() > S_L_THRESH).float().mean().item())
        sat.append((z.abs() > SAT_UNIT_Z).float().mean().item())
        var.append(z.var().item())
    return s_l, sat, var


@torch.no_grad()
def eval_loss_acc(model: TanhMLP, X: torch.Tensor, Y: torch.Tensor,
                  batch_size: int = 1024):
    model.eval()
    crit = nn.CrossEntropyLoss()
    total_loss, correct, total = 0.0, 0, 0
    for start in range(0, X.size(0), batch_size):
        xb = X[start:start+batch_size]
        yb = Y[start:start+batch_size]
        logits = model(xb)
        total_loss += crit(logits, yb).item() * yb.size(0)
        correct    += (logits.argmax(1) == yb).sum().item()
        total      += yb.size(0)
    return total_loss / total, correct / total


# ─────────────────────────────── Bootstrap CI ────────────────────────────────
def bootstrap_ci(values, n_boot=BOOTSTRAP_N, ci=0.95):
    arr = np.array([v for v in values if v is not None and not math.isnan(v)])
    if len(arr) == 0:
        return None, None, None
    boots = [float(np.mean(np.random.choice(arr, len(arr), replace=True)))
             for _ in range(n_boot)]
    lo = float(np.percentile(boots, 100*(1-ci)/2))
    hi = float(np.percentile(boots, 100*(1-(1-ci)/2)))
    return float(arr.mean()), lo, hi


# ─────────────────────────────── Single run ──────────────────────────────────
def run_single(depth: int, seed: int, regime: str,
               X_train, Y_train, X_test, Y_test) -> dict:
    log(f"\n{'─'*60}")
    log(f"Regime={regime}  Depth={depth}  Seed={seed}")
    log(f"{'─'*60}")

    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed); np.random.seed(seed)

    model = TanhMLP(depth=depth).to(DEVICE)
    apply_init(model, regime)

    n_sigma2 = [round(empirical_n_sigma2(l), 4) for l in model.hidden]
    log(f"  n·σ² per layer: {n_sigma2}")

    # Fixed eval subset (same indices every call → deterministic)
    torch.manual_seed(9999)
    eval_idx = torch.randperm(X_train.size(0), device=DEVICE)[:EVAL_N]
    X_eval = X_train[eval_idx]

    # Reset seed for training
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed); np.random.seed(seed)

    init_s_l, init_sat, init_var = measure_sat(model, X_eval)
    init_loss, _ = eval_loss_acc(model, X_train, Y_train)

    log(f"  Init CE: {init_loss:.4f}  (ln10={math.log(10):.4f})")
    log(f"  Init S_l    : {[f'{v:.4f}' for v in init_s_l]}")
    log(f"  Init sat_frc: {[f'{v:.4f}' for v in init_sat]}")
    log(f"  Init Var(z) : {[f'{v:.4f}' for v in init_var]}")

    s_l_decreasing = all(init_s_l[i] >= init_s_l[i+1] for i in range(depth-1))
    s_l_increasing = all(init_s_l[i] <= init_s_l[i+1] for i in range(depth-1))
    log(f"  S_l weakly decreasing: {s_l_decreasing}  increasing: {s_l_increasing}")

    # Training
    lr        = REGIMES[regime]["lr"]
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    sat_history    = [list(init_sat)]
    crossing_epoch = [None] * depth
    diverged       = False

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        ep_loss, ep_n = 0.0, 0
        for xb, yb in make_batches(X_train, Y_train, BATCH_SIZE, shuffle=True):
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            ep_loss += loss.item() * yb.size(0)
            ep_n    += yb.size(0)
        train_loss = ep_loss / ep_n

        if not math.isfinite(train_loss) or train_loss > 1e4:
            log(f"  [DIVERGED] Epoch {epoch}: loss={train_loss:.4f}")
            diverged = True; break

        _, sat, _ = measure_sat(model, X_eval)
        sat_history.append(list(sat))
        for l in range(depth):
            if crossing_epoch[l] is None and sat[l] >= SAT_LAYER_FRAC:
                crossing_epoch[l] = epoch

        if epoch % 5 == 0 or epoch == 1:
            log(f"  Epoch {epoch:3d} | loss={train_loss:.4f} | "
                f"sat={[f'{s:.3f}' for s in sat]}")

    _, test_acc = eval_loss_acc(model, X_test, Y_test)
    log(f"  Final test acc: {test_acc*100:.2f}%")

    ce_values = [crossing_epoch[l] if crossing_epoch[l] is not None else CENSOR_EPOCH
                 for l in range(depth)]
    n_censored = sum(1 for c in crossing_epoch if c is None)
    n_crossed  = depth - n_censored

    crossed_layers = [l for l in range(depth) if crossing_epoch[l] is not None]
    if crossed_layers:
        order = sorted(crossed_layers, key=lambda l: crossing_epoch[l])
        log(f"  Sat order (first→last): {[l+1 for l in order]}")
        first_to_cross = order[0] + 1
    else:
        log("  No layers crossed 80% within 30 epochs.")
        first_to_cross = None

    for l in range(depth):
        status = f"epoch {ce_values[l]}" if crossing_epoch[l] is not None else f"CENSORED(>{MAX_EPOCHS})"
        log(f"  Layer {l+1}: {status}")

    # Spearman
    spearman_sl = spearman_depth = None
    sl_vals  = list(init_s_l)
    di_vals  = list(range(1, depth+1))

    if n_crossed >= 3:
        sl_for_sp = [init_s_l[l] for l in crossed_layers]
        ce_for_sp = [crossing_epoch[l] for l in crossed_layers]
        di_for_sp = [l+1 for l in crossed_layers]
        if len(set(sl_for_sp)) > 1:
            rho, p = stats.spearmanr(sl_for_sp, ce_for_sp)
            spearman_sl = float(rho)
            log(f"  Spearman(S_l, crossing)[n_crossed={n_crossed}]: {rho:.4f} p={p:.4f}")
        rho2, p2 = stats.spearmanr(di_for_sp, ce_for_sp)
        spearman_depth = float(rho2)
        log(f"  Spearman(depth, crossing): {rho2:.4f} p={p2:.4f}")
    else:
        # Use censored values for all layers
        if len(set(ce_values)) > 1 and len(set(sl_vals)) > 1:
            rho, _   = stats.spearmanr(sl_vals, ce_values)
            rho2, _  = stats.spearmanr(di_vals, ce_values)
            spearman_sl    = float(rho)
            spearman_depth = float(rho2)
            log(f"  Spearman(S_l, ce_censored)[{n_crossed} crossed]: {rho:.4f}")
            log(f"  Spearman(depth, ce_censored): {rho2:.4f}")
        else:
            log(f"  Spearman undefined ({n_crossed} crossed, all ce equal).")

    return {
        "regime": regime, "depth": depth, "seed": seed,
        "init_loss": float(init_loss), "test_acc": float(test_acc),
        "n_sigma2": n_sigma2,
        "init_S_l": [round(v,6) for v in init_s_l],
        "init_sat_fracs": [round(v,6) for v in init_sat],
        "init_variances": [round(v,6) for v in init_var],
        "s_l_decreasing": s_l_decreasing,
        "s_l_increasing": s_l_increasing,
        "crossing_epochs": crossing_epoch,
        "crossing_epochs_censored": ce_values,
        "n_censored": n_censored, "n_crossed": n_crossed,
        "first_to_cross": first_to_cross,
        "spearman_sl": spearman_sl,
        "spearman_depth": spearman_depth,
        "sanity_ce": 2.0 <= init_loss <= 3.0,
        "diverged": diverged,
        "sat_frac_ep0":   list(sat_history[0])  if len(sat_history) > 0  else None,
        "sat_frac_ep10":  list(sat_history[10]) if len(sat_history) > 10 else None,
        "sat_frac_ep20":  list(sat_history[20]) if len(sat_history) > 20 else None,
        "sat_frac_final": list(sat_history[-1]) if sat_history else None,
    }


# ─────────────────────────────── Main ────────────────────────────────────────
def main():
    t0 = time.time()
    log(f"Device: {DEVICE}")
    log(f"Regimes: {list(REGIMES.keys())}")
    log(f"Depths: {DEPTHS}  Seeds: {SEEDS}  MaxEpochs: {MAX_EPOCHS}")
    log(f"S_l threshold: |z|>{S_L_THRESH}  Saturation: |tanh(z)|>0.99  Layer: {SAT_LAYER_FRAC*100:.0f}%")
    log(f"EVAL_N={EVAL_N} (preloaded GPU tensor, fixed per run)")

    X_train, Y_train, X_test, Y_test = load_mnist_to_gpu(DATA_DIR)
    log(f"Data loaded to GPU in {time.time()-t0:.1f}s")

    all_results = []
    for regime in REGIMES:
        for depth in DEPTHS:
            for seed in SEEDS:
                r = run_single(depth, seed, regime,
                               X_train, Y_train, X_test, Y_test)
                all_results.append(r)
                elapsed = time.time() - t0
                done = len(all_results)
                total = len(REGIMES) * len(DEPTHS) * len(SEEDS)
                eta = elapsed / done * (total - done)
                log(f"  → {done}/{total} done | elapsed={elapsed:.0f}s | ETA={eta:.0f}s")

    # ── Aggregate per (regime, depth) ────────────────────────────────────────
    log("\n" + "="*60)
    log("AGGREGATED RESULTS")
    log("="*60)
    agg = {}
    np.random.seed(42)

    for regime in REGIMES:
        for depth in DEPTHS:
            key = f"{regime}_d{depth}"
            runs = [r for r in all_results
                    if r["regime"] == regime and r["depth"] == depth]
            sp_sl_list    = [r["spearman_sl"]    for r in runs]
            sp_depth_list = [r["spearman_depth"]  for r in runs]
            first_list    = [r["first_to_cross"]  for r in runs]
            cens_list     = [r["n_censored"]      for r in runs]

            mean_sl, lo_sl, hi_sl = bootstrap_ci(sp_sl_list)
            mean_dp, lo_dp, hi_dp = bootstrap_ci(sp_depth_list)

            log(f"\n{key}:")
            log(f"  Spearman(S_l, crossing)    = {mean_sl} 95%CI [{lo_sl},{hi_sl}]")
            log(f"  Spearman(depth, crossing)  = {mean_dp} 95%CI [{lo_dp},{hi_dp}]")
            log(f"  first_to_cross (layer#)    = {first_list}")
            log(f"  n_censored per seed        = {cens_list}")
            all_sl = np.array([r["init_S_l"] for r in runs])
            log(f"  Mean init S_l              = {[f'{v:.4f}' for v in all_sl.mean(0)]}")

            agg[key] = {
                "regime": regime, "depth": depth, "n_seeds": len(runs),
                "spearman_sl_mean":       float(mean_sl)    if mean_sl    is not None else None,
                "spearman_sl_ci95":       [float(lo_sl), float(hi_sl)] if lo_sl is not None else None,
                "spearman_depth_mean":    float(mean_dp)    if mean_dp    is not None else None,
                "spearman_depth_ci95":    [float(lo_dp), float(hi_dp)] if lo_dp is not None else None,
                "first_to_cross":         first_list,
                "n_censored_per_seed":    cens_list,
                "mean_init_S_l":          [float(v) for v in all_sl.mean(0)],
                "mean_init_sat_frac":     [float(v) for v in
                    np.array([r["init_sat_fracs"] for r in runs]).mean(0)],
                "mean_init_var":          [float(v) for v in
                    np.array([r["init_variances"] for r in runs]).mean(0)],
                "mean_crossing_epochs":   [float(np.mean([r["crossing_epochs_censored"][l]
                                            for r in runs])) for l in range(depth)],
                "spearman_sl_per_seed":   sp_sl_list,
                "spearman_depth_per_seed":sp_depth_list,
            }

    # ── Order-reversal check ──────────────────────────────────────────────────
    log("\n" + "="*60)
    log("ORDER-REVERSAL SUMMARY")
    log("="*60)
    reversal_found = {}
    for depth in DEPTHS:
        log(f"\nDepth={depth}:")
        for regime in REGIMES:
            key = f"{regime}_d{depth}"
            fl = agg[key]["first_to_cross"]
            valid = [f for f in fl if f is not None]
            l1_first  = sum(1 for f in valid if f == 1)
            lL_first  = sum(1 for f in valid if f == depth)
            log(f"  {regime:12s}: first={fl}  layer1_first={l1_first}/{len(valid)}  "
                f"layerL_first={lL_first}/{len(valid)}")

        # Check sign reversal: std Spearman(S_l) < 0, super_4x Spearman(S_l) > 0
        std_sp  = agg.get(f"standard_d{depth}", {}).get("spearman_sl_mean")
        s4x_sp  = agg.get(f"super_4x_d{depth}", {}).get("spearman_sl_mean")
        sign_rev = (std_sp is not None and s4x_sp is not None and
                    std_sp * s4x_sp < 0)
        reversal_found[f"d{depth}"] = sign_rev
        log(f"  Sign reversal (standard<0 & super_4x>0): {sign_rev} "
            f"(std_sp={std_sp}, s4x_sp={s4x_sp})")

    # ── Write RESULTS.json ────────────────────────────────────────────────────
    elapsed = time.time() - t0
    any_crossed  = any(r["n_crossed"] > 0 for r in all_results)
    any_diverged = any(r["diverged"]      for r in all_results)

    results_json = {
        "status": "SUCCESS" if any_crossed else "PARTIAL",
        "scale": "probe",
        "metrics": {
            "per_config": agg,
            "order_reversal_by_depth": reversal_found,
            "any_crossings": any_crossed,
            "any_diverged": any_diverged,
            "elapsed_seconds": round(elapsed, 1),
        },
        "subject_executed": (
            f"tanh MLP, MNIST std-norm, depths={DEPTHS}, seeds={SEEDS}, "
            f"regimes=[standard(n·σ²=1/3), super_2x(n·σ²=2), super_4x(n·σ²=4)], "
            f"lr=1e-2, momentum=0.9, batch={BATCH_SIZE}, max_epochs={MAX_EPOCHS}, "
            f"device={DEVICE}, preloaded GPU tensors"
        ),
        "notes": (
            "Probe. Goal: does super-standard init reverse saturation order "
            "(deep-first vs standard init's shallow-first)? "
            "order_reversal_by_depth: sign(Spearman(S_l,crossing)) flips "
            "from standard to super_4x regime. "
            "Right-censored crossings (no 80% within 30 epochs) assigned "
            f"epoch={CENSOR_EPOCH}. "
            "Baseline-00 (Glorot n·σ²≈1): no crossings in 50 epochs. "
            "Standard init (n·σ²=1/3): expect near-zero S_l everywhere, "
            "likely right-censored. Super_4x (n·σ²=4): expect some crossings, "
            "deep layers first due to gradient flow mechanism."
        ),
        "baseline_comparison": {
            "baseline00_glorot_no_crossings": True,
            "baseline00_layer1_max_sat": 0.196,
            "this_round_new_regimes": "standard(1/3) and super_2x(2) and super_4x(4)",
        },
    }

    with open(RESULTS_PATH, "w") as f:
        json.dump(results_json, f, indent=2)
    log(f"\nResults written → {RESULTS_PATH}")
    log(f"Total elapsed: {elapsed:.1f}s  Status: {results_json['status']}")
    log("\n" + json.dumps(results_json, indent=2))


if __name__ == "__main__":
    main()
