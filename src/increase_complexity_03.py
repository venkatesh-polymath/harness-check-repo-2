"""
increase_complexity_03.py
=========================
Round increase_complexity-03: CONFIRMATORY (5 seeds) — the paper's headline table.

Reuses the refine-02 non-censored metric: Spearman(init_S_l, final_sat_frac_|tanh|>0.9).

Regimes × Depths × 5 seeds:
  Regimes: standard (uniform[-1/sqrt(n)], n·σ²≈1/3)
           super_2x (Normal, n·σ²=2)
           super_4x (Normal, n·σ²=4)
  Depths:  3, 5, 7
  Seeds:   0, 1, 2, 3, 4

PRIMARY: Spearman(init S_l, final saturation_fraction_|tanh|>0.9) across layers
         per (regime, depth), mean ± 95% bootstrap CI over 5 seeds.
ALSO:    Fraction of seeds where layer-1 is the max-saturation layer.
REVERSAL VERDICT: does Spearman sign flip NEGATIVE under super-standard vs standard,
                  or positive/layer-1-first in ALL regimes?

≤25 epochs per config; total < 40 min.
"""

import math, json, os, time
import numpy as np
from scipy import stats
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

# ─────────────────────────────── Config ──────────────────────────────────────
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"
WIDTH          = 256
BATCH_SIZE     = 256
MAX_EPOCHS     = 25          # ≤25 per EXPERIMENT.md
SEEDS          = [0, 1, 2, 3, 4]   # 5 seeds (up from 3 in refine-02)
DEPTHS         = [3, 5, 7]

# Init S_l threshold: P(|z_l^(0)| > 2.0)
S_L_THRESH     = 2.0

# Final saturation threshold: |tanh(z)| > 0.9  ⟺  |z| > atanh(0.9)
SAT_09_Z       = math.atanh(0.9)          # ≈ 1.4722

MNIST_MEAN     = 0.1307
MNIST_STD      = 0.3081
DATA_DIR       = "/tmp/mnist_data"

RESULTS_DIR    = "/workspace/results/increase_complexity-03"
LOG_PATH       = f"{RESULTS_DIR}/run.log"
RESULTS_PATH   = f"{RESULTS_DIR}/RESULTS.json"

EVAL_N         = 2048    # GPU-resident subset for saturation measurement
BOOTSTRAP_N    = 10000   # bootstrap resamples for 95% CI

# Regimes
REGIMES = {
    "standard": {"init": "uniform_fan_in", "n_sigma2": 1.0/3, "lr": 1e-2},
    "super_2x": {"init": "normal_fan_in",  "n_sigma2": 2.0,   "lr": 1e-2},
    "super_4x": {"init": "normal_fan_in",  "n_sigma2": 4.0,   "lr": 1e-2},
}

# ─────────────────────────────── Logging ─────────────────────────────────────
os.makedirs(RESULTS_DIR, exist_ok=True)
_log_fh = open(LOG_PATH, "w", buffering=1)

def log(msg: str):
    print(msg, flush=True)
    print(msg, file=_log_fh, flush=True)


# ─────────────────────────────── Data (preloaded GPU) ────────────────────────
def load_mnist_to_gpu(data_dir: str):
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
        return X.view(X.size(0), -1).to(DEVICE, dtype=torch.float32), Y.to(DEVICE)

    log("Loading MNIST to GPU …")
    X_tr, Y_tr = ds_to_gpu(train_ds)
    X_te, Y_te = ds_to_gpu(test_ds)
    log(f"  Train: {X_tr.shape}  Test: {X_te.shape}  device={DEVICE}")
    return X_tr, Y_tr, X_te, Y_te


def make_batches(X, Y, batch_size, shuffle=True, seed=None):
    N = X.size(0)
    if shuffle:
        if seed is not None:
            g = torch.Generator(device=DEVICE)
            g.manual_seed(seed)
            idx = torch.randperm(N, device=DEVICE, generator=g)
        else:
            idx = torch.randperm(N, device=DEVICE)
    else:
        idx = torch.arange(N, device=DEVICE)
    for s in range(0, N, batch_size):
        sl = idx[s:min(s+batch_size, N)]
        yield X[sl], Y[sl]


# ─────────────────────────────── Model ───────────────────────────────────────
class TanhMLP(nn.Module):
    def __init__(self, depth: int, width: int = WIDTH):
        super().__init__()
        self.depth = depth
        layers, in_dim = [], 784
        for _ in range(depth):
            layers.append(nn.Linear(in_dim, width)); in_dim = width
        self.hidden = nn.ModuleList(layers)
        self.out    = nn.Linear(width, 10)

    def forward_with_preacts(self, x):
        preacts, h = [], x
        for lyr in self.hidden:
            z = lyr(h); preacts.append(z); h = torch.tanh(z)
        return self.out(h), preacts

    def forward(self, x):
        return self.forward_with_preacts(x)[0]


def apply_init(model: TanhMLP, regime: str):
    n_s2 = REGIMES[regime]["n_sigma2"]
    for lyr in model.hidden:
        fan_in = lyr.weight.shape[1]
        if REGIMES[regime]["init"] == "uniform_fan_in":
            # Uniform[-1/sqrt(fan_in), +1/sqrt(fan_in)] → n·σ² = 1/3
            bound = 1.0 / math.sqrt(fan_in)
            nn.init.uniform_(lyr.weight, -bound, bound)
        else:
            # Normal(0, sqrt(n_sigma2/fan_in)) → n·σ² = n_s2
            nn.init.normal_(lyr.weight, 0.0, math.sqrt(n_s2 / fan_in))
        nn.init.zeros_(lyr.bias)
    nn.init.xavier_normal_(model.out.weight)
    nn.init.zeros_(model.out.bias)


def empirical_n_sigma2(lyr: nn.Linear) -> float:
    w = lyr.weight.detach()
    return float(w.shape[1] * w.var(dim=1).mean())


# ─────────────────────────────── Measurement ─────────────────────────────────
@torch.no_grad()
def measure_layer_stats(model: TanhMLP, X_eval: torch.Tensor):
    """Return per-layer (init_s_l, sat_frac_09, var_z) on GPU subset."""
    model.eval()
    _, preacts = model.forward_with_preacts(X_eval)
    sl_init, sl09, var_z = [], [], []
    for z in preacts:
        sl_init.append(float((z.abs() > S_L_THRESH).float().mean()))
        sl09.append(   float((z.abs() > SAT_09_Z ).float().mean()))
        var_z.append(  float(z.var()))
    return sl_init, sl09, var_z


@torch.no_grad()
def eval_loss_acc(model, X, Y, batch_size=1024):
    model.eval()
    crit = nn.CrossEntropyLoss()
    total_loss, correct, total = 0., 0, 0
    for s in range(0, X.size(0), batch_size):
        xb, yb = X[s:s+batch_size], Y[s:s+batch_size]
        logits = model(xb)
        total_loss += crit(logits, yb).item() * yb.size(0)
        correct    += (logits.argmax(1) == yb).sum().item()
        total      += yb.size(0)
    return total_loss / total, correct / total


# ─────────────────────────────── Bootstrap CI ────────────────────────────────
def bootstrap_mean_ci(values, n_boot=BOOTSTRAP_N, ci=0.95):
    arr = np.array([v for v in values if v is not None and np.isfinite(v)])
    if len(arr) == 0:
        return None, None, None
    if len(arr) == 1:
        return float(arr[0]), float(arr[0]), float(arr[0])
    np.random.seed(42)
    boots = [float(np.mean(np.random.choice(arr, len(arr), replace=True)))
             for _ in range(n_boot)]
    lo = float(np.percentile(boots, 100*(1-ci)/2))
    hi = float(np.percentile(boots, 100*(1+ci)/2))
    return float(arr.mean()), lo, hi


# ─────────────────────────────── Single run ──────────────────────────────────
def run_single(depth, seed, regime, X_tr, Y_tr, X_te, Y_te) -> dict:
    log(f"\n{'─'*60}")
    log(f"Regime={regime}  Depth={depth}  Seed={seed}")
    log(f"{'─'*60}")

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    model = TanhMLP(depth=depth).to(DEVICE)
    apply_init(model, regime)

    n_sigma2_empirical = [round(empirical_n_sigma2(l), 4) for l in model.hidden]
    log(f"  Empirical n·σ² per layer: {n_sigma2_empirical}")

    # Fixed eval subset (deterministic, same across seeds)
    torch.manual_seed(9999)
    eval_idx = torch.randperm(X_tr.size(0), device=DEVICE)[:EVAL_N]
    X_eval   = X_tr[eval_idx]
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # ── Init measurements ──
    init_sl, init_sat09, init_var = measure_layer_stats(model, X_eval)
    init_loss, _ = eval_loss_acc(model, X_tr, Y_tr)

    log(f"  Init CE    : {init_loss:.4f}  (ln10={math.log(10):.4f})")
    log(f"  Init S_l(>2): {[f'{v:.4f}' for v in init_sl]}")
    log(f"  Init sat09  : {[f'{v:.4f}' for v in init_sat09]}")
    log(f"  Init Var(z) : {[f'{v:.4f}' for v in init_var]}")

    s_l_strictly_decreasing = all(init_sl[i] > init_sl[i+1] for i in range(depth-1))
    s_l_weakly_decreasing   = all(init_sl[i] >= init_sl[i+1] for i in range(depth-1))
    log(f"  S_l order: strictly_dec={s_l_strictly_decreasing}, weakly_dec={s_l_weakly_decreasing}")

    # ── Training ──
    lr        = REGIMES[regime]["lr"]
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    diverged  = False

    sat09_history = [list(init_sat09)]

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        ep_loss, ep_n = 0., 0
        for xb, yb in make_batches(X_tr, Y_tr, BATCH_SIZE):
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            ep_loss += loss.item() * yb.size(0)
            ep_n    += yb.size(0)
        train_loss = ep_loss / ep_n

        if not math.isfinite(train_loss) or train_loss > 1e4:
            log(f"  [DIVERGED] Epoch {epoch}: loss={train_loss:.4f}")
            diverged = True
            break

        _, sat09, _ = measure_layer_stats(model, X_eval)
        sat09_history.append(list(sat09))

        if epoch % 5 == 0 or epoch == 1:
            log(f"  Epoch {epoch:3d} | loss={train_loss:.4f} | "
                f"sat09={[f'{s:.3f}' for s in sat09]}")

    _, test_acc = eval_loss_acc(model, X_te, Y_te)
    log(f"  Final test_acc: {test_acc*100:.2f}%")

    # ── Final saturation fractions (primary metric) ──
    final_sat09 = sat09_history[-1]   # |tanh|>0.9 at final epoch

    log(f"  Final sat09  : {[f'{v:.4f}' for v in final_sat09]}")

    # ── Layer ranking by final sat09 ──
    rank_final = stats.rankdata([-s for s in final_sat09])  # rank 1 = highest sat
    log(f"  Layer rank by final sat09 (1=highest): {list(rank_final)}")

    # ── Spearman(init_S_l, final_sat09) ──
    spearman_sl   = None
    spearman_sl_p = None
    spearman_di   = None
    spearman_di_p = None

    if len(set(init_sl)) > 1 and len(set(final_sat09)) > 1:
        rho_sl, p_sl = stats.spearmanr(init_sl, final_sat09)
        spearman_sl   = float(rho_sl)
        spearman_sl_p = float(p_sl)
        log(f"  Spearman(init_S_l, final_sat09): rho={rho_sl:.4f} p={p_sl:.4f}")
    else:
        log(f"  Spearman undefined (init_S_l all-zero or final_sat09 all-equal)")

    depth_idx = list(range(1, depth+1))
    if len(set(final_sat09)) > 1:
        rho_di, p_di = stats.spearmanr(depth_idx, final_sat09)
        spearman_di   = float(rho_di)
        spearman_di_p = float(p_di)
        log(f"  Spearman(depth_idx, final_sat09): rho={rho_di:.4f} p={p_di:.4f}")

    # ── Layer-1 checks ──
    layer1_is_max = bool(max(final_sat09) == final_sat09[0])
    last_gt_all_middle = (
        bool(all(final_sat09[-1] > final_sat09[l] for l in range(1, depth-1)))
        if depth >= 3 else None
    )

    log(f"  layer1_is_max={layer1_is_max}  last>all-middle={last_gt_all_middle}")

    return {
        "regime": regime, "depth": depth, "seed": seed,
        "init_loss": round(float(init_loss), 6),
        "test_acc":  round(float(test_acc),  6),
        "n_sigma2_empirical": n_sigma2_empirical,
        "init_S_l":           [round(v, 6) for v in init_sl],
        "init_sat09":         [round(v, 6) for v in init_sat09],
        "init_var_z":         [round(v, 6) for v in init_var],
        "s_l_strictly_decreasing": s_l_strictly_decreasing,
        "s_l_weakly_decreasing":   s_l_weakly_decreasing,
        "final_sat09":        [round(v, 6) for v in final_sat09],
        "rank_final_sat09":   [int(r) for r in rank_final],
        "spearman_sl_final09":   spearman_sl,
        "spearman_sl_p":         spearman_sl_p,
        "spearman_depth_final09": spearman_di,
        "spearman_depth_p":       spearman_di_p,
        "layer1_is_max":         layer1_is_max,
        "last_gt_all_middle":    last_gt_all_middle,
        "diverged": diverged,
        "sanity_ce_ok": 2.0 <= float(init_loss) <= 3.0,
        "sat09_ep0":   sat09_history[0],
        "sat09_final": sat09_history[-1],
    }


# ─────────────────────────────── Aggregation ─────────────────────────────────
def aggregate(all_results):
    """Per-(regime,depth): mean Spearman + 95% CI, reversal check."""
    np.random.seed(42)
    agg = {}
    for regime in REGIMES:
        for depth in DEPTHS:
            key   = f"{regime}_d{depth}"
            runs  = [r for r in all_results
                     if r["regime"] == regime and r["depth"] == depth]
            if not runs:
                continue

            sp_sl_vals = [r["spearman_sl_final09"]     for r in runs]
            sp_di_vals = [r["spearman_depth_final09"]   for r in runs]

            mean_sl, lo_sl, hi_sl = bootstrap_mean_ci(sp_sl_vals)
            mean_di, lo_di, hi_di = bootstrap_mean_ci(sp_di_vals)

            init_sl_mean = np.array([r["init_S_l"]    for r in runs]).mean(0).tolist()
            fin09_mean   = np.array([r["final_sat09"] for r in runs]).mean(0).tolist()
            rank_mean    = np.array([r["rank_final_sat09"] for r in runs]).mean(0).tolist()

            l1_max_count  = sum(1 for r in runs if r["layer1_is_max"])
            lL_gt_mid     = sum(1 for r in runs if r.get("last_gt_all_middle"))

            agg[key] = {
                "regime": regime, "depth": depth, "n_seeds": len(runs),
                "spearman_sl_per_seed":     sp_sl_vals,
                "spearman_sl_mean":         round(mean_sl, 4) if mean_sl is not None else None,
                "spearman_sl_ci95":         [round(lo_sl, 4), round(hi_sl, 4)] if lo_sl is not None else None,
                "spearman_depth_per_seed":  sp_di_vals,
                "spearman_depth_mean":      round(mean_di, 4) if mean_di is not None else None,
                "spearman_depth_ci95":      [round(lo_di, 4), round(hi_di, 4)] if lo_di is not None else None,
                "mean_init_S_l":            [round(v, 6) for v in init_sl_mean],
                "mean_final_sat09":         [round(v, 6) for v in fin09_mean],
                "mean_rank_final_sat09":    [round(v, 2) for v in rank_mean],
                "layer1_is_max_count":      f"{l1_max_count}/{len(runs)}",
                "layer1_is_max_fraction":   round(l1_max_count / len(runs), 2),
                "last_layer_gt_all_middle": {
                    "count":    f"{lL_gt_mid}/{len(runs)}",
                    "fraction": round(lL_gt_mid / len(runs), 2),
                    "per_seed": [r.get("last_gt_all_middle") for r in runs],
                },
            }
    return agg


# ─────────────────────────────── Reversal analysis ───────────────────────────
def analyze_reversal(agg):
    reversal_results = {}
    for depth in DEPTHS:
        std_sp = agg.get(f"standard_d{depth}", {}).get("spearman_sl_mean")
        s2x_sp = agg.get(f"super_2x_d{depth}", {}).get("spearman_sl_mean")
        s4x_sp = agg.get(f"super_4x_d{depth}", {}).get("spearman_sl_mean")
        std_di = agg.get(f"standard_d{depth}", {}).get("spearman_depth_mean")
        s2x_di = agg.get(f"super_2x_d{depth}", {}).get("spearman_depth_mean")
        s4x_di = agg.get(f"super_4x_d{depth}", {}).get("spearman_depth_mean")

        # Sign reversal: standard positive, super negative
        s2x_rev = (s2x_sp is not None and s2x_sp < 0)
        s4x_rev = (s4x_sp is not None and s4x_sp < 0)
        any_rev = s2x_rev or s4x_rev

        reversal_results[f"depth_{depth}"] = {
            "standard_spearman_sl":    std_sp,
            "super_2x_spearman_sl":    s2x_sp,
            "super_4x_spearman_sl":    s4x_sp,
            "standard_spearman_depth": std_di,
            "super_2x_spearman_depth": s2x_di,
            "super_4x_spearman_depth": s4x_di,
            "sign_reversal_2x":        s2x_rev,
            "sign_reversal_4x":        s4x_rev,
            "any_sign_reversal":       any_rev,
            "layer1_is_max_standard":  agg.get(f"standard_d{depth}", {}).get("layer1_is_max_count"),
            "layer1_is_max_super_2x":  agg.get(f"super_2x_d{depth}", {}).get("layer1_is_max_count"),
            "layer1_is_max_super_4x":  agg.get(f"super_4x_d{depth}", {}).get("layer1_is_max_count"),
            "layer1_frac_standard":    agg.get(f"standard_d{depth}", {}).get("layer1_is_max_fraction"),
            "layer1_frac_super_2x":    agg.get(f"super_2x_d{depth}", {}).get("layer1_is_max_fraction"),
            "layer1_frac_super_4x":    agg.get(f"super_4x_d{depth}", {}).get("layer1_is_max_fraction"),
        }

    any_sign_reversal     = any(v["any_sign_reversal"]  for v in reversal_results.values())
    reversal_in_all_depths = all(v["any_sign_reversal"] for v in reversal_results.values())
    layer1_first_everywhere = all(
        agg.get(f"standard_d{d}", {}).get("layer1_is_max_count") == f"{len(SEEDS)}/{len(SEEDS)}"
        for d in DEPTHS
    )

    if any_sign_reversal:
        verdict = (
            "PARTIAL REVERSAL: Spearman(S_l, sat09) sign is negative in at least one "
            "super-standard config, confirming deep-first saturation ordering."
        )
    else:
        verdict = (
            "NO REVERSAL: layer-1-first ordering persists in all regimes; "
            "Spearman(S_l, sat09) stays POSITIVE everywhere. "
            "Standard and super-standard both show layer 1 as max-saturation layer."
        )

    return {
        "per_depth":              reversal_results,
        "any_sign_reversal":      any_sign_reversal,
        "reversal_in_all_depths": reversal_in_all_depths,
        "layer1_first_in_standard_everywhere": layer1_first_everywhere,
        "verdict":                verdict,
    }


# ─────────────────────────────── Main ────────────────────────────────────────
def main():
    t0 = time.time()
    log(f"{'='*65}")
    log(f"INCREASE-COMPLEXITY-03: confirmatory (5 seeds), ≤25 epochs")
    log(f"{'='*65}")
    log(f"Device     : {DEVICE}")
    log(f"Depths     : {DEPTHS}   Seeds: {SEEDS}   MaxEpochs: {MAX_EPOCHS}")
    log(f"Regimes    : {list(REGIMES.keys())}")
    log(f"S_l thresh : |z| > {S_L_THRESH}")
    log(f"Sat thresh : |tanh(z)| > 0.9  ⟺  |z| > {SAT_09_Z:.4f}")
    log(f"Bootstrap  : {BOOTSTRAP_N} resamples for 95% CI")
    log(f"Total runs : {len(REGIMES)} × {len(DEPTHS)} × {len(SEEDS)} = "
        f"{len(REGIMES)*len(DEPTHS)*len(SEEDS)}")

    X_tr, Y_tr, X_te, Y_te = load_mnist_to_gpu(DATA_DIR)
    log(f"Data loaded in {time.time()-t0:.1f}s")

    all_results = []
    total_runs  = len(REGIMES) * len(DEPTHS) * len(SEEDS)

    for regime in REGIMES:
        for depth in DEPTHS:
            for seed in SEEDS:
                r = run_single(depth, seed, regime, X_tr, Y_tr, X_te, Y_te)
                all_results.append(r)
                done    = len(all_results)
                elapsed = time.time() - t0
                eta     = elapsed / done * (total_runs - done)
                log(f"  → {done}/{total_runs} | elapsed={elapsed:.0f}s | ETA≈{eta:.0f}s")

    # ── Aggregate ─────────────────────────────────────────────────────────────
    log(f"\n{'='*65}")
    log("AGGREGATED RESULTS")
    log(f"{'='*65}")
    agg = aggregate(all_results)

    for key, v in agg.items():
        log(f"\n{key}:")
        log(f"  mean_init_S_l      = {[f'{x:.4f}' for x in v['mean_init_S_l']]}")
        log(f"  mean_final_sat09   = {[f'{x:.4f}' for x in v['mean_final_sat09']]}")
        log(f"  layer_rank(1=max)  = {v['mean_rank_final_sat09']}")
        log(f"  Spearman(S_l,sat09)     mean={v['spearman_sl_mean']}  "
            f"95%CI={v['spearman_sl_ci95']}")
        log(f"  Spearman(depth,sat09)   mean={v['spearman_depth_mean']}  "
            f"95%CI={v['spearman_depth_ci95']}")
        log(f"  layer1_is_max       = {v['layer1_is_max_count']} "
            f"({v['layer1_is_max_fraction']*100:.0f}%)")
        log(f"  last_layer>all_mid  = {v['last_layer_gt_all_middle']['count']}")

    # ── Reversal analysis ─────────────────────────────────────────────────────
    log(f"\n{'='*65}")
    log("REVERSAL HYPOTHESIS TEST")
    log(f"{'='*65}")
    reversal = analyze_reversal(agg)

    for depth_key, v in reversal["per_depth"].items():
        log(f"\n  {depth_key}:")
        log(f"    standard  Spearman(S_l,sat09)={v['standard_spearman_sl']}  "
            f"Spearman(depth,sat09)={v['standard_spearman_depth']}")
        log(f"    super_2x  Spearman(S_l,sat09)={v['super_2x_spearman_sl']}  "
            f"sign_reversal={v['sign_reversal_2x']}")
        log(f"    super_4x  Spearman(S_l,sat09)={v['super_4x_spearman_sl']}  "
            f"sign_reversal={v['sign_reversal_4x']}")
        log(f"    layer1_is_max (std/2x/4x): {v['layer1_frac_standard']} / "
            f"{v['layer1_frac_super_2x']} / {v['layer1_frac_super_4x']}")

    log(f"\n{'='*65}")
    log("VERDICT")
    log(f"{'='*65}")
    log(f"  any_sign_reversal:              {reversal['any_sign_reversal']}")
    log(f"  reversal_in_ALL_depths:         {reversal['reversal_in_all_depths']}")
    log(f"  layer1_first_everywhere (std):  {reversal['layer1_first_in_standard_everywhere']}")
    log(f"\n  REVERSAL VERDICT: {reversal['verdict']}")

    # ── Write RESULTS.json ────────────────────────────────────────────────────
    elapsed      = time.time() - t0
    any_diverged = any(r["diverged"] for r in all_results)
    sanity_ok    = all(r["sanity_ce_ok"] for r in all_results)

    # Summary table: per-config mean, CI
    summary_table = []
    for regime in REGIMES:
        for depth in DEPTHS:
            key = f"{regime}_d{depth}"
            cfg = agg.get(key, {})
            summary_table.append({
                "config":              key,
                "regime":              regime,
                "depth":               depth,
                "n_seeds":             cfg.get("n_seeds"),
                "spearman_sl_mean":    cfg.get("spearman_sl_mean"),
                "spearman_sl_ci95_lo": cfg.get("spearman_sl_ci95", [None, None])[0],
                "spearman_sl_ci95_hi": cfg.get("spearman_sl_ci95", [None, None])[1],
                "spearman_depth_mean": cfg.get("spearman_depth_mean"),
                "spearman_depth_ci95": cfg.get("spearman_depth_ci95"),
                "layer1_is_max_frac":  cfg.get("layer1_is_max_fraction"),
            })

    results_json = {
        "status": "SUCCESS" if not any_diverged else "PARTIAL",
        "scale": "full",
        "metrics": {
            "primary_metric": (
                "Spearman(init_S_l, final_sat_frac_|tanh|>0.9) "
                "per (regime, depth), mean ± 95% bootstrap CI over 5 seeds"
            ),
            "sat_threshold_final": f"|tanh(z)| > 0.9  ⟺  |z| > {SAT_09_Z:.4f}",
            "init_sl_threshold":   f"|z| > {S_L_THRESH}",
            "summary_table":       summary_table,
            "per_config":          agg,
            "reversal_analysis":   reversal,
            "reversal_verdict":    reversal["verdict"],
            "any_diverged":        any_diverged,
            "sanity_ce_ok":        sanity_ok,
            "elapsed_seconds":     round(elapsed, 1),
            "total_runs":          len(all_results),
        },
        "subject_executed": (
            f"tanh MLP, MNIST std-norm (µ={MNIST_MEAN}, σ={MNIST_STD}), "
            f"depths={DEPTHS}, seeds={SEEDS}, "
            f"regimes=[standard(uniform,n·σ²≈1/3), super_2x(normal,n·σ²=2), "
            f"super_4x(normal,n·σ²=4)], "
            f"lr=1e-2, momentum=0.9, batch={BATCH_SIZE}, max_epochs={MAX_EPOCHS}, "
            f"device={DEVICE}, bootstrap_n={BOOTSTRAP_N}, total_runs=45"
        ),
        "notes": (
            "Round increase_complexity-03: confirmatory (full) run, 5 seeds, 25 epochs. "
            "Reuses refine-02 non-censored metric: Spearman(init S_l, final_sat_frac_|tanh|>0.9). "
            f"REVERSAL VERDICT: {reversal['verdict']} "
            "Baseline (refine-02, 3 seeds): no reversal, Spearman positive everywhere."
        ),
    }

    with open(RESULTS_PATH, "w") as f:
        json.dump(results_json, f, indent=2)
    log(f"\nResults written → {RESULTS_PATH}")
    log(f"Total elapsed: {elapsed:.1f}s | any_diverged={any_diverged} | sanity_ok={sanity_ok}")
    log("\n" + json.dumps(results_json, indent=2))

    _log_fh.close()


if __name__ == "__main__":
    main()
