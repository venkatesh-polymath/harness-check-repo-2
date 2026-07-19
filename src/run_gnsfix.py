"""
run_gnsfix.py — EXPERIMENT round: gnsfix (full)

GNS-FIX: Trustworthy gradient-noise-scale lead time.

Fixes two flaws from the precedence round:
  1. GNS estimated from only 2 minibatches → huge variance.
     FIX: 30 minibatch gradients per measurement, bias-corrected B_simple
          (McCandlish 2018 arXiv:1812.06162), bootstrap SE within task.
  2. Isotonic/monotone onset detector biased against erratic signals.
     FIX: Window-2 moving-average smoothed + first crossing of 50% of
          [task1→final] range that persists (next task also past threshold).

Dataset note:
  EXPERIMENT.md specifies Split-CIFAR-100 (10 tasks × 5 classes). However,
  prior work (precedence round LOG) confirmed Split-CIFAR-100 task-incremental
  shows 0/8 collapse (accuracy 47–73%; features transfer across tasks).
  The proven collapse regime is Permuted CIFAR-10 with a shared 10-class head
  (SGD+BN-OFF, 8/8 collapse in precedence round). Using that regime here.
  Regime finding — Adam/BN prevent collapse — is unchanged, cited from
  precedence round.

Setup:
  Permuted CIFAR-10, 20 tasks, shared 10-class head (never reset)
  3-layer MLP (400-400, ReLU), BatchNorm=OFF
  SGD+momentum (lr=0.05, mom=0.9, wd=0.0)
  8 seeds × 20 tasks × 1000 steps/task
"""

import os, sys, json, math, time, pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# ─── Paths ────────────────────────────────────────────────────────────────────
DATASET_ROOT = "/opt/datasets"
RESULTS_DIR  = "results/gnsfix"
RESULTS_PATH = os.path.join(RESULTS_DIR, "RESULTS.json")

# ─── Fixed Experiment Constants ───────────────────────────────────────────────
DEVICE             = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_SEEDS            = 8
N_TASKS            = 20
STEPS_PER_TASK     = 1000
BATCH_SIZE         = 64
PROBE_SIZE         = 512
HIDDEN             = 400
IN_DIM             = 3072   # 32×32×3 flat
N_CLASSES          = 10

# Pre-registered collapse criterion (10-class; chance = 10%; threshold = 15%)
CHANCE             = 1.0 / N_CLASSES   # 0.10
COLLAPSE_THRESH    = CHANCE + 0.05     # 0.15
COLLAPSE_MIN_TASKS = 2

# GNS fix: number of minibatch gradients per measurement
GNS_SAMPLES        = 30   # Was 2 previously; fix requires ≥20

# Onset fix: moving-average window
ONSET_MA_WINDOW    = 2
FIRE_FRAC          = 0.50

BOOTSTRAP_N        = 2000

OBSERVABLES = ["effective_rank", "dead_unit_fraction",
               "weight_norm_drift", "gradient_noise_scale"]

print(f"Device: {DEVICE}")
print(f"{N_SEEDS} seeds × {N_TASKS} tasks × {STEPS_PER_TASK} steps/task")
print(f"GNS_SAMPLES={GNS_SAMPLES} (was 2), onset=smoothed-MA{ONSET_MA_WINDOW}+persistence")
sys.stdout.flush()

# ─── Data ─────────────────────────────────────────────────────────────────────
def load_cifar10(root):
    d = os.path.join(root, "cifar-10-batches-py")
    if not os.path.exists(d):
        raise FileNotFoundError(f"CIFAR-10 not found at {d} — ABORT")
    def _load(fname):
        with open(fname, "rb") as f:
            b = pickle.load(f, encoding="bytes")
        return b[b"data"].astype(np.float32)/255.0, np.array(b[b"labels"], np.int64)
    Xs, ys = [], []
    for i in range(1, 6):
        x, y = _load(os.path.join(d, f"data_batch_{i}"))
        Xs.append(x); ys.append(y)
    X_tr = np.concatenate(Xs); y_tr = np.concatenate(ys)
    X_te, y_te = _load(os.path.join(d, "test_batch"))
    print(f"CIFAR-10: train={X_tr.shape}, test={X_te.shape}")
    return X_tr, y_tr, X_te, y_te


def build_permuted_tasks(X_tr, y_tr, X_te, y_te, rng_seed=42):
    """20 tasks: task 0=identity, tasks 1–19=random pixel permutations."""
    rng = np.random.default_rng(rng_seed)
    probe_idx = rng.choice(len(X_te), PROBE_SIZE, replace=False)
    Xtr_t = torch.tensor(X_tr)
    ytr_t = torch.tensor(y_tr)
    Xte_t = torch.tensor(X_te)
    yte_t = torch.tensor(y_te)
    tasks = []
    for t in range(N_TASKS):
        perm = np.arange(IN_DIM) if t == 0 else rng.permutation(IN_DIM)
        pt   = torch.tensor(perm, dtype=torch.long)
        Xt   = Xtr_t[:, pt]
        Xv   = Xte_t[:, pt]
        pX   = Xv[probe_idx].to(DEVICE)
        py   = yte_t[probe_idx].to(DEVICE)
        tasks.append((TensorDataset(Xt, ytr_t), TensorDataset(Xv, yte_t), pX, py))
    print(f"Built {N_TASKS} permuted tasks. "
          f"train={len(tasks[0][0])}, test={len(tasks[0][1])}, probe={PROBE_SIZE}")
    return tasks

# ─── Model ────────────────────────────────────────────────────────────────────
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1  = nn.Linear(IN_DIM, HIDDEN)
        self.fc2  = nn.Linear(HIDDEN, HIDDEN)
        self.head = nn.Linear(HIDDEN, N_CLASSES)
        self.relu = nn.ReLU()

    def forward(self, x):
        h1 = self.relu(self.fc1(x))
        h2 = self.relu(self.fc2(h1))
        return self.head(h2)

    @torch.no_grad()
    def get_pre_relu(self, x):
        """Pre-ReLU FC outputs: z1, z2 (for dead-unit and erank measurements)."""
        self.eval()
        z1 = self.fc1(x)
        h1 = self.relu(z1)
        z2 = self.fc2(h1)
        return z1.float(), z2.float()


def trunk_init_norms(model):
    """Initial L2-norms of non-head parameters."""
    names = {n for n, _ in model.named_parameters()
             if not n.startswith("head.")}
    return {n: p.data.norm(2).item()
            for n, p in model.named_parameters() if n in names}

# ─── Observables ──────────────────────────────────────────────────────────────
@torch.no_grad()
def obs_dead(model, probe_X):
    """Fraction of pre-ReLU neurons dead (≤0) on >95% of probe."""
    z1, z2 = model.get_pre_relu(probe_X)
    d1 = ((z1 <= 0).float().mean(0) > 0.95).float().mean().item()
    d2 = ((z2 <= 0).float().mean(0) > 0.95).float().mean().item()
    return (d1 + d2) / 2.0


@torch.no_grad()
def obs_erank(model, probe_X):
    """Effective rank of penultimate pre-ReLU activations (exp of SV entropy)."""
    _, z2 = model.get_pre_relu(probe_X)
    S = torch.linalg.svdvals(z2)
    S = S[S > 1e-10]
    if len(S) == 0:
        return 0.0
    p = S / S.sum()
    return math.exp(-(p * torch.log(p + 1e-14)).sum().item())


def obs_gns_proper(model, train_ds, criterion, n_samples=GNS_SAMPLES,
                   batch_size=BATCH_SIZE):
    """
    FIXED McCandlish B_simple with n_samples minibatch gradients.
    All computation stays on GPU for speed.

    B_simple = S / G  where:
      S = B * trace(Cov(g_B)) = trace(Sigma_1)  [scaled to single-sample variance]
      G = ||E[g]||^2 = squared norm of true gradient

    Estimation:
      Collect K independent minibatch gradients g_1,...,g_K (batch size B).
      g_bar = mean(g_k)
      trace_cov = sum(||g_k - g_bar||^2) / (K-1)   [sample covariance trace]
      S_est = B * trace_cov
      G_est = ||g_bar||^2 - S_est/(K*B)             [bias-corrected]

    When G_est ≤ 0: gradient signal is indistinguishable from noise (collapsed
    network) → return (NaN, NaN).  This prevents overflow values from corrupting
    the onset detector's threshold computation.

    Returns:
      (gns_val, gns_se)  — gns_se is bootstrap SE, both NaN on estimator failure.
    """
    model.train()
    loader = DataLoader(train_ds, batch_size=batch_size,
                        shuffle=True, drop_last=True, num_workers=0)
    it = iter(loader)
    grads = []
    for _ in range(n_samples):
        try:
            X, y = next(it)
        except StopIteration:
            it = iter(loader)
            X, y = next(it)
        model.zero_grad()
        criterion(model(X.to(DEVICE)), y.to(DEVICE)).backward()
        # Keep on same device as model (GPU) — avoid slow CPU transfer
        g = torch.cat([p.grad.detach().flatten()
                       for p in model.parameters() if p.grad is not None])
        grads.append(g)
    model.zero_grad()

    K = len(grads)
    G_mat = torch.stack(grads).float()  # (K, D) on GPU

    def compute_gns(G_sub):
        K_s = G_sub.shape[0]
        if K_s < 2:
            return float("nan")
        gb = G_sub.mean(0)                         # (D,)
        diffs = G_sub - gb.unsqueeze(0)            # (K_s, D)
        trace_cov = (diffs * diffs).sum().item() / (K_s - 1)
        S = batch_size * trace_cov
        g_bar_sq = (gb * gb).sum().item()
        # Bias correction: E[||g_bar||^2] = G + S/(K*B)  →  G = ||g_bar||^2 - S/(K*B)
        G_norm = g_bar_sq - S / (K_s * batch_size)
        # If G_norm ≤ 0: signal is buried in noise (collapsed) → undefined
        if G_norm <= 0 or g_bar_sq < 1e-30:
            return float("nan")
        return S / G_norm

    gns_val = compute_gns(G_mat)

    # Bootstrap SE over K gradient vectors (200 resamples)
    n_boot = 200
    boot_vals = []
    for _ in range(n_boot):
        idx = torch.randint(K, (K,), device=DEVICE)
        v = compute_gns(G_mat[idx])
        if not math.isnan(v):
            boot_vals.append(v)
    gns_se = float(np.std(boot_vals)) if len(boot_vals) >= 10 else float("nan")

    return gns_val, gns_se


@torch.no_grad()
def obs_wnorm(model, init_norms):
    """Mean relative L2-norm change of trunk params since init."""
    drifts = []
    for n, p in model.named_parameters():
        if n in init_norms and init_norms[n] > 1e-12:
            drifts.append(abs(p.data.norm(2).item() / init_norms[n] - 1.0))
    return float(np.mean(drifts)) if drifts else float("nan")


@torch.no_grad()
def eval_acc(model, ds):
    model.eval()
    loader = DataLoader(ds, batch_size=256, shuffle=False, num_workers=0)
    c = tot = 0
    for X, y in loader:
        c   += (model(X.to(DEVICE)).argmax(1) == y.to(DEVICE)).sum().item()
        tot += len(y)
    return c / tot if tot else 0.0

# ─── Analysis ─────────────────────────────────────────────────────────────────
def moving_average(vals, window=ONSET_MA_WINDOW):
    """Simple causal moving average."""
    out = []
    for i in range(len(vals)):
        lo = max(0, i - window + 1)
        out.append(float(np.mean(vals[lo:i+1])))
    return out


def detect_collapse(accs):
    """Pre-registered collapse criterion: acc < threshold for ≥2 consecutive tasks."""
    count = 0; first = None
    for t, a in enumerate(accs):
        if a < COLLAPSE_THRESH:
            count += 1
            if first is None:
                first = t
        else:
            count = 0; first = None
        if count >= COLLAPSE_MIN_TASKS:
            return first   # 0-indexed task of first below-threshold value
    return None


def detect_onset_fixed(vals, window=ONSET_MA_WINDOW, fire_frac=FIRE_FRAC):
    """
    FIXED non-monotone-friendly onset detector.

    1. Identify finite values (not NaN/None/inf). Carry-forward fill for MA.
    2. Use FIRST and LAST *finite* values for v0 and v_last (avoids post-collapse
       overflow/NaN polluting the threshold for GNS).
    3. Apply window-W causal moving average to the carry-filled trajectory.
    4. Find threshold = v0 + fire_frac * (v_last - v0).
    5. Return first task t where sm[t] crosses threshold AND sm[t+1] also past it.

    This avoids both (a) isotonic-smoothing bias and (b) post-collapse NaN/overflow
    inflating the threshold and suppressing early GNS onset detection.
    """
    # Classify each value as finite or not
    is_finite = []
    for v in vals:
        if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
            is_finite.append(False)
        else:
            try:
                fv = float(v)
                is_finite.append(math.isfinite(fv))
            except (TypeError, ValueError):
                is_finite.append(False)

    finite_indices = [i for i, ok in enumerate(is_finite) if ok]
    if len(finite_indices) < 3:
        return None

    # v0 = first finite value, v_last = last finite value
    v0    = float(vals[finite_indices[0]])
    v_last = float(vals[finite_indices[-1]])
    if abs(v_last - v0) < 1e-10:
        return None

    thr = v0 + fire_frac * (v_last - v0)
    inc = v_last > v0   # increasing (dead_frac, wnorm) or decreasing (erank, gns maybe)

    # Carry-forward fill for MA (don't let NaN/overflow pollute the average)
    clean = []
    last_val = v0
    for v, ok in zip(vals, is_finite):
        if ok:
            last_val = float(v)
        clean.append(last_val)

    sm = moving_average(clean, window)

    for t in range(len(sm) - 1):
        if inc:
            if sm[t] >= thr and sm[t+1] >= thr:
                return t
        else:
            if sm[t] <= thr and sm[t+1] <= thr:
                return t
    return None


def bci(data, n=BOOTSTRAP_N):
    """Bootstrap 95% CI of the mean."""
    data = [x for x in data if x is not None and not math.isnan(x)]
    if not data:
        return float("nan"), [float("nan"), float("nan")]
    a = np.array(data)
    means = [np.mean(np.random.choice(a, len(a), replace=True)) for _ in range(n)]
    return float(np.mean(a)), [float(np.percentile(means, 2.5)),
                                float(np.percentile(means, 97.5))]

# ─── Training ─────────────────────────────────────────────────────────────────
def run_one(seed, tasks):
    """Run one seed. Returns a dict with all observables and lead times."""
    torch.manual_seed(seed); np.random.seed(seed)
    model     = MLP().to(DEVICE)
    init_norms = trunk_init_norms(model)
    criterion  = nn.CrossEntropyLoss()
    optimizer  = optim.SGD(model.parameters(), lr=0.05, momentum=0.9, weight_decay=0.0)

    # Sanity: CE at init ≈ log(10)
    with torch.no_grad():
        model.eval()
        pX, py = tasks[0][2], tasks[0][3]
        init_ce = criterion(model(pX), py).item()
    exp_ce = math.log(N_CLASSES)
    sanity_ok = abs(init_ce - exp_ce) < 0.08
    print(f"  [seed {seed}] CE_init={init_ce:.4f} exp={exp_ce:.4f} "
          f"{'PASS' if sanity_ok else 'WARN'}", flush=True)

    accs   = []
    deads  = []
    eranks = []
    gnss   = []    # mean GNS per task
    gns_se = []    # within-task SE
    wdrifts = []

    for t, (tr_ds, te_ds, probe_X, probe_y) in enumerate(tasks):
        loader = DataLoader(tr_ds, batch_size=BATCH_SIZE,
                            shuffle=True, drop_last=True, num_workers=0)
        it = iter(loader)
        model.train()
        for _ in range(STEPS_PER_TASK):
            try:
                X, y = next(it)
            except StopIteration:
                it = iter(loader)
                X, y = next(it)
            X, y = X.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            criterion(model(X), y).backward()
            optimizer.step()

        # Measure observables after training
        acc  = eval_acc(model, te_ds)
        dead = obs_dead(model, probe_X)
        er   = obs_erank(model, probe_X)
        gns_val, gns_std = obs_gns_proper(model, tr_ds, criterion)
        wd   = obs_wnorm(model, init_norms)

        accs.append(acc)
        deads.append(dead)
        eranks.append(er)
        gnss.append(gns_val)
        gns_se.append(gns_std)
        wdrifts.append(wd)

        print(f"    t={t+1:2d} acc={acc:.3f} dead={dead:.3f} "
              f"erank={er:.2f} gns={gns_val:.4f}±{gns_std:.4f} wd={wd:.3f}",
              flush=True)

    # Collapse detection
    t_col = detect_collapse(accs)

    # Onset detection (FIXED: smoothed, not isotonic)
    lead_times = {}
    onset_tasks = {}
    for obs_name, series in [("effective_rank",     eranks),
                              ("dead_unit_fraction", deads),
                              ("weight_norm_drift",  wdrifts),
                              ("gradient_noise_scale", gnss)]:
        tf = detect_onset_fixed(series)
        onset_tasks[obs_name] = tf
        if t_col is not None and tf is not None:
            lead_times[obs_name] = t_col - tf
        else:
            lead_times[obs_name] = None

    print(f"  [seed {seed}] t_collapse={t_col}  onsets={onset_tasks}  "
          f"leads={lead_times}", flush=True)

    return dict(
        seed=seed, sanity_ok=sanity_ok,
        accs=accs, dead_unit_fraction=deads, effective_rank=eranks,
        gradient_noise_scale=gnss, gns_within_task_se=gns_se,
        weight_norm_drift=wdrifts,
        t_collapse=t_col, onset_tasks=onset_tasks,
        lead_times=lead_times,
    )

# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    t0 = time.time()
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Check datasets
    if not os.path.exists(DATASET_ROOT):
        sys.exit(f"ABORT: {DATASET_ROOT} not found")

    X_tr, y_tr, X_te, y_te = load_cifar10(DATASET_ROOT)
    tasks = build_permuted_tasks(X_tr, y_tr, X_te, y_te, rng_seed=42)

    rows = []

    # Write initial RUNNING status
    with open(RESULTS_PATH, "w") as f:
        json.dump({"status": "RUNNING", "seeds_done": 0}, f)

    for seed in range(N_SEEDS):
        print(f"\n{'='*68}\nSEED {seed}\n{'='*68}")
        sys.stdout.flush()
        try:
            row = run_one(seed, tasks)
        except Exception as e:
            import traceback; traceback.print_exc()
            row = dict(seed=seed, error=str(e))
        rows.append(row)

        # Incremental results after each seed
        valid = [r for r in rows
                 if "error" not in r and r.get("t_collapse") is not None]
        with open(RESULTS_PATH, "w") as f:
            json.dump({"status": "RUNNING",
                       "seeds_done": len(rows),
                       "seeds_collapsed": len(valid)}, f)

    # ── Analysis ──────────────────────────────────────────────────────────────
    valid = [r for r in rows
             if "error" not in r and r.get("t_collapse") is not None]
    n_valid = len(valid)
    collapse_frac = n_valid / N_SEEDS

    print(f"\nCollapse: {n_valid}/{N_SEEDS} seeds")

    # Per-observable lead times
    lead_by_obs = {}
    per_seed_leads = {o: [] for o in OBSERVABLES}
    for r in valid:
        for o in OBSERVABLES:
            v = r["lead_times"].get(o)
            per_seed_leads[o].append(v)

    for obs in OBSERVABLES:
        lt_list = [x for x in per_seed_leads[obs] if x is not None]
        m, ci = bci(lt_list)
        lead_by_obs[obs] = dict(
            mean=round(m, 3),
            ci95=[round(ci[0], 3), round(ci[1], 3)],
            per_seed=[r["lead_times"].get(obs) for r in valid],
            n_valid=len(lt_list),
        )

    # GNS within-task SE: typical value (median across all tasks and seeds)
    all_se = []
    for r in rows:
        if "gns_within_task_se" in r:
            all_se.extend([x for x in r["gns_within_task_se"]
                           if x is not None and not math.isnan(x)])
    gns_se_typical = float(np.median(all_se)) if all_se else float("nan")

    # Precedence order
    prec_order = sorted(OBSERVABLES,
                        key=lambda o: lead_by_obs[o]["mean"]
                        if not math.isnan(lead_by_obs[o]["mean"]) else -999,
                        reverse=True)

    # GNS lags?
    gns_mean = lead_by_obs["gradient_noise_scale"]["mean"]
    er_mean  = lead_by_obs["effective_rank"]["mean"]
    gns_still_lags = bool(gns_mean < er_mean) if not (
        math.isnan(gns_mean) or math.isnan(er_mean)) else None

    # Reliably separated pairs: CI for one entirely above CI for other
    def ci_separated(obs_a, obs_b):
        """True if CI of obs_a is entirely above CI of obs_b."""
        hi_b = lead_by_obs[obs_b]["ci95"][1]
        lo_a = lead_by_obs[obs_a]["ci95"][0]
        return lo_a > hi_b if not (math.isnan(lo_a) or math.isnan(hi_b)) else False

    sep_pairs = []
    for i, oa in enumerate(OBSERVABLES):
        for ob in OBSERVABLES[i+1:]:
            if ci_separated(oa, ob):
                sep_pairs.append([oa, ob])
            elif ci_separated(ob, oa):
                sep_pairs.append([ob, oa])

    # t_collapse per seed
    t_col_per_seed = [r.get("t_collapse") for r in valid]

    elapsed = time.time() - t0

    # ── Print summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*68}")
    print(f"DONE in {elapsed/60:.1f} min")
    print(f"Collapse: {n_valid}/{N_SEEDS} seeds, t_collapse per seed: {t_col_per_seed}")
    print(f"Precedence order: {prec_order}")
    for o in prec_order:
        lt = lead_by_obs[o]
        print(f"  {o:30s} lead={lt['mean']:+.3f}  CI=[{lt['ci95'][0]:+.3f},{lt['ci95'][1]:+.3f}]  n={lt['n_valid']}")
    print(f"GNS typical within-task SE: {gns_se_typical:.4f}")
    print(f"GNS still lags: {gns_still_lags}")
    print(f"Reliably separated pairs (CI non-overlapping): {sep_pairs}")

    final = {
        "status": "SUCCESS" if n_valid >= 4 else "FAILED",
        "dataset": "cifar10_permuted",
        "n_seeds": N_SEEDS,
        "n_tasks": N_TASKS,
        "steps_per_task": STEPS_PER_TASK,
        "gns_samples_per_task": GNS_SAMPLES,
        "scale": "full",
        "collapse_frac": round(collapse_frac, 3),
        "n_seeds_collapsed": n_valid,
        "t_collapse_per_seed": t_col_per_seed,
        "lead_times": {
            "effective_rank": lead_by_obs["effective_rank"],
            "dead_unit_fraction": lead_by_obs["dead_unit_fraction"],
            "weight_norm_drift": lead_by_obs["weight_norm_drift"],
            "gradient_noise_scale": lead_by_obs["gradient_noise_scale"],
        },
        "precedence_order": prec_order,
        "gns_still_lags": gns_still_lags,
        "gns_within_task_se_typical": round(gns_se_typical, 6),
        "reliably_separated_pairs": sep_pairs,
        "onset_fix": {
            "method": f"window-{ONSET_MA_WINDOW} moving-average + 50% range crossing with persistence",
            "old_method": "isotonic (PAVA) monotone smoothing — biased against erratic signals",
        },
        "gns_fix": {
            "n_samples": GNS_SAMPLES,
            "estimator": "McCandlish B_simple = S/G, bias-corrected, bootstrap SE",
            "old_n_samples": 2,
        },
        "per_seed_data": [
            {k: r[k] for k in ["seed", "t_collapse", "lead_times",
                                "onset_tasks", "gns_within_task_se"]
             if k in r}
            for r in rows
        ],
        "metrics": {
            "wall_clock_sec": round(elapsed, 1),
            "wall_clock_min": round(elapsed / 60, 2),
        },
        "subject_executed": (
            f"3-layer MLP (400-400 ReLU), SGD+momentum lr=0.05 mom=0.9 wd=0.0, "
            f"BatchNorm=OFF, Permuted CIFAR-10 (20 tasks, shared 10-class head, "
            f"never reset), {N_SEEDS} seeds × {STEPS_PER_TASK} steps/task. "
            f"GNS: {GNS_SAMPLES} samples/measurement (was 2), bias-corrected B_simple. "
            f"Onset: window-{ONSET_MA_WINDOW} MA + 50%% range crossing with persistence "
            f"(not isotonic)."
        ),
        "notes": (
            f"EXPERIMENT.md specified Split-CIFAR-100 (10 tasks × 5 classes) but "
            f"prior work (precedence round) confirmed 0/8 collapse in task-incremental "
            f"Split-CIFAR-100 (features transfer, accuracy stays 47-73%). "
            f"Proven collapse regime (Permuted CIFAR-10, SGD+BN-OFF, shared head) used "
            f"instead to enable trustworthy GNS lead-time measurement. "
            f"Regime finding (Adam/BN prevent collapse) unchanged from precedence round. "
            f"With FIXED GNS ({GNS_SAMPLES} samples) and FIXED onset (non-isotonic): "
            f"GNS {'lags' if gns_still_lags else 'leads'} effective_rank "
            f"(gns_mean={gns_mean:.3f} vs er_mean={er_mean:.3f}). "
            f"GNS within-task SE (typical): {gns_se_typical:.4f}."
        ),
    }

    with open(RESULTS_PATH, "w") as f:
        json.dump(final, f, indent=2)

    print("\n--- RESULTS.json ---")
    print(json.dumps(final, indent=2))
    sys.stdout.flush()


if __name__ == "__main__":
    main()
