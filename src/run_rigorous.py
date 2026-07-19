"""
run_rigorous.py — EXPERIMENT round: precedence (full)

Observable Temporal Precedence Map: 4 configs × 8 seeds × Permuted-CIFAR-10

WHY PERMUTED (not split-CIFAR-100 task-incremental):
  Split-CIFAR-100 with a fresh head per task showed 0/8 collapse across all 10 tasks
  because even a trunk with 90% dead units can support a fresh 5-class head.
  The WORKING collapse mechanism (validated in baseline_obs) is permuted inputs
  with a SHARED output head never reset: each permutation requires NEW representations
  from the increasingly-dead trunk.

Design (matches pre-registered observables + analysis):
  Dataset  : Permuted CIFAR-10, 20 tasks (task 0 = identity perm, tasks 1–19 random)
  Model    : 3-layer MLP (400-400, ReLU), with or without BatchNorm1d
  Configs  : {SGD+momentum, Adam} × {BN off, BN on}  — 4 configs total
  Seeds    : 8 per config
  Steps    : 1000 / task  (same dynamics as confirmed baseline_obs collapse)
  Head     : SHARED 10-class, NEVER reset — collapse mechanism
  Observables: dead_unit_fraction, effective_rank (pre-BN), gradient_noise_scale,
               weight_norm_drift — all measured on per-task probe (current permutation)

Pre-registered collapse criterion:
  t_col = first task (0-indexed) where new-task acc < 20%−10pp = 10%? NO:
  acc < (chance + 5pp) where chance = 1/10 = 10%, so threshold = 15%.
  Must stay below for ≥ 2 consecutive tasks.

Pre-registered onset criterion:
  t_fire = first task crossing 50% of [v_task1 → v_final] range (monotone-smoothed)
  lead_time = t_col − t_fire  (positive = fires before collapse)

Analysis:
  Per (config, observable): mean lead_time ± 95% bootstrap CI over 8 seeds
  Optimizer-conditional: SGD vs Adam GNS lead time difference + CI
  Overall precedence order: ranked by pooled mean lead time
"""

import os, sys, json, math, time, pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# ─── Paths ────────────────────────────────────────────────────────────────────
DATASET_ROOT = "/opt/datasets"
RESULTS_DIR  = "results/precedence"
ROWS_PATH    = os.path.join(RESULTS_DIR, "rows.jsonl")
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
COLLAPSE_THRESH    = CHANCE + 0.05     # 0.15  (chance + 5pp)
COLLAPSE_MIN_TASKS = 2

# Pre-registered onset criterion
FIRE_FRAC          = 0.50

BOOTSTRAP_N        = 2000

# Configs
CONFIGS = {
    "sgd_bnoff":  dict(opt="sgd",  bn=False, lr=0.05, momentum=0.9, wd=0.0),
    "adam_bnoff": dict(opt="adam", bn=False, lr=1e-3, wd=0.0),
    "sgd_bnon":   dict(opt="sgd",  bn=True,  lr=0.05, momentum=0.9, wd=0.0),
    "adam_bnon":  dict(opt="adam", bn=True,  lr=1e-3, wd=0.0),
}

OBSERVABLES = ["dead_unit_fraction", "effective_rank",
               "gradient_noise_scale", "weight_norm_drift"]

print(f"Device: {DEVICE}")
print(f"Configs: {list(CONFIGS.keys())}")
print(f"{N_SEEDS} seeds × {N_TASKS} tasks × {STEPS_PER_TASK} steps/task | "
      f"collapse_thresh={COLLAPSE_THRESH:.2f}")
sys.stdout.flush()

# ─── Data ────────────────────────────────────────────────────────────────────
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
        x, y = _load(os.path.join(d, f"data_batch_{i}")); Xs.append(x); ys.append(y)
    X_tr = np.concatenate(Xs); y_tr = np.concatenate(ys)
    X_te, y_te = _load(os.path.join(d, "test_batch"))
    print(f"CIFAR-10: train={X_tr.shape}, test={X_te.shape}")
    return X_tr, y_tr, X_te, y_te


def build_permuted_tasks(X_tr, y_tr, X_te, y_te, rng_seed=42):
    """
    20 tasks: task 0 = identity permutation, tasks 1-19 = random permutations.
    Permutations are FIXED (same seed=42 across all configs/model-seeds).
    Returns list of (train_ds, test_ds, probe_X, probe_y).
    probe_X = test images with current permutation applied (500 samples).
    probe_y = same labels (permuting pixels doesn't change class).
    """
    rng = np.random.default_rng(rng_seed)
    # Fixed probe indices from test set
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
          f"train={len(tasks[0][0])}, test={len(tasks[0][1])}, probe={len(tasks[0][2])}")
    return tasks

# ─── Model ───────────────────────────────────────────────────────────────────
class MLP(nn.Module):
    def __init__(self, use_bn=False):
        super().__init__()
        self.use_bn = use_bn
        self.fc1  = nn.Linear(IN_DIM, HIDDEN)
        self.fc2  = nn.Linear(HIDDEN, HIDDEN)
        self.head = nn.Linear(HIDDEN, N_CLASSES)
        if use_bn:
            self.bn1 = nn.BatchNorm1d(HIDDEN)
            self.bn2 = nn.BatchNorm1d(HIDDEN)
        self.relu = nn.ReLU()

    def forward(self, x):
        z1 = self.fc1(x)
        h1 = self.relu(self.bn1(z1) if self.use_bn else z1)
        z2 = self.fc2(h1)
        h2 = self.relu(self.bn2(z2) if self.use_bn else z2)
        return self.head(h2)

    @torch.no_grad()
    def get_prebn(self, x):
        """Pre-BN (pre-ReLU) FC outputs: z1, z2."""
        self.eval()
        z1 = self.fc1(x)
        h1 = self.relu(self.bn1(z1) if self.use_bn else z1)
        z2 = self.fc2(h1)
        return z1.float(), z2.float()


def trunk_init_norms(model):
    """Initial L2-norms of non-head parameters (snapshot at start)."""
    names = {n for n, _ in model.named_parameters()} - {"head.weight", "head.bias"}
    return {n: p.data.norm(2).item()
            for n, p in model.named_parameters() if n in names}

# ─── Observables ──────────────────────────────────────────────────────────────
@torch.no_grad()
def obs_dead(model, probe_X):
    """Fraction of pre-BN neurons dead (value ≤ 0) on > 95% of probe."""
    z1, z2 = model.get_prebn(probe_X)
    d1 = ((z1 <= 0).float().mean(0) > 0.95).float().mean().item()
    d2 = ((z2 <= 0).float().mean(0) > 0.95).float().mean().item()
    return (d1 + d2) / 2.0


@torch.no_grad()
def obs_erank(model, probe_X):
    """Effective rank of penultimate pre-BN activations (exp of SV entropy)."""
    _, z2 = model.get_prebn(probe_X)
    S = torch.linalg.svdvals(z2)
    S = S[S > 1e-10]
    if len(S) == 0: return 0.0
    p = S / S.sum()
    return math.exp(-(p * torch.log(p + 1e-14)).sum().item())


def obs_gns(model, train_ds, criterion):
    """McCandlish B_simple: two mini-batches, noise/signal ratio."""
    model.train()
    loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                        shuffle=True, drop_last=True, num_workers=0)
    it = iter(loader)
    try:
        X1, y1 = next(it); X2, y2 = next(it)
    except StopIteration:
        return float("nan")
    def gv(X, y):
        model.zero_grad()
        criterion(model(X.to(DEVICE)), y.to(DEVICE)).backward()
        return torch.cat([p.grad.detach().flatten()
                          for p in model.parameters() if p.grad is not None])
    g1, g2 = gv(X1, y1), gv(X2, y2)
    g_bar = (g1 + g2) * 0.5
    noise = (g1 - g2).norm(2).pow(2) * 0.5
    sig   = g_bar.norm(2).pow(2)
    return float("nan") if sig < 1e-20 else (noise / sig).item()


@torch.no_grad()
def obs_wnorm(model, init_norms):
    """Mean relative L2-norm change of trunk params since init."""
    drifts = [abs(p.data.norm(2).item() / w - 1.0)
              for (n, p), w in zip(
                  ((n, p) for n, p in model.named_parameters() if n in init_norms),
                  (init_norms[n] for n in init_norms))
              if w > 1e-12]
    return float(np.mean(drifts)) if drifts else float("nan")


@torch.no_grad()
def eval_acc(model, ds):
    model.eval()
    loader = DataLoader(ds, batch_size=256, shuffle=False, num_workers=0)
    c = tot = 0
    for X, y in loader:
        c += (model(X.to(DEVICE)).argmax(1) == y.to(DEVICE)).sum().item()
        tot += len(y)
    return c / tot if tot else 0.0

# ─── Analysis ─────────────────────────────────────────────────────────────────
def pava(vals, increasing=True):
    """Pool Adjacent Violators Algorithm (isotonic regression)."""
    v = [x if increasing else -x for x in vals]
    blocks = [[v[0], 1]]
    for x in v[1:]:
        blocks.append([x, 1])
        while len(blocks) >= 2 and blocks[-2][0] > blocks[-1][0]:
            b = blocks.pop(); a = blocks.pop()
            tot = a[1] + b[1]
            blocks.append([(a[0]*a[1] + b[0]*b[1]) / tot, tot])
    out = [m for m, c in blocks for _ in range(c)]
    return [x if increasing else -x for x in out]


def detect_collapse(accs):
    count = 0; first = None
    for t, a in enumerate(accs):
        if a < COLLAPSE_THRESH:
            count += 1
            if first is None: first = t
        else:
            count = 0; first = None
        if count >= COLLAPSE_MIN_TASKS:
            return first
    return None


def detect_onset(vals):
    clean = [v for v in vals if not (isinstance(v, float) and math.isnan(v))]
    if len(clean) < 3: return None
    v0, vf = clean[0], clean[-1]
    if abs(vf - v0) < 1e-10: return None
    thr = v0 + FIRE_FRAC * (vf - v0)
    inc = vf > v0
    sm = pava(clean, increasing=inc)
    for t, sv in enumerate(sm):
        if inc and sv >= thr: return t
        if not inc and sv <= thr: return t
    return None


def bci(data, n=BOOTSTRAP_N):
    data = [x for x in data if x is not None and not math.isnan(x)]
    if not data: return float("nan"), [float("nan"), float("nan")]
    a = np.array(data)
    means = [np.mean(np.random.choice(a, len(a), replace=True)) for _ in range(n)]
    return float(np.mean(a)), [float(np.percentile(means,2.5)),
                                float(np.percentile(means,97.5))]

# ─── Training ─────────────────────────────────────────────────────────────────
def run_one(cfg_name, cfg, tasks, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    model = MLP(use_bn=cfg["bn"]).to(DEVICE)
    init_norms = trunk_init_norms(model)
    criterion  = nn.CrossEntropyLoss()

    # Sanity: CE at init ≈ log(10)
    with torch.no_grad():
        model.eval()
        pX, py = tasks[0][2], tasks[0][3]
        init_ce = criterion(model(pX), py).item()
    exp_ce = math.log(N_CLASSES)
    ok = abs(init_ce - exp_ce) < 0.08
    print(f"  [sanity] CE_init={init_ce:.4f} exp={exp_ce:.4f} {'PASS' if ok else 'WARN'}")

    if cfg["opt"] == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=cfg["lr"],
                              momentum=cfg["momentum"], weight_decay=cfg["wd"])
    else:
        optimizer = optim.Adam(model.parameters(), lr=cfg["lr"],
                               weight_decay=cfg["wd"])

    accs = []; deads = []; eranks = []; gnss = []; wdrifts = []

    for t, (tr_ds, te_ds, probe_X, probe_y) in enumerate(tasks):
        loader = DataLoader(tr_ds, batch_size=BATCH_SIZE,
                            shuffle=True, drop_last=True, num_workers=0)
        it = iter(loader)
        model.train()
        for _ in range(STEPS_PER_TASK):
            try: X, y = next(it)
            except StopIteration: it = iter(loader); X, y = next(it)
            X, y = X.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            criterion(model(X), y).backward()
            optimizer.step()

        acc   = eval_acc(model, te_ds)
        dead  = obs_dead(model, probe_X)
        er    = obs_erank(model, probe_X)
        gns   = obs_gns(model, tr_ds, criterion)
        wd    = obs_wnorm(model, init_norms)

        accs.append(acc); deads.append(dead); eranks.append(er)
        gnss.append(gns); wdrifts.append(wd)
        print(f"    t={t+1:2d} acc={acc:.3f} dead={dead:.3f} "
              f"erank={er:.2f} gns={gns:.3f} wd={wd:.3f}", flush=True)

    t_col = detect_collapse(accs)
    lead_times = {}
    for obs_name, series in [("dead_unit_fraction", deads),
                              ("effective_rank",     eranks),
                              ("gradient_noise_scale", gnss),
                              ("weight_norm_drift",  wdrifts)]:
        tf = detect_onset(series)
        if t_col is not None and tf is not None:
            lead_times[obs_name] = t_col - tf
        else:
            lead_times[obs_name] = None

    return dict(config=cfg_name, seed=seed, sanity_ok=ok,
                accs=accs, dead_unit_fraction=deads, effective_rank=eranks,
                gradient_noise_scale=gnss, weight_norm_drift=wdrifts,
                t_collapse=t_col, lead_times=lead_times)

# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    t0 = time.time()
    os.makedirs(RESULTS_DIR, exist_ok=True)

    X_tr, y_tr, X_te, y_te = load_cifar10(DATASET_ROOT)
    tasks = build_permuted_tasks(X_tr, y_tr, X_te, y_te, rng_seed=42)

    all_rows = []
    config_results = {}
    rows_fh = open(ROWS_PATH, "w")

    for cfg_name, cfg in CONFIGS.items():
        print(f"\n{'='*68}\nCONFIG: {cfg_name}  {cfg}\n{'='*68}")
        sys.stdout.flush()
        seed_rows = []

        for seed in range(N_SEEDS):
            print(f"\n  --- SEED {seed} ---")
            sys.stdout.flush()
            try:
                row = run_one(cfg_name, cfg, tasks, seed)
            except Exception as e:
                import traceback; traceback.print_exc()
                row = dict(config=cfg_name, seed=seed, error=str(e))
            seed_rows.append(row); all_rows.append(row)
            rows_fh.write(json.dumps(row) + "\n"); rows_fh.flush()

        valid = [r for r in seed_rows
                 if "error" not in r and r.get("t_collapse") is not None]
        collapse_frac = len(valid) / N_SEEDS

        lead_by_obs = {}
        for obs in OBSERVABLES:
            lts = [r["lead_times"].get(obs) for r in valid]
            lts = [x for x in lts if x is not None]
            m, ci = bci(lts)
            lead_by_obs[obs] = dict(mean=round(m, 3),
                                    ci95=[round(ci[0],3), round(ci[1],3)],
                                    n_valid=len(lts))

        prec = sorted(OBSERVABLES,
                      key=lambda o: lead_by_obs[o]["mean"]
                      if not math.isnan(lead_by_obs[o]["mean"]) else -999,
                      reverse=True)

        config_results[cfg_name] = dict(
            collapse_frac=round(collapse_frac, 3),
            n_seeds_collapsed=len(valid),
            lead_times=lead_by_obs,
            precedence_order=prec,
        )

        print(f"\n  {cfg_name} SUMMARY: collapse {len(valid)}/{N_SEEDS}")
        for o in prec:
            lt = lead_by_obs[o]
            print(f"    {o:30s} lead={lt['mean']:+.2f}  CI=[{lt['ci95'][0]:+.2f},{lt['ci95'][1]:+.2f}]  n={lt['n_valid']}")
        sys.stdout.flush()

        # Write incremental RESULTS.json
        with open(RESULTS_PATH, "w") as f:
            json.dump(dict(status="RUNNING", configs_done=list(config_results.keys()),
                           configs=config_results), f, indent=2)

    rows_fh.close()

    # ── Overall analysis ───────────────────────────────────────────────────────
    def get_lts(cfg_prefix, obs):
        lts = []
        for r in all_rows:
            if r.get("config","").startswith(cfg_prefix) and r.get("t_collapse") is not None:
                v = r.get("lead_times",{}).get(obs)
                if v is not None: lts.append(v)
        return lts

    overall_means = {}
    for obs in OBSERVABLES:
        all_lts = [r.get("lead_times",{}).get(obs)
                   for r in all_rows
                   if r.get("t_collapse") is not None and r.get("lead_times",{}).get(obs) is not None]
        overall_means[obs] = float(np.mean(all_lts)) if all_lts else float("nan")

    overall_order = sorted(OBSERVABLES,
                           key=lambda o: overall_means[o] if not math.isnan(overall_means[o]) else -999,
                           reverse=True)

    # Optimizer-conditional GNS
    sgd_gns  = get_lts("sgd",  "gradient_noise_scale")
    adam_gns = get_lts("adam", "gradient_noise_scale")
    m_sgd,  ci_sgd  = bci(sgd_gns)
    m_adam, ci_adam = bci(adam_gns)
    diff = m_sgd - m_adam
    if sgd_gns and adam_gns:
        diff_samples = [
            np.mean(np.random.choice(sgd_gns,  len(sgd_gns),  True))
          - np.mean(np.random.choice(adam_gns, len(adam_gns), True))
            for _ in range(BOOTSTRAP_N)]
        diff_ci = [float(np.percentile(diff_samples, 2.5)),
                   float(np.percentile(diff_samples, 97.5))]
    else:
        diff_ci = [float("nan"), float("nan")]
    adam_supp = bool(diff > 0) if not math.isnan(diff) else None

    elapsed = time.time() - t0

    final = dict(
        status="SUCCESS",
        dataset="cifar10_permuted",
        n_seeds=N_SEEDS, n_tasks=N_TASKS,
        steps_per_task=STEPS_PER_TASK,
        scale="full",
        collapse_thresh=COLLAPSE_THRESH,
        configs=config_results,
        optimizer_conditional=dict(
            gns_lead_sgd=round(m_sgd, 3),  gns_lead_sgd_ci=[round(x,3) for x in ci_sgd],
            gns_lead_adam=round(m_adam,3), gns_lead_adam_ci=[round(x,3) for x in ci_adam],
            sgd_minus_adam_pp=round(diff, 3),
            ci95=[round(diff_ci[0],3), round(diff_ci[1],3)],
            adam_suppresses_gns=adam_supp,
        ),
        overall_precedence_order=overall_order,
        overall_lead_times={o: round(overall_means[o],3) for o in overall_order},
        metrics=dict(wall_clock_sec=round(elapsed,1), wall_clock_min=round(elapsed/60,2)),
        subject_executed=(
            f"3-layer MLP (400-400 ReLU), 4 configs × {N_SEEDS} seeds × "
            f"{N_TASKS} permuted-CIFAR-10 tasks × {STEPS_PER_TASK} steps/task. "
            f"Shared 10-class head (never reset). Permuted pixel inputs destroy "
            f"prior features each task, forcing new representations from degraded trunk. "
            f"Observables on pre-BN FC outputs. Collapse threshold = {COLLAPSE_THRESH:.2f}."
        ),
        notes=(
            f"Split-CIFAR-100 task-incremental tested first but produced 0/8 collapse "
            f"(fresh head per task lets even 90%-dead trunk succeed at 5-class). "
            f"Permuted-CIFAR-10 with shared head confirmed collapse in baseline_obs. "
            f"Rigorous run extends baseline to 4 configs × 8 seeds. "
            f"Overall ordering (True): {overall_order}. "
            f"Adam suppresses GNS: {adam_supp}."
        ),
    )

    with open(RESULTS_PATH, "w") as f:
        json.dump(final, f, indent=2)

    print(f"\n{'='*68}")
    print(f"DONE in {elapsed/60:.1f} min")
    print(f"Overall precedence: {overall_order}")
    print(f"GNS leads: SGD={m_sgd:+.2f} Adam={m_adam:+.2f} diff={diff:+.2f} adam_supp={adam_supp}")
    print(json.dumps(final, indent=2))


if __name__ == "__main__":
    main()
