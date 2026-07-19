#!/usr/bin/env python3
"""
run_gen.py — GENERALITY experiment (Phase 3 of the plasticity-precedence paper)
================================================================================
Repeats the VALIDATED precedence + predictive-AUC analysis (see run_valprec.py)
in NEW regimes, to test whether the finding generalises beyond the single
hidden=100 MLP / permuted-MNIST setup the main result used:

  dead_unit_fraction + effective_rank LEAD and PREDICT plasticity collapse;
  gradient_noise_scale FAILS as a predictor.

Which regime is selected by the env var SH_EXP:

  g1_fashion : 2ND DATASET      — online Permuted-*Fashion*-MNIST, MLP hidden=100
  g2_narrow  : 2ND ARCHITECTURE — online Permuted-MNIST, narrower MLP hidden=50
  g3_cnn     : 2ND ARCH FAMILY  — online *Label-permuted* MNIST, small ConvNet
               (input images unchanged, target labels permuted each task — a
                spatially-meaningful continual stressor, so a CNN is a genuine
                architecture generalisation, not a permuted-pixel degenerate.)

Everything downstream of the trajectory (collapse detection, onset, lead times,
bootstrap CIs, paired erank-vs-GNS Wilcoxon, predictive AUC k=5, threshold
sensitivity, reproducibility table) is the SAME code as run_valprec.py so the
comparison is apples-to-apples.

Outputs (exp = SH_EXP):
  results/<exp>/RESULTS.json   (incremental + final)
  results/<exp>/trajectories.json
  results/<exp>/LOG.md
"""

import os, sys, json, math, time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from scipy.stats import wilcoxon as scipy_wilcoxon
from sklearn.metrics import roc_auc_score

# ─── Regime selection ──────────────────────────────────────────────────────────
EXP = os.environ.get("SH_EXP", "g1_fashion").strip()

CONFIGS = {
    # 2nd dataset: permuted Fashion-MNIST, same MLP as the main result
    "g1_fashion": dict(
        family="mlp", dataset="fashion", task="perm_input",
        hidden=100, lr=0.10, n_seeds=6, n_tasks=200,
        label="2nd dataset — online Permuted-Fashion-MNIST, MLP hidden=100",
    ),
    # 2nd architecture size: narrower MLP, permuted MNIST
    "g2_narrow": dict(
        family="mlp", dataset="mnist", task="perm_input",
        hidden=50, lr=0.10, n_seeds=6, n_tasks=200,
        label="2nd architecture — online Permuted-MNIST, narrower MLP hidden=50",
    ),
    # 2nd architecture family: small ConvNet, label-permuted MNIST
    "g3_cnn": dict(
        family="cnn", dataset="mnist", task="perm_label",
        hidden=64, lr=0.05, n_seeds=5, n_tasks=150,
        label="2nd arch family — online Label-permuted MNIST, small ConvNet",
    ),
}
if EXP not in CONFIGS:
    print(f"FATAL: unknown SH_EXP={EXP!r}; valid: {list(CONFIGS)}", flush=True)
    sys.exit(2)
C = dict(CONFIGS[EXP])
if os.environ.get("SH_SMOKE") == "1":
    C["n_seeds"], C["n_tasks"] = 1, 4   # fast CPU sanity run

RESULTS_DIR  = f"results/{EXP}"
RESULTS_PATH = os.path.join(RESULTS_DIR, "RESULTS.json")
TRAJ_PATH    = os.path.join(RESULTS_DIR, "trajectories.json")
LOG_PATH     = os.path.join(RESULTS_DIR, "LOG.md")
DATA_ROOT    = "/tmp/tvdata"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs("_weights", exist_ok=True)

# ─── Fixed hyperparameters (matched to run_valprec.py) ─────────────────────────
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HIDDEN         = C["hidden"]
N_SEEDS        = C["n_seeds"]
N_TASKS        = C["n_tasks"]
LR             = C["lr"]
STEPS_PER_TASK = 200
BATCH_SIZE     = 128
PROBE_SIZE     = 1000
N_CLASSES      = 10
IN_DIM         = 784
MOMENTUM       = 0.9
WEIGHT_DECAY   = 0.0

COLLAPSE_THRESH_PP = 20.0
COLLAPSE_MIN_TASKS = 2
GNS_BATCHES   = 20
GNS_BATCH     = 64
ONSET_FRACS   = [0.30, 0.50, 0.70]
ONSET_WINDOW  = 2
N_BOOT        = 2000
BOOT_ALPHA    = 0.05
LOOKAHEAD_K   = 5
DATA_SEED     = 42

print(f"EXP:      {EXP} — {C['label']}", flush=True)
print(f"Device:   {DEVICE}", flush=True)
print(f"Config:   family={C['family']} dataset={C['dataset']} task={C['task']} "
      f"hidden={HIDDEN} lr={LR} {N_SEEDS}seeds x {N_TASKS}tasks", flush=True)


# ─── Models ────────────────────────────────────────────────────────────────────
class MLP(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.fc1  = nn.Linear(IN_DIM, hidden)
        self.fc2  = nn.Linear(hidden, hidden)
        self.head = nn.Linear(hidden, N_CLASSES)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.head(self.relu(self.fc2(self.relu(self.fc1(x)))))

    def probe_activations(self, x):
        """Return [pre-pool activation tensors [N, F]] for every ReLU layer.
        Last element is the penultimate representation used for effective rank."""
        h1 = self.relu(self.fc1(x))
        h2 = self.relu(self.fc2(h1))
        return [h1, h2]


class SmallCNN(nn.Module):
    """Small ConvNet for 28x28x1 MNIST. Two conv blocks + one FC hidden layer."""
    def __init__(self, ch=64):
        super().__init__()
        self.c1   = nn.Conv2d(1, ch, 3, padding=1)
        self.c2   = nn.Conv2d(ch, ch, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.fc1  = nn.Linear(ch * 7 * 7, 128)
        self.head = nn.Linear(128, N_CLASSES)
        self.relu = nn.ReLU()

    def _feat(self, x):
        x = x.view(-1, 1, 28, 28)
        a1 = self.relu(self.c1(x))          # [N, ch, 28, 28]
        p1 = self.pool(a1)
        a2 = self.relu(self.c2(p1))         # [N, ch, 14, 14]
        p2 = self.pool(a2)                  # [N, ch, 7, 7]
        flat = p2.flatten(1)
        a3 = self.relu(self.fc1(flat))      # [N, 128]
        return a1, a2, a3

    def forward(self, x):
        _, _, a3 = self._feat(x)
        return self.head(a3)

    def probe_activations(self, x):
        a1, a2, a3 = self._feat(x)
        # spatially average conv maps -> per-channel activation [N, C]
        g1 = a1.mean(dim=(2, 3))
        g2 = a2.mean(dim=(2, 3))
        return [g1, g2, a3]   # a3 (penultimate FC) used for effective rank


def build_model():
    if C["family"] == "cnn":
        return SmallCNN(HIDDEN).to(DEVICE)
    return MLP(HIDDEN).to(DEVICE)


# ─── Data ──────────────────────────────────────────────────────────────────────
def load_data():
    if C["dataset"] == "fashion":
        ds_tr = torchvision.datasets.FashionMNIST(DATA_ROOT, train=True,  download=True)
        ds_te = torchvision.datasets.FashionMNIST(DATA_ROOT, train=False, download=True)
    else:
        ds_tr = torchvision.datasets.MNIST(DATA_ROOT, train=True,  download=True)
        ds_te = torchvision.datasets.MNIST(DATA_ROOT, train=False, download=True)
    X_tr = ds_tr.data.numpy().reshape(-1, IN_DIM).astype(np.float32) / 255.0
    y_tr = ds_tr.targets.numpy().astype(np.int64)
    X_te = ds_te.data.numpy().reshape(-1, IN_DIM).astype(np.float32) / 255.0
    y_te = ds_te.targets.numpy().astype(np.int64)
    print(f"Data loaded ({C['dataset']}): train={X_tr.shape}, test={X_te.shape}", flush=True)
    return X_tr, y_tr, X_te, y_te


def make_input_perms(n_tasks, seed=DATA_SEED):
    rng = np.random.default_rng(seed)
    perms = [np.arange(IN_DIM)]
    for _ in range(n_tasks - 1):
        perms.append(rng.permutation(IN_DIM))
    return perms


def make_label_perms(n_tasks, seed=DATA_SEED):
    rng = np.random.default_rng(seed)
    perms = [np.arange(N_CLASSES)]
    for _ in range(n_tasks - 1):
        perms.append(rng.permutation(N_CLASSES))
    return perms


def apply_task(X_tr, y_tr, X_te, y_te, t_idx, iperms, lperms):
    """Return (X_tr_task, y_tr_task, X_te_task, y_te_task) for task t_idx."""
    if C["task"] == "perm_input":
        p = iperms[t_idx]
        return X_tr[:, p], y_tr, X_te[:, p], y_te
    else:  # perm_label — images unchanged, labels remapped
        lp = lperms[t_idx]
        return X_tr, lp[y_tr], X_te, lp[y_te]


# ─── Training (matches valprec global-state pattern) ───────────────────────────
def train_task(model, optimizer, X_task, y_task, criterion, steps, bs):
    model.train()
    N   = len(y_task)
    Xg  = torch.from_numpy(np.ascontiguousarray(X_task)).to(DEVICE)
    yg  = torch.from_numpy(np.ascontiguousarray(y_task)).to(DEVICE)
    idx = np.arange(N)
    np.random.shuffle(idx)
    ptr = 0
    for _ in range(steps):
        if ptr + bs > N:
            np.random.shuffle(idx)
            ptr = 0
        b = idx[ptr:ptr + bs]
        ptr += bs
        optimizer.zero_grad()
        criterion(model(Xg[b]), yg[b]).backward()
        optimizer.step()


# ─── Observables ───────────────────────────────────────────────────────────────
@torch.no_grad()
def compute_dead(model, probe_X):
    """Dead-unit fraction: fraction of units inactive on >95% of probe, averaged
    over ALL ReLU layers (conv channels are spatially averaged first)."""
    model.eval()
    acts = model.probe_activations(probe_X)
    ds = []
    for a in acts:
        ds.append(((a <= 0).float().mean(0) > 0.95).float().mean().item())
    return float(np.mean(ds))


@torch.no_grad()
def compute_erank(model, probe_X):
    """Effective rank exp(H(singular-value distribution)) of penultimate activations."""
    model.eval()
    a = model.probe_activations(probe_X)[-1].float()
    try:
        S = torch.linalg.svdvals(a)
        S = S[S > 1e-10]
        if len(S) == 0:
            return 1.0
        p = S / S.sum()
        H = -(p * torch.log(p + 1e-12)).sum().item()
        return max(1.0, math.exp(H))
    except Exception:
        return 1.0


def compute_gns(model, X_task, y_task, criterion):
    """McCandlish B_opt = tr(Sigma)/||g_mean||^2 over GNS_BATCHES mini-batches.
    One np.random.permutation call (matches valprec global-state advancement)."""
    model.train()
    N = len(y_task)
    n, b = GNS_BATCHES, GNS_BATCH
    if N < n * b:
        return float("nan")
    idx = np.random.permutation(N)
    Xg  = torch.from_numpy(np.ascontiguousarray(X_task)).to(DEVICE)
    yg  = torch.from_numpy(np.ascontiguousarray(y_task)).to(DEVICE)
    grads = []
    for i in range(n):
        bidx = idx[i * b:(i + 1) * b]
        model.zero_grad()
        criterion(model(Xg[bidx]), yg[bidx]).backward()
        g = torch.cat([p.grad.detach().clone().flatten()
                       for p in model.parameters() if p.grad is not None])
        grads.append(g)
    model.zero_grad()
    G      = torch.stack(grads, dim=0)
    g_mean = G.mean(dim=0)
    dev    = G - g_mean.unsqueeze(0)
    tr_sig = dev.pow(2).sum(dim=1).sum().item() / (n - 1)
    signal = g_mean.pow(2).sum().item()
    if signal < 1e-20 or not math.isfinite(tr_sig):
        return float("nan")
    val = tr_sig / signal
    return float(val) if math.isfinite(val) and val > 0 else float("nan")


@torch.no_grad()
def compute_wdrift(model, init_norms):
    drifts = []
    for (name, p), w0 in zip(model.named_parameters(), init_norms.values()):
        if w0 > 1e-12:
            drifts.append(abs(p.data.norm(2).item() / w0 - 1.0))
    return float(np.mean(drifts)) if drifts else float("nan")


@torch.no_grad()
def eval_acc(model, X_te_task, y_te_task):
    model.eval()
    Xg = torch.from_numpy(np.ascontiguousarray(X_te_task)).to(DEVICE)
    yg = torch.from_numpy(np.ascontiguousarray(y_te_task)).to(DEVICE)
    correct = 0
    N = len(y_te_task)
    for i in range(0, N, 1024):
        correct += (model(Xg[i:i+1024]).argmax(1) == yg[i:i+1024]).sum().item()
    return correct / N if N > 0 else 0.0


def make_probe(X_te_task, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X_te_task), size=min(PROBE_SIZE, len(X_te_task)), replace=False)
    return torch.from_numpy(np.ascontiguousarray(X_te_task[idx])).to(DEVICE)


# ─── Analysis helpers (identical to run_valprec.py) ────────────────────────────
def detect_collapse(accs, thresh_pp=COLLAPSE_THRESH_PP, min_tasks=COLLAPSE_MIN_TASKS):
    if len(accs) < min_tasks + 1:
        return None
    ref, count, first = accs[0], 0, None
    for t, a in enumerate(accs):
        if (ref - a) * 100.0 >= thresh_pp:
            count += 1
            if first is None:
                first = t
        else:
            count, first = 0, None
        if count >= min_tasks:
            return first
    return None


def moving_avg(vals, window=ONSET_WINDOW):
    out = []
    for i in range(len(vals)):
        chunk = [v for v in vals[max(0, i - window + 1):i + 1] if math.isfinite(v)]
        out.append(float(np.mean(chunk)) if chunk else float("nan"))
    return out


def compute_onset(vals_raw, frac=0.50, window=ONSET_WINDOW):
    vals = moving_avg(vals_raw, window)
    clean = [(i, v) for i, v in enumerate(vals) if math.isfinite(v)]
    if len(clean) < 4:
        return None
    v0, vf = clean[0][1], clean[-1][1]
    if abs(vf - v0) < 1e-10:
        return None
    thr = v0 + frac * (vf - v0)
    direction = 1 if vf > v0 else -1
    for i, v in clean:
        if direction == 1 and v >= thr:
            return i
        if direction == -1 and v <= thr:
            return i
    return None


def bootstrap_ci(data, n_boot=N_BOOT, alpha=BOOT_ALPHA, seed=99):
    arr = np.array([x for x in data if x is not None and math.isfinite(float(x))], dtype=float)
    if len(arr) == 0:
        return [None, None]
    rng = np.random.default_rng(seed)
    boots = [float(np.mean(rng.choice(arr, size=len(arr), replace=True))) for _ in range(n_boot)]
    return [float(np.percentile(boots, 100 * alpha / 2)),
            float(np.percentile(boots, 100 * (1 - alpha / 2)))]


def compute_predictive_auc(recs, obs_key, k=LOOKAHEAD_K):
    features, labels = [], []
    for r in recs:
        accs, vals = r["accs"], r[obs_key]
        thresh = accs[0] - COLLAPSE_THRESH_PP / 100.0
        T = len(accs)
        for t in range(T - 1):
            v = vals[t]
            if not math.isfinite(v):
                continue
            end = min(t + k + 1, T)
            lbl = int(any(accs[tt] < thresh for tt in range(t + 1, end)))
            features.append(v)
            labels.append(lbl)
    if len(labels) < 10 or len(set(labels)) < 2:
        return float("nan")
    feats = np.array(features, dtype=float)
    lbls  = np.array(labels, dtype=int)
    if obs_key == "erank":
        feats = -feats
    try:
        return float(roc_auc_score(lbls, feats))
    except Exception:
        return float("nan")


def _j(x):
    if x is None:
        return None
    try:
        v = float(x)
        return None if not math.isfinite(v) else v
    except Exception:
        return x


def _jlist(lst):
    return [_j(x) for x in lst] if lst is not None else None


def save_partial(recs, seed, task):
    partial = {"status": "RUNNING", "exp": EXP, "current_seed": seed,
               "current_task": task, "per_seed_so_far": []}
    for r in recs:
        n = len(r["accs"])
        partial["per_seed_so_far"].append(dict(
            seed=r["seed"], tasks_done=n, task1_acc=r.get("task1_acc"),
            dead_after_task1=r.get("dead_after_task1"), healthy=r.get("healthy", False),
            t_collapse=r["t_collapse"],
            acc_last20=(float(np.mean(r["accs"][-20:])) if n >= 20 else None)))
    with open(RESULTS_PATH, "w") as f:
        json.dump(partial, f, indent=2)


# ─── Main ──────────────────────────────────────────────────────────────────────
def main():
    t0 = time.time()
    with open(LOG_PATH, "w") as f:
        f.write(f"# LOG — generality round {EXP}\n\n{C['label']}\n\n")
        f.write("Repeats validated precedence+AUC analysis (run_valprec.py) in a new regime.\n")
        f.write(f"Started {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

    X_tr, y_tr, X_te, y_te = load_data()
    criterion = nn.CrossEntropyLoss()
    iperms = make_input_perms(N_TASKS)
    lperms = make_label_perms(N_TASKS)

    obs_keys  = ["dead", "erank", "gns", "wdrift"]
    obs_names = ["dead_unit_fraction", "effective_rank", "gradient_noise_scale", "weight_norm_drift"]

    recs, trajectories = [], {}
    for seed in range(N_SEEDS):
        print(f"\n{'='*66}\nSEED {seed}/{N_SEEDS-1}  (elapsed {time.time()-t0:.0f}s)\n{'='*66}", flush=True)
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = build_model()
        optimizer = optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
        init_norms = {n: p.data.norm(2).item() for n, p in model.named_parameters()}

        Xtr0, ytr0, Xte0, yte0 = apply_task(X_tr, y_tr, X_te, y_te, 0, iperms, lperms)
        dead_init = compute_dead(model, make_probe(Xte0, seed=seed))
        print(f"  dead_init={dead_init:.4f}", flush=True)

        rec = dict(seed=seed, dead_at_init=dead_init, dead_after_task1=None,
                   task1_acc=None, healthy=False, t_collapse=None,
                   accs=[], dead=[], erank=[], gns=[], wdrift=[])
        recs.append(rec)

        t_seed = time.time()
        for t_idx in range(N_TASKS):
            Xtr_t, ytr_t, Xte_t, yte_t = apply_task(X_tr, y_tr, X_te, y_te, t_idx, iperms, lperms)
            train_task(model, optimizer, Xtr_t, ytr_t, criterion, STEPS_PER_TASK, BATCH_SIZE)
            pr_X = make_probe(Xte_t, seed=seed * 10_000 + t_idx)
            acc  = eval_acc(model, Xte_t, yte_t)
            dead = compute_dead(model, pr_X)
            er   = compute_erank(model, pr_X)
            gns  = compute_gns(model, Xtr_t, ytr_t, criterion)
            wd   = compute_wdrift(model, init_norms)
            rec["accs"].append(acc); rec["dead"].append(dead); rec["erank"].append(er)
            rec["gns"].append(gns);  rec["wdrift"].append(wd)
            if t_idx == 0:
                rec["dead_after_task1"] = dead
                rec["task1_acc"] = acc
                rec["healthy"] = (acc > 0.70) and (dead < 0.25)
            rec["t_collapse"] = detect_collapse(rec["accs"])
            if t_idx < 3 or (t_idx + 1) % 25 == 0 or t_idx == N_TASKS - 1:
                print(f"  Task {t_idx+1:3d}/{N_TASKS} acc={acc:.4f} dead={dead:.4f} "
                      f"erank={er:7.3f} gns={gns:8.3f} wd={wd:.4f} [{time.time()-t_seed:.0f}s]", flush=True)
            if (t_idx + 1) % 50 == 0 or t_idx == N_TASKS - 1:
                save_partial(recs, seed, t_idx + 1)

        f20 = float(np.mean(rec["accs"][:20])); l20 = float(np.mean(rec["accs"][-20:]))
        print(f"  Seed {seed} DONE dead_t1={rec['dead_after_task1']:.4f} "
              f"task1_acc={rec['task1_acc']:.4f} drop={ (f20-l20)*100:.2f}pp "
              f"t_collapse={rec['t_collapse']}", flush=True)
        trajectories[f"seed_{seed}"] = {k: rec[k] for k in ["accs", "dead", "erank", "gns", "wdrift"]}
        with open(TRAJ_PATH, "w") as f:
            json.dump(trajectories, f)

    # ── Analysis ────────────────────────────────────────────────────────────────
    task1_accs = [r["task1_acc"] for r in recs]
    dead_t1s   = [r["dead_after_task1"] for r in recs]
    healthy_all = all(r["healthy"] for r in recs)
    first20 = [float(np.mean(r["accs"][:20])) for r in recs]
    last20  = [float(np.mean(r["accs"][-20:])) for r in recs]
    acc_drops = [(f - l) * 100.0 for f, l in zip(first20, last20)]
    mean_drop = float(np.mean(acc_drops))
    plasticity = mean_drop > 5.0
    t_collapses = [r["t_collapse"] for r in recs]
    n_collapsed = sum(t is not None for t in t_collapses)

    lead_per_obs = {k: [] for k in obs_keys}
    for r in recs:
        tc = r["t_collapse"]
        for k in obs_keys:
            if tc is None:
                lead_per_obs[k].append(None)
            else:
                onset = compute_onset(r[k], frac=0.50)
                lead_per_obs[k].append((tc - onset) if onset is not None else None)

    lead_summary = {}
    for k, nm in zip(obs_keys, obs_names):
        valid = [x for x in lead_per_obs[k] if x is not None]
        if valid:
            arr = np.array(valid, float)
            lead_summary[nm] = dict(mean=float(arr.mean()), median=float(np.median(arr)),
                                    iqr=[float(np.percentile(arr, 25)), float(np.percentile(arr, 75))],
                                    ci95=bootstrap_ci(valid, seed=42))
        else:
            lead_summary[nm] = dict(mean=None, median=None, iqr=[None, None], ci95=[None, None])

    ranked = [(nm, lead_summary[nm]["median"]) for nm in obs_names if lead_summary[nm]["median"] is not None]
    ranked.sort(key=lambda x: x[1], reverse=True)
    precedence_order = [x[0] for x in ranked]

    paired = [(e, g) for e, g in zip(lead_per_obs["erank"], lead_per_obs["gns"]) if e is not None and g is not None]
    if len(paired) >= 4:
        diff = np.array([e - g for e, g in paired], float)
        median_diff = float(np.median(diff))
        try:
            _, w_p = scipy_wilcoxon(diff)
        except Exception:
            w_p = float("nan")
    else:
        median_diff, w_p = float("nan"), float("nan")

    predictive_auc = {}
    for k, nm in zip(obs_keys, obs_names):
        predictive_auc[nm] = _j(compute_predictive_auc(recs, k))

    thresh_orders = {}
    for frac in ONSET_FRACS:
        meds = {}
        for k, nm in zip(obs_keys, obs_names):
            leads = []
            for r in recs:
                tc = r["t_collapse"]
                if tc is None:
                    continue
                onset = compute_onset(r[k], frac=frac)
                if onset is not None:
                    leads.append(tc - onset)
            meds[nm] = float(np.median(leads)) if leads else None
        srt = [(nm, meds[nm]) for nm in obs_names if meds[nm] is not None]
        srt.sort(key=lambda x: x[1], reverse=True)
        thresh_orders[f"{int(frac*100)}pct"] = [x[0] for x in srt]

    results = {
        "status": "DONE", "exp": EXP, "regime": C["label"],
        "family": C["family"], "dataset": C["dataset"], "task": C["task"],
        "hidden": HIDDEN, "lr": LR, "n_seeds": N_SEEDS, "n_tasks": N_TASKS,
        "healthy": bool(healthy_all),
        "mean_dead_after_task1": _j(float(np.mean(dead_t1s))),
        "mean_task1_acc": _j(float(np.mean(task1_accs))),
        "mean_acc_drop_pp": _j(mean_drop),
        "plasticity_loss_confirmed": bool(plasticity),
        "n_collapsed": int(n_collapsed),
        "lead_times": {nm: {"mean": _j(lead_summary[nm]["mean"]), "median": _j(lead_summary[nm]["median"]),
                            "iqr": _jlist(lead_summary[nm]["iqr"]), "ci95": _jlist(lead_summary[nm]["ci95"])}
                       for nm in obs_names},
        "precedence_order": precedence_order,
        "erank_vs_gns_paired": {"median_diff": _j(median_diff), "wilcoxon_p": _j(w_p)},
        "predictive_auc": predictive_auc,
        "threshold_sensitivity": thresh_orders,
        "per_seed": [dict(seed=r["seed"], task1_acc=_j(r["task1_acc"]),
                          dead_after_task1=_j(r["dead_after_task1"]), healthy=r["healthy"],
                          t_collapse=r["t_collapse"],
                          acc_drop_pp=_j((np.mean(r["accs"][:20]) - np.mean(r["accs"][-20:])) * 100.0))
                     for r in recs],
        "total_wall_time_sec": float(time.time() - t0),
        "subject_executed": C["label"],
        "metrics": {
            "regime": C["label"],
            "healthy": bool(healthy_all),
            "plasticity_loss_confirmed": bool(plasticity),
            "mean_acc_drop_pp": _j(mean_drop),
            "predictive_auc": predictive_auc,
            "precedence_order": precedence_order,
        },
        "notes": (f"Generality of the plasticity-precedence finding in regime {EXP}. "
                  "Same analysis as run_valprec.py. Reports honestly whether dead+erank "
                  "still lead/predict and GNS still fails."),
    }
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    with open(LOG_PATH, "a") as f:
        f.write(f"\n## Results\nhealthy={healthy_all} plasticity={plasticity} "
                f"drop={mean_drop:.2f}pp n_collapsed={n_collapsed}/{N_SEEDS}\n")
        f.write(f"precedence={precedence_order}\npredictive_auc={predictive_auc}\n")
    print(f"\n{'='*66}\nDONE {EXP} in {time.time()-t0:.0f}s", flush=True)
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
