#!/usr/bin/env python3
"""
run_eos.py — Does CURVATURE (sharpness / Edge-of-Stability departure) predict
plasticity collapse, and how does it rank against representation-statistics
signals?  (Idea 2, goal-driven)
=============================================================================
Adds a Hessian-based observable to the validated healthy-collapse regime
(hidden=100 MLP, online Permuted-MNIST, SGD lr=0.10, dead_t1~6.7%):

  S(t)      = top Hessian eigenvalue (sharpness), via power iteration with
              Hessian-vector products on a fixed probe batch of the current task.
  eos(t)    = eta * S(t), the Edge-of-Stability product (EoS regime ~ 2; a value
              falling well below 2 means the loss landscape is flattening — the
              network is *leaving* the Edge of Stability).

We track S(t) and eos(t) alongside dead-unit fraction and effective rank, and
compute the SAME within-trajectory 5-task-ahead predictive AUC + lead-time
analysis as the main precedence study, to answer:
  (Q1) does sharpness predict collapse (AUC), and
  (Q2) does it LEAD the representation signals (dead-unit, effective rank)?

Outputs: results/eos/RESULTS.json (+ trajectories, LOG).
"""
import os, sys, json, math, time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from scipy.stats import wilcoxon as scipy_wilcoxon
from sklearn.metrics import roc_auc_score

_SUF = "_smoke" if os.environ.get("SH_SMOKE") == "1" else ""
RESULTS_DIR  = f"results/eos{_SUF}"
RESULTS_PATH = os.path.join(RESULTS_DIR, "RESULTS.json")
TRAJ_PATH    = os.path.join(RESULTS_DIR, "trajectories.json")
LOG_PATH     = os.path.join(RESULTS_DIR, "LOG.md")
MNIST_ROOT   = "/tmp/mnist"
os.makedirs(RESULTS_DIR, exist_ok=True); os.makedirs("_weights", exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HIDDEN, N_SEEDS, N_TASKS = 100, 6, 250
STEPS_PER_TASK, BATCH_SIZE, PROBE_SIZE = 200, 128, 1000
N_CLASSES, IN_DIM = 10, 784
LR, MOMENTUM, WEIGHT_DECAY = 0.10, 0.9, 0.0
DATA_SEED = 42
COLLAPSE_THRESH_PP, COLLAPSE_MIN_TASKS = 20.0, 2
ONSET_FRACS, ONSET_WINDOW = [0.30, 0.50, 0.70], 2
N_BOOT, LOOKAHEAD_K = 2000, 5
HESS_ITERS = 12          # power-iteration steps for top Hessian eigenvalue
HESS_BATCH = 512         # fixed batch size for the Hessian estimate
if os.environ.get("SH_SMOKE") == "1":
    N_SEEDS, N_TASKS = 1, 6

print(f"EOS predictor | hidden={HIDDEN} {N_SEEDS}seeds x {N_TASKS}tasks lr={LR} "
      f"| sharpness via {HESS_ITERS}-step power iteration", flush=True)


class MLP(nn.Module):
    def __init__(self, hidden=100):
        super().__init__()
        self.fc1 = nn.Linear(IN_DIM, hidden); self.fc2 = nn.Linear(hidden, hidden)
        self.head = nn.Linear(hidden, N_CLASSES); self.relu = nn.ReLU()
    def forward(self, x): return self.head(self.relu(self.fc2(self.relu(self.fc1(x)))))
    def penultimate(self, x):
        h1 = self.relu(self.fc1(x)); h2 = self.relu(self.fc2(h1)); return h1, h2


def load_mnist():
    tr = torchvision.datasets.MNIST(MNIST_ROOT, train=True, download=True)
    te = torchvision.datasets.MNIST(MNIST_ROOT, train=False, download=True)
    return (tr.data.numpy().reshape(-1, IN_DIM).astype(np.float32)/255.0, tr.targets.numpy().astype(np.int64),
            te.data.numpy().reshape(-1, IN_DIM).astype(np.float32)/255.0, te.targets.numpy().astype(np.int64))


def make_perms(n, seed=DATA_SEED):
    rng = np.random.default_rng(seed); perms = [np.arange(IN_DIM)]
    for _ in range(n-1): perms.append(rng.permutation(IN_DIM))
    return perms


def train_task(model, opt, X, y, crit, steps, bs):
    model.train(); N = len(y)
    Xg = torch.from_numpy(X).to(DEVICE); yg = torch.from_numpy(y).to(DEVICE)
    idx = np.arange(N); np.random.shuffle(idx); ptr = 0
    for _ in range(steps):
        if ptr+bs > N: np.random.shuffle(idx); ptr = 0
        b = idx[ptr:ptr+bs]; ptr += bs
        opt.zero_grad(); crit(model(Xg[b]), yg[b]).backward(); opt.step()


@torch.no_grad()
def compute_dead(model, probe_X):
    model.eval(); h1, h2 = model.penultimate(probe_X)
    d1 = ((h1 <= 0).float().mean(0) > 0.95).float().mean().item()
    d2 = ((h2 <= 0).float().mean(0) > 0.95).float().mean().item()
    return (d1+d2)/2.0


@torch.no_grad()
def compute_erank(model, probe_X):
    model.eval(); _, h2 = model.penultimate(probe_X)
    try:
        S = torch.linalg.svdvals(h2.float()); S = S[S > 1e-10]
        if len(S) == 0: return 1.0
        p = S/S.sum(); return max(1.0, math.exp(-(p*torch.log(p+1e-12)).sum().item()))
    except Exception:
        return 1.0


def compute_sharpness(model, X, y, crit, iters=HESS_ITERS):
    """Top Hessian eigenvalue via power iteration with Hessian-vector products.
    Hv computed by double backprop on a fixed probe batch of the current task."""
    model.eval()
    params = [p for p in model.parameters() if p.requires_grad]
    Xg = torch.from_numpy(X).to(DEVICE); yg = torch.from_numpy(y).to(DEVICE)
    loss = crit(model(Xg), yg)
    grads = torch.autograd.grad(loss, params, create_graph=True)
    # random init direction
    v = [torch.randn_like(p) for p in params]
    nrm = math.sqrt(sum((vi*vi).sum().item() for vi in v)) + 1e-12
    v = [vi/nrm for vi in v]
    lam = 0.0
    for _ in range(iters):
        Hv = torch.autograd.grad(grads, params, grad_outputs=v, retain_graph=True)
        lam = float(sum((hv*vi).sum().item() for hv, vi in zip(Hv, v)))
        nrm = math.sqrt(sum((hv*hv).sum().item() for hv in Hv)) + 1e-12
        v = [hv.detach()/nrm for hv in Hv]
    model.zero_grad()
    return max(0.0, lam)


@torch.no_grad()
def eval_acc(model, X, y):
    model.eval(); Xg = torch.from_numpy(X).to(DEVICE); yg = torch.from_numpy(y).to(DEVICE)
    c = 0
    for i in range(0, len(y), 1024):
        c += (model(Xg[i:i+1024]).argmax(1) == yg[i:i+1024]).sum().item()
    return c/len(y)


def make_probe(X, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=min(PROBE_SIZE, len(X)), replace=False)
    return torch.from_numpy(X[idx]).to(DEVICE), idx


# ── analysis (identical to run_valprec) ──
def detect_collapse(accs, thresh_pp=COLLAPSE_THRESH_PP, min_tasks=COLLAPSE_MIN_TASKS):
    if len(accs) < min_tasks+1: return None
    ref, count, first = accs[0], 0, None
    for t, a in enumerate(accs):
        if (ref-a)*100.0 >= thresh_pp:
            count += 1
            if first is None: first = t
        else: count, first = 0, None
        if count >= min_tasks: return first
    return None

def moving_avg(vals, window=ONSET_WINDOW):
    out = []
    for i in range(len(vals)):
        chunk = [v for v in vals[max(0, i-window+1):i+1] if math.isfinite(v)]
        out.append(float(np.mean(chunk)) if chunk else float("nan"))
    return out

def compute_onset(vals_raw, frac=0.50, window=ONSET_WINDOW):
    vals = moving_avg(vals_raw, window)
    clean = [(i, v) for i, v in enumerate(vals) if math.isfinite(v)]
    if len(clean) < 4: return None
    v0, vf = clean[0][1], clean[-1][1]
    if abs(vf-v0) < 1e-10: return None
    thr = v0 + frac*(vf-v0); direction = 1 if vf > v0 else -1
    for i, v in clean:
        if direction == 1 and v >= thr: return i
        if direction == -1 and v <= thr: return i
    return None

def bootstrap_ci(data, n_boot=N_BOOT, seed=99):
    arr = np.array([x for x in data if x is not None and math.isfinite(float(x))], float)
    if len(arr) == 0: return [None, None]
    rng = np.random.default_rng(seed)
    boots = [float(np.mean(rng.choice(arr, len(arr), replace=True))) for _ in range(n_boot)]
    return [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]

def predictive_auc(recs, key, invert, k=LOOKAHEAD_K):
    feats, labs = [], []
    for r in recs:
        accs, vals = r["accs"], r[key]; thr = accs[0] - COLLAPSE_THRESH_PP/100.0; T = len(accs)
        for t in range(T-1):
            v = vals[t]
            if not math.isfinite(v): continue
            lbl = int(any(accs[tt] < thr for tt in range(t+1, min(t+k+1, T))))
            feats.append(v); labs.append(lbl)
    if len(labs) < 10 or len(set(labs)) < 2: return float("nan")
    f = np.array(feats, float)
    if invert: f = -f
    try: return float(roc_auc_score(np.array(labs, int), f))
    except Exception: return float("nan")

def _j(x):
    if x is None: return None
    try:
        v = float(x); return None if not math.isfinite(v) else v
    except Exception: return x


def main():
    t0 = time.time()
    with open(LOG_PATH, "w") as f:
        f.write(f"# LOG — EOS/sharpness predictor\nStarted {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    X_tr, y_tr, X_te, y_te = load_mnist()
    crit = nn.CrossEntropyLoss(); perms = make_perms(N_TASKS)

    # observables: dead, erank, sharpness S, eos = eta*S  (+ inverse sharpness for AUC direction)
    obs = ["dead", "erank", "sharp", "eos"]
    obs_names = ["dead_unit_fraction", "effective_rank", "sharpness", "eos_product"]
    invert = {"dead": False, "erank": True, "sharp": True, "eos": True}  # lower sharp/eos/erank -> collapse

    recs, trajectories = [], {}
    for seed in range(N_SEEDS):
        print(f"\n{'='*60}\nSEED {seed} ({time.time()-t0:.0f}s)\n{'='*60}", flush=True)
        torch.manual_seed(seed); np.random.seed(seed)
        model = MLP(HIDDEN).to(DEVICE)
        opt = optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
        rec = dict(seed=seed, task1_acc=None, dead_after_task1=None, healthy=False, t_collapse=None,
                   accs=[], dead=[], erank=[], sharp=[], eos=[])
        recs.append(rec)
        tseed = time.time()
        for t in range(N_TASKS):
            p = perms[t]; Xtr_t, Xte_t = X_tr[:, p], X_te[:, p]
            train_task(model, opt, Xtr_t, y_tr, crit, STEPS_PER_TASK, BATCH_SIZE)
            pr_X, pr_idx = make_probe(Xte_t, seed=seed*10000+t)
            acc = eval_acc(model, Xte_t, y_te)
            dead = compute_dead(model, pr_X); er = compute_erank(model, pr_X)
            # sharpness on a fixed training batch of this task
            hb = np.random.default_rng(seed*777+t).choice(len(y_tr), size=min(HESS_BATCH, len(y_tr)), replace=False)
            sharp = compute_sharpness(model, Xtr_t[hb], y_tr[hb], crit)
            eos = LR * sharp
            rec["accs"].append(acc); rec["dead"].append(dead); rec["erank"].append(er)
            rec["sharp"].append(sharp); rec["eos"].append(eos)
            if t == 0:
                rec["dead_after_task1"] = dead; rec["task1_acc"] = acc
                rec["healthy"] = (acc > 0.80) and (dead < 0.25)
            rec["t_collapse"] = detect_collapse(rec["accs"])
            if t < 3 or (t+1) % 25 == 0 or t == N_TASKS-1:
                print(f"  t{t+1:3d}/{N_TASKS} acc={acc:.4f} dead={dead:.4f} erank={er:6.2f} "
                      f"S={sharp:8.3f} eos={eos:6.3f} [{time.time()-tseed:.0f}s]", flush=True)
            if (t+1) % 50 == 0 or t == N_TASKS-1:
                with open(RESULTS_PATH, "w") as f:
                    json.dump({"status": "RUNNING", "seed": seed, "task": t+1}, f)
        trajectories[f"seed_{seed}"] = {k: rec[k] for k in ["accs", "dead", "erank", "sharp", "eos"]}
        with open(TRAJ_PATH, "w") as f: json.dump(trajectories, f)
        f20 = float(np.mean(rec["accs"][:20])); l20 = float(np.mean(rec["accs"][-20:]))
        print(f"  Seed {seed} DONE dead_t1={rec['dead_after_task1']:.3f} drop={(f20-l20)*100:.1f}pp "
              f"t_collapse={rec['t_collapse']} S0={rec['sharp'][0]:.2f} Sf={rec['sharp'][-1]:.2f}", flush=True)

    # analysis
    dead_t1 = [r["dead_after_task1"] for r in recs]; task1 = [r["task1_acc"] for r in recs]
    healthy_all = all(r["healthy"] for r in recs)
    drops = [(np.mean(r["accs"][:20]) - np.mean(r["accs"][-20:]))*100.0 for r in recs]
    n_coll = sum(r["t_collapse"] is not None for r in recs)

    lead = {k: [] for k in obs}
    for r in recs:
        tc = r["t_collapse"]
        for k in obs:
            if tc is None: lead[k].append(None); continue
            on = compute_onset(r[k], 0.50); lead[k].append((tc-on) if on is not None else None)
    lead_sum = {}
    for k, nm in zip(obs, obs_names):
        val = [x for x in lead[k] if x is not None]
        lead_sum[nm] = dict(median=(float(np.median(val)) if val else None),
                            mean=(float(np.mean(val)) if val else None),
                            ci95=bootstrap_ci(val)) if val else dict(median=None, mean=None, ci95=[None, None])
    auc = {nm: _j(predictive_auc(recs, k, invert[k])) for k, nm in zip(obs, obs_names)}

    # paired: does sharpness lead dead-unit fraction?
    paired = [(s, d) for s, d in zip(lead["sharp"], lead["dead"]) if s is not None and d is not None]
    if len(paired) >= 4:
        diff = np.array([s-d for s, d in paired], float); md = float(np.median(diff))
        try: _, wp = scipy_wilcoxon(diff)
        except Exception: wp = float("nan")
    else: md, wp = float("nan"), float("nan")

    ranked = sorted([(nm, lead_sum[nm]["median"]) for nm in obs_names if lead_sum[nm]["median"] is not None],
                    key=lambda x: x[1], reverse=True)
    results = {
        "status": "DONE", "regime": "hidden=100 MLP online Permuted-MNIST SGD lr=0.10 (validated healthy regime)",
        "n_seeds": N_SEEDS, "n_tasks": N_TASKS, "healthy": bool(healthy_all),
        "mean_dead_after_task1": _j(float(np.mean(dead_t1))), "mean_task1_acc": _j(float(np.mean(task1))),
        "mean_acc_drop_pp": _j(float(np.mean(drops))), "n_collapsed": int(n_coll),
        "predictive_auc": auc,
        "lead_times": {nm: {"median": _j(lead_sum[nm]["median"]), "mean": _j(lead_sum[nm]["mean"]),
                            "ci95": [_j(x) for x in lead_sum[nm]["ci95"]]} for nm in obs_names},
        "precedence_order": [x[0] for x in ranked],
        "sharpness_vs_dead_paired": {"median_diff": _j(md), "wilcoxon_p": _j(wp)},
        "mean_sharpness_init": _j(float(np.mean([r["sharp"][0] for r in recs]))),
        "mean_sharpness_final": _j(float(np.mean([r["sharp"][-1] for r in recs]))),
        "mean_eos_init": _j(float(np.mean([r["eos"][0] for r in recs]))),
        "mean_eos_final": _j(float(np.mean([r["eos"][-1] for r in recs]))),
        "per_seed": [dict(seed=r["seed"], task1_acc=_j(r["task1_acc"]), dead_after_task1=_j(r["dead_after_task1"]),
                          healthy=r["healthy"], t_collapse=r["t_collapse"],
                          sharp_init=_j(r["sharp"][0]), sharp_final=_j(r["sharp"][-1])) for r in recs],
        "subject_executed": "EOS/sharpness predictor on validated regime; top-Hessian-eig via power iteration",
        "metrics": {"healthy": bool(healthy_all), "predictive_auc": auc, "precedence_order": [x[0] for x in ranked],
                    "mean_sharpness_init": _j(float(np.mean([r["sharp"][0] for r in recs]))),
                    "mean_sharpness_final": _j(float(np.mean([r["sharp"][-1] for r in recs])))},
        "notes": "Q1 does sharpness predict collapse (AUC)? Q2 does it LEAD dead-unit/erank? Honest either way.",
        "total_wall_time_sec": float(time.time()-t0),
    }
    with open(RESULTS_PATH, "w") as f: json.dump(results, f, indent=2)
    print(f"\n{'='*60}\nDONE {time.time()-t0:.0f}s", flush=True)
    print(json.dumps({k: results[k] for k in ["predictive_auc", "precedence_order", "sharpness_vs_dead_paired",
          "mean_sharpness_init", "mean_sharpness_final", "mean_eos_init", "mean_eos_final", "healthy", "n_collapsed"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
