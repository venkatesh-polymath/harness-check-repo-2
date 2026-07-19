"""
run_gns200_perm10.py — Permuted-CIFAR-10 leg for gns200

Proven collapse regime: 8/8 seeds collapse in prior rounds.
Uses B=200 GNS samples (the key fix in this round).
Same onset detector as the CIFAR-100 leg.
"""

import os, sys, json, math, time, pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

C10_ROOT = "/opt/datasets/cifar-10-batches-py"
if not os.path.isdir(C10_ROOT):
    print(f"ABORT: {C10_ROOT} missing", file=sys.stderr)
    sys.exit(1)

RESULTS_DIR  = "results/gns200"
PERM10_PATH  = os.path.join(RESULTS_DIR, "perm10_results.json")
os.makedirs(RESULTS_DIR, exist_ok=True)

DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_SEEDS      = 8
N_TASKS      = 20
STEPS_TASK   = 1000
BATCH        = 64
PROBE        = 512
HIDDEN       = 400
IN_DIM       = 3072
GNS_B        = 200
GNS_BOOT     = 50
LR           = 0.05
MOM          = 0.9
WD           = 0.0
DEAD_THRESH  = 0.95
COLLAPSE_PP  = 15.0
N_OUT        = 10

print(f"Permuted-CIFAR-10 B=200 run on {DEVICE}")
sys.stdout.flush()

# ─── Load data ───────────────────────────────────────────────────────────────
def load_cifar10():
    def _lb(p):
        with open(p, 'rb') as f:
            d = pickle.load(f, encoding='bytes')
        return d[b'data'].astype(np.float32)/255., np.array(d[b'labels'],dtype=np.int64)
    Xs, ys = [], []
    for i in range(1, 6):
        X, y = _lb(os.path.join(C10_ROOT, f"data_batch_{i}"))
        Xs.append(X); ys.append(y)
    Xtr = np.concatenate(Xs); ytr = np.concatenate(ys)
    Xte, yte = _lb(os.path.join(C10_ROOT, 'test_batch'))
    m = Xtr.mean(0); s = Xtr.std(0) + 1e-8
    return (torch.from_numpy((Xtr-m)/s), torch.from_numpy(ytr),
            torch.from_numpy((Xte-m)/s), torch.from_numpy(yte))

print("Loading CIFAR-10 ... ", end='', flush=True)
Xtr, ytr, Xte, yte = load_cifar10()
print("done.")

# Permutation per task (deterministic from task_id)
_perms = {}
def get_perm(task_id):
    if task_id not in _perms:
        _perms[task_id] = np.random.default_rng(task_id).permutation(IN_DIM)
    return _perms[task_id]

def task_data(task_id, train=True):
    p = get_perm(task_id)
    X = Xtr if train else Xte
    y = ytr if train else yte
    return X[:, p], y

# ─── Model ───────────────────────────────────────────────────────────────────
class Trunk(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(IN_DIM, HIDDEN)
        self.fc2 = nn.Linear(HIDDEN, HIDDEN)
    def forward(self, x):
        return torch.relu(self.fc2(torch.relu(self.fc1(x))))

class Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(HIDDEN, N_OUT)
    def forward(self, h):
        return self.fc(h)

# ─── Observables ─────────────────────────────────────────────────────────────
def dead_frac(trunk, Xp):
    trunk.eval()
    with torch.no_grad():
        x = Xp.to(DEVICE)
        h1 = torch.relu(trunk.fc1(x))
        h2 = torch.relu(trunk.fc2(h1))
    d1 = (h1==0).float().mean(0) > DEAD_THRESH
    d2 = (h2==0).float().mean(0) > DEAD_THRESH
    return (d1.sum()+d2.sum()).item() / (HIDDEN*2)

def erank(trunk, Xp):
    trunk.eval()
    with torch.no_grad():
        h = trunk(Xp.to(DEVICE))
    S = torch.linalg.svdvals(h.float())
    S = S[S>0]
    if len(S)==0: return 1.0
    p = S/S.sum()
    H = -(p*torch.log(p+1e-12)).sum().item()
    return max(math.exp(H), 1.0)

def wdrift(trunk, init_norms):
    ds = [abs(p.data.norm().item()/init_norms[n]-1.)
          for n,p in trunk.named_parameters()
          if 'weight' in n and n in init_norms and init_norms[n]>0]
    return float(np.mean(ds)) if ds else 0.

def gns(trunk, head, Xdata, ydata, rng):
    idx = rng.choice(len(Xdata), size=min(GNS_B,len(Xdata)), replace=False)
    Xb = Xdata[idx].to(DEVICE); yb = ydata[idx].to(DEVICE)
    crit = nn.CrossEntropyLoss()
    params = list(trunk.parameters()) + list(head.parameters())
    trunk.eval(); head.eval()
    grads = []
    for i in range(len(idx)):
        for p in params:
            if p.grad is not None: p.grad.zero_()
        logits = head(trunk(Xb[i:i+1]))
        crit(logits, yb[i:i+1]).backward()
        g = torch.cat([p.grad.detach().flatten() for p in params if p.grad is not None])
        grads.append(g.cpu())
    trunk.train(); head.train()
    G = torch.stack(grads)
    Gm = G.mean(0); Gn2 = (Gm**2).sum().item()
    if Gn2 < 1e-30: return float('nan'), float('nan')
    c = G - Gm.unsqueeze(0)
    tr_cov = (c**2).sum().item() / (len(idx)-1)
    B_est = tr_cov / Gn2
    rng2 = np.random.default_rng(99)
    boots = []
    for _ in range(GNS_BOOT):
        bi = rng2.choice(len(idx), size=len(idx), replace=True)
        Gb = G[bi]; Gbm = Gb.mean(0); n2 = (Gbm**2).sum().item()
        if n2<1e-30: continue
        cb = Gb - Gbm.unsqueeze(0)
        boots.append((cb**2).sum().item()/(len(idx)-1)/n2)
    se = float(np.std(boots)) if len(boots)>2 else float('nan')
    return float(B_est), se

def evaluate(trunk, head, X, y):
    trunk.eval(); head.eval()
    correct = total = 0
    with torch.no_grad():
        for i in range(0, len(X), 256):
            xb=X[i:i+256].to(DEVICE); yb=y[i:i+256].to(DEVICE)
            correct += (head(trunk(xb)).argmax(1)==yb).sum().item()
            total += len(yb)
    return correct/total if total>0 else 0.

# ─── Onset ───────────────────────────────────────────────────────────────────
def onset(init_v, traj, direction=None):
    if len(traj)<2: return None
    full = [init_v] + list(traj)
    sm = [full[0]]
    for i in range(1,len(full)): sm.append((full[i]+full[i-1])/2.)
    v0,vf = full[0],full[-1]
    if abs(vf-v0)<1e-10: return None
    if direction is None: direction = +1 if vf>v0 else -1
    thr = v0 + .5*(vf-v0)
    def past(v): return v>=thr if direction>0 else v<=thr
    for t in range(1,len(sm)-1):
        if past(sm[t]) and past(sm[t+1]): return t
    return None

def collapse(accs, t1):
    thr = t1 - COLLAPSE_PP/100.
    for t in range(len(accs)-1):
        if accs[t]<thr and accs[t+1]<thr: return t+1
    return None

# ─── LR sanity ───────────────────────────────────────────────────────────────
Xp0, _ = task_data(0, train=True)
Xp0 = Xp0[:PROBE]
trunk_tmp = Trunk().to(DEVICE)
d_init = dead_frac(trunk_tmp, Xp0)
er_init = erank(trunk_tmp, Xp0)
del trunk_tmp
print(f"LR sanity: dead_at_init={d_init:.4f} (must be <0.15), erank_at_init={er_init:.3f}")
assert d_init < 0.15, f"FAIL: dead_at_init={d_init}"
sys.stdout.flush()

# ─── Main loop ───────────────────────────────────────────────────────────────
print(f"\n=== {N_SEEDS} seeds × {N_TASKS} tasks × {STEPS_TASK} steps, lr={LR}, B={GNS_B} ===")
sys.stdout.flush()

all_results = []
t0 = time.time()

for seed in range(N_SEEDS):
    torch.manual_seed(seed); np.random.seed(seed)
    rng = np.random.default_rng(seed)

    print(f"\n--- Seed {seed} ---")
    sys.stdout.flush()

    trunk = Trunk().to(DEVICE)
    head  = Head().to(DEVICE)
    init_norms = {n: p.data.norm().item() for n,p in trunk.named_parameters() if 'weight' in n}
    Xp = Xp0  # same probe set

    d0  = dead_frac(trunk, Xp)
    er0 = erank(trunk, Xp)
    print(f"  INIT: dead={d0:.4f}, erank={er0:.3f}")
    sys.stdout.flush()

    accs=[]; dead_t=[]; er_t=[]; wd_t=[]; gns_t=[]; gns_se_t=[]

    for task_id in range(N_TASKS):
        Xtr_t, ytr_t = task_data(task_id, train=True)
        Xte_t, yte_t = task_data(task_id, train=False)

        # NOTE: Permuted CIFAR-10 uses SHARED head (never reset)
        opt = optim.SGD(list(trunk.parameters())+list(head.parameters()),
                        lr=LR, momentum=MOM, weight_decay=WD)
        crit = nn.CrossEntropyLoss()
        ds = TensorDataset(Xtr_t, ytr_t)
        loader = DataLoader(ds, batch_size=BATCH, shuffle=True, drop_last=True)
        trunk.train(); head.train()
        step=0; it=iter(loader)
        while step<STEPS_TASK:
            try: xb,yb=next(it)
            except StopIteration: it=iter(loader); xb,yb=next(it)
            xb,yb=xb.to(DEVICE),yb.to(DEVICE)
            opt.zero_grad()
            crit(head(trunk(xb)),yb).backward()
            opt.step()
            step+=1

        acc = evaluate(trunk, head, Xte_t, yte_t)
        d   = dead_frac(trunk, Xp)
        er  = erank(trunk, Xp)
        wd  = wdrift(trunk, init_norms)
        g, g_se = gns(trunk, head, Xtr_t, ytr_t, rng)

        accs.append(acc); dead_t.append(d); er_t.append(er)
        wd_t.append(wd); gns_t.append(g); gns_se_t.append(g_se)

        print(f"  Task {task_id+1:2d}: acc={acc:.3f} dead={d:.3f} erank={er:.2f} "
              f"wd={wd:.3f} gns={g:.1f}±{g_se:.1f}")
        sys.stdout.flush()

    t_col = collapse(accs, accs[0])
    o_dead  = onset(d0,   dead_t,  +1)
    o_er    = onset(er0,  er_t,    -1)
    o_wd    = onset(0.,   wd_t,    +1)
    gns_v0 = gns_t[0] if gns_t and not math.isnan(gns_t[0]) else 1.
    gns_vf = gns_t[-1] if gns_t and not math.isnan(gns_t[-1]) else gns_v0
    o_gns   = onset(gns_v0, gns_t, +1 if gns_vf>gns_v0 else -1)

    def lt(o,t): return (t-o) if o is not None and t is not None else None

    print(f"  t_col={t_col} | onsets: dead={o_dead} er={o_er} wd={o_wd} gns={o_gns}")
    print(f"  LTs: dead={lt(o_dead,t_col)} er={lt(o_er,t_col)} wd={lt(o_wd,t_col)} gns={lt(o_gns,t_col)}")
    sys.stdout.flush()

    all_results.append({
        "seed": seed,
        "t_collapse": t_col, "task1_acc": float(accs[0]),
        "task_accs": [float(a) for a in accs],
        "dead_init": float(d0), "erank_init": float(er0),
        "dead_traj":  [float(v) for v in dead_t],
        "erank_traj": [float(v) for v in er_t],
        "wdrift_traj":[float(v) for v in wd_t],
        "gns_traj":   [float(v) if not math.isnan(v) else None for v in gns_t],
        "gns_se_traj":[float(v) if not math.isnan(v) else None for v in gns_se_t],
        "onsets": {
            "dead_unit_fraction":   o_dead,
            "effective_rank":       o_er,
            "weight_norm_drift":    o_wd,
            "gradient_noise_scale": o_gns,
        },
        "lead_times": {
            "dead_unit_fraction":   lt(o_dead, t_col),
            "effective_rank":       lt(o_er,   t_col),
            "weight_norm_drift":    lt(o_wd,   t_col),
            "gradient_noise_scale": lt(o_gns,  t_col),
        },
    })

    # Save intermediate
    with open(PERM10_PATH, 'w') as f:
        json.dump({"seeds_done": seed+1, "results": all_results}, f, indent=2)

# ─── Aggregate ───────────────────────────────────────────────────────────────
OBS = ["dead_unit_fraction","effective_rank","weight_norm_drift","gradient_noise_scale"]

def bci(values, n=1000):
    vals = [v for v in values if v is not None and not (isinstance(v,float) and math.isnan(v))]
    if not vals: return float('nan'), [float('nan'), float('nan')]
    if len(vals)==1: return float(vals[0]), [float(vals[0]), float(vals[0])]
    rng = np.random.default_rng(0)
    arr = np.array(vals,float)
    boots = [rng.choice(arr,size=len(arr),replace=True).mean() for _ in range(n)]
    return float(arr.mean()), [float(np.percentile(boots,2.5)),float(np.percentile(boots,97.5))]

lead_times = {}
for obs in OBS:
    lts = [sr["lead_times"][obs] for sr in all_results]
    mean, ci = bci(lts)
    n_v = len([v for v in lts if v is not None])
    lead_times[obs] = {
        "mean": round(mean,3) if not math.isnan(mean) else None,
        "ci95": [round(ci[0],3) if not math.isnan(ci[0]) else None,
                 round(ci[1],3) if not math.isnan(ci[1]) else None],
        "per_seed": lts, "n_valid": n_v,
    }

n_col = sum(1 for sr in all_results if sr["t_collapse"] is not None)
wall = time.time() - t0

valid = [(o,lead_times[o]["mean"]) for o in OBS
         if lead_times[o]["mean"] is not None and not math.isnan(lead_times[o]["mean"])]
valid.sort(key=lambda x:x[1], reverse=True)
prec_order = [o for o,_ in valid]

reliable = [o for o in OBS
            if lead_times[o]["ci95"][0] is not None
            and not math.isnan(lead_times[o]["ci95"][0])
            and lead_times[o]["ci95"][0] > 0]

gns_m = lead_times["gradient_noise_scale"]["mean"]
er_m  = lead_times["effective_rank"]["mean"]
if gns_m is None or math.isnan(gns_m):
    glo = "inconclusive"
elif er_m is not None and not math.isnan(er_m):
    glo = "leads" if gns_m > er_m else "lags"
else:
    glo = "inconclusive"

ec_vals = [sr["erank_traj"][sr["t_collapse"]-1]
           for sr in all_results
           if sr["t_collapse"] is not None and sr["t_collapse"] <= len(sr["erank_traj"])]
erank_col = float(np.mean(ec_vals)) if ec_vals else float('nan')

all_se = [se for sr in all_results for se in sr["gns_se_traj"]
          if se is not None and not math.isnan(se)]
gns_se_typ = float(np.median(all_se)) if all_se else float('nan')

print("\n=== PERM-10 AGGREGATE ===")
for obs in OBS:
    v = lead_times[obs]
    print(f"  {obs}: mean={v['mean']} ci95={v['ci95']} n_valid={v['n_valid']}")
print(f"  precedence: {prec_order}")
print(f"  reliable: {reliable}")
print(f"  gns_leads_or_lags: {glo}")
print(f"  collapse: {n_col}/{N_SEEDS}")
print(f"  wall: {wall:.1f}s")

output = {
    "dataset": "permuted_cifar10",
    "n_seeds": N_SEEDS, "n_tasks": N_TASKS, "steps_per_task": STEPS_TASK,
    "lr_used": LR, "gns_B": GNS_B,
    "dead_at_init": round(d_init, 4), "erank_at_init": round(er_init, 3),
    "erank_at_collapse": round(erank_col, 3) if not math.isnan(erank_col) else None,
    "collapse_reproduced": n_col >= 4,
    "n_seeds_collapsed": n_col,
    "lead_times": lead_times,
    "precedence_order": prec_order,
    "reliable_leaders": reliable,
    "gns_leads_or_lags": glo,
    "gns_se_typical_median": round(gns_se_typ, 3) if not math.isnan(gns_se_typ) else None,
    "per_seed_data": all_results,
    "wall_clock_sec": round(wall, 1),
}

with open(PERM10_PATH, 'w') as f:
    json.dump(output, f, indent=2)

print(f"\nSaved to {PERM10_PATH}")
print("DONE.")
