#!/usr/bin/env python3
"""
run_intervention.py — INTERVENTION experiment (Phase 4 / reviewer Q1)
=====================================================================
Turns the passive early-warning diagnostic into an ACTIVE lever, in the exact
VALIDATED healthy regime of the main result (hidden=100 MLP, online
Permuted-MNIST, SGD, dead_t1~6.5%).

Question (reviewer's "most powerful version"): does using the diagnostic to
TIME a plasticity-restoring reset preserve future-task learning better than
resetting on a fixed schedule or at random times, *at a matched reset budget*?
If yes, the timing carries value — the diagnostic is causal, not merely
correlational.

Design — one ARM per pod (env SH_ARM), matched budget K reset EVENTS:
  none       : K=0   (control — must collapse, reproduces the main result)
  triggered  : K<=8  events fired ONLINE when effective rank drops below
               TRIG_FRAC * its running max (refractory GAP tasks) — DIAGNOSTIC timing
  fixed      : K=8   events evenly spaced over the run                — FIXED timing
  random     : K=8   events at seeded-random tasks                    — RANDOM timing

At every event, ALL arms apply the SAME reset rule: reinitialize the units that
are currently dead on the probe (fresh incoming weights, zeroed outgoing weights
so the function is not disrupted, momentum cleared) — a coarse continual-backprop
step. Only the TIMING of the K events differs across arms, so the contrast
isolates the value of the diagnostic's timing. Seeds + task permutations are
identical across arms (torch.manual_seed(seed), DATA_SEED), so arms run in
separate pods are comparable seed-by-seed.

Primary metrics per (arm, seed):
  ss_acc  : mean new-task accuracy over the LAST 50 tasks (steady-state plasticity)
  auc_acc : mean new-task accuracy over ALL tasks (area under the learning curve)
  final_dead : dead-unit fraction at the end

Outputs:
  results/iv_<arm>/RESULTS.json   (per-seed arrays; merged & tested locally)
"""

import os, sys, json, math, time, copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision

ARM = os.environ.get("SH_ARM", "none").strip()
if ARM not in ("none", "triggered", "fixed", "random", "smart"):
    print(f"FATAL unknown SH_ARM={ARM!r}", flush=True); sys.exit(2)
# arm semantics:
#   none      : no resets (control)
#   triggered : fire when erank drops below TRIG_FRAC*running-max (naive early-warning;
#               front-loads because erank falls fast early)
#   smart     : fire when dead-fraction rises >= DEAD_STEP since the last event
#               (budget-aware diagnostic timing — spreads events across the degradation)
#   fixed     : K evenly spaced events
#   random    : K seeded-random events

_SUF = "_smoke" if os.environ.get("SH_SMOKE") == "1" else ""
RESULTS_DIR  = f"results/iv_{ARM}{_SUF}"
RESULTS_PATH = os.path.join(RESULTS_DIR, "RESULTS.json")
TRAJ_PATH    = os.path.join(RESULTS_DIR, "trajectories.json")
LOG_PATH     = os.path.join(RESULTS_DIR, "LOG.md")
MNIST_ROOT   = "/tmp/mnist"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs("_weights", exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HIDDEN, N_SEEDS, N_TASKS = 100, 5, 280
if os.environ.get("SH_SMOKE") == "1":
    N_SEEDS, N_TASKS = 1, 12   # fast CPU sanity run
STEPS_PER_TASK, BATCH_SIZE, PROBE_SIZE = 200, 128, 1000
N_CLASSES, IN_DIM = 10, 784
LR, MOMENTUM, WEIGHT_DECAY = 0.10, 0.9, 0.0
DATA_SEED = 42

# reset-event budget / trigger
K_EVENTS   = 8       # matched budget for fixed/random; cap for triggered/smart
TRIG_FRAC  = 0.92    # fire when erank < TRIG_FRAC * running-max erank
TRIG_GAP   = 8       # refractory: >= this many tasks between triggered events
DEAD_STEP  = 0.10    # smart arm: fire each time dead-fraction rises this much since last event
DEAD_THRESH = 0.50   # a unit is "resettable" if inactive on > this frac of probe
SS_WINDOW  = 50      # steady-state window (last N tasks)

print(f"ARM={ARM}  hidden={HIDDEN} {N_SEEDS}seeds x {N_TASKS}tasks  "
      f"K={K_EVENTS} TRIG_FRAC={TRIG_FRAC}", flush=True)


class MLP(nn.Module):
    def __init__(self, hidden=100):
        super().__init__()
        self.fc1  = nn.Linear(IN_DIM, hidden)
        self.fc2  = nn.Linear(hidden, hidden)
        self.head = nn.Linear(hidden, N_CLASSES)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.head(self.relu(self.fc2(self.relu(self.fc1(x)))))

    def penultimate(self, x):
        h1 = self.relu(self.fc1(x))
        h2 = self.relu(self.fc2(h1))
        return h1, h2


def load_mnist():
    tr = torchvision.datasets.MNIST(MNIST_ROOT, train=True,  download=True)
    te = torchvision.datasets.MNIST(MNIST_ROOT, train=False, download=True)
    X_tr = tr.data.numpy().reshape(-1, IN_DIM).astype(np.float32) / 255.0
    y_tr = tr.targets.numpy().astype(np.int64)
    X_te = te.data.numpy().reshape(-1, IN_DIM).astype(np.float32) / 255.0
    y_te = te.targets.numpy().astype(np.int64)
    return X_tr, y_tr, X_te, y_te


def make_perms(n, seed=DATA_SEED):
    rng = np.random.default_rng(seed)
    perms = [np.arange(IN_DIM)]
    for _ in range(n - 1):
        perms.append(rng.permutation(IN_DIM))
    return perms


def train_task(model, opt, X, y, crit, steps, bs):
    model.train()
    N = len(y)
    Xg = torch.from_numpy(X).to(DEVICE); yg = torch.from_numpy(y).to(DEVICE)
    idx = np.arange(N); np.random.shuffle(idx); ptr = 0
    for _ in range(steps):
        if ptr + bs > N:
            np.random.shuffle(idx); ptr = 0
        b = idx[ptr:ptr+bs]; ptr += bs
        opt.zero_grad(); crit(model(Xg[b]), yg[b]).backward(); opt.step()


@torch.no_grad()
def dead_mask(model, probe_X, thresh=DEAD_THRESH):
    """Return (mask_h1, mask_h2): bool arrays, True where the unit is dead."""
    model.eval()
    h1, h2 = model.penultimate(probe_X)
    m1 = ((h1 <= 0).float().mean(0) > thresh).cpu().numpy()
    m2 = ((h2 <= 0).float().mean(0) > thresh).cpu().numpy()
    return m1, m2


@torch.no_grad()
def dead_frac(model, probe_X):
    model.eval()
    h1, h2 = model.penultimate(probe_X)
    d1 = ((h1 <= 0).float().mean(0) > 0.95).float().mean().item()
    d2 = ((h2 <= 0).float().mean(0) > 0.95).float().mean().item()
    return (d1 + d2) / 2.0


@torch.no_grad()
def erank(model, probe_X):
    model.eval()
    _, h2 = model.penultimate(probe_X)
    try:
        S = torch.linalg.svdvals(h2.float()); S = S[S > 1e-10]
        if len(S) == 0:
            return 1.0
        p = S / S.sum()
        return max(1.0, math.exp(-(p * torch.log(p + 1e-12)).sum().item()))
    except Exception:
        return 1.0


@torch.no_grad()
def eval_acc(model, X, y):
    model.eval()
    Xg = torch.from_numpy(X).to(DEVICE); yg = torch.from_numpy(y).to(DEVICE)
    c = 0
    for i in range(0, len(y), 1024):
        c += (model(Xg[i:i+1024]).argmax(1) == yg[i:i+1024]).sum().item()
    return c / len(y)


def make_probe(X, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=min(PROBE_SIZE, len(X)), replace=False)
    return torch.from_numpy(X[idx]).to(DEVICE)


def _reset_momentum(opt, param, rows=None, cols=None):
    st = opt.state.get(param, {})
    buf = st.get("momentum_buffer")
    if buf is None:
        return
    if rows is not None:
        buf[rows] = 0.0
    if cols is not None:
        buf[:, cols] = 0.0


@torch.no_grad()
def apply_reset(model, opt, probe_X, gen):
    """Reinitialize currently-dead units: fresh incoming weights (kaiming-uniform
    like nn.Linear default), zeroed outgoing weights, cleared momentum. Returns
    number of units reset."""
    m1, m2 = dead_mask(model, probe_X)
    n_reset = 0
    # layer-1 dead units -> reinit fc1 row, zero fc2 col
    idx1 = np.where(m1)[0]
    if len(idx1):
        fan_in = model.fc1.weight.shape[1]
        bound = 1.0 / math.sqrt(fan_in)
        for j in idx1:
            model.fc1.weight[j].uniform_(-bound, bound, generator=gen)
            model.fc1.bias[j].uniform_(-bound, bound, generator=gen)
            model.fc2.weight[:, j] = 0.0
        _reset_momentum(opt, model.fc1.weight, rows=idx1)
        _reset_momentum(opt, model.fc1.bias,   rows=idx1)
        _reset_momentum(opt, model.fc2.weight, cols=idx1)
        n_reset += len(idx1)
    # layer-2 dead units -> reinit fc2 row, zero head col
    idx2 = np.where(m2)[0]
    if len(idx2):
        fan_in = model.fc2.weight.shape[1]
        bound = 1.0 / math.sqrt(fan_in)
        for j in idx2:
            model.fc2.weight[j].uniform_(-bound, bound, generator=gen)
            model.fc2.bias[j].uniform_(-bound, bound, generator=gen)
            model.head.weight[:, j] = 0.0
        _reset_momentum(opt, model.fc2.weight, rows=idx2)
        _reset_momentum(opt, model.fc2.bias,   rows=idx2)
        _reset_momentum(opt, model.head.weight, cols=idx2)
        n_reset += len(idx2)
    return n_reset


def fixed_event_tasks(k, n_tasks):
    # k evenly spaced tasks in (0, n_tasks), avoiding task 0
    step = n_tasks / (k + 1)
    return sorted(set(int(round(step * (i + 1))) for i in range(k)))


def random_event_tasks(k, n_tasks, seed):
    rng = np.random.default_rng(1000 + seed)
    return sorted(rng.choice(np.arange(5, n_tasks), size=k, replace=False).tolist())


def main():
    t0 = time.time()
    with open(LOG_PATH, "w") as f:
        f.write(f"# LOG — intervention arm={ARM}\n\nValidated regime hidden=100 "
                f"PMNIST SGD. K={K_EVENTS} matched budget.\nStarted "
                f"{time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

    X_tr, y_tr, X_te, y_te = load_mnist()
    crit = nn.CrossEntropyLoss()
    perms = make_perms(N_TASKS)

    per_seed = []
    trajectories = {}
    for seed in range(N_SEEDS):
        print(f"\n{'='*66}\nARM={ARM} SEED {seed} (elapsed {time.time()-t0:.0f}s)\n{'='*66}", flush=True)
        torch.manual_seed(seed); np.random.seed(seed)
        gen = torch.Generator(device=DEVICE); gen.manual_seed(10_000 + seed)
        model = MLP(HIDDEN).to(DEVICE)
        opt = optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)

        # planned events for fixed/random (triggered is decided online)
        if ARM == "fixed":
            planned = set(fixed_event_tasks(K_EVENTS, N_TASKS))
        elif ARM == "random":
            planned = set(random_event_tasks(K_EVENTS, N_TASKS, seed))
        else:
            planned = set()

        accs, deads, eranks, event_tasks, event_counts = [], [], [], [], []
        run_max_erank = 0.0
        last_event = -TRIG_GAP - 1
        last_event_dead = None   # dead-fraction at last smart event (set after task 0)

        for t in range(N_TASKS):
            p = perms[t]
            Xtr_t, Xte_t = X_tr[:, p], X_te[:, p]
            train_task(model, opt, Xtr_t, y_tr, crit, STEPS_PER_TASK, BATCH_SIZE)
            pr = make_probe(Xte_t, seed=seed * 10_000 + t)
            acc = eval_acc(model, Xte_t, y_te)
            er  = erank(model, pr)
            df  = dead_frac(model, pr)
            accs.append(acc); eranks.append(er); deads.append(df)
            run_max_erank = max(run_max_erank, er)

            if last_event_dead is None:
                last_event_dead = df   # baseline dead level (after task 0)

            fire = False
            if ARM == "triggered":
                if (er < TRIG_FRAC * run_max_erank and (t - last_event) > TRIG_GAP
                        and len(event_tasks) < K_EVENTS and t >= 5):
                    fire = True
            elif ARM == "smart":
                if (df - last_event_dead >= DEAD_STEP and (t - last_event) > TRIG_GAP
                        and len(event_tasks) < K_EVENTS and t >= 5):
                    fire = True
            elif ARM in ("fixed", "random"):
                fire = t in planned

            if fire:
                nrst = apply_reset(model, opt, pr, gen)
                event_tasks.append(t); event_counts.append(nrst)
                last_event = t
                last_event_dead = df   # next smart event needs another DEAD_STEP rise
                print(f"  [event] task {t}: reset {nrst} units (erank={er:.2f} dead={df:.2f})", flush=True)

            if t < 3 or (t + 1) % 40 == 0 or t == N_TASKS - 1:
                print(f"  t{t+1:3d}/{N_TASKS} acc={acc:.4f} dead={df:.4f} erank={er:6.2f}", flush=True)

        ss_acc  = float(np.mean(accs[-SS_WINDOW:]))
        auc_acc = float(np.mean(accs))
        task1_acc = accs[0]
        final_dead = deads[-1]
        rec = dict(seed=seed, arm=ARM, task1_acc=task1_acc, ss_acc=ss_acc,
                   auc_acc=auc_acc, final_dead=final_dead,
                   n_events=len(event_tasks), event_tasks=event_tasks,
                   total_units_reset=int(sum(event_counts)),
                   ss_minus_task1_pp=(ss_acc - task1_acc) * 100.0)
        per_seed.append(rec)
        trajectories[f"seed_{seed}"] = dict(accs=accs, dead=deads, erank=eranks,
                                            event_tasks=event_tasks)
        print(f"  Seed {seed} DONE ss_acc={ss_acc:.4f} task1={task1_acc:.4f} "
              f"events={len(event_tasks)} reset={sum(event_counts)}", flush=True)
        # incremental save
        with open(RESULTS_PATH, "w") as f:
            json.dump({"status": "RUNNING", "arm": ARM, "per_seed": per_seed}, f, indent=2)
        with open(TRAJ_PATH, "w") as f:
            json.dump(trajectories, f)

    ss = [r["ss_acc"] for r in per_seed]
    auc = [r["auc_acc"] for r in per_seed]
    fd = [r["final_dead"] for r in per_seed]
    results = {
        "status": "DONE", "arm": ARM,
        "regime": "hidden=100 MLP, online Permuted-MNIST, SGD, validated healthy regime",
        "n_seeds": N_SEEDS, "n_tasks": N_TASKS, "K_events": K_EVENTS,
        "trig_frac": TRIG_FRAC, "dead_thresh": DEAD_THRESH,
        "mean_ss_acc": float(np.mean(ss)), "std_ss_acc": float(np.std(ss)),
        "mean_auc_acc": float(np.mean(auc)),
        "mean_final_dead": float(np.mean(fd)),
        "mean_events": float(np.mean([r["n_events"] for r in per_seed])),
        "mean_units_reset": float(np.mean([r["total_units_reset"] for r in per_seed])),
        "per_seed": per_seed,
        "total_wall_time_sec": float(time.time() - t0),
        "subject_executed": f"intervention arm={ARM}, {N_SEEDS} seeds x {N_TASKS} tasks, K={K_EVENTS}",
        "metrics": {"arm": ARM, "mean_ss_acc": float(np.mean(ss)),
                    "mean_final_dead": float(np.mean(fd)),
                    "mean_events": float(np.mean([r["n_events"] for r in per_seed]))},
        "notes": ("Matched-budget reset-timing comparison. Compare mean_ss_acc across "
                  "arms none/triggered/fixed/random seed-by-seed to test whether the "
                  "diagnostic's TIMING preserves plasticity. Honest either way."),
    }
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    with open(LOG_PATH, "a") as f:
        f.write(f"\n## Results arm={ARM}\nmean_ss_acc={np.mean(ss):.4f} "
                f"mean_final_dead={np.mean(fd):.4f} mean_events={np.mean([r['n_events'] for r in per_seed]):.1f}\n")
    print(f"\n{'='*66}\nDONE arm={ARM} in {time.time()-t0:.0f}s", flush=True)
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
