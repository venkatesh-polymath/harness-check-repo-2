"""
Grokking Ablation v3 — BatchNorm sublayer decomposition

Resolves "is BN's implicit L2 absent or sub-threshold?" confound.

Arms (5 seeds each: 42,123,7,99,2024; BS=256; noWD unless stated; 15k-step cap):
  1. BN_all_noWD      — both sublayers BN, no WD (replicate null from v2)
  2. BN_mlp_only_noWD — norm2=BN (MLP sublayer), norm1=LN (attention)  [KEY]
  3. BN_attn_only_noWD— norm1=BN (attention), norm2=LN (MLP)           [complementary]
  4. LN_wd0.25        — pure LN, WD=0.25 (finer sweep, WD in (0.1,1.0) gap)
  5. LN_wd0.5         — pure LN, WD=0.5
  6. LN_wd0.75        — pure LN, WD=0.75

Effective-WD estimate: match final wnorm_ratio of BN_all+noWD to LN+WD curves.

Output: results/grok_v3/RESULTS.json, run.log, curves/
"""

import math
import json
import time
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# ─── Constants (FIXED — do not change) ───────────────────────────────────────
P          = 97
TRAIN_FRAC = 0.4
LR         = 1e-3
MAX_STEPS  = 15_000
EVAL_EVERY = 100
LOG_EVERY  = 5_000   # per-cell progress print
BATCH_SIZE = 256
SEEDS      = [42, 123, 7, 99, 2024]

RESULTS_DIR  = "results/grok_v3"
RESULTS_PATH = f"{RESULTS_DIR}/RESULTS.json"
CURVES_DIR   = f"{RESULTS_DIR}/curves"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}", flush=True)


# ─── Helpers ─────────────────────────────────────────────────────────────────

class TransposedBN(nn.Module):
    """BatchNorm1d wrapper for (B, T, D) tensors — handles variable batch sizes."""
    def __init__(self, d_model):
        super().__init__()
        self.bn = nn.BatchNorm1d(d_model)

    def forward(self, x):
        B, T, D = x.shape
        x = x.reshape(B * T, D)
        x = self.bn(x)
        return x.reshape(B, T, D)


def make_norm(norm_type, d_model):
    if norm_type == "LayerNorm":
        return nn.LayerNorm(d_model)
    elif norm_type == "BatchNorm":
        return TransposedBN(d_model)
    else:
        raise ValueError(f"Unknown norm_type: {norm_type}")


# ─── Architecture ─────────────────────────────────────────────────────────────

D_MODEL  = 128
N_HEADS  = 4
MLP_HID  = 512
VOCAB_SZ = P + 1   # 98 (P residues + separator)
SEQ_LEN  = 3
NUM_CLS  = P       # 97


class TransformerBlock(nn.Module):
    """
    Pre-norm transformer block.
    norm1_type → used before attention (attn sublayer).
    norm2_type → used before MLP (mlp sublayer).
    """
    def __init__(self, norm1_type, norm2_type):
        super().__init__()
        self.norm1 = make_norm(norm1_type, D_MODEL)
        self.norm2 = make_norm(norm2_type, D_MODEL)
        self.attn  = nn.MultiheadAttention(D_MODEL, N_HEADS, batch_first=True)
        self.mlp   = nn.Sequential(
            nn.Linear(D_MODEL, MLP_HID),
            nn.GELU(),
            nn.Linear(MLP_HID, D_MODEL),
        )

    def forward(self, x):
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class GrokTransformer(nn.Module):
    def __init__(self, norm1_type, norm2_type):
        super().__init__()
        self.tok_emb = nn.Embedding(VOCAB_SZ, D_MODEL)
        self.pos_emb = nn.Embedding(SEQ_LEN, D_MODEL)
        # Use the SAME norm type for output norm as norm2 (MLP side) for consistency
        self.block   = TransformerBlock(norm1_type, norm2_type)
        self.ln_out  = make_norm(norm2_type, D_MODEL)
        self.head    = nn.Linear(D_MODEL, NUM_CLS)

    def forward(self, x):
        B, T = x.shape
        pos  = torch.arange(T, device=x.device).unsqueeze(0)
        emb  = self.tok_emb(x) + self.pos_emb(pos)
        h    = self.block(emb)
        h    = self.ln_out(h)
        return self.head(h[:, -1, :])

    def weight_norm(self):
        """L2 norm of all weight matrices (excluding bias, norm, emb params)."""
        total = sum(
            p.data.norm(2).item() ** 2
            for n, p in self.named_parameters()
            if 'weight' in n and p.ndim >= 2
        )
        return math.sqrt(total)


# ─── Dataset ──────────────────────────────────────────────────────────────────

def make_dataset(seed):
    rng = torch.Generator().manual_seed(seed)
    SEP = P
    pairs = [(a, b, (a + b) % P) for a in range(P) for b in range(P)]
    pairs = torch.tensor(pairs, dtype=torch.long)
    perm  = torch.randperm(len(pairs), generator=rng)
    pairs = pairs[perm]
    n_train = int(len(pairs) * TRAIN_FRAC)
    tr, va = pairs[:n_train], pairs[n_train:]

    def make_xy(s):
        a   = s[:, 0:1]
        b   = s[:, 1:2]
        sep = torch.full((len(s), 1), SEP, dtype=torch.long)
        return torch.cat([a, sep, b], 1), s[:, 2]

    return make_xy(tr), make_xy(va), n_train, len(va)


# ─── Single cell ──────────────────────────────────────────────────────────────

def run_cell(arm_name, norm1_type, norm2_type, weight_decay, seed):
    label = f"{arm_name} seed={seed}"
    print(f"\n{'='*60}", flush=True)
    print(f"CELL: {label}", flush=True)
    print(f"  norm1={norm1_type} norm2={norm2_type} WD={weight_decay}", flush=True)
    print(f"{'='*60}", flush=True)

    # Reproducibility
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Data
    (x_tr, y_tr), (x_va, y_va), n_train, n_val = make_dataset(seed)
    x_va = x_va.to(DEVICE); y_va = y_va.to(DEVICE)
    x_tr_full = x_tr.to(DEVICE); y_tr_full = y_tr.to(DEVICE)

    train_ds = TensorDataset(x_tr, y_tr)
    loader   = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        generator=torch.Generator().manual_seed(seed)
    )

    # Model
    torch.manual_seed(seed)
    model = GrokTransformer(norm1_type, norm2_type).to(DEVICE)
    init_wnorm = model.weight_norm()

    # Optimizer — exclude norm params and embeddings from WD
    decay_p   = [p for n, p in model.named_parameters()
                 if 'weight' in n and p.ndim >= 2
                 and 'norm' not in n and 'emb' not in n and 'bn' not in n]
    nodecay_p = [p for n, p in model.named_parameters()
                 if not ('weight' in n and p.ndim >= 2
                         and 'norm' not in n and 'emb' not in n and 'bn' not in n)]
    optimizer = torch.optim.AdamW(
        [{'params': decay_p,   'weight_decay': weight_decay},
         {'params': nodecay_p, 'weight_decay': 0.0}],
        lr=LR, betas=(0.9, 0.98), eps=1e-8
    )

    grok_onset     = None
    grok_confirmed = None
    history        = []
    step           = 0
    data_iter      = iter(loader)
    t0             = time.time()

    while step < MAX_STEPS:
        # fetch batch
        try:
            xb, yb = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            xb, yb = next(data_iter)
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)

        model.train()
        optimizer.zero_grad()
        loss = F.cross_entropy(model(xb), yb)
        loss.backward()
        optimizer.step()
        step += 1

        # Evaluate
        if step % EVAL_EVERY == 0:
            model.eval()
            with torch.no_grad():
                val_logits = model(x_va)
                val_loss   = F.cross_entropy(val_logits, y_va).item()
                val_acc    = (val_logits.argmax(-1) == y_va).float().mean().item()
                tr_logits  = model(x_tr_full)
                tr_loss    = F.cross_entropy(tr_logits, y_tr_full).item()
                tr_acc     = (tr_logits.argmax(-1) == y_tr_full).float().mean().item()

            wnorm = model.weight_norm()
            history.append(dict(
                step=step, tr_acc=tr_acc, tr_loss=round(tr_loss, 6),
                val_acc=val_acc, val_loss=round(val_loss, 6),
                wnorm=round(wnorm, 4)
            ))

            if step % LOG_EVERY == 0:
                elapsed = time.time() - t0
                print(f"  step={step:5d} | tr={tr_acc:.3f} | val={val_acc:.3f} "
                      f"| wnorm={wnorm:.2f} | {elapsed:.0f}s", flush=True)

            # grokking detection
            if val_acc > 0.95 and grok_onset is None:
                grok_onset = step
                print(f"  *** GROKKING ONSET step={step} val_acc={val_acc:.4f}", flush=True)

            if grok_onset is not None and grok_confirmed is None:
                recent = [r['val_acc'] for r in history[-3:]]
                if len(recent) >= 3 and all(v > 0.95 for v in recent):
                    grok_confirmed = step
                    print(f"  *** GROKKING CONFIRMED step={step}", flush=True)

        # Early stop after confirmed grokking + high accuracy
        if grok_confirmed is not None:
            model.eval()
            with torch.no_grad():
                va = (model(x_va).argmax(-1) == y_va).float().mean().item()
            if va > 0.99:
                print(f"  Early stop: val_acc={va:.4f} at step={step}", flush=True)
                break

    # Final eval
    model.eval()
    with torch.no_grad():
        final_val_acc = (model(x_va).argmax(-1) == y_va).float().mean().item()
        final_tr_acc  = (model(x_tr_full).argmax(-1) == y_tr_full).float().mean().item()
    final_wnorm = model.weight_norm()
    elapsed = time.time() - t0

    grokkd = grok_onset is not None
    print(f"  DONE | steps={step} | grok={'YES @'+str(grok_onset) if grokkd else 'NO'} "
          f"| val={final_val_acc:.3f} | wnorm_ratio={final_wnorm/init_wnorm:.3f} "
          f"| {elapsed:.0f}s", flush=True)

    # Weight-norm trajectory (every 500 steps = every 5th eval)
    wnorm_traj = [(r['step'], r['wnorm']) for r in history[::5]]

    return {
        "arm":          arm_name,
        "norm1_type":   norm1_type,
        "norm2_type":   norm2_type,
        "weight_decay": weight_decay,
        "seed":         seed,
        "grokked":      grokkd,
        "grok_onset":   grok_onset,
        "grok_confirmed": grok_confirmed,
        "total_steps":  step,
        "final_val_acc": round(final_val_acc, 6),
        "final_tr_acc":  round(final_tr_acc, 6),
        "init_wnorm":    round(init_wnorm, 4),
        "final_wnorm":   round(final_wnorm, 4),
        "wnorm_ratio":   round(final_wnorm / init_wnorm, 4),
        "elapsed_s":     round(elapsed, 1),
        "wnorm_traj":    wnorm_traj,
    }


# ─── Results aggregation ──────────────────────────────────────────────────────

def aggregate_arm(cells):
    """Compute per-arm grok rate and wnorm summary."""
    n = len(cells)
    if n == 0:
        return {}
    grokkd = [c for c in cells if c["grokked"]]
    onsets = [c["grok_onset"] for c in grokkd if c["grok_onset"]]
    wnorm_ratios = [c["wnorm_ratio"] for c in cells]
    return {
        "n_seeds":          n,
        "grok_rate":        f"{len(grokkd)}/{n}",
        "grok_rate_float":  len(grokkd) / n,
        "mean_onset":       round(sum(onsets)/len(onsets), 0) if onsets else None,
        "std_onset":        round(
            (sum((x - sum(onsets)/len(onsets))**2 for x in onsets)/len(onsets))**0.5, 0
        ) if len(onsets) > 1 else None,
        "final_wnorm_ratio_mean": round(sum(wnorm_ratios)/n, 4),
        "final_wnorm_ratio_std":  round(
            (sum((x - sum(wnorm_ratios)/n)**2 for x in wnorm_ratios)/n)**0.5, 4
        ) if n > 1 else 0.0,
        "wnorm_ratios": wnorm_ratios,
    }


def estimate_effective_wd(arms_data):
    """
    Estimate BN_all+noWD's effective WD by matching its final wnorm_ratio
    to the LN+WD sweep cells.

    Returns: (wd_estimate, method_note, comparison_table)
    """
    bn_all = arms_data.get("BN_all_noWD", {})
    bn_wnorm = bn_all.get("final_wnorm_ratio_mean")
    if bn_wnorm is None:
        return None, "insufficient data", {}

    # Reference: LN+WD cells
    ln_refs = {
        0.0:  arms_data.get("LN_wd0.0",   {}),
        0.1:  arms_data.get("LN_wd0.1",   {}),   # from grok_v2
        0.25: arms_data.get("LN_wd0.25",  {}),
        0.5:  arms_data.get("LN_wd0.5",   {}),
        0.75: arms_data.get("LN_wd0.75",  {}),
        1.0:  arms_data.get("LN_wd1.0",   {}),   # from grok_v2
    }

    best_wd, best_diff = None, float("inf")
    comparison = {}
    for wd, agg in ln_refs.items():
        ln_wnorm = agg.get("final_wnorm_ratio_mean")
        if ln_wnorm is None:
            continue
        diff = abs(bn_wnorm - ln_wnorm)
        comparison[wd] = {"ln_wnorm_ratio_mean": ln_wnorm, "abs_diff_from_bn": round(diff, 4)}
        if diff < best_diff:
            best_diff = diff
            best_wd   = wd

    method = (
        f"Matched BN_all+noWD final_wnorm_ratio_mean={bn_wnorm:.4f} "
        f"to nearest LN+WD arm by absolute difference in wnorm_ratio. "
        f"Best match: LN+WD={best_wd} (diff={best_diff:.4f})."
    )
    return best_wd, method, comparison


def save_results(arms_data, all_cells, status="RUNNING"):
    """Write incremental RESULTS.json in the required format."""
    # Check if BN_mlp_only groks
    mlp_arm  = arms_data.get("BN_mlp_only_noWD", {})
    bn_mlp_groks = mlp_arm.get("grok_rate_float", 0.0) > 0.0 if mlp_arm else None

    # WD threshold interval from LN finer sweep
    # grok_v2: LN+WD=0.1 → 0/5, LN+WD=1.0 → 5/5; now adding 0.25, 0.5, 0.75
    wd_threshold_lo = 0.1
    wd_threshold_hi = 1.0
    for wd in [0.25, 0.5, 0.75]:
        arm_key = f"LN_wd{wd}"
        agg = arms_data.get(arm_key, {})
        rate = agg.get("grok_rate_float", None)
        if rate is not None:
            if rate > 0.0 and wd < wd_threshold_hi:
                wd_threshold_hi = wd
            if rate == 0.0 and wd > wd_threshold_lo:
                wd_threshold_lo = wd

    # Effective WD estimate
    wd_eff, wd_eff_method, wd_eff_table = estimate_effective_wd(arms_data)

    out = {
        "status": status,
        "scale":  "probe",
        "arms":   {k: v for k, v in arms_data.items()},
        "bn_mlp_only_groks": bn_mlp_groks,
        "grokking_wd_threshold_interval": [wd_threshold_lo, wd_threshold_hi],
        "bn_effective_wd_estimate": wd_eff,
        "estimate_method": wd_eff_method,
        "wd_eff_comparison_table": wd_eff_table,
        "n_seeds": 5,
        "metrics": {
            "n_cells_completed": len(all_cells),
        },
        "subject_executed": (
            "Arms: BN_all_noWD, BN_mlp_only_noWD, BN_attn_only_noWD (norm1=BN attn, norm2=BN MLP, "
            "mixed), LN_wd{0.25,0.5,0.75}. 5 seeds each. 1-layer Transformer d=128 h=4 mlp=512, "
            "P=97 mod-add, BS=256, AdamW LR=1e-3, 15k-step cap. "
            "grok_v2 reused for LN+WD=0.0, 0.1, 1.0 comparison anchors."
        ),
        "notes": "",
    }

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(out, f, indent=2)


# ─── Main ─────────────────────────────────────────────────────────────────────

# Cell definitions: (arm_name, norm1_type, norm2_type, weight_decay)
# norm1 = before attention; norm2 = before MLP (and output)
ARMS = [
    # 1. BN_all+noWD — replicate from v2 (both sublayers BN)
    ("BN_all_noWD",       "BatchNorm", "BatchNorm", 0.0),
    # 2. BN_mlp_only+noWD — KEY: BN only on MLP, LN on attention
    ("BN_mlp_only_noWD",  "LayerNorm", "BatchNorm", 0.0),
    # 3. BN_attn_only+noWD — BN only on attention, LN on MLP
    ("BN_attn_only_noWD", "BatchNorm", "LayerNorm", 0.0),
    # 4. LN + finer WD sweep
    ("LN_wd0.25",         "LayerNorm", "LayerNorm", 0.25),
    ("LN_wd0.5",          "LayerNorm", "LayerNorm", 0.50),
    ("LN_wd0.75",         "LayerNorm", "LayerNorm", 0.75),
]


def main():
    t_total = time.time()
    print("Grokking Ablation v3 — BN sublayer decomposition", flush=True)
    print(f"Arms: {[a[0] for a in ARMS]}", flush=True)
    print(f"Seeds: {SEEDS}, BS={BATCH_SIZE}, max_steps={MAX_STEPS}", flush=True)
    print(f"Device: {DEVICE}", flush=True)
    print(f"Total cells: {len(ARMS) * len(SEEDS)}", flush=True)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(CURVES_DIR, exist_ok=True)

    all_cells  = []   # flat list of cell results
    arms_data  = {}   # arm_name → aggregated stats
    arm_cells  = {}   # arm_name → list of cell results

    for arm_name, norm1, norm2, wd in ARMS:
        arm_cells[arm_name] = []

    save_results(arms_data, all_cells, status="RUNNING")

    for arm_idx, (arm_name, norm1, norm2, wd) in enumerate(ARMS):
        print(f"\n{'#'*60}", flush=True)
        print(f"ARM [{arm_idx+1}/{len(ARMS)}]: {arm_name}", flush=True)
        print(f"  norm1={norm1}, norm2={norm2}, WD={wd}", flush=True)

        for s_idx, seed in enumerate(SEEDS):
            print(f"\n  [{s_idx+1}/{len(SEEDS)}] seed={seed}", flush=True)
            result = run_cell(arm_name, norm1, norm2, wd, seed)
            all_cells.append(result)
            arm_cells[arm_name].append(result)

        # Aggregate this arm
        arms_data[arm_name] = aggregate_arm(arm_cells[arm_name])

        # Save weight-norm trajectory for this arm (all seeds, sampled)
        traj_path = f"{CURVES_DIR}/{arm_name}.json"
        with open(traj_path, "w") as f:
            json.dump({
                "arm": arm_name, "norm1": norm1, "norm2": norm2, "wd": wd,
                "seeds": [c["seed"] for c in arm_cells[arm_name]],
                "wnorm_trajs": {
                    str(c["seed"]): c["wnorm_traj"] for c in arm_cells[arm_name]
                },
            }, f, indent=2)

        print(f"\n  ARM SUMMARY: {arm_name} → {arms_data[arm_name]['grok_rate']} grokked "
              f"| mean_onset={arms_data[arm_name]['mean_onset']} "
              f"| mean_wnorm_ratio={arms_data[arm_name]['final_wnorm_ratio_mean']:.4f}",
              flush=True)

        # Save incremental results after each arm
        is_last_arm = arm_idx == len(ARMS) - 1
        save_results(arms_data, all_cells,
                     status="DONE" if is_last_arm else "RUNNING")

    total_elapsed = time.time() - t_total
    print(f"\n{'='*60}", flush=True)
    print(f"ALL DONE: {len(all_cells)} cells in {total_elapsed:.0f}s ({total_elapsed/60:.1f} min)",
          flush=True)

    save_results(arms_data, all_cells, status="DONE")

    # Final summary
    print("\n=== SUMMARY ===", flush=True)
    for arm_name, agg in arms_data.items():
        print(f"  {arm_name:25s}: {agg['grok_rate']:5s} groks "
              f"| onset={str(agg['mean_onset']):8s} "
              f"| wnorm_ratio={agg['final_wnorm_ratio_mean']:.4f}", flush=True)

    # Print final RESULTS.json
    with open(RESULTS_PATH) as f:
        r = json.load(f)
    print("\nFinal RESULTS.json:", flush=True)
    print(json.dumps(r, indent=2), flush=True)


if __name__ == "__main__":
    main()
