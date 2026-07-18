"""
Grokking Grid Search — round increase_complexity-01 (probe)

Grid: norm {LayerNorm, BatchNorm} x weight_decay {0, 1.0} x batch_size {64, 256, 512}
Seeds: {42, 123}  →  24 cells total

Tests:
  (a) BN+no-WD groks ONLY at small batch (implicit noise reg ~ 1/batch)?
  (b) BN+no-WD consistent with LN+WD (implicit-L2 equivalence)?
  (c) CONTROL: LN+no-WD should NOT grok.

Writes incremental RESULTS.json after every cell.
Cap: 15k steps per cell or early-stop on sustained grokking (val_acc>0.95, 3 evals).
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

# ─── Constants ───────────────────────────────────────────────────────────────
P          = 97
TRAIN_FRAC = 0.4
LR         = 1e-3
MAX_STEPS  = 15_000
EVAL_EVERY = 100
LOG_EVERY  = 1_000   # per-cell progress print

RESULTS_PATH = "results/increase_complexity-01/RESULTS.json"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}", flush=True)

# Grid axes
NORMS         = ["LayerNorm", "BatchNorm"]
WEIGHT_DECAYS = [0.0, 1.0]
BATCH_SIZES   = [64, 256, 512]
SEEDS         = [42, 123]

# ─── Helpers ────────────────────────────────────────────────────────────────

class TransposedBN(nn.Module):
    """BatchNorm1d wrapper for (B, T, D) tensors."""
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
        raise ValueError(f"Unknown norm: {norm_type}")


# ─── Architecture ────────────────────────────────────────────────────────────

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, mlp_hidden, norm_type):
        super().__init__()
        self.norm1 = make_norm(norm_type, d_model)
        self.norm2 = make_norm(norm_type, d_model)
        self.attn  = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.mlp   = nn.Sequential(
            nn.Linear(d_model, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, d_model),
        )

    def forward(self, x):
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class GrokTransformer(nn.Module):
    D_MODEL   = 128
    N_HEADS   = 4
    MLP_HID   = 512
    VOCAB_SZ  = P + 1   # 98
    SEQ_LEN   = 3
    NUM_CLS   = P       # 97

    def __init__(self, norm_type):
        super().__init__()
        d = self.D_MODEL
        self.tok_emb = nn.Embedding(self.VOCAB_SZ, d)
        self.pos_emb = nn.Embedding(self.SEQ_LEN, d)
        self.block   = TransformerBlock(d, self.N_HEADS, self.MLP_HID, norm_type)
        self.ln_out  = make_norm(norm_type, d)
        self.head    = nn.Linear(d, self.NUM_CLS)

    def forward(self, x):
        B, T = x.shape
        pos  = torch.arange(T, device=x.device).unsqueeze(0)
        emb  = self.tok_emb(x) + self.pos_emb(pos)
        h    = self.block(emb)
        h    = self.ln_out(h)
        return self.head(h[:, -1, :])

    def weight_norm(self):
        total = sum(p.data.norm(2).item() ** 2
                    for n, p in self.named_parameters()
                    if 'weight' in n)
        return math.sqrt(total)


# ─── Dataset ─────────────────────────────────────────────────────────────────

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
        a = s[:, 0:1]; b = s[:, 1:2]
        sep = torch.full((len(s), 1), SEP, dtype=torch.long)
        return torch.cat([a, sep, b], 1), s[:, 2]

    return make_xy(tr), make_xy(va), n_train, len(va)


# ─── Single cell ─────────────────────────────────────────────────────────────

def run_cell(norm_type, weight_decay, batch_size, seed):
    label = f"norm={norm_type} WD={weight_decay} BS={batch_size} seed={seed}"
    print(f"\n{'='*60}", flush=True)
    print(f"CELL: {label}", flush=True)
    print(f"{'='*60}", flush=True)

    # Reproducibility
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Data
    (x_tr, y_tr), (x_va, y_va), n_train, n_val = make_dataset(seed)
    x_va = x_va.to(DEVICE); y_va = y_va.to(DEVICE)
    x_tr_full = x_tr.to(DEVICE); y_tr_full = y_tr.to(DEVICE)

    train_ds = TensorDataset(x_tr, y_tr)
    loader   = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          generator=torch.Generator().manual_seed(seed))

    # Model
    torch.manual_seed(seed)
    model = GrokTransformer(norm_type).to(DEVICE)
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

    grok_onset = None
    grok_confirmed = None
    history = []
    step = 0
    data_iter = iter(loader)
    t0 = time.time()

    while step < MAX_STEPS:
        # batch
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

        # Eval
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
            history.append(dict(step=step, tr_acc=tr_acc, tr_loss=tr_loss,
                                val_acc=val_acc, val_loss=val_loss, wnorm=wnorm))

            if step % LOG_EVERY == 0:
                elapsed = time.time() - t0
                print(f"  step={step:5d} | tr={tr_acc:.3f} | val={val_acc:.3f} "
                      f"| wnorm={wnorm:.2f} | {elapsed:.0f}s", flush=True)

            if val_acc > 0.95 and grok_onset is None:
                grok_onset = step
                print(f"  *** GROKKING ONSET step={step} val_acc={val_acc:.4f}", flush=True)

            if grok_onset is not None and grok_confirmed is None:
                recent = [r['val_acc'] for r in history[-3:]]
                if len(recent) >= 3 and all(v > 0.95 for v in recent):
                    grok_confirmed = step
                    print(f"  *** GROKKING CONFIRMED step={step}", flush=True)

        # Early stop
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
        final_val_acc  = (model(x_va).argmax(-1) == y_va).float().mean().item()
        final_tr_acc   = (model(x_tr_full).argmax(-1) == y_tr_full).float().mean().item()
    final_wnorm = model.weight_norm()
    elapsed = time.time() - t0

    grokkd = grok_onset is not None
    print(f"  DONE | steps={step} | grok={'YES @'+str(grok_onset) if grokkd else 'NO'} "
          f"| val={final_val_acc:.3f} | wnorm_ratio={final_wnorm/init_wnorm:.2f} "
          f"| {elapsed:.0f}s", flush=True)

    # Weight norm at key steps for trajectory
    wnorm_traj = [(r['step'], r['wnorm']) for r in history[::5]]  # every 500 steps

    return {
        "norm_type":     norm_type,
        "weight_decay":  weight_decay,
        "batch_size":    batch_size,
        "seed":          seed,
        "grokked":       grokkd,
        "grok_onset":    grok_onset,
        "grok_confirmed": grok_confirmed,
        "total_steps":   step,
        "final_val_acc": round(final_val_acc, 6),
        "final_tr_acc":  round(final_tr_acc, 6),
        "init_wnorm":    round(init_wnorm, 4),
        "final_wnorm":   round(final_wnorm, 4),
        "wnorm_ratio":   round(final_wnorm / init_wnorm, 4),
        "elapsed_s":     round(elapsed, 1),
        "wnorm_traj_sampled": wnorm_traj,
    }


# ─── Incremental save ────────────────────────────────────────────────────────

def save_results(cells_done, status="RUNNING"):
    """Write incremental RESULTS.json."""
    # Compute verdicts if we have enough data
    # (a) BN+WD=0: groks at small batch but not 512?
    # (b) BN+WD=0 grokking consistent with LN+WD=1.0?
    # (c) LN+WD=0 should NOT grok

    def cells_matching(**kwargs):
        return [c for c in cells_done
                if all(c.get(k) == v for k, v in kwargs.items())]

    # Build grid table
    grid = []
    for c in cells_done:
        grid.append({
            "norm": c["norm_type"],
            "WD": c["weight_decay"],
            "BS": c["batch_size"],
            "seed": c["seed"],
            "grokked": c["grokked"],
            "grok_onset": c["grok_onset"],
            "final_val_acc": c["final_val_acc"],
            "wnorm_ratio": c["wnorm_ratio"],
        })

    # Verdicts (only if enough cells done to evaluate)
    bn_nowd = cells_matching(norm_type="BatchNorm", weight_decay=0.0)
    ln_wd   = cells_matching(norm_type="LayerNorm", weight_decay=1.0)
    ln_nowd = cells_matching(norm_type="LayerNorm", weight_decay=0.0)

    # (a) BN+noWD batch-size effect
    bn_nowd_64  = [c for c in bn_nowd if c["batch_size"] == 64]
    bn_nowd_256 = [c for c in bn_nowd if c["batch_size"] == 256]
    bn_nowd_512 = [c for c in bn_nowd if c["batch_size"] == 512]

    def grok_rate(lst): return sum(1 for c in lst if c["grokked"]) / max(len(lst), 1)
    def mean_onset(lst):
        onsets = [c["grok_onset"] for c in lst if c["grokked"] and c["grok_onset"]]
        return round(sum(onsets)/len(onsets), 0) if onsets else None

    verdict_a = "PENDING"
    if bn_nowd_64 and bn_nowd_256 and bn_nowd_512:
        r64, r256, r512 = grok_rate(bn_nowd_64), grok_rate(bn_nowd_256), grok_rate(bn_nowd_512)
        verdict_a = (
            f"BN+noWD grok rates: BS64={r64:.1f}, BS256={r256:.1f}, BS512={r512:.1f}. "
            f"Mean onset: BS64={mean_onset(bn_nowd_64)}, BS256={mean_onset(bn_nowd_256)}, "
            f"BS512={mean_onset(bn_nowd_512)}. "
        )
        if r64 > r512:
            verdict_a += "CONFIRMED: small batch groks more than large batch."
        elif r64 == r512 == 1.0:
            verdict_a += "PARTIAL: BN+noWD groks at all batch sizes (implicit reg stronger than expected)."
        elif r64 == r512 == 0.0:
            verdict_a += "REFUTED: BN+noWD does not grok at any batch size."
        else:
            verdict_a += f"INCONCLUSIVE."

    # (b) BN+noWD vs LN+WD=1.0 consistency
    verdict_b = "PENDING"
    if bn_nowd and ln_wd:
        bn_groks = [c for c in bn_nowd if c["grokked"] and c["grok_onset"]]
        ln_groks = [c for c in ln_wd  if c["grokked"] and c["grok_onset"]]
        bn_onsets = [c["grok_onset"] for c in bn_groks]
        ln_onsets = [c["grok_onset"] for c in ln_groks]
        bn_mean = round(sum(bn_onsets)/len(bn_onsets), 0) if bn_onsets else None
        ln_mean = round(sum(ln_onsets)/len(ln_onsets), 0) if ln_onsets else None
        if bn_mean and ln_mean:
            ratio = abs(bn_mean - ln_mean) / max(ln_mean, 1)
            verdict_b = (
                f"BN+noWD mean onset={bn_mean} vs LN+WD=1.0 mean onset={ln_mean}. "
                f"Ratio diff={ratio:.2f}. "
            )
            if ratio < 0.25:
                verdict_b += "CONSISTENT (within 25%): BN implicit reg ~ LN+WD."
            elif ratio < 0.5:
                verdict_b += "SOMEWHAT CONSISTENT (25-50% diff): partial equivalence."
            else:
                verdict_b += "INCONSISTENT (>50% diff): BN != LN+WD in onset timing."
        elif bn_onsets and not ln_onsets:
            verdict_b = "INCONSISTENT: BN+noWD groks but LN+WD=1.0 does not."
        elif ln_onsets and not bn_onsets:
            verdict_b = "INCONSISTENT: LN+WD=1.0 groks but BN+noWD does not."
        else:
            verdict_b = "BOTH NO-GROK: neither BN+noWD nor LN+WD=1.0 grokked."

    # (c) LN+noWD control
    verdict_c = "PENDING"
    if ln_nowd:
        grok_count = sum(1 for c in ln_nowd if c["grokked"])
        verdict_c = (
            f"LN+noWD: {grok_count}/{len(ln_nowd)} cells grokked. "
        )
        if grok_count == 0:
            verdict_c += "CONFIRMED: LN+noWD never groks (no regularization)."
        else:
            verdict_c += f"VIOLATED: {grok_count} cells grokked unexpectedly."

    results = {
        "status": status,
        "scale":  "probe",
        "metrics": {
            "n_cells_completed": len(cells_done),
            "n_cells_total":     24,
            "grid_table":        grid,
            "summary_by_condition": {
                "BN_noWD": {
                    "n_cells": len(bn_nowd),
                    "grok_rate": round(grok_rate(bn_nowd), 3),
                    "mean_onset": mean_onset(bn_nowd),
                    "BS_breakdown": {
                        "64":  {"n": len(bn_nowd_64),  "grok_rate": round(grok_rate(bn_nowd_64), 2),  "mean_onset": mean_onset(bn_nowd_64)},
                        "256": {"n": len(bn_nowd_256), "grok_rate": round(grok_rate(bn_nowd_256), 2), "mean_onset": mean_onset(bn_nowd_256)},
                        "512": {"n": len(bn_nowd_512), "grok_rate": round(grok_rate(bn_nowd_512), 2), "mean_onset": mean_onset(bn_nowd_512)},
                    },
                },
                "LN_WD1p0": {
                    "n_cells": len(ln_wd),
                    "grok_rate": round(grok_rate(ln_wd), 3),
                    "mean_onset": mean_onset(ln_wd),
                },
                "LN_noWD": {
                    "n_cells": len(ln_nowd),
                    "grok_rate": round(grok_rate(ln_nowd), 3),
                    "mean_onset": mean_onset(ln_nowd),
                },
                "BN_WD1p0": {
                    "n_cells": len(cells_matching(norm_type="BatchNorm", weight_decay=1.0)),
                    "grok_rate": round(grok_rate(cells_matching(norm_type="BatchNorm", weight_decay=1.0)), 3),
                    "mean_onset": mean_onset(cells_matching(norm_type="BatchNorm", weight_decay=1.0)),
                },
            },
            "mechanistic_verdicts": {
                "(a) BN+noWD_batch_size_effect": verdict_a,
                "(b) BN+noWD_vs_LN+WD_consistency": verdict_b,
                "(c) LN+noWD_control": verdict_c,
            },
        },
        "subject_executed": (
            "Grid: norm {LN,BN} x WD {0,1.0} x BS {64,256,512} x seeds {42,123} = 24 cells. "
            "1-layer transformer (d=128, h=4, mlp=512) on mod-97 addition. "
            "AdamW LR=1e-3, max 15k steps, early-stop at sustained val_acc>0.95."
        ),
        "notes": (
            "Building on baseline-00 (LN+WD=1.0 grokked at step 5500). "
            "BN applied as BatchNorm1d over (B*T) dim in pre-norm positions. "
            f"Status: {len(cells_done)}/24 cells done."
        ),
    }

    os.makedirs("results/increase_complexity-01", exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    t_total = time.time()
    print(f"Grokking Grid Search — increase_complexity-01", flush=True)
    print(f"Grid: norm={NORMS}, WD={WEIGHT_DECAYS}, BS={BATCH_SIZES}, seeds={SEEDS}", flush=True)
    print(f"Max steps/cell={MAX_STEPS}, EVAL_EVERY={EVAL_EVERY}", flush=True)
    print(f"Device: {DEVICE}", flush=True)

    # Build ordered list of cells
    # Order: WD=1.0 first (fast grokking confirms setup), then WD=0 cells
    cells_todo = []
    for norm in NORMS:
        for wd in WEIGHT_DECAYS:
            for bs in BATCH_SIZES:
                for seed in SEEDS:
                    cells_todo.append((norm, wd, bs, seed))

    print(f"Total cells: {len(cells_todo)}", flush=True)

    cells_done = []
    save_results(cells_done, status="RUNNING")

    for i, (norm, wd, bs, seed) in enumerate(cells_todo):
        print(f"\n[{i+1}/{len(cells_todo)}] Starting cell: {norm} WD={wd} BS={bs} seed={seed}",
              flush=True)
        result = run_cell(norm, wd, bs, seed)
        cells_done.append(result)
        save_results(cells_done, status="RUNNING" if i < len(cells_todo)-1 else "SUCCESS")
        print(f"  → Saved incremental results ({len(cells_done)}/24 done)", flush=True)

    total_elapsed = time.time() - t_total
    print(f"\n{'='*60}", flush=True)
    print(f"ALL DONE: {len(cells_done)} cells in {total_elapsed:.0f}s ({total_elapsed/60:.1f} min)",
          flush=True)
    save_results(cells_done, status="SUCCESS")

    # Print summary
    with open(RESULTS_PATH) as f:
        r = json.load(f)
    print("\nFinal RESULTS.json:", flush=True)
    print(json.dumps(r, indent=2), flush=True)


if __name__ == "__main__":
    main()
