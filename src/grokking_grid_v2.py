"""
Grokking Grid v2 — round grok_v2 (probe)

Grid: norm {LayerNorm, BatchNorm} × WD {0.0, 0.01, 0.1, 1.0} × seed {42,123,7,99,2024}
Fixed batch_size = 256. Total: 2 × 4 × 5 = 40 cells.

Changes vs increase_complexity-01:
- 5 seeds (was 2)
- WD ∈ {0.0, 0.01, 0.1, 1.0} (was {0, 1.0})
- Fixed BS=256 (was {64,256,512})
- Full training curves for seed=42 cells (8 representative, no early stop)
- Wilson 95% CI on grok_rate per condition
- init_val_ce sanity check at step 0
- Aggregate stats: grok_rate, mean±std onset, final_val_accs list
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
LOG_EVERY  = 1_000

RESULTS_PATH = "results/grok_v2/RESULTS.json"
CURVES_DIR   = "results/grok_v2/curves"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}", flush=True)

# Grid axes
NORMS         = ["LayerNorm", "BatchNorm"]
WEIGHT_DECAYS = [0.0, 0.01, 0.1, 1.0]
SEEDS         = [42, 123, 7, 99, 2024]
BATCH_SIZE    = 256    # fixed for this round

# Representative seed (save full curves, no early stop)
CURVE_SEED    = 42

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


def wilson_ci(k, n, z=1.96):
    """Wilson 95% CI for a proportion k/n."""
    if n == 0:
        return [0.0, 1.0]
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    margin = (z / denom) * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
    lo = max(0.0, center - margin)
    hi = min(1.0, center + margin)
    return [round(lo, 4), round(hi, 4)]


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

def run_cell(norm_type, weight_decay, seed, save_curve=False):
    """
    Train one cell.
    save_curve=True: run to full 15k steps (no early stop), return full history.
    save_curve=False: apply early stop on sustained grokking.
    """
    label = f"norm={norm_type} WD={weight_decay} seed={seed}"
    print(f"\n{'='*60}", flush=True)
    print(f"CELL: {label} {'[CURVE]' if save_curve else ''}", flush=True)
    print(f"{'='*60}", flush=True)

    # Reproducibility
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Data
    (x_tr, y_tr), (x_va, y_va), n_train, n_val = make_dataset(seed)
    x_va = x_va.to(DEVICE); y_va = y_va.to(DEVICE)
    x_tr_full = x_tr.to(DEVICE); y_tr_full = y_tr.to(DEVICE)

    train_ds = TensorDataset(x_tr, y_tr)
    loader   = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          generator=torch.Generator().manual_seed(seed))

    # Model
    torch.manual_seed(seed)
    model = GrokTransformer(norm_type).to(DEVICE)
    init_wnorm = model.weight_norm()

    # Measure init_val_ce at step 0 (sanity check)
    model.eval()
    with torch.no_grad():
        logits0 = model(x_va)
        init_val_ce = F.cross_entropy(logits0, y_va).item()
    print(f"  init_val_ce={init_val_ce:.4f}  ln(P={P})={math.log(P):.4f}", flush=True)

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
            history.append(dict(step=step, train_acc=tr_acc, train_loss=tr_loss,
                                val_acc=val_acc, val_loss=val_loss, weight_norm=wnorm))

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

        # Early stop only for non-curve cells
        if not save_curve and grok_confirmed is not None:
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

    cell_result = {
        "norm_type":      norm_type,
        "weight_decay":   weight_decay,
        "seed":           seed,
        "grokked":        grokkd,
        "grok_onset":     grok_onset,
        "grok_confirmed": grok_confirmed,
        "total_steps":    step,
        "final_val_acc":  round(final_val_acc, 6),
        "final_tr_acc":   round(final_tr_acc, 6),
        "init_wnorm":     round(init_wnorm, 4),
        "final_wnorm":    round(final_wnorm, 4),
        "wnorm_ratio":    round(final_wnorm / init_wnorm, 4),
        "init_val_ce":    round(init_val_ce, 4),
        "elapsed_s":      round(elapsed, 1),
    }

    # Save curve if requested
    if save_curve:
        curve_name = f"{norm_type.lower()}_wd{weight_decay}"
        curve_path = os.path.join(CURVES_DIR, f"{curve_name}.json")
        curve_data = {
            "norm": norm_type,
            "WD": weight_decay,
            "seed": seed,
            "history": history,
        }
        with open(curve_path, "w") as f:
            json.dump(curve_data, f)
        print(f"  Saved curve: {curve_path}", flush=True)
        cell_result["curve_saved"] = curve_path

    return cell_result


# ─── Aggregate stats ─────────────────────────────────────────────────────────

def compute_by_condition(cells_done):
    """Aggregate per (norm, WD) condition."""
    conditions = {}
    for norm in NORMS:
        for wd in WEIGHT_DECAYS:
            key = f"{norm.lower().replace('norm', '')}_wd{wd}"
            cells = [c for c in cells_done
                     if c["norm_type"] == norm and c["weight_decay"] == wd]
            n = len(cells)
            grok_count = sum(1 for c in cells if c["grokked"])
            onsets = [c["grok_onset"] for c in cells if c["grokked"] and c["grok_onset"]]
            final_accs = [c["final_val_acc"] for c in cells]
            wnorm_ratios = [c["wnorm_ratio"] for c in cells]

            # mean onset / std onset
            if onsets:
                mean_onset = round(sum(onsets) / len(onsets), 1)
                var = sum((x - mean_onset)**2 for x in onsets) / len(onsets)
                std_onset = round(math.sqrt(var), 1)
            else:
                mean_onset = None
                std_onset = None

            conditions[key] = {
                "norm": norm,
                "WD": wd,
                "n_cells": n,
                "grok_rate": f"{grok_count}/{n}",
                "grok_rate_float": round(grok_count / n, 4) if n > 0 else 0.0,
                "wilson_ci95": wilson_ci(grok_count, n) if n > 0 else [0.0, 1.0],
                "mean_onset": mean_onset,
                "std_onset": std_onset,
                "final_val_accs": [round(a, 6) for a in final_accs],
                "wnorm_ratios": [round(r, 4) for r in wnorm_ratios],
            }
    return conditions


def determine_implicit_wd_equivalent(by_condition):
    """
    Find the explicit WD level for LN whose grok_rate matches BN+noWD.
    BN+noWD rate: 0/n. If BN still doesn't grok, this is null.
    """
    bn_nowd_key = "batch_wd0.0"
    # Fix key naming
    bn_nowd = None
    for k, v in by_condition.items():
        if v["norm"] == "BatchNorm" and v["WD"] == 0.0:
            bn_nowd = v
            break

    if bn_nowd is None or bn_nowd["n_cells"] == 0:
        return None

    # Extract BN+noWD grok rate
    bn_rate_str = bn_nowd["grok_rate"]
    bn_k = int(bn_rate_str.split("/")[0])
    bn_n = int(bn_rate_str.split("/")[1])
    bn_rate = bn_k / bn_n if bn_n > 0 else 0.0

    # Find LN condition with closest grok_rate to BN+noWD
    # Only complete (n=5) conditions
    best_match = None
    best_diff = float('inf')
    for k, v in by_condition.items():
        if v["norm"] != "LayerNorm":
            continue
        if v["n_cells"] < 5:
            continue  # not complete
        ln_rate = v["grok_rate_float"]
        diff = abs(ln_rate - bn_rate)
        if diff < best_diff:
            best_diff = diff
            best_match = v["WD"]

    # Only report if it's an exact match (same rate) or within 1 seed
    if best_diff <= 1/5 + 0.01:  # within 1/5 = one seed
        return best_match
    return None


# ─── Incremental save ────────────────────────────────────────────────────────

def save_results(cells_done, status="RUNNING", init_val_ce=None, curves_saved=None):
    """Write incremental RESULTS.json in final format."""

    # Grid table
    grid = []
    for c in cells_done:
        grid.append({
            "norm": c["norm_type"],
            "WD": c["weight_decay"],
            "seed": c["seed"],
            "grokked": c["grokked"],
            "grok_onset": c["grok_onset"],
            "final_val_acc": c["final_val_acc"],
            "wnorm_ratio": c["wnorm_ratio"],
        })

    by_condition = compute_by_condition(cells_done)

    # BN+noWD verdict
    bn_nowd_cells = [c for c in cells_done
                     if c["norm_type"] == "BatchNorm" and c["weight_decay"] == 0.0]
    bn_nowd_groks = any(c["grokked"] for c in bn_nowd_cells) if bn_nowd_cells else None

    # implicit WD equivalent
    implicit_wd_equiv = determine_implicit_wd_equivalent(by_condition) if len(cells_done) >= 20 else None

    # Curves saved
    if curves_saved is None:
        curves_saved = []

    # init_val_ce — use first cell's value (should all be same model init)
    if init_val_ce is None and cells_done:
        init_val_ce = cells_done[0].get("init_val_ce", None)

    # Build notes
    n_done = len(cells_done)
    n_total = len(NORMS) * len(WEIGHT_DECAYS) * len(SEEDS)  # 40

    ln_wd1_cells = [c for c in cells_done
                    if c["norm_type"] == "LayerNorm" and c["weight_decay"] == 1.0]
    ln_nowd_cells = [c for c in cells_done
                     if c["norm_type"] == "LayerNorm" and c["weight_decay"] == 0.0]

    notes_parts = [
        f"Build on increase_complexity-01: BN+noWD 0/6 groks (BS{{64,256,512}}, 2 seeds). "
        f"This round: fixed BS=256, 5 seeds, WD expanded to {{0,0.01,0.1,1.0}}.",
        f"Status: {n_done}/{n_total} cells done.",
    ]
    if bn_nowd_cells:
        bn_nowd_rate = sum(1 for c in bn_nowd_cells if c["grokked"])
        notes_parts.append(f"BN+noWD: {bn_nowd_rate}/{len(bn_nowd_cells)} grokked.")
    if ln_wd1_cells:
        ln_wd1_rate = sum(1 for c in ln_wd1_cells if c["grokked"])
        notes_parts.append(f"LN+WD=1.0: {ln_wd1_rate}/{len(ln_wd1_cells)} grokked.")
    if ln_nowd_cells:
        ln_nowd_rate = sum(1 for c in ln_nowd_cells if c["grokked"])
        notes_parts.append(f"LN+noWD: {ln_nowd_rate}/{len(ln_nowd_cells)} grokked.")

    # Check for anomalies
    for c in cells_done:
        if not c["grokked"] and c["final_val_acc"] > 0.5:
            notes_parts.append(
                f"ANOMALY: {c['norm_type']}+WD={c['weight_decay']} seed={c['seed']} "
                f"stalled at val_acc={c['final_val_acc']:.3f} (never crossed 0.95 threshold)."
            )

    results = {
        "status": status,
        "scale":  "probe",
        "grid":   grid,
        "by_condition": by_condition,
        "bn_nowd_groks": bn_nowd_groks,
        "implicit_wd_equivalent_level": implicit_wd_equiv,
        "init_val_ce": round(init_val_ce, 4) if init_val_ce else None,
        "vocab_size": P,
        "ln_vocab": round(math.log(P), 4),
        "n_seeds": len(SEEDS),
        "curves_saved": curves_saved,
        "metrics": {
            "n_cells_completed": n_done,
            "n_cells_total":     n_total,
        },
        "subject_executed": (
            f"Grid: norm {{LN,BN}} × WD {{0,0.01,0.1,1.0}} × seed {{42,123,7,99,2024}} "
            f"= 40 cells. Fixed BS=256. 1-layer transformer (d=128, h=4, mlp=512) on "
            f"mod-97 addition. AdamW LR=1e-3, max 15k steps. Seed=42 cells: full 15k "
            f"(no early stop), curves saved. Other seeds: early-stop on sustained val_acc>0.95."
        ),
        "notes": " ".join(notes_parts),
    }

    os.makedirs("results/grok_v2", exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    t_total = time.time()
    print(f"Grokking Grid v2 — grok_v2 (probe)", flush=True)
    print(f"Grid: norm={NORMS}, WD={WEIGHT_DECAYS}, seeds={SEEDS}, BS={BATCH_SIZE}", flush=True)
    print(f"Max steps/cell={MAX_STEPS}, EVAL_EVERY={EVAL_EVERY}", flush=True)
    print(f"Device: {DEVICE}", flush=True)

    os.makedirs(CURVES_DIR, exist_ok=True)

    # Cell ordering: run seed=42 (curve cells) first so we fail fast if broken,
    # then remaining seeds. Within each seed, order: WD desc (fast grokkers first).
    cells_todo = []
    # Curve cells first: seed=42, all norm×WD
    for norm in NORMS:
        for wd in WEIGHT_DECAYS:
            cells_todo.append((norm, wd, CURVE_SEED, True))
    # Then remaining seeds (non-curve)
    for seed in SEEDS:
        if seed == CURVE_SEED:
            continue
        for norm in NORMS:
            for wd in sorted(WEIGHT_DECAYS, reverse=True):  # WD=1.0 first (fast grokking)
                cells_todo.append((norm, wd, seed, False))

    n_total = len(cells_todo)
    print(f"Total cells: {n_total}", flush=True)

    cells_done = []
    curves_saved = []
    first_init_val_ce = None

    save_results(cells_done, status="RUNNING")

    for i, (norm, wd, seed, save_curve) in enumerate(cells_todo):
        print(f"\n[{i+1}/{n_total}] Starting: {norm} WD={wd} seed={seed} curve={save_curve}",
              flush=True)
        result = run_cell(norm, wd, seed, save_curve=save_curve)
        cells_done.append(result)

        if first_init_val_ce is None:
            first_init_val_ce = result.get("init_val_ce")

        if save_curve and "curve_saved" in result:
            curves_saved.append(result["curve_saved"])

        is_last = (i == n_total - 1)
        save_results(
            cells_done,
            status="DONE" if is_last else "RUNNING",
            init_val_ce=first_init_val_ce,
            curves_saved=curves_saved,
        )
        print(f"  → Saved incremental results ({len(cells_done)}/{n_total} done)", flush=True)

    total_elapsed = time.time() - t_total
    print(f"\n{'='*60}", flush=True)
    print(f"ALL DONE: {len(cells_done)} cells in {total_elapsed:.0f}s ({total_elapsed/60:.1f} min)",
          flush=True)

    # Final print
    with open(RESULTS_PATH) as f:
        r = json.load(f)
    print("\nFinal RESULTS.json:", flush=True)
    print(json.dumps(r, indent=2), flush=True)


if __name__ == "__main__":
    main()
