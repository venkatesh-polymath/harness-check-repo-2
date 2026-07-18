"""
Grokking Baseline — round baseline-00 (probe)

Modular arithmetic dataset (a+b mod 97) + 1-layer transformer.
Baseline cell: LayerNorm + WD=1e-3 + AdamW + batch_size=512 + LR=1e-3
Goal: verify grokking occurs (train acc→100% early, val acc groks later).
Report: grokking-onset step (first step val_acc>0.9), final train/val acc,
        weight-norm trajectory.
Budget: ≤20k steps or stop after grokking confirmed.
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

# ─── Config ──────────────────────────────────────────────────────────────────
P = 97               # prime modulus
SEED = 42
TRAIN_FRAC = 0.4     # 40% train / 60% val (Power et al. standard)
BATCH_SIZE = 512
LR = 1e-3
WEIGHT_DECAY = 1e-3
MAX_STEPS = 100_000
EVAL_EVERY = 100
LOG_EVERY = 500      # heavier logging

# Architecture (study spec)
D_MODEL = 128
N_HEADS = 4
MLP_HIDDEN = 512
VOCAB_SIZE = P + 1   # 97 residues + 1 separator token

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}", flush=True)

# ─── Reproducibility ─────────────────────────────────────────────────────────
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# ─── Dataset ─────────────────────────────────────────────────────────────────
def make_dataset(p, train_frac, seed):
    """Return (train_dataset, val_dataset) as TensorDatasets.

    Input format: [a, SEP, b]  →  label: (a+b) mod p
    SEP token = p (index 97)
    """
    rng = torch.Generator()
    rng.manual_seed(seed)

    SEP = p  # separator token
    pairs = []
    for a in range(p):
        for b in range(p):
            pairs.append((a, b, (a + b) % p))

    pairs = torch.tensor(pairs, dtype=torch.long)   # shape [p^2, 3]
    n = len(pairs)  # 9409

    # Shuffle
    perm = torch.randperm(n, generator=rng)
    pairs = pairs[perm]

    n_train = int(n * train_frac)
    train_pairs = pairs[:n_train]
    val_pairs   = pairs[n_train:]

    sep = torch.full((len(pairs), 1), SEP, dtype=torch.long)

    # Build inputs: [a, SEP, b]
    def make_xy(p_slice):
        a = p_slice[:, 0:1]
        b = p_slice[:, 1:2]
        sep_ = torch.full((len(p_slice), 1), SEP, dtype=torch.long)
        x = torch.cat([a, sep_, b], dim=1)  # [N, 3]
        y = p_slice[:, 2]                   # [N]
        return x, y

    x_tr, y_tr = make_xy(train_pairs)
    x_va, y_va = make_xy(val_pairs)

    return (TensorDataset(x_tr, y_tr),
            TensorDataset(x_va, y_va),
            n_train, len(val_pairs))


# ─── Architecture ────────────────────────────────────────────────────────────
class TransformerBlock(nn.Module):
    """Single transformer block: self-attention + MLP, with LayerNorm."""

    def __init__(self, d_model, n_heads, mlp_hidden):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, d_model),
        )

    def forward(self, x):
        # Pre-LN style
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class GrokTransformer(nn.Module):
    """1-layer transformer for modular arithmetic."""

    def __init__(self, vocab_size, d_model, n_heads, mlp_hidden, seq_len=3, num_classes=97):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(seq_len, d_model)
        self.block   = TransformerBlock(d_model, n_heads, mlp_hidden)
        self.ln_out  = nn.LayerNorm(d_model)
        self.head    = nn.Linear(d_model, num_classes)
        self.seq_len = seq_len

    def forward(self, x):
        # x: [B, 3]
        B, T = x.shape
        pos  = torch.arange(T, device=x.device).unsqueeze(0)  # [1, T]
        emb  = self.tok_emb(x) + self.pos_emb(pos)            # [B, T, D]
        h    = self.block(emb)                                  # [B, T, D]
        h    = self.ln_out(h)
        # Use last token position for classification
        logits = self.head(h[:, -1, :])                        # [B, num_classes]
        return logits

    def weight_norm(self):
        """Return L2 norm of all trainable weight parameters (no biases)."""
        total = 0.0
        for name, p in self.named_parameters():
            if 'weight' in name:
                total += p.data.norm(2).item() ** 2
        return math.sqrt(total)


# ─── Training ────────────────────────────────────────────────────────────────
def train():
    print("=" * 60, flush=True)
    print("Grokking Baseline — probe run", flush=True)
    print(f"P={P}, seed={SEED}, train_frac={TRAIN_FRAC}", flush=True)
    print(f"Arch: d_model={D_MODEL}, n_heads={N_HEADS}, mlp_hidden={MLP_HIDDEN}", flush=True)
    print(f"Optim: AdamW, LR={LR}, WD={WEIGHT_DECAY}, BS={BATCH_SIZE}", flush=True)
    print(f"Max steps: {MAX_STEPS}", flush=True)
    print("=" * 60, flush=True)

    # Data
    train_ds, val_ds, n_train, n_val = make_dataset(P, TRAIN_FRAC, SEED)
    print(f"Train size: {n_train}, Val size: {n_val}", flush=True)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              generator=torch.Generator().manual_seed(SEED))
    # Val: evaluate full val set at once (fits in memory)
    val_x  = val_ds.tensors[0].to(DEVICE)
    val_y  = val_ds.tensors[1].to(DEVICE)
    train_x_full = train_ds.tensors[0].to(DEVICE)
    train_y_full = train_ds.tensors[1].to(DEVICE)

    # Model
    model = GrokTransformer(
        vocab_size=VOCAB_SIZE, d_model=D_MODEL, n_heads=N_HEADS,
        mlp_hidden=MLP_HIDDEN, seq_len=3, num_classes=P
    ).to(DEVICE)

    # Weight decay only on weight matrices (not biases, embeddings, layernorm)
    decay_params  = [p for n, p in model.named_parameters()
                     if 'weight' in n and p.ndim >= 2
                     and 'norm' not in n and 'emb' not in n]
    nodecay_params = [p for n, p in model.named_parameters()
                      if not ('weight' in n and p.ndim >= 2
                              and 'norm' not in n and 'emb' not in n)]
    optimizer = torch.optim.AdamW(
        [{'params': decay_params,   'weight_decay': WEIGHT_DECAY},
         {'params': nodecay_params, 'weight_decay': 0.0}],
        lr=LR, betas=(0.9, 0.98), eps=1e-8
    )

    # Sanity check: initial CE
    model.eval()
    with torch.no_grad():
        init_logits = model(val_x[:256])
        init_ce = F.cross_entropy(init_logits, val_y[:256]).item()
    print(f"\nSanity — init val CE: {init_ce:.4f}  (expected ~{math.log(P):.4f})", flush=True)
    assert 4.0 <= init_ce <= 5.2, f"Init CE out of range: {init_ce}"

    # Initial weight norm
    init_wnorm = model.weight_norm()
    print(f"Sanity — init weight norm: {init_wnorm:.4f}", flush=True)

    # ── Training loop ──
    history = []          # list of dicts per eval step
    wnorm_traj = []       # (step, wnorm) every LOG_EVERY steps

    grokking_onset = None  # first step val_acc > 0.9
    grokking_confirmed = None  # sustained > 0.9 for 3 consecutive evals

    step = 0
    epoch = 0
    t_start = time.time()
    data_iter = iter(train_loader)

    model.train()

    while step < MAX_STEPS:
        # Get next batch (cycle through data)
        try:
            x_batch, y_batch = next(data_iter)
        except StopIteration:
            epoch += 1
            data_iter = iter(train_loader)
            x_batch, y_batch = next(data_iter)

        x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)

        optimizer.zero_grad()
        logits = model(x_batch)
        loss = F.cross_entropy(logits, y_batch)
        loss.backward()
        optimizer.step()
        step += 1

        # Eval
        if step % EVAL_EVERY == 0:
            model.eval()
            with torch.no_grad():
                # Val
                val_logits = model(val_x)
                val_loss   = F.cross_entropy(val_logits, val_y).item()
                val_acc    = (val_logits.argmax(-1) == val_y).float().mean().item()
                # Train (full)
                tr_logits  = model(train_x_full)
                tr_loss    = F.cross_entropy(tr_logits, train_y_full).item()
                tr_acc     = (tr_logits.argmax(-1) == train_y_full).float().mean().item()

            wnorm = model.weight_norm()

            rec = dict(step=step, tr_loss=tr_loss, tr_acc=tr_acc,
                       val_loss=val_loss, val_acc=val_acc, wnorm=wnorm)
            history.append(rec)

            if step % LOG_EVERY == 0:
                elapsed = time.time() - t_start
                print(f"step={step:6d} | tr_acc={tr_acc:.3f} tr_loss={tr_loss:.4f} "
                      f"| val_acc={val_acc:.3f} val_loss={val_loss:.4f} "
                      f"| wnorm={wnorm:.3f} | {elapsed:.0f}s", flush=True)
                wnorm_traj.append((step, wnorm))

            # Grokking detection
            if val_acc > 0.9 and grokking_onset is None:
                grokking_onset = step
                print(f"\n*** GROKKING ONSET at step {step}: val_acc={val_acc:.4f} ***\n", flush=True)

            # Confirm sustained grokking (3 consecutive evals > 0.9)
            if grokking_onset is not None and grokking_confirmed is None:
                recent = [r['val_acc'] for r in history[-3:]]
                if len(recent) >= 3 and all(v > 0.9 for v in recent):
                    grokking_confirmed = step
                    print(f"*** GROKKING CONFIRMED (sustained) at step {step}: "
                          f"val_acc={val_acc:.4f} ***", flush=True)

            model.train()

        # Early stopping: grokking confirmed AND val_acc > 0.99
        if grokking_confirmed is not None:
            model.eval()
            with torch.no_grad():
                va = (model(val_x).argmax(-1) == val_y).float().mean().item()
            model.train()
            if va > 0.99:
                print(f"\nStopping early: grokking confirmed & val_acc={va:.4f} > 0.99 at step={step}", flush=True)
                # Do one final eval
                step_final = step
                break

    # ── Final eval ──
    model.eval()
    with torch.no_grad():
        val_logits = model(val_x)
        final_val_loss = F.cross_entropy(val_logits, val_y).item()
        final_val_acc  = (val_logits.argmax(-1) == val_y).float().mean().item()
        tr_logits      = model(train_x_full)
        final_tr_loss  = F.cross_entropy(tr_logits, train_y_full).item()
        final_tr_acc   = (tr_logits.argmax(-1) == train_y_full).float().mean().item()
    final_wnorm = model.weight_norm()
    elapsed = time.time() - t_start

    print("\n" + "=" * 60, flush=True)
    print(f"DONE  |  steps={step}  |  elapsed={elapsed:.1f}s", flush=True)
    print(f"Final train acc={final_tr_acc:.4f}  loss={final_tr_loss:.4f}", flush=True)
    print(f"Final val   acc={final_val_acc:.4f}  loss={final_val_loss:.4f}", flush=True)
    print(f"Grokking onset step: {grokking_onset}", flush=True)
    print(f"Grokking confirmed step: {grokking_confirmed}", flush=True)
    print(f"Init weight norm: {init_wnorm:.4f}  |  Final weight norm: {final_wnorm:.4f}", flush=True)
    print(f"Weight norm ratio (final/init): {final_wnorm/init_wnorm:.4f}", flush=True)
    print("=" * 60, flush=True)

    # ── Save results ──
    wnorm_traj_full = [(r['step'], r['wnorm']) for r in history]
    results = {
        "status": "SUCCESS" if grokking_onset is not None else "FAILED",
        "scale": "probe",
        "metrics": {
            "grokking_onset_step": grokking_onset,
            "grokking_confirmed_step": grokking_confirmed,
            "final_train_acc": round(final_tr_acc, 6),
            "final_val_acc":   round(final_val_acc, 6),
            "final_train_loss": round(final_tr_loss, 6),
            "final_val_loss":   round(final_val_loss, 6),
            "init_val_ce": round(init_ce, 6),
            "init_weight_norm": round(init_wnorm, 6),
            "final_weight_norm": round(final_wnorm, 6),
            "weight_norm_ratio": round(final_wnorm / init_wnorm, 6),
            "total_steps": step,
            "elapsed_seconds": round(elapsed, 1),
            "grokking_reproduced": grokking_onset is not None,
            "wnorm_trajectory_sampled": wnorm_traj,  # every LOG_EVERY steps
        },
        "subject_executed": (
            f"1-layer transformer (d={D_MODEL}, heads={N_HEADS}, mlp={MLP_HIDDEN}) "
            f"on modular addition p={P}, LayerNorm + AdamW WD={WEIGHT_DECAY} "
            f"LR={LR} BS={BATCH_SIZE}, seed={SEED}, "
            f"train_frac={TRAIN_FRAC}, max_steps={MAX_STEPS}"
        ),
        "notes": (
            f"Probe run: baseline cell (LN+WD=1e-3+BS=512). "
            f"Grokking {'reproduced' if grokking_onset else 'NOT observed'} "
            f"at step {grokking_onset}. "
            f"Init CE={init_ce:.3f} (expected ~{math.log(P):.3f}). "
            f"Weight norm grew {final_wnorm/init_wnorm:.2f}x from init."
        )
    }

    os.makedirs("results/baseline-00", exist_ok=True)
    with open("results/baseline-00/RESULTS.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to results/baseline-00/RESULTS.json", flush=True)

    # Save history for later analysis
    with open("results/baseline-00/history.json", "w") as f:
        json.dump(history, f, indent=2)
    print("History saved to results/baseline-00/history.json", flush=True)

    return results


if __name__ == "__main__":
    train()
