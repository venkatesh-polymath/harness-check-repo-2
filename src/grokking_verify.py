"""
Grokking Verification — quick check that grokking CAN occur.
Uses WD=1.0 (Power et al. standard) to verify architecture is correct.
Cap at 20k steps.
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
P = 97
SEED = 42
TRAIN_FRAC = 0.4
BATCH_SIZE = 512
LR = 1e-3
WEIGHT_DECAY = 1.0      # Power et al. standard
MAX_STEPS = 20_000
EVAL_EVERY = 100
LOG_EVERY = 500

D_MODEL = 128
N_HEADS = 4
MLP_HIDDEN = 512
VOCAB_SIZE = P + 1

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}", flush=True)

torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


def make_dataset(p, train_frac, seed):
    rng = torch.Generator()
    rng.manual_seed(seed)
    SEP = p
    pairs = []
    for a in range(p):
        for b in range(p):
            pairs.append((a, b, (a + b) % p))
    pairs = torch.tensor(pairs, dtype=torch.long)
    n = len(pairs)
    perm = torch.randperm(n, generator=rng)
    pairs = pairs[perm]
    n_train = int(n * train_frac)
    train_pairs = pairs[:n_train]
    val_pairs   = pairs[n_train:]

    def make_xy(p_slice):
        a = p_slice[:, 0:1]
        b = p_slice[:, 1:2]
        sep_ = torch.full((len(p_slice), 1), SEP, dtype=torch.long)
        x = torch.cat([a, sep_, b], dim=1)
        y = p_slice[:, 2]
        return x, y

    x_tr, y_tr = make_xy(train_pairs)
    x_va, y_va = make_xy(val_pairs)
    return (TensorDataset(x_tr, y_tr),
            TensorDataset(x_va, y_va),
            len(train_pairs), len(val_pairs))


class TransformerBlock(nn.Module):
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
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class GrokTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, mlp_hidden, seq_len=3, num_classes=97):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(seq_len, d_model)
        self.block   = TransformerBlock(d_model, n_heads, mlp_hidden)
        self.ln_out  = nn.LayerNorm(d_model)
        self.head    = nn.Linear(d_model, num_classes)
        self.seq_len = seq_len

    def forward(self, x):
        B, T = x.shape
        pos  = torch.arange(T, device=x.device).unsqueeze(0)
        emb  = self.tok_emb(x) + self.pos_emb(pos)
        h    = self.block(emb)
        h    = self.ln_out(h)
        logits = self.head(h[:, -1, :])
        return logits

    def weight_norm(self):
        total = 0.0
        for name, p in self.named_parameters():
            if 'weight' in name:
                total += p.data.norm(2).item() ** 2
        return math.sqrt(total)


def train():
    print("=" * 60, flush=True)
    print("Grokking Verification — WD=1.0 (Power et al. standard)", flush=True)
    print(f"P={P}, seed={SEED}, train_frac={TRAIN_FRAC}", flush=True)
    print(f"Optim: AdamW, LR={LR}, WD={WEIGHT_DECAY}, BS={BATCH_SIZE}", flush=True)
    print("=" * 60, flush=True)

    train_ds, val_ds, n_train, n_val = make_dataset(P, TRAIN_FRAC, SEED)
    print(f"Train size: {n_train}, Val size: {n_val}", flush=True)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              generator=torch.Generator().manual_seed(SEED))
    val_x  = val_ds.tensors[0].to(DEVICE)
    val_y  = val_ds.tensors[1].to(DEVICE)
    train_x_full = train_ds.tensors[0].to(DEVICE)
    train_y_full = train_ds.tensors[1].to(DEVICE)

    model = GrokTransformer(
        vocab_size=VOCAB_SIZE, d_model=D_MODEL, n_heads=N_HEADS,
        mlp_hidden=MLP_HIDDEN, seq_len=3, num_classes=P
    ).to(DEVICE)

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

    model.eval()
    with torch.no_grad():
        init_logits = model(val_x[:256])
        init_ce = F.cross_entropy(init_logits, val_y[:256]).item()
    init_wnorm = model.weight_norm()
    print(f"Init val CE: {init_ce:.4f}  (expected ~{math.log(P):.4f})", flush=True)
    print(f"Init weight norm: {init_wnorm:.4f}", flush=True)

    history = []
    grokking_onset = None
    grokking_confirmed = None

    step = 0
    data_iter = iter(train_loader)
    t_start = time.time()
    model.train()

    while step < MAX_STEPS:
        try:
            x_batch, y_batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            x_batch, y_batch = next(data_iter)

        x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
        optimizer.zero_grad()
        logits = model(x_batch)
        loss = F.cross_entropy(logits, y_batch)
        loss.backward()
        optimizer.step()
        step += 1

        if step % EVAL_EVERY == 0:
            model.eval()
            with torch.no_grad():
                val_logits = model(val_x)
                val_loss   = F.cross_entropy(val_logits, val_y).item()
                val_acc    = (val_logits.argmax(-1) == val_y).float().mean().item()
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

            if val_acc > 0.9 and grokking_onset is None:
                grokking_onset = step
                print(f"\n*** GROKKING ONSET at step {step}: val_acc={val_acc:.4f} ***\n", flush=True)

            if grokking_onset is not None and grokking_confirmed is None:
                recent = [r['val_acc'] for r in history[-3:]]
                if len(recent) >= 3 and all(v > 0.9 for v in recent):
                    grokking_confirmed = step
                    print(f"*** GROKKING CONFIRMED at step {step} ***", flush=True)

            model.train()

        if grokking_confirmed is not None:
            model.eval()
            with torch.no_grad():
                va = (model(val_x).argmax(-1) == val_y).float().mean().item()
            model.train()
            if va > 0.99:
                print(f"Early stop: val_acc={va:.4f} at step={step}", flush=True)
                break

    # Final eval
    model.eval()
    with torch.no_grad():
        val_logits = model(val_x)
        final_val_acc  = (val_logits.argmax(-1) == val_y).float().mean().item()
        final_val_loss = F.cross_entropy(val_logits, val_y).item()
        tr_logits      = model(train_x_full)
        final_tr_acc   = (tr_logits.argmax(-1) == train_y_full).float().mean().item()
        final_tr_loss  = F.cross_entropy(tr_logits, train_y_full).item()
    final_wnorm = model.weight_norm()
    elapsed = time.time() - t_start

    print("\n" + "=" * 60, flush=True)
    print(f"DONE | steps={step} | elapsed={elapsed:.1f}s", flush=True)
    print(f"Final train acc={final_tr_acc:.4f} loss={final_tr_loss:.4f}", flush=True)
    print(f"Final val   acc={final_val_acc:.4f} loss={final_val_loss:.4f}", flush=True)
    print(f"Grokking onset: {grokking_onset}", flush=True)
    print(f"Init wnorm={init_wnorm:.3f} → Final wnorm={final_wnorm:.3f} (ratio={final_wnorm/init_wnorm:.3f})", flush=True)
    print("=" * 60, flush=True)

    # Save verification results
    vresults = {
        "wd_1p0_verification": {
            "grokking_onset_step": grokking_onset,
            "grokking_confirmed_step": grokking_confirmed,
            "final_train_acc": round(final_tr_acc, 6),
            "final_val_acc":   round(final_val_acc, 6),
            "init_wnorm": round(init_wnorm, 6),
            "final_wnorm": round(final_wnorm, 6),
            "elapsed_seconds": round(elapsed, 1),
            "total_steps": step,
        }
    }
    with open("results/baseline-00/verify_wd1p0.json", "w") as f:
        json.dump(vresults, f, indent=2)
    print("Saved verify_wd1p0.json", flush=True)
    return vresults


if __name__ == "__main__":
    train()
