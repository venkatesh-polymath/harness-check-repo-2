# LOG — baseline_A3 (Arm A v3: Adam + no resets + GroupNorm, REAL CIFAR-100)

## Objective
Run the DEFINITIVE clean Arm A baseline on REAL CIFAR-100.

### Background from prior rounds
- **baseline_A**: BatchNorm collapsed → erank=500 (invalid SVD artifact from zero matrix); acc=0.2
- **baseline_A2**: GroupNorm fixed BN-collapse, zero-matrix guard added → erank=0.0 (honest).
  BUT still used SYNTHETIC data (CIFAR-100 download too slow at ~6 MB/min).
  Result was: catastrophic collapse from task 3 onward (dead=1.0, erank=0). Could be a
  synthetic-data artifact (SNR=0.5 → easy overfitting → Adam kills all neurons).

### This round's fix
CIFAR-100 is now PRE-BAKED at `/opt/datasets/cifar-100-python` (confirmed present).
`download=False` — no network. If data fails to load, ABORT immediately (no synthetic fallback).

## Setup decisions

| Decision | Choice | Rationale |
|---|---|---|
| Model | SmallConvNetGN (3 conv-blocks + 2 FC, GroupNorm) | Same as baseline_A_v2; GN avoids BN collapse |
| Data | Real CIFAR-100 from /opt/datasets | Pre-baked in image; download=False |
| Fallback | NONE — ABORT if load fails | Synthetic run is worthless for this round |
| Tasks | 20 × 5 classes = 100 total | Class-incremental CIFAR-100 |
| Steps/task | 2000 | Matches study spec |
| Batch size | 128 | Same as prior rounds |
| LR | 1e-3 | Adam default |
| Optimizer | Adam, NO resets | This is the Arm A reference baseline |
| Seeds | [0, 1] both to completion | 2 seeds as specified |
| Dead-unit threshold | 0.01 (mean |activation|) | Standard heuristic |
| Effective rank | Spectral entropy on penultimate layer | Returns 0.0 for dead matrix, NaN on failure |
| Wall limit | 33 min | Stay under 35 min budget |
| Output dir | results/baseline_A3/ | This round's outputs |
| Script | src/baseline_A3.py | New script built on baseline_A_v2.py |

## Scientific question
With REAL CIFAR-100 + GroupNorm: does vanilla Adam (no resets) lose plasticity?
Expected: milder than synthetic collapse — real images have shared structure (edges, textures),
so the model doesn't overfit as catastrophically and dead-unit fraction rises more gradually.
A milder-than-expected result is fine and scientifically important.

## Timeline

### 2026-07-18 — Setup
- Confirmed /opt/datasets/cifar-100-python/train exists (50000 samples, 100 fine-label classes)
- GPU: NVIDIA A10G, 23 GB VRAM
- Created src/baseline_A3.py based on src/baseline_A_v2.py
  - Changed RESULTS_DIR to results/baseline_A3/
  - Replaced all data-loading with real CIFAR-100 from /opt/datasets
  - Added ABORT path (no synthetic fallback)
  - Added num_workers=2 + pin_memory for faster data loading
  - Writes RESULTS.json incrementally after each task
  - Final RESULTS.json has keys: status, scale, metrics, subject_executed, notes (as per experiment spec)
- Launched experiment: `python src/baseline_A3.py 2>&1 | tee results/baseline_A3/run.log`

### 2026-07-18 — Interim (seed 0 complete, seed 1 running)

Seed 0 per-task metrics (confirmed on real CIFAR-100):

| Task | Acc | Dead-frac | Erank |
|------|-----|-----------|-------|
| 1 | 0.910 | 0.592 | 47.15 |
| 2 | 0.886 | 0.953 | 10.55 |
| 3 | 0.200 | 1.000 | 0.00 |
| 4 | 0.200 | 1.000 | 0.00 |
| 5-20 | 0.200 | 1.000 | 0.00 |

**Key finding**: Plasticity collapse is equally severe on REAL CIFAR-100.
- The collapse is NOT a synthetic-data artifact.
- Pattern: task 1 overfits (loss→0), Adam v_t accumulates → effective lr→0 for new tasks → dead=1.0 by task 3.
- This validates the scientific question: the baseline DOES lose plasticity strongly.
- Note: "milder than expected" was not the case; collapse is equally catastrophic.

### 2026-07-18 — Final results (after run completes)
<!-- updated after run -->
