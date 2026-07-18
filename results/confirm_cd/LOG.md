# LOG — confirm_cd (Round: Confirmatory D-vs-C equivalence, full scale)

## Objective
Finalize the D-vs-C equivalence claim with statistical rigor:
- **C**: explicit empirical-Fisher utility (extra backward pass per reset)
- **D**: Adam exp_avg_sq (v_t) per neuron (ZERO extra compute)
- PAIRED design, 8 seeds, to compute tight paired CI on D-C diff

## Prior round context
- **baseline_A / A2 / A3**: Arm A floor (no resets) → acc≈0.20, dead≈1.0 by task 3. Confirmed.
- **method_arms** (probe, n=3 seeds): B=0.811, C=0.802, D=0.784. D-vs-C = -1.77pp CI [-7.8,+4.3]. Too wide to conclude equivalence with n=3.
- **confirm_arms** (cut before finishing): started B+C+D with 8 seeds, was cut mid-run after 2 seeds of B. Incomplete.
- This round: ONLY C and D (B already characterized at 0.811; A floor fixed at 0.20).

## Design decisions

| Decision | Choice | Rationale |
|---|---|---|
| Arms | C (Fisher), D (Adam-v_t) | EXPERIMENT.md: "Run ONLY C and D here" |
| Tasks | 5 × 5 classes = 25 total | Per spec: "5 tasks/run" |
| Steps/task | 1000 | Per spec: "~1000 steps/task" |
| Seeds | 8 {0..7} | Per spec: "8 seeds" |
| PAIRED | Yes: C then D, same seed | Critical for tight CI |
| Incremental | After EACH seed (both C+D done) | "Write RESULTS.json INCREMENTALLY" |
| Finalization trigger | >=6 seeds done | Per spec |
| Equivalence margin | ±3pp | Per EXPERIMENT.md output spec |
| Per-seed acc metric | mean(tasks 2-5) | Standard: exclude task-1 warmup |
| Architecture | SmallConvNetGN (GroupNorm) | Validated in all prior rounds |
| Data | Real CIFAR-100, /opt/datasets | ABORT if missing |
| Eval | Masked (task-current classes only) | Same as all prior rounds |
| Reset cadence | Every 100 steps | Same as prior rounds |
| Reset fraction | 10% (51/512 neurons) | Same as prior rounds |
| LR | 1e-3 (Adam) | Same as prior rounds |

## Scientific question
Is v_t (Adam's exp_avg_sq, aggregated per neuron) a statistically equivalent
zero-cost substitute for explicit empirical-Fisher in CBP resets?
- Equivalent = paired D-C 95% CI fully within ±3pp

## Timeline

### 2026-07-18 — Setup
- GPU: NVIDIA A10, 23 GB VRAM (free)
- Data verified at /opt/datasets/cifar-100-python/
- Written: src/confirm_cd.py (builds on confirm_arms.py, arms C+D only, 5 tasks, 1000 steps)
- Estimated wall-time: 2 arms × 8 seeds × 5 tasks × ~25s ≈ 30 min total
- Launch: `python src/confirm_cd.py 2>&1 | tee results/confirm_cd/run.log`

## Run timeline

### Seed 0 (4.0 min from start)
- C: tasks 1-5 acc = [0.866, 0.914, 0.818, 0.822, 0.800] → mean(t2-5) = 0.839
- D: tasks 1-5 acc = [0.880, 0.888, 0.844, 0.720, 0.804] → mean(t2-5) = 0.814
- D-C = -2.45pp (n=1: CI undefined)

### Seed 1 (8.0 min)
- C: mean(t2-5) = 0.790
- D: mean(t2-5) = 0.784
- D-C = -0.60pp | Running: mean=-1.52pp CI=[-13.28, +10.23] n=2

### Seed 2 (11.8 min)
- C: mean(t2-5) = 0.833
- D: mean(t2-5) = 0.823
- D-C = -0.95pp | Running: mean=-1.33pp CI=[-3.77, +1.11] n=3

### Seed 3 (15.8 min)
- C: mean(t2-5) = 0.809
- D: mean(t2-5) = 0.793
- D-C = -1.55pp | Running: mean=-1.39pp CI=[-2.68, -0.10] p=0.042 n=4

### Seed 4 (19.8 min)
- C: mean(t2-5) = 0.790
- D: mean(t2-5) = 0.806
- D-C = +1.65pp | Running: mean=-0.58pp CI=[-2.32, +1.15] p=0.44 n=5

### Seed 5 (23.9 min)
- C: mean(t2-5) = 0.783
- D: mean(t2-5) = 0.787
- D-C = +0.40pp | Running: mean=-0.25pp CI=[-1.88, +1.38] p=0.74 n=6
  → >=6 seeds done; early finalization trigger logged

### Seed 6 (27.9 min)
- C: mean(t2-5) = 0.764
- D: mean(t2-5) = 0.789
- D-C = +2.50pp | Running: mean=-0.14pp CI=[-1.77, +1.49] p=0.84 n=7

### Seed 7 (31.8 min — FINAL)
- C: mean(t2-5) = 0.782
- D: mean(t2-5) = 0.776
- D-C = -0.60pp

## Final Results (8 seeds, 31.8 min wall time)

| Arm | Mean acc (tasks 2-5) | Std | Dead (final) | Erank (final) |
|-----|----------------------|-----|--------------|----------------|
| C (explicit Fisher) | **0.7986** | 0.0259 | 0.877 | 12.95 |
| D (Adam v_t)        | **0.7966** | 0.0162 | 0.863 | 15.83 |
| B (prior, n=3)      | ~0.811     | —     | ~0.82 | —     |
| A floor (prior)     | 0.200      | —     | 1.000 | ~0    |

### Paired D-C analysis (n=8 seeds)
- Per-seed diffs (D-C): [-2.45, -0.60, -0.95, -1.55, +1.65, +0.40, +2.50, -0.60] pp
- Mean D-C = **-0.20 pp**
- 95% CI = **[-1.57, +1.17] pp**
- t=-0.345, p=**0.740**
- equivalent_within_3pp = **TRUE**

## Scientific verdict

**YES — Adam's v_t IS a statistically equivalent zero-cost substitute for explicit Fisher.**

The 95% CI for D-C is [-1.57, +1.17] pp, fully within the ±3pp equivalence margin
(in fact within ±2pp). The point estimate is -0.20pp (D essentially identical to C).
p=0.74 strongly rejects any meaningful difference.

This confirms: Adam's stored exp_avg_sq, aggregated per neuron, can replace the extra
backward pass required for explicit Fisher utility in CBP resets — with no measurable
accuracy cost and zero additional compute.

Additional observation: D has slightly lower variance (std=0.0162 vs 0.0259 for C),
suggesting the temporal accumulation in v_t provides a more stable utility estimate than
a single-batch Fisher gradient.
