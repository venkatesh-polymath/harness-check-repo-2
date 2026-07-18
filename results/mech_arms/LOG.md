# LOG — mech_arms (Round: Mechanistic control arm + full 4-arm comparison)

## Objective

MECHANISTIC + PAIRED comparison of four CBP reset strategies, adding a RANDOM
control arm to isolate whether v_t's *ranking* (not just the reset cadence) is
what matters.

**Four arms (ONLY utility criterion differs; everything else identical):**
- **B**: CBP heuristic utility — |mean outgoing weight| × running mean activation
- **C**: explicit empirical-Fisher utility — mean (∂L/∂act_i)² over current batch
- **D (OURS)**: Adam exp_avg_sq (v_t) aggregated per neuron — zero extra compute
- **RANDOM** (mechanistic control): random subset of units each cycle, same cadence + fraction

**Key mechanistic question:** If D >> RANDOM, then v_t's *ranking* carries real
information (not just "any reset helps").

## Prior round context

| Round | Key result |
|-------|-----------|
| baseline_A / A2 / A3 | Arm A floor: acc≈0.20, dead≈1.0 by task 3 (confirmed on real CIFAR-100) |
| method_arms (probe, n=3) | B=0.811, C=0.802, D=0.784; D-C=-1.77pp CI[-7.8,+4.3] — too wide |
| confirm_arms | Incomplete (cut mid-run) |
| confirm_cd (n=8 seeds) | D-C=-0.20pp CI[-1.57,+1.17]; equiv_3pp=TRUE. v_t ≈ Fisher confirmed |

This round adds the RANDOM control arm to answer: "does v_t ranking carry information?"

## Design decisions

| Decision | Choice | Rationale |
|---|---|---|
| Arms | B, C, D, RANDOM | Exactly as per EXPERIMENT.md mech_arms spec |
| Tasks | 4 × 5 classes = 20 classes | Per spec: "4 tasks/run" |
| Steps/task | 1000 | Per spec: "~1000 steps/task" |
| Seeds | 6 {0..5} | Per spec: "6 seeds {0..5}" |
| PAIRED | Yes: all 4 arms on same seed | Critical for tight CIs |
| Finalization trigger | >=4 seeds done | Per spec: "finalize moment >=4 seeds done" |
| Incremental writes | After each seed's 4 arms | "write RESULTS.json INCREMENTALLY" |
| Equivalence margin | ±3pp | Per spec D-C output format |
| Per-seed acc metric | mean(tasks 2-4) | Exclude task-1 warmup; 4 tasks total |
| RANDOM utility | numpy RandomState(1000+seed) | Reproducible but independent of model |
| Architecture | SmallConvNetGN (GroupNorm) | Validated in all prior rounds |
| Data | Real CIFAR-100, /opt/datasets | ABORT if missing |
| Eval | Masked (task-current classes only) | Same as all prior rounds |
| Reset cadence | Every 100 steps | Same as prior rounds |
| Reset fraction | 10% (51/512 neurons) | Same as prior rounds |
| LR | 1e-3 (Adam) | Same as prior rounds |

## Scientific hypotheses

1. **D vs C**: D should match C within ±3pp (already confirmed in confirm_cd,
   but now in a 4-task context with all arms present)
2. **B vs D**: v_t should tie or beat the CBP heuristic (B) on accuracy
3. **D vs RANDOM**: D >> RANDOM → v_t's *ranking* carries real information
   (this is the mechanistic-isolation result — the core of this round)

## Code

- Source: `src/mech_arms.py` (built on `src/confirm_cd.py`)
- Adds `utility_B` (from `src/method_arms.py`) and `utility_RANDOM` functions
- RANDOM arm uses `np.random.RandomState(1000 + seed)` — reproducible, independent
- Writes RESULTS.json after every seed, finalizes at >=4 seeds

## Timeline

### 2026-07-18 — Setup

- GPU verified (NVIDIA A10G, ~23 GB VRAM)
- Data verified at /opt/datasets/cifar-100-python/
- Written: src/mech_arms.py
- Estimated wall-time: 4 arms × 6 seeds × 4 tasks × ~22s ≈ 35 min
- Launch: `python src/mech_arms.py 2>&1 | tee results/mech_arms/run.log`

## Run timeline

### Seed 0 (8.4 min from start)
- B: tasks 1-4 acc = [0.898, 0.912, 0.834, 0.802] → mean(t2-4) = 0.849
- C: tasks 1-4 acc = [0.866, 0.914, 0.818, 0.822] → mean(t2-4) = 0.851
- D: tasks 1-4 acc = [0.880, 0.888, 0.844, 0.720] → mean(t2-4) = 0.817
- RANDOM: tasks 1-4 acc = [0.900, 0.864, 0.488, 0.812] → mean(t2-4) = 0.721
- D-C = -3.40pp  |  D-RANDOM = +9.60pp (STRONG signal)
- Key observation: RANDOM shows high variance (task 3: 0.488!) — ranking matters!

### Seed 1 (16.8 min from start)
- B/s1: tasks 1-4 = [0.806, 0.866, 0.728, 0.730] → mean(t2-4) = 0.775
- C/s1: tasks 1-4 = [0.816, 0.874, 0.694, 0.754] → mean(t2-4) = 0.774
- D/s1: tasks 1-4 = [0.804, 0.872, 0.740, 0.734] → mean(t2-4) = 0.782
- RANDOM/s1: tasks 1-4 = [0.804, 0.836, 0.640, ~0.785] → mean(t2-4) ≈ 0.754
- Running D-C = -1.30pp | D-RANDOM = +6.20pp (n=2, CI wide)

*(Seeds 2-5 in progress)*
