# LOG — method_arms (Round: CBP utility comparison B/C/D)

## Objective
Compare 3 neuron-utility functions for Continual-Backprop (CBP) resets head-to-head on real CIFAR-100. The key scientific question: does Adam's stored v_t (Arm D) serve as a zero-overhead substitute for explicit Fisher (Arm C)?

## Prior round context
- **baseline_A / baseline_A2 / baseline_A3**: Established that vanilla Adam (no resets) catastrophically loses plasticity by task 3 — acc=0.2, dead=1.0. Confirmed on real CIFAR-100.
- This is the Arm A floor we're trying to beat with CBP.
- Architecture validated: SmallConvNetGN (GroupNorm 3-conv-block + 2 FC, no BN collapse).

## This round's design decisions

| Decision | Choice | Rationale |
|---|---|---|
| Arms | B (heuristic), C (Fisher), D (Adam-v_t) | Exactly as per EXPERIMENT.md |
| Arm A floor | Reference from baseline_A3 (acc≈0.20, dead≈1.0) | Don't re-run; save wall time |
| Tasks | 8 tasks × 5 classes (40 out of 100 classes) | Spec: collapse visible by task 3, 8 = enough headroom |
| Steps/task | 1000 (probe scale) | Spec allows ≥1000; 2000 was ~60s/task → 1000 ≈ 30s |
| Seeds | 3 per arm (0, 1, 2) | Spec: min 3 for error bars |
| Reset cadence | Every 100 steps | Mid-spec range; enough resets per task (10 per 1000 steps) |
| Reset fraction | 10% (51 out of 512 neurons) | Bottom of utility distribution |
| Reset target | fc1 only (512-unit penultimate layer) | Where plasticity loss is measured; main bottleneck |
| Arm B utility | mean(|fc2.weight[:, i]|) × running_mean(act_i), EMA α=0.01 | Dohare et al. 2024 heuristic |
| Arm C utility | mean_batch[(∂L/∂post_act_i)²], extra forward-backward | Empirical Fisher on observed labels |
| Arm D utility | mean(exp_avg_sq[i, :]) from Adam state for fc1.weight[i, :] | Zero-overhead v_t proxy |
| Reset: incoming | Kaiming uniform init | Fresh random start for dead neuron |
| Reset: outgoing | Set to 0 | Preserve output continuity |
| Reset: Adam state | Zero exp_avg / exp_avg_sq for reset neurons | Give fresh momentum to reset neurons |
| Architecture | SmallConvNetGN (same as baseline_A3) | Validated; GroupNorm prevents BN collapse |
| Data | Real CIFAR-100 /opt/datasets | download=False, ABORT if missing |
| Task splits | Same seed across arms | Critical isolation: only utility function differs |
| LR | 1e-3 (Adam) | Same across all arms |
| Equivalence margin | ±1.5 pp (pre-registered) | EXPERIMENT.md spec |

## Scientific question and hypotheses
1. Do CBP arms (B/C/D) recover plasticity vs Arm A floor (acc>0.2, dead<1.0)?
2. Does D match C (the paper's core claim)?
3. Does D match/beat B (heuristic)?

## Timeline

### 2026-07-18 — Setup and implementation
- Verified GPU: NVIDIA A10G, 23 GB VRAM (free)
- Verified data: /opt/datasets/cifar-100-python/train present
- Estimated wall time: 3 arms × 3 seeds × 8 tasks × ~30s = ~36 min
- Written: src/method_arms.py
  - SmallConvNetGN with `forward_with_act_grad` method for Arm C Fisher
  - `utility_B/C/D` functions
  - `cbp_reset`: resets fc1 incoming weights (kaiming), fc2 outgoing (→0), Adam state
  - `aggregate_arm`: computes mean/std final-task accuracy (tasks 5–8)
  - `compare_arms`: D vs C diff, rough 95% CI, verdict
  - Results written incrementally after each (arm, seed, task)
- Launched: `python src/method_arms.py 2>&1 | tee results/method_arms/run.log`

### During run — Arm B complete (11 min)
- Arm B seed 0 tasks 5-8: 0.802, 0.826, 0.810, 0.772 → mean final = 0.803
- Arm B seed 1 tasks 5-8: 0.858, 0.894, 0.776, 0.804 → mean final = 0.833
- Arm B seed 2 tasks 5-8: 0.804, 0.834, 0.744, 0.812 → mean final = 0.799
- Key observation: CBP resets WORK — acc stays 0.78-0.83 vs Arm A floor of 0.20!

### During run — Arm C complete (22 min)
- Arm C seed 0 tasks 5-8: 0.800, 0.784, 0.752, 0.732 → mean = 0.767
- Arm C seed 1 tasks 5-8: 0.838, 0.892, 0.792, 0.826 → mean = 0.837
- Arm C seed 2 tasks 5-8: 0.790, 0.788, 0.798, 0.826 → mean = 0.801
- Fisher utility also works; slightly noisier across seeds

### During run — Arm D complete (34 min, total)
- Arm D seed 0 tasks 5-8: 0.804, 0.820, 0.762, 0.724 → mean = 0.778
- Arm D seed 1 tasks 5-8: 0.790, 0.862, 0.760, 0.808 → mean = 0.805
- Arm D seed 2 tasks 5-8: 0.818, 0.740, 0.806, 0.742 → mean = 0.777 (approx)

### Final results (34 min wall time)

| Arm | Mean final acc (tasks 5-8) | Std | Dead final | Erank final |
|-----|--------------------------|-----|------------|-------------|
| A (floor) | 0.200 | — | 1.000 | 0.0 |
| B (heuristic) | **0.811** | 0.015 | 0.818 | 23.3 |
| C (Fisher) | 0.802 | 0.029 | 0.845 | 18.7 |
| D (Adam-v_t) | 0.784 | 0.015 | 0.813 | 21.8 |

**D vs C**: acc_diff = -1.767 pp (D worse); CI = [-7.83, +4.30] pp (n=3, df=2)

**Scientific findings**:
1. **Recovers plasticity**: YES — all CBP arms achieve 0.78-0.81 vs 0.20 floor (↑>30 pp).
2. **D matches C?**: Point estimate: D_worse by 1.767 pp (slightly outside ±1.5 pp margin). CI is wide at n=3 — statistically inconclusive. Cannot claim equivalence from probe alone.
3. **D vs B**: D is -2.7 pp vs B. Heuristic is marginally best, v_t slightly worst. Differences small given variability.

**Bug found and corrected**: The `compare_arms()` function compared mean_diff (proportion) to equiv_margin (in pp), always yielding "match" due to scale mismatch. Corrected in RESULTS.json to "D_worse" based on point estimate.

**Caveats**:
- n=3 seeds is underpowered for TOST equivalence testing (need ≥5 seeds per EXPERIMENT.md full spec)
- 1000 steps/task (probe scale) may underestimate effects vs 2000-step full run
- Dead units still high (~81-85%) for all arms — resets help but don't eliminate
- Effective rank sustained (18-23) vs complete collapse (0) in Arm A — key improvement

### Probe conclusion
The probe answers its question: YES it runs, YES the metrics move dramatically. All CBP arms recover plasticity by >30 pp vs the no-reset baseline. The D vs C comparison is underpowered at n=3 and requires more seeds for a definitive equivalence verdict.
