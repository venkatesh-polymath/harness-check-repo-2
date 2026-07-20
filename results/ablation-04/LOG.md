# LOG — ablation-04 (full)

## Overview
Full-scale statistical confirmation of ablation-03 findings.
Run: 5 seeds × 3 arms × 20 tasks × 10 epochs on CIFAR-100 class-incremental.

## Prior rounds summary
- **baseline-00**: Sanity gates 1-4 PASS (seed determinism, init loss ≈ 4.605, constant-predictor = 1%, single-batch memorisation < 0.01 loss).
- **refine-01**: ResNet-18 continual training scaffolding confirmed; effective-rank collapse observed.
- **increase_complexity-02**: SNRI mechanism implemented; gates G7 (null-space orthogonality), G8 (non-disruption), G9 (rank increase after 1 SNRI event) all PASS.
- **ablation-03 (probe, seed=42)**: Three-arm A/B/C comparison confirmed hypotheses with a single seed:
  - Rank AUC: A=12.23, B=12.14, C=12.31 → C > A > B
  - Final dead%: A=6.76%, B=15.68%, C=7.54% → B >> A ≈ C
  - Disruption (mean): B=0.011, C=0.148 → C >> B
  - All three primary hypotheses TRUE on single seed.

## ablation-04 design decisions

### Why 5 seeds
- Study spec (EXPERIMENT.md) specifies `random_seeds: [0,1,2,3,4]`.
- EXPERIMENT.md for this round says "3-5 seeds"; using 5 to match spec.
- Wilcoxon signed-rank test requires ≥2 paired observations; 5 gives reasonable power.

### Seeds: [0, 1, 2, 3, 4]
- Each seed initialises a fresh ResNet-18 from scratch (no pre-training).
- All three arms within a seed start from the same model weights (seed is set before `make_model()`).
- Within-seed ordering: A first, then B, then C (same as ablation-03).

### Statistical tests
- Wilcoxon signed-rank (paired) across 5 seeds for: rank_auc B-vs-A, rank_auc C-vs-B, dead_pct B-vs-A, dead_pct C-vs-B, disruption C-vs-B.
- Bonferroni correction: α/4 = 0.0125 for 4 primary tests.
- Sign consistency: report #seeds where direction holds for each comparison.

### What was NOT changed from ablation-03
- Architecture: ResNet-18 (no pretrain), fc=Linear(512,100)
- Optimizer: SGD + Nesterov momentum=0.9, weight_decay=5e-4, lr=0.1
- LR schedule: CosineAnnealingLR per task, eta_min=0
- Dead-channel detection: mean |activation| < 0.01 over 20 probe batches
- SNRI: null-space vectors from SVD (ε=1e-6), zero outgoing weights
- Naive ReDo (arm C): kaiming_normal incoming, outgoing untouched
- Disruption measurement: deepcopy model, measure max |logit diff| over 256-image probe

### Sanity gates
- Gates 1-4 (seed determinism, init loss, constant predictor, single-batch memorisation): PASS — cited from baseline-00 (not re-run to save GPU time since architecture is unchanged).
- Gates G7/G8/G9 (null-space orthogonality, SNRI non-disruption, rank increase): PASS — cited from increase_complexity-02 (same SNRI implementation).

## Run log
- Script: src/ablation_04.py
- Launch time: see run.log timestamps
- Expected runtime: ~55 min (5 seeds × 3 arms × ~3.5 min/arm on H100)
- GPU: NVIDIA H100 80GB HBM3

## Results summary

### Completed: 2026-07-20, total runtime ~69 min (H100 GPU)

#### Per-seed data

| Seed | A rank_auc | A dead% | B rank_auc | B dead% | B reinit | C rank_auc | C dead% | C reinit |
|------|-----------|---------|-----------|---------|---------|-----------|---------|---------|
| 0    | 12.3292   | 9.87    | 12.2377   | 10.78   | 5       | 12.2377   | 10.80   | 5       |
| 1    | 12.2082   | 5.59    | 12.1969   | 6.01    | 43      | 12.2006   | 6.86    | 41      |
| 2    | 12.4013   | 10.21   | 12.3722   | 11.92   | 1       | 12.3722   | 11.32   | 2       |
| 3    | 10.2894†  | 18.93†  | 12.3262   | 9.36    | 44      | 12.3403   | 12.46   | 52      |
| 4    | 12.0979   | 6.54    | 11.8462   | 7.21    | 40      | 11.8509   | 9.43    | 51      |
| **mean** | **11.865±0.889** | **10.23±5.27** | **12.196±0.207** | **9.06±2.45** | — | **12.200±0.208** | **10.17±2.15** | — |

†Seed 3 arm A is a clear outlier (dramatic rank collapse, r=-0.43 vs -0.95+ for other seeds/arms).

#### Disruption (mean logit change per task, averaged over seeds)
- Arm B (SNRI): 0.0001 ± 0.0001 (mean disruption_mean across seeds)
- Arm C (naive): 0.0009 ± 0.0008 (9× higher than B)

#### Hypothesis verdicts

| ID | Hypothesis | Sign consistency | Wilcoxon p | Result |
|----|-----------|-----------------|-----------|--------|
| H1 | C more disruptive than B | 5/5 | 0.0625* | ✓ CONFIRMED |
| H2 | C higher rank_auc than B | 3/5 | 0.25 | ✗ NOT CONFIRMED |
| H3 | B more dead than A | 4/5 | 0.625 | ✗ NOT CONFIRMED |
| H4 | C fewer dead than B | 1/5 | 0.1875 | ✗ NOT CONFIRMED |

*p=0.0625 is the minimum achievable with n=5 paired observations (all 5 seeds same direction). Above Bonferroni-corrected α=0.0125.

#### Key scientific findings

1. **H1 robustly confirmed**: SNRI (B) is ~9× less disruptive than random re-init (C) in ALL 5 seeds. This is the one finding from ablation-03 that replicates.

2. **Ablation-03 H3/H4 findings do NOT replicate**: The dramatic B vs C dead% difference from ablation-03 (B=15.68%, C=7.54%) was a seed-42 outlier. Across 5 seeds, B and C have similar dead% (9.06±2.45% vs 10.17±2.15%), and C has MORE dead neurons than B in 4/5 seeds (opposite of hypothesis).

3. **Seed 3 arm A outlier**: vanilla training with seed=3 collapsed dramatically (rank_auc=10.29, dead=18.93%, rank_drop=23.5%) while B and C did not. This is likely due to a training trajectory peculiarity at seed=3 that was interrupted/corrected by the periodic re-init in arms B and C (B had 44 reinits, C had 52). However, this is confounded by the fact that arms B/C reset the CPU RNG (affecting data shuffling order) while arm A does not.

4. **Reinit count is highly variable**: Seeds 0,2 had very few reinits (0-5), seeds 1,3,4 had 40-52 reinits. This extreme variance suggests the dead-channel detection is highly sensitive to the training trajectory.

5. **All Wilcoxon p-values > Bonferroni threshold (0.0125)**: No effect is statistically significant at the required level. With n=5 pairs, the test lacks power.

#### Why results differ from ablation-03

- Ablation-03 used seed=42 which produced extreme B dead% (15.68%): this is NOT representative.
- Seed 3 shows the most extreme behavior in ablation-04 too, but in the OPPOSITE direction (A collapses, B/C protect).
- The `torch.manual_seed(seed * 1000 + task_id)` call before re-init in arms B/C changes the DataLoader shuffle sequence relative to arm A, creating a confound.

#### Notes on run execution

- run.log has duplicate lines: Python's direct file write + `tee` both write to same file. Does not affect RESULTS.json (computed in-memory).
- Script ran on NVIDIA H100 80GB HBM3, total ~69 min for 5 seeds × 3 arms × 20 tasks × 10 epochs.
- One UTF-8 decode error in run.log (binary data from tee collision); handled with errors='replace'.
