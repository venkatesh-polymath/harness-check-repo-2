# LOG — valprec round

## Why this round
Builds on the smallnet round (results/smallnet/) which confirmed:
- hidden=100 MLP on online Permuted-MNIST with LR=0.10 is HEALTHY at task 1
  (dead_after_task1~6.5%, task1_acc~0.93-0.94)
- Plasticity collapses over 300 tasks (acc drops 14-30pp, t_collapse found for all seeds)
This round runs the FULL 8-seed experiment with proper 20-batch GNS, complete
lead-time analysis, predictive AUC, threshold sensitivity, and reproducibility table.

## Config
- Architecture: hidden=100 MLP (784→100→100→10), ReLU, 3 layers
- SGD: lr=0.10 (selected by verify_lr in smallnet round for hidden=100),
  momentum=0.9, weight_decay=0.0
- N_SEEDS=8, N_TASKS=300, STEPS_PER_TASK=200, BATCH_SIZE=128
- GNS: 20 batches × 64 samples = 1280 samples/task (proper McCandlish estimator)
  - ONE np.random.permutation(N) call per GNS computation (advances global state
    by same amount as reference smallnet code, ensuring matching training dynamics)
- Collapse criterion: 20pp below task1_acc, sustained 2 consecutive tasks
- DATA_SEED=42 (fixed task permutations, same for all model seeds)

## Debugging history

### Attempt 1 (FAILED): LR=0.05, independent GNS RNG
- LR=0.05 was wrong — the smallnet verify_lr selected LR=0.10
- GNS used np.random.default_rng (independent) instead of global state
- Result: ~5-7pp drop, t_collapse=None for all 8 seeds
- Why different from smallnet: BOTH wrong LR and different global state progression

### Attempt 2 (FAILED): LR=0.05, global state GNS
- Corrected GNS to use global np.random.permutation (matching smallnet)
- Still wrong LR=0.05
- Result: ~5-7pp drop, t_collapse=None for all 8 seeds
- Root cause: LR=0.05 simply doesn't produce enough plasticity loss in 300 tasks

### Attempt 3 (SUCCESS): LR=0.10, global state GNS
- Fixed LR to 0.10 (from smallnet LOG.md: "Selected LR=0.10 for hidden=100")
- Confirmed: task1_acc=0.9238 EXACTLY matching smallnet seed 0 result
- All 8 seeds collapsed with 20pp threshold
- Wall time: ~1172s (~20 min)

## Per-seed results (FINAL RUN)
Started: 2026-07-19 19:16:53

- seed=0: dead_t1=0.0600  task1_acc=0.9238  drop=13.84pp  t_collapse=269
- seed=1: dead_t1=0.0650  task1_acc=0.9354  drop=21.84pp  t_collapse=210
- seed=2: dead_t1=0.0750  task1_acc=0.9309  drop=14.34pp  t_collapse=219
- seed=3: dead_t1=0.0750  task1_acc=0.9370  drop=15.01pp  t_collapse=257
- seed=4: dead_t1=0.0650  task1_acc=0.9294  drop=15.84pp  t_collapse=249
- seed=5: dead_t1=0.0850  task1_acc=0.9386  drop=22.69pp  t_collapse=168
- seed=6: dead_t1=0.0750  task1_acc=0.9389  drop=30.08pp  t_collapse=200
- seed=7: dead_t1=0.0350  task1_acc=0.9396  drop=17.65pp  t_collapse=208

## Key results
- mean_dead_after_task1: 6.7% (healthy: < 15%) ✓
- mean_task1_acc: 0.934 (~0.94 confirmed) ✓
- mean_acc_drop_pp: 18.91pp (plasticity loss confirmed) ✓
- n_collapsed: 8/8 with 20pp threshold ✓

## Analysis

### Lead times (median tasks before collapse)
- dead_unit_fraction:    median=210, mean=217, iqr=[200,246], ci95=[197,238]
- gradient_noise_scale:  median=208, mean=212, iqr=[198,238], ci95=[194,230]
- effective_rank:         median=203, mean=209, iqr=[185,238], ci95=[186,231]
- weight_norm_drift:      median=157, mean=166, iqr=[140,199], ci95=[141,193]

### Precedence order (all threshold levels 30%/50%/70%)
dead_unit_fraction > gradient_noise_scale > effective_rank > weight_norm_drift

### Prediction vs. reality
PREDICTED: GNS > erank > dead > wdrift
OBSERVED:  dead > GNS > erank > wdrift
→ Dead unit fraction leads, not GNS as predicted.
→ The top 3 (dead, GNS, erank) are very close (within 7 tasks of median).

### Paired erank vs GNS
- median_diff = -7.0 tasks (GNS leads erank by 5 tasks on average)
- Wilcoxon p = 0.37 (not significant at 0.05)

### Predictive AUC (k=5 tasks ahead)
- dead_unit_fraction:   0.919 (excellent)
- effective_rank:        0.907 (excellent)
- weight_norm_drift:     0.880 (good)
- gradient_noise_scale:  0.290 (POOR — highly variable, anti-predictive without inversion)
  Note: inverted AUC = 0.71 (lower GNS = predicts collapse), still below dead/erank

### Threshold sensitivity
Order is IDENTICAL at 30%, 50%, 70% thresholds:
dead_unit_fraction > gradient_noise_scale > effective_rank > weight_norm_drift
→ Result is robust to onset threshold choice

Finished: 2026-07-19 19:36:26 (1172s)
