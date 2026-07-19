# smallnet Experiment Log

## Setup
- Date: 2026-07-19 18:20:20
- Script: src/run_small.py
- Hidden sizes: [100, 256]
- N_TASKS=300, N_SEEDS=3, STEPS_PER_TASK=200
- SGD lr=0.05 mom=0.9 WD=0.0
- Device: cuda

## Decisions
- Starting LR=0.05 from prior pmnist run (task1_acc=0.975 at hidden=2000)
- Running LR verification for each hidden size to confirm health
- STEPS_PER_TASK=200 ('a few hundred', online per brief)
- Health criterion: dead_after_task1 < 0.25, task1_acc > 0.80
- Plasticity loss: acc_drop_pp > 3.0pp (primary outcome = acc_first20 vs acc_last20)
- Collapse criterion: acc drops >= 15.0pp for >= 2 consecutive tasks (for t_collapse)
- Incremental save every 25 tasks
- 4 observables: dead_unit_fraction, effective_rank (penultimate SVD), GNS (McCandlish B_opt, 2×64 mini-batches), weight_norm_drift

## LR Verification (hidden=100)
- Tested LR ∈ {0.01, 0.05, 0.10} on seed=0, task=0
- LR=0.01 → task1_acc=0.862, dead_t1=0.120 (healthy but low acc)
- LR=0.05 → task1_acc=0.920, dead_t1=0.070 (healthy)
- LR=0.10 → task1_acc=0.924, dead_t1=0.060 (healthy, best)
- Selected LR=0.10 for hidden=100

## Early Results (hidden=100, LR=0.10, INTERIM — 2 seeds done)
- Seed 0: task1_acc=0.9238, dead_t1=0.06, healthy=True
  acc_first20=0.9172, acc_last20=0.7789, drop=13.84pp, t_collapse=269 (COLLAPSE!)
- Seed 1: task1_acc=0.9354, dead_t1=0.065, healthy=True
  acc_first20=0.9162, acc_last20=0.6978, drop=21.84pp, t_collapse=112 (COLLAPSE!)
- Seed 2: running...
- KEY FINDING: GENUINE plasticity loss in healthy small (hidden=100) network!
  Dead units grow rapidly: 0.06→0.89 (task1→task300)
  Effective rank collapses: ~38→8 (task1→task300)
  Contrast with hidden=2000 (prior run): drop only 0.41pp, no collapse

