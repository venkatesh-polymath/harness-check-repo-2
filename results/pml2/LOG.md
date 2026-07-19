# pml2 Experiment Log

## Setup
- Date: 2026-07-19 17:55:14
- Script: src/run_pml2.py
- N_TASKS=300, N_SEEDS=3, HIDDEN=2000, LR=0.05
- Device: cuda
- Extending prior pmnist 100-task run (no collapse found)

## Decisions
- LR=0.05 reused from pmnist tuning (task1_acc=0.975, dead_after_t1=0.023)
- N_SEEDS=3 (experiment brief specifies 3)
- STEPS_PER_TASK=1000 (same as pmnist, fast+healthy)
- GNS_BATCH=64 (two mini-batches of 64, > 20 required)
- Incremental save every 25 tasks


## Results
- Wall-clock: 959.0s (16.0 min)
- healthy=True, task1_acc=0.9745
- acc_first20=0.9768, acc_last20=0.9728, drop=0.41pp
- plasticity_loss_occurred=False (0/3 seeds collapsed)
- t_collapse_per_seed=[None, None, None]
- precedence_order=[]

## Conclusion
NO plasticity loss even at 300 tasks: 0/3 seeds collapsed. acc_drop=0.4pp (first20→last20). Network remains plastic. Permuted-MNIST with SGD momentum is a robust benchmark where plasticity is NOT lost.

## Key Observations
- Dead-unit fraction grows monotonically from 0.023 after task-1 to ~0.75 after task-300 (a 33× increase), yet accuracy does not decline.
- This dissociation shows dead-unit accumulation does NOT necessarily cause plasticity loss in permuted-MNIST.
- Effective rank of penultimate layer activations INCREASES from ~255 to ~375 across 300 tasks (the network's representational diversity expands, not contracts).
- Gradient-noise scale remains stable (1.5–2.2 range) across all 300 tasks, confirming optimization health.
- Weight-norm drift grows continuously (1.8 → 33), indicating the parameters drift far from initialization without losing plasticity.
- The combination: SGD + momentum + permuted-MNIST (label-preserving permutations) may be too forgiving for plasticity loss — the label structure doesn't change, only pixel ordering.
- Dohare et al. show loss of plasticity primarily in *split* tasks (disjoint class sets) where the output head must be reused across incompatible distributions.

## Why No Collapse?
Permuted-MNIST keeps the same label set (digits 0-9) across all tasks. The permutation only reorders pixels. This means:
1. The output head never needs to re-specialize.
2. Dead units in hidden layers don't prevent classification because enough live units remain (>25% active in both layers at task-300).
3. SGD+momentum on a smooth, well-conditioned loss (MNIST) with a large hidden width (2000) is robust to dead-unit accumulation.

