# LOG — lrgrid (full) round

## 2026-07-19

### Context from Prior Rounds

**baseline_obs** (probe): Permuted-CIFAR-10, SGD lr=0.05, 5 seeds × 20 tasks × 1000 steps.
- Collapse: 5/5 seeds. Mean dead_after_task1 = **76.7%** (pathological dying-ReLU)
- Preliminary lead ordering: erank(2.6) > wnorm(2.0) > dead(1.8) > gns(0.8)

**precedence** (full): 4 configs × 8 seeds × Permuted-CIFAR-10 (shared head, never reset).
- SGD+BN-OFF: 8/8 collapsed. LR=0.05 caused 77% dead units after Task 1 — already pathological.
- Adam+BN-OFF: 0/8 collapsed. SGD+BN-ON: 0/8 collapsed.

### Validity Threat This Round Addresses

The prior rounds used lr=0.05 which produced 76.7% dead units after just Task 1. This raises
the question: is the measured "collapse" genuine continual-learning plasticity loss, or is it
optimizer-induced degeneracy (dying-ReLU pathology from task 1 onward)?

This lrgrid round investigates: is there an LR where
1. The network is HEALTHY after Task 1 (dead_after_task1 < 30%)  
2. Task 1 is learned normally (val_acc >> 20% chance)
3. Plasticity still declines over the 10-task stream (eventual collapse)

### Design Decisions

**Dataset**: Split-CIFAR-100, 10 tasks × 5 classes (as specified in EXPERIMENT.md)
- Classes 5t..5t+4 for task t (fine labels 0-49 used)
- ~2500 train / ~500 test per task
- 5-class problem: chance = 20%

**Setting**: Task-incremental — fresh 5-class head per task, shared trunk never reset
- "New-task accuracy" = ability to learn each new task (trainability measure)
- Collapse criterion: new-task acc < 25% (chance + 5pp) for ≥2 consecutive tasks

**LR grid**: {1e-4, 3e-4, 1e-3, 3e-3, 1e-2}, 3 seeds each (15 runs total)

**Steps per task**: 500 (>= 300 minimum from EXPERIMENT.md)

**Started**: 2026-07-19
**Command**: `python src/run_lrgrid.py 2>&1 | tee results/lrgrid/run.log`
**Runtime**: 1.35 minutes (H100 GPU — very fast because MLPs are cheap)

### Results Summary

| LR | dead_at_init | dead_after_t1 | task1_val_acc | final_acc | collapse_frac |
|---|---|---|---|---|---|
| 1e-4 | 20.8% | **22.5%** | 30.5% | 36.7% | 0/3 |
| 3e-4 | 20.8% | **25.0%** | 45.7% | 48.0% | 0/3 |
| 1e-3 | 20.8% | **26.2%** | 53.0% | 58.5% | 0/3 |
| 3e-3 | 20.8% | **21.8%** | 57.9% | 64.4% | 0/3 |
| 1e-2 | 20.8% | **19.9%** | 60.7% | 66.8% | 0/3 |

### Key Findings

1. **All LRs produce HEALTHY networks after Task 1**: dead_after_task1 ranges from 19.9% to 26.2%,
   all well below the 30% threshold. Notably, lr=0.01 is actually HEALTHIER than lr=0.001 
   (19.9% vs 26.2% dead). This is because higher LR drives stronger gradient flow that 
   keeps units active.

2. **NO collapse at any LR**: New-task accuracy remains 25-79% throughout all 10 tasks for
   all 15 runs. Collapse (new-task acc < 25%) never occurs.

3. **Network IMPROVES with more tasks**: Final-task accuracy is HIGHER than task-1 accuracy
   at all LRs (e.g., lr=0.01: task1=60.7% → final=66.8%). The trunk benefits from
   multi-task exposure — more diverse training produces better general features.

4. **Effective rank does NOT crash**: Erank stays in the range 20-75 throughout (vs. 
   the prior round's crash from 35 → 0 at lr=0.05). The network remains representationally
   rich throughout all 10 tasks.

### Interpretation: Why No Collapse Here?

Two key differences from prior rounds explain the absence of collapse:

**A. Task setting (most important)**: Task-incremental with fresh 5-class heads is much
   easier than shared-head Permuted CIFAR-10. Even a degraded trunk (e.g., 30-40% dead)
   provides enough signal for a fresh linear head to achieve good 5-class accuracy.
   With 400 units/layer and 20-26% dead → ~300-320 active units → easily sufficient
   for 5-class linear separation.

**B. LR range**: The highest LR tested here (0.01) is 5× lower than the lr=0.05 from
   prior rounds. At lr=0.05, dying-ReLU accumulated rapidly (77% dead after Task 1).
   At lr=0.01, dying-ReLU is essentially absent (20% dead = similar to at-init levels).

### What This Tells Us About the Prior Round's Results

The precedence round (Permuted CIFAR-10, shared head, lr=0.05) had TWO confounds:
1. Dying-ReLU pathology: 77% dead units after Task 1 = already catastrophically damaged
2. Hard collapse setting: shared head, 10-class, random permutations each task

This lrgrid result does NOT rule out genuine plasticity collapse. It rules out collapse
in the specific (Split-CIFAR-100, task-incremental, 10-task, LR≤0.01) regime.

To study genuine healthy-regime collapse without dying-ReLU confound, one should use:
- Permuted CIFAR-10 (shared head, 20+ tasks) at lower LR (e.g., lr=3e-3 or 1e-2)
- This would maintain < 30% dead units while still producing collapse through shared-head 
  catastrophic forgetting

But that experiment was not run here.

### Answer to the Experiment's Central Question

**Is there a "healthy-yet-collapsing" regime?**
- In Split-CIFAR-100 task-incremental with ≤10 tasks and LR∈{1e-4..1e-2}: **NO**
- All LRs are healthy (dead<30%) AND none collapse
- The criteria cannot ALL be met simultaneously in this specific setup
- healthy_collapsing_lr = null
- recommended_lr_for_precedence = null (with explicit honest explanation)

### Files

- `src/run_lrgrid.py` — experiment code
- `results/lrgrid/run.log` — full stdout/stderr
- `results/lrgrid/RESULTS.json` — final results with per-run data
- `results/lrgrid/LOG.md` — this file
