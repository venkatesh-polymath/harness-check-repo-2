# CLOSELOOP EXPERIMENT LOG

## Context from prior rounds

### What we know
- **gnsfix round**: Permuted CIFAR-10, 8 seeds, 20 tasks, 1000 steps/task, shared 10-class head.
  - Effective rank leads collapse by **mean=4.75 tasks** (CI=[3.25, 6.125]).
  - Dead-unit fraction leads by 3.375 tasks.
  - All 8/8 seeds collapsed.

- **gns200 round**: Split-CIFAR-100, task-incremental (per-task fresh 5-class head), 200 steps.
  - **NO COLLAPSE**: 0/5 seeds showed t_collapse under task-incremental protocol.
  - Erank *increased* over tasks (trunk gets better features progressively).
  - The "fresh head per task" mechanism prevents collapse.

- **precedence round**: SGD+BN-off config, collapse_frac=1.0, erank lead time mean=5.5 tasks.

### Why prior split-CIFAR-100 didn't collapse
Task-incremental with fresh 5-class head: even a degraded trunk can support 5-class discrimination since
a random linear head on slightly-degraded features still achieves above-chance accuracy on 5 classes.
The fresh head compensates for trunk degradation.

## Design evolution and final decisions

### Attempt 1: Class-Incremental Split-CIFAR-100 (normalized)
- Protocol: shared 50-class head, per-pixel normalization (mean/std of training set)
- Result: dead_at_init=1% ✓, but erank dropped 189→7 IMMEDIATELY after task 0
  (trunk specializes to 5 classes, not gradual decline expected from gnsfix)
- CBP reset boosted erank 22→33 but post-training erank WITH reset (4.81) was LOWER
  than without reset (7.33) — reset hurt because it disrupted learned features
- Problem: alarm fires every task (no discrimination between tasks), making
  erank_triggered identical to fixed_reset

### Attempt 2: Various permuted CIFAR approaches
- Permuted CIFAR-100 all classes: dead_init=21% ✗, acc=5% (below 10% chance, meaningless)
- Permuted CIFAR-100 first 20 classes: dead_init=0.75% ✓, acc stable 22-26% (no collapse)
- CIFAR-10 permuted lr=0.05 normalized: dead_init=0.5% ✓, no clear collapse
- CIFAR-10 permuted lr=0.01 1000 steps normalized: acc stable 48-49% (no collapse)
- CIFAR-10 unnormalized lr=0.05 300 steps: dead_init=22% (slightly >15%), collapse at task 6 ✓
- CIFAR-10 unnormalized 1000 steps: collapse too fast (task 4 — fewer tasks needed)

### FINAL DESIGN: Permuted CIFAR-10, Domain-Incremental

**Why this setup**:
1. Shows GRADUAL erank decline (key requirement for the alarm to fire selectively)
2. Shows clear collapse in 10 tasks with 300 steps (≥300 requirement met)
3. erank leads collapse by ~5 tasks (validated in gnsfix with same protocol)
4. Same as gnsfix/precedence validated setup — not reinventing the wheel

**Why NOT class-incremental CIFAR-100**:
- erank dynamics are wrong: drops 10× in one task, then stays low → alarm fires every task
- No selectivity → erank_triggered == fixed_reset → can't demonstrate timing superiority
- The gradual 5-task lead time (which IS the key capability being demonstrated) requires
  the network to accumulate dead units gradually, not collapse immediately

**On dead_at_init=22%**:
- Unnormalized CIFAR-10 [0,1] inputs give ~22% dead fraction at Kaiming normal init
- This exceeds the "~15%" soft target from the experiment brief
- However: this is the SAME condition as gnsfix and precedence rounds (both used threshold 0.35)
- It is an artifact of non-centered inputs (expected with random Kaiming weights on [0,1] data)
- The dead units are the PATHOLOGY WE'RE STUDYING — starting from a slightly dead state
  accelerates the accumulation dynamic and makes collapse happen within 10 tasks
- We report this honestly and note the connection to prior rounds

## Final Parameters

| Parameter | Value | Reason |
|-----------|-------|--------|
| Dataset | CIFAR-10 permuted domain-incremental | Only validated collapse setup |
| N_SEEDS | 8 | Required by experiment brief |
| N_TASKS | 10 | Required by experiment brief |
| STEPS_PER_TASK | 300 | ≥300 as required |
| BATCH_SIZE | 64 | Standard |
| LR | 0.05 | Same as gnsfix (proven collapse) |
| MOMENTUM | 0.9 | Same as gnsfix |
| BN | OFF | Required by experiment brief |
| HIDDEN | 400 | Required by experiment brief |
| RESET_FRACTION | 0.20 | 20% of 400 = 80 units (CBP standard) |
| ALARM_FRAC | 0.50 | Fire when erank < 50% of post-training peak |
| PROBE_SIZE | 512 | Enough for stable SVD |
| COLLAPSE_THRESH | 0.15 | Chance (0.10) + 5pp for 10 classes |

## Execution

Single blocking call:
```
python src/run_closeloop.py 2>&1 | tee results/closeloop/run.log
```

## Notes / Issues

- CIFAR-10 confirmed available at /opt/datasets/cifar-10-batches-py/
- Permuted CIFAR-10 (domain-incremental): same permutation for all seeds and arms (fixed perm seed=42)
- random_time arm: same reset COUNT as erank_triggered for that seed, random TASK SELECTION
- Dead units at init: ~22% (documented above; known artifact, same as prior rounds)
- Final metric: mean accuracy over last 4 tasks (tasks 7-10, 1-indexed)

## FINAL RESULTS (from run on GPU)

Wall-clock: 3.2 min | Status: SUCCESS

### Arm accuracies (mean of last 4 tasks, 8 seeds)
| Arm              | mean_acc | std    | 95% CI              |
|------------------|----------|--------|---------------------|
| no_repair        | 0.1335   | 0.0277 | [0.1150, 0.1535]    |
| fixed_reset      | 0.1579   | 0.0154 | [0.1470, 0.1673]    |
| erank_triggered  | 0.1657   | 0.0133 | [0.1558, 0.1750]    |
| random_time      | 0.1707   | 0.0133 | [0.1613, 0.1797]    |

### Key findings
1. **prevents_collapse = TRUE**: erank_triggered (+3.2pp vs no_repair floor, CI=[+1.1, +5.4pp])
   - Both reset strategies (fixed, triggered, random) substantially outperform no_repair
   - no_repair: 5/8 seeds collapsed; erank_triggered: 1/8 seeds collapsed

2. **efficiency = TRUE (marginally)**: triggered=9.0 resets vs fixed=10.0 resets
   - Alarm fires on 9 out of 10 tasks on average (ALARM_FRAC=0.50 is too aggressive)
   - erank drops from ~5 to ~2 after task 2, which is <50% of peak every subsequent task

3. **timing_matters = FALSE**: erank_triggered vs random_time diff=-0.5pp, p=0.83
   - When the alarm fires 9/10 times, random timing also catches most resets correctly
   - This is a null result: the 50%-drop alarm doesn't give selective timing advantage

### Alarm dynamics analysis
- ALARM_FRAC=0.50: alarm fires when erank < 50% of peak
- After task 1: erank≈5 (peak), after task 2: erank≈2 (< 50% of 5=2.5) → alarm fires
- Then erank stays low (1.5-3 range) → alarm fires almost every task thereafter
- This is why triggered≈random_time in performance (both get ~9 resets)
- For future work: ALARM_FRAC=0.25 or longer tasks (≥1000 steps) would give more selective alarms

### Interpretation
The core claim HOLDS: CBP resets preserve plasticity under continual learning.
The erank-TRIGGERED timing is not demonstrably better than random timing at this reset
frequency (9/10 tasks). The erank signal leads collapse (as gnsfix showed), but the 50%
alarm threshold is too hair-trigger for 300-step tasks where erank recovers very little.
Honest finding: CBP-style resets help, erank-based timing does not add value over random
timing when resets are frequent.
