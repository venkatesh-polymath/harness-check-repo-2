# LOG — baseline_obs round

## 2026-07-18

### Goal
Establish plasticity collapse baseline and instrument 4 leading-indicator
observables (dead_unit_fraction, effective_rank, grad_noise_scale, weight_norm_drift)
per task on a sequential task stream. Report preliminary lead times.

---

### Attempt 1: Split-CIFAR-10 (5 tasks × 2 classes, 500 steps/task)
**Result: NO collapse**

Accuracy varied by task pair difficulty (binary classification; inter-class
similarity dominates). Task 4 actually had HIGHER accuracy than task 1.
Dead fraction increased 32%→50% (observable moves) but accuracy did NOT decline.

Decision: 5 tasks too few; binary task difficulty too variable; switch to CIFAR-100.

---

### Attempt 2: Split-CIFAR-100 (10 tasks × 10 classes, 2000 steps/task)
**Result: NO collapse**

Accuracy stayed flat at ~55-60% across 10 tasks. Dead fraction increased 17%→53%
(large movement!) but accuracy did not decline monotonically because:
- Tasks have varying difficulty (different CIFAR-100 class groups)
- 10-class output head adapts to each task (no reset needed)
- 47% active units still sufficient for 10-class classification

---

### Attempt 3: CIFAR-100 binary tasks (20 tasks × 2 classes, output head RESET, LR=0.05)
**Result: PARTIAL — 2/5 seeds showed collapse by criterion**

Binary task pair difficulty varies wildly across CIFAR-100 class pairs. Some pairs
(e.g., visually similar classes) give 60% accuracy even with a dead network; others
give 97% from task 7 even late in training. Task-difficulty noise completely masks
the plasticity signal.

Decision: The canonical fix is PERMUTED inputs (identical theoretical difficulty per task).

---

### Attempt 4 (FINAL): Permuted CIFAR-10 (20 tasks, same 10-class problem, different pixel permutations)
**Result: SUCCESS — 5/5 seeds show collapse**

**Why this works:**
- All tasks are the SAME 10-class CIFAR-10 classification, just with different FIXED
  pixel permutations. Identical theoretical difficulty across all tasks.
- Each new permutation destroys prior learned features, forcing the network to
  learn new ones using the same (increasingly dead) hidden layers.
- Output head is NEVER reset — must adapt in-place to each new permutation.

**Config chosen:**
- 3-layer MLP (400-400, ReLU), no BatchNorm — matches CBP paper spec
- SGD + momentum (LR=0.05, mom=0.9, WD=0) — no weight decay for raw collapse
- 20 tasks × 1000 steps/task — fast probe, enough to see severe collapse
- 5 seeds

**Key results:**
- Task 1 mean acc: 29.1% (fresh network, 10-class above chance)
- Task 10+ mean acc: 10.0% (CHANCE — complete collapse)
- Dead fraction: 76.6% (task 1) → 99.9% (task 20) — catastrophic
- Effective rank: 36.0 (task 1) → 0.0 (task 20) — complete rank collapse
- Weight norm drift: ~2.3x (task 1) → ~4.5x (task 20)
- GNS: measurable but noisy (1-16 range)

**Collapse detection (15pp drop from task-1, ≥2 tasks):**
- t_collapse per seed (0-based): 2, 1, 9, 3, 3 (all non-null)

**Preliminary lead times (50% threshold, mean over 5 seeds):**
- effective_rank: +2.6 tasks (LONGEST LEAD)
- weight_norm_drift: +2.0 tasks
- dead_unit_fraction: +1.8 tasks
- grad_noise_scale: +0.8 tasks (shortest, noisy)

Note: collapse happens FAST with LR=0.05 (within 1-9 tasks), so lead times are
short (0-9 tasks range). The rigorous run will use a longer stream / smaller LR
to spread out the collapse and better measure lead times.

**Sanity checks:**
- All 5 seeds: init CE ≈ 2.303 ≈ log(10) PASS (diff < 0.01)
- Wall-clock: 3.7 min (well within 40 min budget)

---

### Observables evaluation
| Observable          | Measurable? | Monotone? | Notes |
|---------------------|-------------|-----------|-------|
| dead_unit_fraction  | YES         | YES       | 77%→99.9% monotone increase |
| effective_rank      | YES         | YES       | 36→0 monotone decrease, saturates at 0 |
| grad_noise_scale    | YES         | NO        | Noisy (1-16 range), but tends to increase |
| weight_norm_drift   | YES         | YES       | Monotone increase until collapse, then plateaus |

### Conclusion
All 4 observables are measurable. Collapse is reproducible (5/5 seeds).
Preliminary lead time ordering: erank > wnorm_drift ≈ dead > gns.
This is DIFFERENT from the pre-registered prediction (gns > erank > dead > wnorm),
suggesting gns may NOT be the leading indicator under this regime.
However, this is a FAST collapse regime (LR=0.05, permuted-CIFAR, 1000 steps/task)
— the rigorous run needs more tasks and tuned LR to be definitive.
