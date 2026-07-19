# LOG — pmnist round (full)

## 2026-07-19

### Context
Prior rounds (baseline_obs, lrgrid, gnsfix, gns200, closeloop, precedence) worked on
Permuted-CIFAR-10. The CIFAR-10 setup DID show plasticity collapse under SGD, but:
- Split-CIFAR-10 (binary, 5 tasks): no collapse (too few tasks, binary difficulty varies)
- CIFAR-100 (binary, 20 tasks): no collapse (task difficulty varies too much)
- Permuted CIFAR-10 (20 tasks): collapse in 5/5 seeds BUT collapse was very fast (tasks 1-9)
  and the "healthy" condition was marginal under some configs.

This round: **Online Permuted-MNIST** — the canonical Dohare et al. benchmark.
MNIST is small (784-dim), so we can run 100 tasks × 5 seeds comfortably.

### Design decisions

1. **Architecture**: 2000-2000 MLP (2 hidden layers, ReLU). Matches Dohare et al. exactly.
   - IN_DIM=784 (28×28 grayscale, normalized 0-1)
   - No BatchNorm

2. **LR tuning**: grid [0.05, 0.01, 0.005, 0.001] with momentum=0.9.
   - Health criterion: task1_acc > 0.9 AND dead_after_task1 < 0.20
   - Tune on seed=0, task=0, then hold fixed

3. **100 tasks**: long stream to clearly show plasticity loss.
   - 1000 steps/task (≈1.7 epochs of 60k MNIST)
   - MNIST MLP on H100 is extremely fast (<5 min total)

4. **Observables** (measured after training each task):
   - `dead_unit_fraction`: fraction of ReLU units inactive on >95% of 1000 probe samples
   - `effective_rank`: exp(H) of penultimate (fc2) activation SVD; always >=1
   - `gradient_noise_scale`: McCandlish B_opt with two 64-sample batches (>>20)
   - `weight_norm_drift`: mean |‖w_t‖/‖w_0‖ - 1| across layers

5. **Collapse**: first task where new-task acc drops >=15pp below task-1 acc
   for >=2 consecutive tasks.

6. **Onset**: 50% of range crossing with moving-avg window 2.

7. **Lead-time analysis**: bootstrap CI (2000 resamples), Wilcoxon signed-rank
   for paired erank vs GNS lead differences.

### Script
`src/run_pmnist.py` — single blocking run, output captured to run.log.

### Prediction (from EXPERIMENT.md)
- GNS will show longest lead (5-15 tasks before collapse)
- Effective rank second (3-10 tasks)
- Dead-unit fraction third (1-5 tasks)
- Weight-norm drift near-contemporaneous (0-2 tasks)

### Run command
```
python src/run_pmnist.py 2>&1 | tee results/pmnist/run.log
```

---

## Run 1 results (2026-07-19)

### Outcome: NO PLASTICITY LOSS

Config: HIDDEN=2000, momentum=0.9, lr=0.05, 1000 steps/task, 100 tasks, 5 seeds.

- dead_at_init = 5.2%, dead_after_task1 = 2.3%, task1_acc = 97.5% → **HEALTHY**
- acc_task100 ≈ 97.5% (virtually identical to task 1)
- acc_drop = only 0.16pp → **NO PLASTICITY LOSS**
- dead fraction DID increase: 2.3% → ~62% by task 100
- BUT: 2000-unit layers with 62% dead → 1520 alive units, still plenty for MNIST

### Root cause
The 2000-2000 network has massive overcapacity. Even with 62% dead units, 1520 alive
units is far more than needed for MNIST 10-class. Also, SGD+momentum maintains the 
plasticity of alive units (momentum helps escape flat regions). The effective rank
actually INCREASED (253→340), indicating alive units develop richer representations.

### Diagnosis from Dohare et al.
They use **SGD without momentum** (pure gradient descent). Without momentum:
- Dead units accumulate faster (no gradient history to escape)
- Alive units can't maintain as diverse representations
- Result: accuracy drops over task stream

### Fix for Run 2
- MOMENTUM = 0.0 (pure SGD, matching Dohare et al.)
- Increase STEPS_PER_TASK to 2000 (compensate for weaker optimizer)
- LR grid: [0.1, 0.05, 0.01, 0.001]
- Try both HIDDEN=2000 and HIDDEN=400

---

## Run 2 (2026-07-19)

Config: HIDDEN=2000, momentum=0.0, STEPS_PER_TASK=2000, 100 tasks, 5 seeds.

### LR tuning outcome
[See run.log]

### Health check
[See RESULTS.json]

### Plasticity loss
[See RESULTS.json]
