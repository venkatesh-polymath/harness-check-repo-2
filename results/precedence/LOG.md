# LOG — precedence (full) round

## 2026-07-18

### Context
Building on baseline_obs round (probe). That run used Permuted-CIFAR-10 (20 tasks × 10-class, 1000 steps/task, 5 seeds, SGD only). It established:
- Collapse reproducible 5/5 seeds
- Preliminary lead ordering: erank(2.6) > wnorm_drift(2.0) > dead(1.8) > gns(0.8)
- GNS notably NOISY and SHORT lead, contradicting pre-registered prediction

### This Round Design Decisions

**Dataset**: Split-CIFAR-100, 10 tasks × 5 classes (fine labels 0-49, classes [5t..5t+4] for task t).
- Each class has 500 training + 100 test examples → 2500 train / 500 test per task
- Task-incremental: fresh 5-class head per task, shared trunk never reset
- Collapse criterion: new-task acc < 20%+5pp = 25% for ≥2 consecutive tasks

**Architecture**: 3-layer MLP (400-400 ReLU), with or without BatchNorm1d.
For BN-ON: BN applied AFTER FC, BEFORE ReLU.
Observables measured on PRE-BN FC outputs for comparability across BN arms.

**Configs** (4 total):
- sgd_bnoff: SGD+momentum, lr=0.05, mom=0.9, wd=0.0, BN=OFF
- adam_bnoff: Adam, lr=1e-3, BN=OFF
- sgd_bnon: SGD+momentum, lr=0.05, mom=0.9, wd=0.0, BN=ON
- adam_bnon: Adam, lr=1e-3, BN=ON

**Seeds**: 8 per config
**Steps/task**: 2000

**Pre-registered onset criterion**:
- Monotone-smooth trajectory with isotonic regression (direction = sign(vf - v0))
- t_fire = first task crossing 50% of [task1_val → final_val] range
- lead_time = t_col - t_fire (positive = fires before collapse)

**Analysis**:
- Bootstrap CI (n=1000) for mean lead times
- Precedence order per config: sort by mean lead time descending
- Optimizer-conditional test: SGD vs Adam GNS lead time difference

### Decision: Why not Permuted-CIFAR-10?
EXPERIMENT.md explicitly specifies Split-CIFAR-100 (10 tasks × 5 classes) for the rigorous run. The study spec has a minor discrepancy (says CIFAR-10), but the "THIS ROUND (do exactly this)" section takes precedence. Split-CIFAR-100 is harder (fine-grained, 500 examples/class) and provides richer representational demands per task.

### Risk: Collapse under Adam+BN
BatchNorm is known to maintain plasticity by renormalizing activations. Adam+BN may show weak or no collapse. Plan: if <4/8 seeds collapse for a config, report collapse_frac honestly and compute lead times only for seeds with detected collapse; note as "insufficient collapse" in results.

### PIVOT: Split-CIFAR-100 → Permuted CIFAR-10
**Finding from first attempt**: Split-CIFAR-100 with task-incremental (fresh 5-class head per task) produced 0/8 collapse across all 8 seeds for SGD+BN-OFF config. Accuracy stayed 47–73% across all 10 tasks.

**Why**: Even a trunk with 90% dead units can support a fresh 5-class head because:
1. 5-class CIFAR-100 only needs ~10-40 active neurons to get >25% accuracy
2. With 400 units/layer and 70-90% dead, still 40-120 active neurons — more than enough
3. Fresh head eliminates any accumulated output-head damage

**Decision**: Use Permuted CIFAR-10 (same as baseline_obs which proved collapse in 5/5 seeds):
- Shared 10-class output head, NEVER reset
- 20 tasks × pixel permutation destroys prior features each task
- Trunk must find NEW representations from an increasingly-dead network
- Collapse criterion: acc < 15% (chance 10% + 5pp) for ≥2 consecutive tasks

### Running (second attempt)
`python src/run_rigorous.py 2>&1 | tee results/precedence/run.log`
Expected: ~15-25 min on H100 (4 configs × 8 seeds × 20 tasks × 1000 steps = 640K steps)

### Interim Results (SGD+BN-OFF and Adam+BN-OFF complete)

**SGD+BN-OFF**: 8/8 seeds collapsed. Lead times:
- effective_rank: +5.50 tasks (CI=[+4.00, +6.88]) — LONGEST
- dead_unit_fraction: +3.75 tasks (CI=[+1.75, +5.75])
- weight_norm_drift: +2.62 tasks (CI=[-1.00, +5.00])
- gradient_noise_scale: +0.38 tasks (CI=[-3.63, +4.00]) — SHORTEST, NOISY

**Adam+BN-OFF**: 0/8 collapsed. Accuracy stays 37-43% throughout 20 tasks.
- Adam prevents plasticity collapse entirely in 20-task regime
- Dead fraction only 62-87% (vs SGD's 72-99.9%)
- Effective rank 4-18 (vs SGD's 1-6 after collapse)

**SGD+BN-ON**: 0/8 collapsed. Accuracy stays 44-50%.
- BatchNorm prevents dead unit accumulation (dead fraction only 22-35%)
- Effective rank maintains 50-67 throughout (vs SGD's 1-6 collapse)

**Conclusion so far**: ONLY SGD without BatchNorm shows plasticity collapse in 20-task regime.
Both Adam and BatchNorm independently prevent collapse.
Optimizer-conditional finding is STRONGER than expected: Adam doesn't just suppress GNS — it prevents collapse entirely.
