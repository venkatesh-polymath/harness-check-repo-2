# LOG.md — scale_grid round

## Round description
SCALE-OUT EXPERIMENT: Generalize the Adam-v_t plasticity-reset result across DATASETS and ARCHITECTURES.

Prior finding (confirm_cd + mech_rand rounds on CIFAR-100 GroupNorm ConvNet):
- All reset arms recover plasticity ~4× over no-reset floor (0.20)
- D(Adam-v_t) ≡ C(explicit Fisher): D-C = -0.20pp, CI=[-1.57,+1.17]pp ✓ within ±3pp
- D beats RANDOM: +4.33pp, p=0.0198 ✓ utility ranking carries real signal

This round extends the result to:
- **Datasets**: CIFAR-100 (5 cls/task × 10 tasks) AND CIFAR-10 (2 cls/task × 5 tasks)
- **Architectures**: (1) SmallConvNetGN, (2) MLP3GN, (3) TinyResNetGN
- **Arms**: A_floor (no reset), B (heuristic), C (Fisher), D (Adam-v_t), RANDOM
- **Seeds**: 8 (paired per seed — same task split for all arms)
- **~800 steps/task**, Adam, masked eval

Total runs: 2 datasets × 3 architectures × 5 arms × 8 seeds = 240 training runs.

## Setup

### Environment
- GPU: NVIDIA H100 80GB HBM3 (much faster than prior A10 rounds)
- PyTorch 2.13.0+cu130
- scipy available

### Architecture details
1. **SmallConvNetGN**: 3 conv-blocks + 2 FC, GroupNorm, ~3.3M params, fc1=512 units (penultimate)
2. **MLP3GN**: 3 hidden layers (512 each) with GroupNorm, ~2.2M params, fc1=512 units (penultimate)
3. **TinyResNetGN**: stem + 4 residual blocks (64ch→128ch) + global avg pool + FC-256, GroupNorm, ~800K params, fc1=256 units (penultimate)

### Hyperparameters (fixed across all arms/seeds/datasets/architectures)
- batch_size = 128
- LR = 1e-3 (Adam)
- STEPS_PER_TASK = 800 (within 800-1000 spec range)
- RESET_EVERY = 100 steps
- RESET_FRAC = 0.10 (bottom 10% of penultimate units)
- Primary metric = mean(acc[tasks 2..end]) — exclude task-1 warmup

### PAIRED design
For each (dataset, arch, seed): task splits generated ONCE from `seed`, then ALL arms run on the SAME splits. This enables tight paired statistics (D-C, D-RANDOM, D-B).

### Data
- CIFAR-100: loaded from /opt/datasets (download=False) — verified present
- CIFAR-10: loaded from /opt/datasets/cifar-10-batches-py (download=False) — verified present
- CIFAR-100: 10 tasks × 5 classes = 100 classes total
- CIFAR-10: 5 tasks × 2 classes = 10 classes total

## Decisions

| Decision | Choice | Rationale |
|---|---|---|
| steps/task | 800 | Within 800-1000 spec; ensures full run ≤ 2.5h on H100 |
| Seeds | 8 | Per spec (min 6 if time-pressed); starting with full 8 |
| Penultimate layer | fc1 for all archs | Where CBP resets happen; enables consistent comparison |
| MLP GroupNorm | GroupNorm(8, 512) on each hidden layer | Prevents dead-unit collapse via normalization |
| ResNet pen. size | 256 units | Modest size; enough for srank measurement |
| Equivalence margin | ±3pp (per prior rounds spec) | Consistent with confirm_cd round |
| Eval | Masked (current task classes only) | Consistent with prior rounds |
| RANDOM arm RNG | np.RandomState(1000+seed) | Deterministic, independent of model training RNG |

## Timeline

### 2026-07-18 — Setup
- Verified GPU: H100 80GB
- Verified data: /opt/datasets/cifar-100-python/train ✓, /opt/datasets/cifar-10-batches-py/ ✓
- Syntax check: PASSED
- Smoke tests:
  - All 3 model architectures: PASSED (correct output shapes)
  - Mini training loop (5 steps): PASSED (loss=4.68 at init, acc=0.20)
  - All 5 arms on 2-task MLP3 (50 steps/task): PASSED
- Estimated wall-time: ~2 hours on H100
- Written: src/run_grid.py
- Launch: `python src/run_grid.py 2>&1 | tee results/scale_grid/run.log`

## Run results

### 2026-07-18 — Run in progress (job bbb20xsx8)

**Launch command**: `python src/run_grid.py 2>&1 | tee results/scale_grid/run.log`

**JSON serialization issue resolved**: Python 3.12 `round(numpy.float64, n)` returns
`numpy.float64`, not Python float. Comparison `numpy.float64 >= -3.0` returns `numpy.bool_`
which crashes the standard json encoder. Fix: `NumpyEncoder` class + `cls=NumpyEncoder` in
`write_results` + explicit `float()` conversions + `bool()` wrappers for booleans.
Test: `b6o7h2d8t` JSON test PASSED before full launch.

**Speed**: ~4.5s/task on H100. CIFAR-100 (10 tasks): ~40s/run. CIFAR-10 (5 tasks): ~20s/run.
Estimated total: ~120 min.

**Progress at context compaction** (~run 10/240):
- CIFAR-100 × convnet × seed 0: A_floor=0.2776, B=0.7844, C=0.7884, D=0.7978, RANDOM=0.7576
- CIFAR-100 × convnet × seed 1: A_floor=0.3538, B=0.7771, C=0.7258, D=0.7607, RANDOM=0.7584
- RESULTS.json written after both seeds (JSON fix held ✓)

**Seed 1 paired_stats (n=2, CIs too wide — expected)**:
- D-C = +2.22pp, CI=[-14.0, +18.4]pp — will tighten to ~±2pp with n=8
- D-RANDOM = +2.13pp, CI=[-22.0, +26.2]pp

**Estimated completion**: ~120 min from launch (~17:20 UTC)

---
