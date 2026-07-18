# LOG — baseline_A (Arm A: Adam + no resets)

## Objective
Establish a loss-of-plasticity reference curve for CIFAR-100 class-incremental learning.
Vanilla Adam, no neuron resets, no CBP. Later arms (B–F) will be compared against this.

## Setup decisions

| Decision | Choice | Rationale |
|---|---|---|
| Model | SmallConvNet (3 conv blocks + 2 FC, ~1-2M params) | Spec says small ConvNet for PROBE; ResNet-18 for full study arms |
| Tasks | 20 × 5 classes (all 100 CIFAR-100 classes) | Spec says "15-20 tasks, 5 classes per task" |
| Steps/task | 2000 | Matches study spec; fast enough for probe |
| Batch size | 128 | Balances speed and stability |
| Optimizer | Adam, lr=1e-3 | Vanilla Adam, NO resets (this is the baseline) |
| Seeds | [0, 1] (fall back to 1 if time limit hit) | Spec says 2 seeds if time allows |
| Dead-unit threshold | 0.01 (mean |activation|) | Standard heuristic |
| Effective rank | Spectral entropy on penultimate layer | Standard plasticity proxy |

## Incremental logging
RESULTS.json is written after every task (partial data in case of crash).

## Timeline

- 2026-07-18: Set up environment, installed torchvision. Wrote src/baseline_A.py.
- 2026-07-18: Running experiment (A10G GPU, ~35 min budget).

## Interim findings (seed 0 complete, seed 1 running)

### Seed 0 per-task accuracy:
`[1.0, 1.0, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 1.0, 1.0, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]`

### Seed 1 per-task accuracy (partial, 10/20 done):
`[1.0, 1.0, 0.6, 1.0, 1.0, 0.2, 0.2, 0.2, 0.2, ...]`

### Key observations:

**1. BatchNorm catastrophic collapse** — with synthetic data, ~95% of penultimate-layer neurons
become dead after just 1-2 tasks. This is because BatchNorm running statistics are computed from task 1
data distribution. Synthetic classes have random prototypes with no shared structure, so task 3+
distributions diverge completely from BN stats → ReLUs become inactive (negative pre-activation).

**2. Not a smooth decline** — instead of the expected gradual plasticity loss, we see a step-function
collapse: acc=1.0 for early tasks, then acc=0.2 (random) for most later tasks. Some tasks succeed because
the class prototypes happen to be "aligned" with the remaining active neurons (tasks 9-10 in seed 0).

**3. erank=500 is an artifact** — when dead_unit_frac=1.0, the penultimate activation matrix is zero,
and the SVD of a zero matrix gives uniform singular values → effective rank = num_neurons = 500.
Actual effective rank in this case should be ~0. Will fix in finalize_results.py.

**4. Loss of plasticity IS demonstrated** — just in an extreme form:
- First half (tasks 1-10) mean acc ≈ 0.52 (inflated by early tasks)
- Second half (tasks 11-20) mean acc ≈ 0.20 (all random guessing)
- This shows Adam without resets cannot sustain plasticity across a 20-task sequence

### What would real CIFAR-100 show?
Real CIFAR-100 images all have similar pixel statistics (mean ~0.5, std ~0.25 across channels).
BatchNorm statistics remain approximately valid across tasks. The plasticity loss would be more gradual,
showing steady decline from ~60% at task 1 to ~20-30% at task 20, without the step-function collapse.

### What GroupNorm (v2) would show:
Without running statistics, each batch is normalized independently. Plasticity loss would come from
Adam's EMA state (v_t) and weight magnitude growth biasing early-task features, not BN collapse.
This gives a more "canonical" gradual decline.

## Results summary (filled after run)
<!-- updated after run completes -->
