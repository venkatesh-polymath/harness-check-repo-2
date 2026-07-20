# Adam Experiment — LOG

## Context
This round tests whether the plasticity-loss findings from the paper (which used SGD) replicate under **Adam optimizer** (lr=1e-3). Same MLP architecture (hidden=100), same Permuted-MNIST protocol, 8 seeds × 300 tasks. Also measures predictive AUC for each signal and a monotone-null baseline.

## Why this round
PAPER_FINAL.md identifies "Optimizer and task family — all regimes use SGD" as a limitation (Section 5, point 4). This Adam run directly addresses that critique.

## Setup
- GPU: NVIDIA A10, 23 GB
- PyTorch 2.13.0+cu130, CUDA available
- torchvision 0.28.0, sklearn 1.9.0, scipy 1.18.0
- Script: src/run_adam.py (not modified per instructions)
- Results dir: results/adam/

## Step 1 — Smoke Test
Running: `SH_SMOKE=1 python src/run_adam.py`

**Result:** PASSED (exit 0). Smoke run: 1 seed × 6 tasks, acc=0.93, healthy=True. 
With only 6 tasks no plasticity loss yet (expected). AUC metrics null (insufficient data).

## Step 2 — Full Run
Running: `python src/run_adam.py 2>&1 | tee results/adam/run.log`

Config: hidden=100 MLP, Adam lr=1e-3, 8 seeds × 300 tasks, Permuted-MNIST.
GPU: NVIDIA A10.

### Observations during run (captured live)

**Partial results by seed (completed seeds):**

| Seed | Task 1 acc | Dead@T1 | Drop (pp) | Collapsed? |
|------|-----------|---------|-----------|-----------|
| 0 | 0.9142 | 0.080 | 4.1 | No |
| 1 | 0.9163 | 0.090 | 4.0 | No |
| 2 | 0.9187 | 0.075 | 4.3 | No |
| 3 | 0.9144 | 0.085 | 4.5 | No |
| 4 | 0.9173 | 0.040 | 4.1 | No |

**Key observation:** Under Adam (lr=1e-3), dead unit fraction still rises dramatically
(0.04-0.09 at task 1 → ~0.60-0.65 by task 300), but accuracy drop is only ~4pp
compared to ~19pp under SGD. Adam appears to maintain accuracy better than SGD
despite accumulating dead units. No collapses (tc=None) seen yet — the 20pp threshold
is not crossed under Adam in these seeds.

**Trajectory of seed 0 (representative):**
- Task 1: acc=0.914, dead=0.080
- Task 50: acc=0.923, dead=0.430
- Task 100: acc=0.916, dead=0.495
- Task 150: acc=0.908, dead=0.515
- Task 200: acc=0.889, dead=0.585
- Task 300: acc=0.885, dead=0.605

Dead units accumulate but accuracy stays high — Adam's adaptive learning rates
compensate for dead units in a way SGD cannot.

## Final Results (all 8 seeds)

**Per-seed summary:**

| Seed | Task 1 acc | Dead@T1 | Dead@T300 | Drop (pp) | Collapsed? |
|------|-----------|---------|-----------|-----------|-----------|
| 0 | 0.9142 | 0.080 | 0.605 | 4.1 | No |
| 1 | 0.9163 | 0.090 | 0.585 | 4.0 | No |
| 2 | 0.9187 | 0.075 | 0.605 | 4.3 | No |
| 3 | 0.9144 | 0.085 | 0.595 | 4.5 | No |
| 4 | 0.9173 | 0.040 | 0.620 | 4.1 | No |
| 5 | 0.9143 | 0.100 | 0.585 | 4.2 | No |
| 6 | 0.9126 | 0.080 | 0.620 | 4.1 | No |
| 7 | 0.9113 | 0.090 | 0.580 | 3.8 | No |

**Aggregate:**
- Mean task-1 acc: 0.9149 (vs 0.934 SGD)
- Mean dead@T1: 0.08 (vs 0.067 SGD) — similar healthy start
- Mean acc drop: 4.2pp (vs 18.9pp SGD) — dramatically less plasticity loss!
- n_collapsed: 0/8 (vs 8/8 for SGD)

**Key finding: Adam does NOT exhibit accuracy-based plasticity collapse.**
Despite dead units accumulating to similar final levels (~60%), Adam maintains
accuracy within 4.2pp. The 20pp collapse threshold is never crossed. This means:
- `plasticity_loss_confirmed = False`
- All predictive AUC values are NULL (no positive labels possible)
- The SGD-specific findings from PAPER_FINAL.md do not transfer to Adam

**Why predictive AUC is null:** The pauc() function requires at least 10 observations
with BOTH positive (collapse imminent) and negative (safe) labels. Since no seed
ever collapses, all labels are 0 (safe), making AUC undefined.

## Analysis and Interpretation

This result directly addresses PAPER_FINAL.md Section 5, Limitation 4:
> "Optimizer and task family. All regimes use SGD and permutation-based task 
> streams. Adam and non-permutation continual streams are not covered."

**Finding:** Adam's adaptive per-parameter learning rates effectively compensate
for the dead units that accumulate, maintaining plasticity at the accuracy level.
This is consistent with known properties of Adam: its per-parameter learning rate
scaling can boost gradients for nearly-dead units, preventing the feedback loop
that causes SGD-based collapse.

**Mechanism:** Under SGD, dead units receive near-zero gradients → permanently
stay dead → effective capacity drops → collapse. Under Adam, even near-dead units
have their effective step size boosted by the 1/sqrt(v+eps) scaling, allowing
recovery. The adaptive optimizer acts as an implicit plasticity preserving mechanism.

**Implication for the paper:** The paper's claim about dead-unit fraction being a
predictor is specific to SGD regimes. Under Adam, dead units accumulate (structural
signal) but don't predict collapse (because collapse doesn't happen). This is an
important qualification: the early-warning system addresses a problem (SGD 
plasticity collapse) that Adam largely avoids.

## Files Written
- results/adam/run.log: full stdout/stderr of the run
- results/adam/trajectories.json: per-task accuracy, dead, erank, gns, wdrift for all 8 seeds
- results/adam/RESULTS.json: final results (status=SUCCESS)
- results/adam/LOG.md: this file

## Notes on Weights
No model weights were saved (the script does not checkpoint). The _weights/ 
directory exists but is empty. No weights were committed to git.
