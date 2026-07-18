# LOG.md — bd_paired round

## What this round is
PAIRED B-vs-D run: is Adam v_t (free) as good as the original CBP contribution heuristic?

## Design decisions

### Arms
- **B**: CBP contribution heuristic = `|outgoing weight| × mean post-activation` (running EMA α=0.01). Reset lowest-utility neurons.
- **D**: Adam `exp_avg_sq` (v_t) aggregated per neuron (mean over fc1.weight[i, :] fan). Reset lowest. Zero extra compute.

Only the utility function differs; all other hyperparameters are identical.

### Setup
- 8 seeds {0..7}, PAIRED: same task split for B and D at each seed (critical for paired stats)
- 4 tasks × 5 classes/task = 20 classes, class-incremental
- ~1000 steps/task
- GroupNorm ConvNet (SmallConvNetGN — same architecture as all prior rounds)
- Real CIFAR-100 from /opt/datasets (download=False; ABORT if missing)
- Adam lr=1e-3, reset_every=100 steps, reset_frac=0.10
- Masked eval (argmax restricted to current task's classes only)

### Why 4 tasks (not 5 or 8)
EXPERIMENT.md says "4 tasks/run, ~1000 steps/task (~20 min)". Using exactly 4.

### Per-seed metric: mean_tasks_2to4
Task 1 is warm-up (model learns first task from scratch with no prior knowledge).
The meaningful plasticity signal comes from tasks 2–4.
Statistical comparison uses mean over tasks 2–4 per seed.

### Building on prior work
- Architecture, data loading, CBP reset procedure: copied from confirm_cd.py (validated).
- Arm B utility: copied from method_arms.py (validated against prior probe run).
- Arm D utility: copied from confirm_cd.py (validated, shown equivalent to explicit Fisher).
- Statistical analysis: paired t-test with 95% CI, equivalence within ±3pp.

### Code file
`/workspace/src/bd_paired.py`

## Run log

### 2026-07-18
- Wrote bd_paired.py based on confirm_cd.py + method_arms.py
- Started GPU run on NVIDIA A10 (23GB VRAM)
- Expected runtime: ~20–25 min (8 seeds × 2 arms × 4 tasks × 1000 steps)
