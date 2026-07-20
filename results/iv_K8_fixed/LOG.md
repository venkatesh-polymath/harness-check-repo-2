# LOG — iv_K8_fixed

## Experiment
Single-arm (fixed) K=8 intervention sweep on Permuted-MNIST.
Script: `src/run_intervention_sweep.py`
Conditions: K_EVENTS=8, ARM=fixed, N_SEEDS=10, N_TASKS=280
Model: hidden=100 MLP, SGD lr=0.10, momentum=0.9

## Setup
- GPU: NVIDIA A10 (23GB)
- CUDA: 13.0
- Working dir: /workspace

## Step 1: Sanity smoke test
Running: `SH_K=8 SH_ARM=fixed SH_SMOKE=1 python src/run_intervention_sweep.py`
Purpose: verify the script exits cleanly before committing to ~20 min full run.

## Step 2: Full run
Running: `SH_K=8 SH_ARM=fixed python src/run_intervention_sweep.py 2>&1 | tee results/iv_K8_fixed/run.log`
This runs 10 seeds × 280 tasks with 8 fixed-interval resets.
Expected duration: ~20 minutes.

## Decisions
- No code changes — EXPERIMENT.md explicitly says "Do NOT edit code"
- Building on existing repo; iv_K8_fixed dir was empty
- Running single-arm (fixed) pod as instructed by EXPERIMENT.md
