# LOG — iv_sweep_K8

## Experiment
Intervention budget sweep at K=8 reset events, 5 arms × 10 seeds × 280 tasks on Permuted-MNIST.
Arms: none, triggered, smart, fixed, random.

## Setup
- GPU: NVIDIA A10 (23 GB)
- Code: `src/run_intervention_sweep.py` (pre-committed, not edited)
- Prior rounds already in `results/` — this is a fresh K=8 run

## Step 1 — Smoke test
`SH_K=8 SH_SMOKE=1 python src/run_intervention_sweep.py`

## Step 2 — Full run
`SH_K=8 python src/run_intervention_sweep.py 2>&1 | tee results/iv_sweep_K8/run.log`

## Decisions
- No code changes per EXPERIMENT.md ("Do NOT edit code")
- All weights stay in `_weights/` (git-ignored)
- Waiting for full blocking run to complete before writing RESULTS.json (the script writes it itself)
