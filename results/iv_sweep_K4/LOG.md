# LOG — iv_sweep_K4

## Round description
Experiment: `iv_sweep_K4` — Powered (10-seed), matched-budget reset-timing sweep at K=4.  
Code: `src/run_intervention_sweep.py` (pre-committed, not edited).  
Hardware: NVIDIA A10 (23 GB VRAM).

## Steps

### Step 0 — Environment check (2026-07-20)
- CUDA available: A10 GPU confirmed.
- Dependencies: numpy, torch, torchvision, scipy all present.
- No prior RESULTS.json in results/iv_sweep_K4/ — fresh run.

### Step 1 — Sanity / smoke test
Run: `SH_K=4 SH_SMOKE=1 python src/run_intervention_sweep.py`  
- SH_SMOKE=1 → N_SEEDS=2, N_TASKS=20 (fast sanity)
- Expect: exit 0, writes to results/iv_sweep_K4_smoke/

### Step 2 — Full run
Run: `SH_K=4 python src/run_intervention_sweep.py 2>&1 | tee results/iv_sweep_K4/run.log`  
- N_SEEDS=10, N_TASKS=280, 5 arms × 10 seeds = 50 training runs
- Writes results/iv_sweep_K4/RESULTS.json with status DONE on success

## Decision log
- Using K=4 as specified in EXPERIMENT.md (env SH_K=4).
- Code is pre-committed and must NOT be edited.
- Weights stay in _weights/ (git-ignored).
