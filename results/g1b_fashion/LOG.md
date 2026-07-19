# LOG — generality round g1b_fashion

2nd dataset — online Permuted-Fashion-MNIST, MLP hidden=100 (lr=0.05)

Repeats validated precedence+AUC analysis (run_valprec.py) in a new regime.
Started 2026-07-19 23:22:00

## Setup

- Read EXPERIMENT.md: no code to write (src/run_gen.py already committed).
- Step 1: ran smoke check `SH_EXP=g1b_fashion SH_SMOKE=1 python src/run_gen.py` — printed DONE, exit 0 in ~8s.
- Step 2: ran full experiment `SH_EXP=g1b_fashion python src/run_gen.py 2>&1 | tee results/g1b_fashion/run.log` on H100 GPU.
  - 6 seeds × 250 tasks each, lr=0.05 (hedge from g1_fashion lr=0.03).
  - Wall time: ~1466s (~24 min).

## Decisions

- No code changes made; EXPERIMENT.md explicitly says "Code is ALREADY committed at src/run_gen.py. Do NOT write or edit code."
- LR=0.05 (this g1b variant) was designed to ensure collapse occurs faster than g1_fashion (lr=0.03). It worked: all 6 seeds collapsed.

## Observations

- All 6 seeds started healthy (task1_acc ~0.80, dead_t1 ~12%). 
- All 6 collapsed (n_collapsed=6/6), collapse tasks: 95, 88, 90, 61, 105, 31.
- Mean accuracy drop: 36.6pp over 250 tasks.

## Results
healthy=True plasticity=True drop=36.59pp n_collapsed=6/6
precedence=['dead_unit_fraction', 'effective_rank', 'weight_norm_drift', 'gradient_noise_scale']
predictive_auc={'dead_unit_fraction': 0.8911797677234206, 'effective_rank': 0.8964979148917089, 'gradient_noise_scale': 0.6103277879915698, 'weight_norm_drift': 0.6111295457602799}

## Interpretation

- dead_unit_fraction and effective_rank lead collapse by median ~85.5 and ~78.0 tasks, respectively.
- Both have high predictive AUC (~0.891, ~0.896) — strong early-warning signals.
- gradient_noise_scale leads by only ~47 tasks and AUC=0.610, confirming it is NOT a good predictor.
- Precedence order is consistent across all three threshold fractions (30/50/70%).
- Finding generalises to Fashion-MNIST dataset at lr=0.05.
