# LOG — EOS/sharpness predictor
Started 2026-07-20 05:00:42

## Session notes (2026-07-20)

### What we ran
- Script: `src/run_eos.py` (committed, unmodified per EXPERIMENT.md)
- Experiment: EOS/sharpness predictor on the validated healthy-collapse regime
  - MLP hidden=100, online Permuted-MNIST, SGD lr=0.10
  - 6 seeds × 250 tasks (full statistically meaningful run)
  - Observables: dead-unit fraction, effective rank, sharpness (top Hessian eigenvalue), EoS product (η·S)

### Step 1: Smoke test
- Command: `SH_SMOKE=1 python src/run_eos.py`
- Result: DONE exit 0 in ~10 seconds (1 seed × 6 tasks)
- Confirmed: sharpness computation working, all metrics computed

### Step 2: Full run
- Command: `python src/run_eos.py 2>&1 | tee results/eos/run.log`
- Configuration: 6 seeds × 250 tasks, HESS_ITERS=12, HESS_BATCH=512, PROBE_SIZE=1000
- GPU: CUDA available and used for all tensor computations
- MNIST: downloaded to /tmp/mnist on first access

### Decisions
- No code modifications needed per EXPERIMENT.md ("Code is ALREADY committed at src/run_eos.py. Do NOT write or edit code.")
- All pip deps (scipy, scikit-learn, torchvision) already installed
- Weights kept in _weights/ (git-ignored), not committed

### Key questions answered
- Q1: Does sharpness (top Hessian eigenvalue) predict plasticity collapse? → AUC metric
- Q2: Does sharpness LEAD dead-unit fraction and effective rank? → lead time analysis
- Analysis: same within-trajectory 5-task-ahead predictive AUC + lead-time as main precedence study

## Results (completed ~1000s)

### Regime validated
- All 6 seeds healthy (task1_acc >0.80, dead_t1 <0.25)
- mean_task1_acc = 0.9325, mean_dead_after_task1 = 0.071
- n_collapsed = 6/6 seeds experienced collapse
- mean_acc_drop = 20.4 pp

### Q1: Predictive AUC (5-task-ahead)
| Signal            | AUC   |
|-------------------|-------|
| dead_unit_fraction| 0.910 |  ← best predictor
| effective_rank    | 0.904 |
| sharpness         | 0.191 |  ← poor predictor
| eos_product       | 0.191 |  ← poor predictor

**Interpretation**: Sharpness (and EoS product) are poor predictors of collapse in this regime (AUC ~0.19, near chance). Dead-unit fraction remains the best predictor.

### Q2: Lead times (median tasks ahead of collapse)
| Signal            | Median lead | Mean lead |
|-------------------|-------------|-----------|
| dead_unit_fraction| 209.5       | 203.0     |
| sharpness         | 199.5       | 186.2     |
| eos_product       | 199.5       | 186.2     |
| effective_rank    | 198.0       | 192.8     |

Precedence order: dead_unit_fraction > sharpness ≈ eos_product > effective_rank

### Q2 Paired test (sharpness vs dead-unit)
- Median diff (sharpness lead - dead lead) = -8.5 tasks (sharpness LAGS dead-unit)
- Wilcoxon signed-rank p = 0.031 (significant: dead-unit leads earlier than sharpness)

### Sharpness behavior
- mean sharpness init = 3.44, final = 6.02 (increases ~75% over training, opposite of EoS departure)
- EoS product (η·S) init=0.344, final=0.602 (stays well below 2 throughout — not in EoS regime)

### Conclusion
Sharpness does NOT predict collapse in this regime (AUC≈0.19). Dead-unit fraction remains the dominant early-warning signal, leading sharpness by ~8.5 tasks (p=0.031).

