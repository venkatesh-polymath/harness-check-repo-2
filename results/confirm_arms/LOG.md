# confirm_arms — Running Log

## Round purpose
Confirmatory experiment: tighten the D-vs-C equivalence claim with 8 seeds + PAIRED analysis.
Prior method_arms probe (3 seeds) showed D-vs-C = -1.77 pp with CI [-7.8, +4.3] — too wide to conclude.

## Setup decisions (fixed by EXPERIMENT.md)
- Arms B (CBP-heuristic), C (explicit-Fisher), D (Adam-v_t)
- 8 seeds {0..7}, 6 tasks × 5 classes/task, 1200 steps/task
- Architecture: SmallConvNetGN (GroupNorm ConvNet, same as method_arms)
- Adam lr=1e-3, reset_every=100 steps, reset_frac=10%
- Real CIFAR-100 from /opt/datasets (abort if missing)
- Masked eval (predict only among current task's 5 classes)
- PAIRED: each seed runs B/C/D on SAME task-split → tight paired CI

## Code
- Source: src/confirm_arms.py (new script, builds on src/method_arms.py)
- Uses scipy.stats.ttest_rel for paired t-test
- Final acc metric: mean of last 4 tasks (tasks 3–6), excluding task 1 warmup

## Timeline
- 2026-07-18: Starting confirmatory run (8 seeds × 3 arms × 6 tasks × 1200 steps = 172,800 total steps)
- Estimated wall-time: ~80–90 min (based on method_arms probe: 34 min / 72,000 steps)

## Key statistical design
- Paired diff per seed: d_s = acc_D(s) - acc_C(s), in percentage points
- Report: mean(d_s), paired 95% CI, paired t-test p-value
- Equivalence verdict: CI fully within ±3 pp of zero = "practically equivalent"
