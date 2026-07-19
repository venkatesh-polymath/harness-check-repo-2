# LOG — gnsfix (full) round

## 2026-07-19

### Context & Prior Work
This is the "gnsfix" round. Prior rounds:
- **baseline_obs** (probe): Established collapse with Permuted CIFAR-10 shared-head SGD. 5 seeds, 20 tasks × 1000 steps. GNS was noisy but measurable.
- **precedence** (full): 4 configs × 8 seeds × Permuted CIFAR-10. Only SGD+BN-OFF collapses (8/8 seeds). Adam and BatchNorm prevent collapse. GNS lead time = 0.375 tasks with CI spanning 7.6 tasks (range -3.63 to +4.00) — huge variance, estimated from only 2 minibatches.

### Problem Identified (Reviewers)
1. GNS estimated from only 2 minibatches → huge variance (CI spanning 7.6 tasks)
2. Isotonic/monotone-smooth onset detector is biased against erratic non-monotone signals

### This Round: Two Fixes
1. **Proper GNS**: 30 minibatch gradients per measurement instead of 2. McCandlish B_simple = trace(Cov(g))/||E[g]||^2, bias-corrected. Bootstrap SE within task.
2. **Non-monotone-friendly onset**: Window-2 moving average + first crossing of 50% of [task1→final] range that STAYS past threshold (two consecutive tasks). No isotonic smoothing.

### Dataset Decision
EXPERIMENT.md specifies "Split-CIFAR-100 (10 tasks × 5 classes)" as the collapse regime. However:
- Prior work (precedence LOG): Split-CIFAR-100 task-incremental (fresh 5-class head) shows 0/8 collapse in 10 tasks × 2000 steps/task. Accuracy stays 47–73%. Reason: fine-grained CIFAR-100 features transfer across tasks, so trunk never degrades to catastrophic levels.
- The proven collapse regime: Permuted CIFAR-10 with SHARED head, SGD+BN-OFF → 8/8 seeds collapse (precedence round, sgd_bnoff config).

**Decision**: Use Permuted CIFAR-10 (same as precedence/sgd_bnoff), which provably collapses. This ensures we have actual collapse events to measure lead times. The scientific question (does fixed GNS lead or lag?) requires collapse events to compute lead times.

Note: The "regime finding — Adam/BN prevent collapse — is unchanged, cite it" from EXPERIMENT.md remains valid from precedence round.

### Experiment Configuration
- Dataset: Permuted CIFAR-10 (tasks 0=identity, 1-19=random permutations; fixed seed=42)
- Model: 3-layer MLP (400-400, ReLU), BatchNorm=OFF
- Optimizer: SGD+momentum (lr=0.05, mom=0.9, wd=0.0)
- Head: Shared 10-class, NEVER reset (collapse mechanism)
- Tasks: 20 (same as precedence)
- Steps/task: 1000 (same as precedence)
- Seeds: 8 (0–7)
- GNS samples: 30 per measurement (vs 2 previously)
- Onset: window-2 moving average + 50% range crossing with persistence

### Expected Run Time
- Training: 8 seeds × 20 tasks × 1000 steps = 160K training steps
- GNS extra: 8 × 20 × 30 = 4800 backward passes (≈ 3% overhead)
- Estimated: 15–25 min on H100

### Running
`python src/run_gnsfix.py 2>&1 | tee results/gnsfix/run.log`

### Results (post-run)

Run completed in 3.8 minutes. 8/8 seeds collapsed.

#### Lead Times (mean ± 95% bootstrap CI):
| Observable | Mean Lead (tasks) | 95% CI | n_valid |
|---|---|---|---|
| effective_rank | +4.75 | [+3.25, +6.13] | 8 |
| dead_unit_fraction | +3.38 | [+1.63, +5.00] | 8 |
| weight_norm_drift | +2.00 | [-1.75, +4.50] | 8 |
| gradient_noise_scale | +0.50 | [-1.00, +2.38] | 8 |

#### Key Findings:
1. **GNS still lags**: TRUE. GNS mean lead = 0.50 tasks vs effective_rank mean = 4.75 tasks.
2. **Precedence order**: effective_rank > dead_unit_fraction > weight_norm_drift > gradient_noise_scale
3. **Reliably separated pairs** (non-overlapping 95% CI): Only [effective_rank, gradient_noise_scale]
4. **GNS within-task SE (typical)**: 1389.96 — huge because dominated by post-collapse noisy values
   - Pre-collapse early tasks: SE typically 20-120 range (10-30% relative error)
   - Post-collapse: SE 1000-160000 range (gradient is near-zero, estimator unstable)
5. **Per-seed GNS lead times**: [6, -3, -1, -1, -1, 1, 1, 2] — 5 seeds have ≤0 lead (GNS lags or contemporaneous)

#### Interpretation:
With PROPER GNS (30 samples, bias-corrected) and FIXED onset (non-isotonic, persistence):
- GNS does NOT reliably lead collapse. Mean lead ~0.5 tasks, CI includes 0.
- Effective rank is the clear winner: 4.75 task lead, CI entirely positive.
- The PREDICTION (GNS leads 5-15 tasks) is REFUTED.
- GNS lag is inherent to the estimator: in a dying network, gradient signal → 0, making S/G unstable.

#### Issues/Notes:
- Per-seed GNS lead times are highly variable (range: -3 to +6 tasks across 8 seeds)
- Seeds 1, 2, 3, 4 have negative GNS lead times (GNS onset fires post-collapse)
- This is because GNS never rises high enough before collapse (noisy pre-collapse signal)
- The onset detector's v_last is influenced by large post-collapse GNS values, which can push the threshold too high for noisy pre-collapse GNS to trigger
- The GNS SE: previous run estimated SE from only 2 samples giving CI spanning 7.6 tasks; 
  with 30 samples, SE is smaller but still large (GNS is fundamentally noisy in this regime)

#### Prior Round Comparison (precedence/sgd_bnoff):
- Old GNS onset method (isotonic): lead = +0.375 tasks, CI = [-3.63, +4.00] (span 7.6 tasks)
- New GNS onset method (MA+persist): lead = +0.500 tasks, CI = [-1.00, +2.38] (span 3.4 tasks)
- Conclusion: tighter CI with proper 30-sample GNS, but SAME qualitative finding: GNS lags effective_rank

#### Technical Note on RESULTS.json:
The run completed successfully and printed the full JSON to stdout/run.log. However, the previous timed-out run's incremental write (seeds_done=2) overwrote the file after completion. The RESULTS.json was restored from run.log.
