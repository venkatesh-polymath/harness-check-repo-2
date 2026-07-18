# LOG.md — mech_rand round

## Round description
MECHANISTIC CONTROL: Does v_t's RANKING carry information, or does any reset help equally?

Two arms on identical per-seed task splits (PAIRED):
- **D (OURS)**: reset lowest-utility units by Adam exp_avg_sq (v_t) per-neuron score
- **RANDOM (control)**: reset a RANDOM subset of units of the same size, ignoring utility

If D significantly beats RANDOM → v_t's ranking carries real signal (not just "any reset recycles capacity").

## Setup
- Architecture: SmallConvNetGN (GroupNorm ConvNet, same as all prior rounds — ~3.3M params)
- Data: Real CIFAR-100 from /opt/datasets (download=False, ABORT if missing)
- Class-incremental: 4 tasks × 5 classes = 20 classes
- Steps per task: 1000
- Reset every 100 steps, reset_frac=0.10 (same for both arms)
- Adam lr=1e-3 (same for both arms)
- Seeds: {0..7} — 8 seeds, PAIRED (same task split per seed)
- Primary metric: mean accuracy over tasks 2-4 (exclude task-1 warmup)
- Headline: paired t-test D-RANDOM, one-sided H1: D > RANDOM, p < 0.05

## Decisions
1. Reused SmallConvNetGN from method_arms.py/mech_arms.py (validated architecture).
2. Only 2 arms (D and RANDOM) to match the mech_rand spec exactly — not 4 arms like mech_arms.
3. 8 seeds per spec (vs 6 in mech_arms.py) for better statistical power.
4. RANDOM arm uses separate np.RandomState(1000+seed) so it's reproducible but
   independent of the model's RNG state — ensures fairness.
5. Both arms run with identical model init (same torch seed before model creation)
   and identical task splits (same seed to make_task_splits) — true paired design.
6. Incremental RESULTS.json written after each seed; finalized when all 8 seeds done
   (threshold >=6 seeds met after seed 6).

## Run
- Script: src/mech_rand.py
- Command: `python src/mech_rand.py > results/mech_rand/run.log 2>&1`
- GPU: NVIDIA A10 (23 GB)
- Wall time: 24.14 minutes
- Exit code: 0 (SUCCESS)

## Results

### Per-arm mean ± std (primary metric: mean acc tasks 2-4)
| Arm    | Mean acc | ±Std  | Per-seed accs |
|--------|----------|-------|---------------|
| D      | 0.8066   | 0.022 | [0.817, 0.782, 0.831, 0.827, 0.831, 0.789, 0.783, 0.791] |
| RANDOM | 0.7632   | 0.049 | [0.721, 0.754, 0.843, 0.713, 0.789, 0.755, 0.713, 0.817] |

### Paired D-RANDOM
- Per-seed diffs (pp): [+9.6, +2.8, -1.1, +11.3, +4.2, +3.5, +7.0, -2.6]
- Mean diff: **+4.33pp** (D beats RANDOM)
- 95% CI: [0.275, 8.391] pp (does NOT include 0)
- t-statistic: 2.525
- p-value (one-sided, H1: D > RANDOM): **0.0198** < 0.05
- **v_t_ranking_beats_random: TRUE**

### Conclusion
YES: v_t's ranking carries real information beyond 'any reset helps equally.'
D (Adam-v_t utility CBP) significantly beats RANDOM (same reset cadence+fraction, random selection)
by +4.33pp on mean tasks 2-4 accuracy. 6 of 8 per-seed diffs are positive.
The effect is statistically significant at p=0.02 (one-sided paired t-test).
This confirms that the utility ranking from Adam's exp_avg_sq (v_t) is the operative factor —
not merely the periodic recycling of neurons.
