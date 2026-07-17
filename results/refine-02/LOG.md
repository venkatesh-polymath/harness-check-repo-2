# LOG — refine-02

## What this round fixes

Round 1 (`increase_complexity-01`) hit a critical problem: the **80%/|tanh|>0.99**
crossing-epoch metric was **censored for every single run** (no layer ever reached 80%
saturation within 30 epochs). With all outcomes right-censored, Spearman rank
correlation is undefined — you cannot rank layers by something that never happened.

## Decision: use final-epoch saturation fraction at |tanh|>0.9

The EXPERIMENT.md for refine-02 specifies:
> "per layer, the mean saturation FRACTION (fraction of units with |tanh(z)|>0.9)
>  at the FINAL epoch (these values were non-trivial and differ by layer)"

**Why this works:**
- Round 1 showed non-trivial sat_frac values at epoch 30 (up to 0.29 for layer 1 under
  super_4x) even with the STRICTER 0.99 threshold.
- The LOWER threshold |tanh|>0.9 (|z|>1.472 vs |z|>2.647) will give MUCH higher
  fractions → clear, non-censored differences between layers.
- Spearman(init S_l, final_sat_frac_09) across layers can be computed without censoring.

## Key changes from round 1

| Aspect | Round 1 | Round 2 |
|---|---|---|
| Final sat threshold | \|tanh\|>0.99 (z>2.647) | \|tanh\|>0.9 (z>1.472) |
| Primary metric | crossing epoch (censored) | final sat fraction (non-censored) |
| Spearman target | S_l vs crossing-epoch | S_l vs final_sat_frac |
| CI computation | per-config mean over 3 seeds | 95% bootstrap CI over 3 seeds |

## Setup

- Device: NVIDIA A10 (23 GB)
- Architecture: same as rounds 0-1 (TanhMLP, FC, no BN, no skip, no dropout)
- Regimes: standard (uniform, n·σ²=1/3), super_2x (n·σ²=2), super_4x (n·σ²=4)
- Depths: 3, 5, 7. Seeds: 0, 1, 2. Max epochs: 30.
- Init S_l threshold: |z| > 2.0 (unchanged from rounds 0-1)
- Bootstrap: 10,000 resamples for 95% CI

## Code

Script: `src/refine_02.py` — built on `src/increase_complexity_01.py`.
Changes: new `SAT_09_Z = atanh(0.9)` threshold; Spearman now correlates init S_l with
`final_sat09` (not crossing epoch); bootstrap CI computed per (regime,depth);
explicit reversal verdict (YES/NO).

## Run command

```bash
python3 src/refine_02.py 2>&1 | tee results/refine-02/run.log
```

## Results summary

Completed: 27 runs in 227.8 s on NVIDIA A10.

### Non-censored metric ✓
|tanh(z)| > 0.9 gives non-trivial, well-separated saturation fractions for all
layers (0.00 – 0.58), making Spearman well-defined for every (regime, depth) config.

### Spearman(init S_l, final_sat09) — ALL POSITIVE

| config        | Spearman mean | 95% CI         |
|---------------|---------------|----------------|
| standard_d3   | +0.866        | [0.866, 0.866] |
| standard_d5   | +0.707        | [0.707, 0.707] |
| standard_d7   | +0.612        | [0.612, 0.612] |
| super_2x_d3   | +1.000        | [1.000, 1.000] |
| super_2x_d5   | +0.767        | [0.700, 0.900] |
| super_2x_d7   | +0.571        | [0.500, 0.607] |
| super_4x_d3   | +1.000        | [1.000, 1.000] |
| super_4x_d5   | +0.833        | [0.700, 0.900] |
| super_4x_d7   | +0.738        | [0.536, 0.893] |

### Reversal hypothesis: **NO**
- Layer 1 has the highest final sat09 in 100% of (regime, depth, seed) runs.
- Spearman(S_l, sat09) is POSITIVE across ALL regimes.
- The absolute saturation ordering is ALWAYS layer-1-first (no reversal).
- The deep-first signal from round 1 (relative growth reversal) does not appear
  in absolute saturation levels at 30 epochs.

### Key finding
S_l (init) and final_sat09 both decrease monotonically with depth in all regimes.
Spearman is positive everywhere because layer 1 has both the highest S_l AND the
highest final saturation fraction. The experiment's predicted reversal (S_l ordering
flip under super-standard) is mathematically impossible: Var(z_l) always decreases
through tanh layers (since E[tanh²(z)] < Var(z) for any finite variance), so S_l
can never increase with depth.

Under depth=7 with super regimes, the LAST layer shows a U-shape elevation (sat09
for layer 7 is higher than layers 5-6), but not higher than layer 1. This partial
deep-first elevation hints at gradient-flow effects but doesn't flip the overall order.

