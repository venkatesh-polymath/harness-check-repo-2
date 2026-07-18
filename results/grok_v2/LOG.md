# LOG — grok_v2 (probe)

## Date
2026-07-18

## Round description
Powered re-run of grokking grid with peer-review fixes:
- **5 seeds** (was 2): gives Wilson CI with meaningful width
- **WD ∈ {0.0, 0.01, 0.1, 1.0}** (was {0, 1.0}): directly tests where BN's implicit-reg equivalent lies on explicit-WD axis
- **Fixed BS=256** (was {64,256,512}): BS-robustness already shown in v1
- **Full training curves** for seed=42 × all 8 (norm×WD) cells: disambiguates "never groks" from "needs more steps"
- **Aggregate stats**: grok_rate, Wilson 95% CI, mean±std onset, per-seed final_val_accs
- **init_val_ce** sanity at step 0 (expect ≈ ln(97) ≈ 4.575)

## Prior results (increase_complexity-01)
- LN + WD=0  → 0/6 grokked (all batch sizes) ✓ (no reg = no grok)
- LN + WD=1.0 → 6/6 grokked, mean onset ~6083 steps ✓ (strong WD = grok)
- BN + WD=0  → 0/6 grokked (all batch sizes) — SURPRISING: BN implicit reg not enough
- BN + WD=1.0 → 4/5 grokked (1 cell stalled at 0.818, anomaly)
- Conclusion from v1: BN's implicit regularization alone insufficient; explicit WD needed

## Scientific questions this round answers
- Q1: BN+noWD with 5 seeds — still 0/5? (v1 was 0/6)
- Q2: What WD level makes LN grok? 0.01? 0.1? 1.0?
- Q3: Do grokked cells compress weight norm? (Use saved curves)

## Implementation decisions

### Architecture (unchanged from prior rounds)
- 1-layer transformer: d=128, n_heads=4, MLP_hidden=512
- TransposedBN: BatchNorm1d applied to (B*T, D) reshape for transformer positions
- Decoupled WD: norm/bias/embedding params excluded from weight decay (same as v1)
- LR=1e-3, AdamW β=(0.9, 0.98)

### Curve cells (seed=42)
- 8 cells: {LN,BN} × {0.0,0.01,0.1,1.0}
- NO early stopping: run full 15k steps
- Save per-eval history: step, train_acc, val_acc, train_loss, val_loss, weight_norm
- Purpose: distinguish "never groks" from "needs more time"

### Non-curve cells (seeds 123, 7, 99, 2024)
- Apply early stop when val_acc > 0.99 (after grokking confirmed)
- Faster turnaround for aggregate stats

### Cell ordering
1. Seed=42 curve cells first (fail-fast validation)
2. Then remaining seeds, WD desc (fast grokkers first)

### Output
- RESULTS.json: final format with status="DONE"
- curves/<norm>_wd<wd>.json: per-eval history for 8 representative cells

## Results Summary

### Q1: BN+noWD grok rate with 5 seeds?
**Answer: 0/5 (0.0%, Wilson CI [0.0, 0.435]).** Confirms prior round's 0/6. BN's implicit
regularization is insufficient for grokking in this 1-layer transformer setup at BS=256.

### Q2: At what WD does LN start grokking?
**Answer: WD=1.0 (5/5 grok, CI [0.566, 1.0]).**  
- LN+WD=0.0 → 0/5  
- LN+WD=0.01 → 0/5  
- LN+WD=0.1 → 0/5  
- LN+WD=1.0 → 5/5, mean onset 5580 ± 598 steps  

The grokking threshold lies strictly between WD=0.1 and WD=1.0.  
BN+WD=1.0 also groks 5/5 (mean onset 5980 ± 574 steps, slightly slower than LN+WD=1.0).

### Q3: Weight-norm trajectory from saved curves (seed=42)
| Condition | Init norm | Final norm | Ratio | Groks |
|-----------|-----------|------------|-------|-------|
| LN+WD=0.0 | ~118 | 221 | 1.88× | No |
| LN+WD=0.01 | ~118 | 196 | 1.67× | No |
| LN+WD=0.1 | ~118 | 124 | 1.06× | No |
| LN+WD=1.0 | ~118 | 113 | 0.96× | YES (step 5100) |
| BN+WD=0.0 | ~117 | 135 | 1.15× | No |
| BN+WD=0.01 | ~117 | 134 | 1.14× | No |
| BN+WD=0.1 | ~117 | 125 | 1.06× | No |
| BN+WD=1.0 | ~117 | 118 | 1.00× | YES (step 6900) |

WD=0.1 moderates weight growth (1.06×) but doesn't cross the grokking threshold.
Only WD=1.0 compresses/stabilizes weights enough to trigger generalization.
BN at WD=0.0 shows some natural weight compression relative to LN (1.15× vs 1.88×)
but this is far from the ~1.0× needed for grokking.

### Sanity check
- init_val_ce = 4.7173 (expected ln(97)=4.5747; close to uniform, slight overestimate OK)

### Key conclusion
BN's implicit regularization is equivalent to explicit WD < 0.1 for LN (probably ~0 effective WD),
since both BN+noWD and LN+WD≤0.1 fail to grok. The hypothesis that BN implicit reg ~ LN+matched-WD
is **falsified**: there is no grok-enabling WD level for LN that matches BN+noWD (which never groks).
This is an equally informative negative result: BN's implicit L2 is insufficient for grokking,
period. The experiment shows grokking requires STRONG explicit WD (~1.0) regardless of norm type.

## Status
- [x] Script written (src/grokking_grid_v2.py)
- [x] Grid run completed (40/40 cells)
- [x] 8 training curves saved to results/grok_v2/curves/
- [x] RESULTS.json written with status="DONE"
