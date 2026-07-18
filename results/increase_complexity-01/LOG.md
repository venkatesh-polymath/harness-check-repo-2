# LOG — increase_complexity-01 (probe)

## Date
2026-07-18

## Round description
Grid search: norm {LayerNorm, BatchNorm} × weight_decay {0, 1.0} × batch_size {64, 256, 512},
2 seeds (42, 123) = 24 cells. Max 15k steps per cell or early-stop on grokking.
Tests: (a) BN+no-WD groks only at small batch? (b) BN+no-WD ~ LN+WD? (c) LN+no-WD should NOT grok.

## Building on baseline-00
- baseline-00 established: LN+WD=1.0 groks at step 5500 (BS=512, seed=42)
- WD=1e-3 failed to grok (weight norm 4.8× explosion in 100k steps)
- Therefore using WD ∈ {0, 1.0} (not 1e-3 from study spec draft)
- Architecture verified correct; skipping sanity gates (already passed in baseline-00)

## Implementation decisions

### BatchNorm in transformer
- Applied in same positions as LayerNorm (pre-norm style: norm1 before attn, norm2 before MLP, ln_out after block)
- BN wrapper reshapes (B, T, D) → (B*T, D) → BatchNorm1d → (B, T, D)
- WD mask excludes BN parameters (consistent with LN exclusion; BN params are scale params)
- In eval mode BN uses running statistics

### Grid ordering
- Run cells sorted by expected speed: WD=1.0 first (fastest grokking), then WD=0 cells
- Incremental RESULTS.json write after each cell

### Early stopping
- Stop cell when val_acc > 0.95 AND sustained 3 consecutive evals (every 100 steps)
- Hard cap: 15,000 steps per cell
- Total budget target: < 40 min

## Grid cells (planned)
| norm | WD | BS | seed | expected |
|------|----|----|------|----------|
| LN | 1.0 | 64 | 42 | grok (fast) |
| LN | 1.0 | 64 | 123 | grok (fast) |
| LN | 1.0 | 256 | 42 | grok |
| LN | 1.0 | 256 | 123 | grok |
| LN | 1.0 | 512 | 42 | grok (baseline ~5500) |
| LN | 1.0 | 512 | 123 | grok |
| BN | 1.0 | 64 | 42 | grok (fastest with both regs) |
| BN | 1.0 | 64 | 123 | grok |
| BN | 1.0 | 256 | 42 | grok |
| BN | 1.0 | 256 | 123 | grok |
| BN | 1.0 | 512 | 42 | grok |
| BN | 1.0 | 512 | 123 | grok |
| BN | 0 | 64 | 42 | grok (implicit reg from small batch noise) |
| BN | 0 | 64 | 123 | grok |
| BN | 0 | 256 | 42 | grok (maybe, medium batch) |
| BN | 0 | 256 | 123 | grok (maybe) |
| BN | 0 | 512 | 42 | NO grok predicted (large batch = weak BN implicit reg) |
| BN | 0 | 512 | 123 | NO grok predicted |
| LN | 0 | 64 | 42 | NO grok (no regularization) |
| LN | 0 | 64 | 123 | NO grok |
| LN | 0 | 256 | 42 | NO grok |
| LN | 0 | 256 | 123 | NO grok |
| LN | 0 | 512 | 42 | NO grok (control) |
| LN | 0 | 512 | 123 | NO grok |

## Status
- [ ] Script written
- [ ] Grid run completed
- [ ] RESULTS.json written with verdicts
