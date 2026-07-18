# LOG — grok_v3

## Round: grok_v3 (probe)

### Context from prior rounds
- **baseline-00**: LN+WD=1.0 groks at step ~5500. Confirmed setup works.
- **increase_complexity-01**: BN+noWD = 0/6 at BS∈{64,256,512}, 2 seeds. BN+WD=1.0 groks.
- **grok_v2**: BS=256, 5 seeds, WD∈{0, 0.01, 0.1, 1.0}.
  - BN+noWD = **0/5** (null confirmed)
  - LN+WD=1.0 = 5/5 groks (mean onset 5580 steps)
  - LN+WD∈{0, 0.01, 0.1} = 0/5 each → grokking threshold somewhere in (0.1, 1.0)
  - BN+WD=1.0 = 5/5 groks (mean onset 5980 steps)
  - BN wnorm_ratio ≈ 1.15 (stable); LN+noWD grows to 1.6–2.3×

### This round (grok_v3)

**Goal**: Resolve "is BN implicit L2 absent or sub-threshold?" confound raised by reviewers.

**Key question**: BN is applied to BOTH sublayers in prior rounds. Scale-invariance argument (Zhang et al. 2018) for implicit L2 holds cleanly for FF weights but is muddied for attention (QK^T/√d softmax temperature coupling). So BN_all+noWD null (0/5) could be an attention-BN artifact, not genuinely insufficient implicit L2.

**Arms**:
1. `BN_all_noWD` — replicate null (0/5 expected from grok_v2)
2. `BN_mlp_only_noWD` ⭐KEY — BN on MLP only (norm2), LN on attention (norm1)
3. `BN_attn_only_noWD` — BN on attention only (norm1), LN on MLP (norm2)
4. `LN_wd0.25`, `LN_wd0.5`, `LN_wd0.75` — finer WD sweep in (0.1, 1.0) gap

**Architecture details**:
- `norm1` = pre-attention LayerNorm/BatchNorm
- `norm2` = pre-MLP LayerNorm/BatchNorm (also used for output norm)
- Mixed: `BN_mlp_only` → norm1=LN, norm2=BN
- All other hyperparams same as grok_v2: d=128, h=4, mlp=512, BS=256, LR=1e-3, AdamW

**Effective WD estimate method**: Match BN_all+noWD mean final_wnorm_ratio to LN+WD arm
with closest wnorm_ratio. This is the "matched norm suppression" proxy.

### Decision log

1. **Script**: New `src/grokking_ablation_v3.py`. Builds on grokking_grid.py architecture;
   adds `norm1_type`/`norm2_type` parameters to TransformerBlock for mixed-norm support.

2. **Output norm**: Using `norm2_type` for the output norm (before the classification head)
   to be consistent with "MLP-side" normalization. This ensures the BN_mlp_only arm gets
   BN on all FF-adjacent norms.

3. **BN_all replication**: Re-running rather than copying grok_v2 data, to have matching
   trajectory format and ensure consistency within this experiment.

4. **LN reference data from grok_v2**: Not re-running LN+WD=0.0, 0.1, 1.0 since these
   are complete 5-seed results. Using those wnorm_ratios for effective WD estimation.

5. **15k-step cap**: Same as v2. At ~5s/cell on GPU, 30 cells ≈ 37 minutes total.

### Results (populated after run)
— see RESULTS.json
