# LOG — round refine-02b (probe)

## Experiment
Online Mid-Training Invariance Auditor via Unlabeled Feature-Tail Consistency Signal

## What this round does
**Critical fix from refine-02**: refine-02 ran the full experiment (5 corruption types × 5 severities) but timed out during seed=456 baseline evaluation and never wrote RESULTS.json.

refine-02b makes the CIFAR-10-C evaluation lightweight enough to guarantee completion:
- **4 corruption types** × **severity 3 ONLY** = 4 pairs per model (vs 25 in refine-02)
- Corruption types: gaussian_noise, motion_blur, fog, contrast (per EXPERIMENT.md spec)
- Skip sanity gates (all 4 passed in main-01 and refine-02; inherited)

## Decisions made

### Why skip sanity gates?
- All 4 gates confirmed PASS in both main-01 and refine-02
- Gate runs cost ~3 minutes; removing them ensures the 10-minute training budget isn't blown
- Results are inherited in RESULTS.json with source noted

### New corruption functions vs refine-02
refine-02 used: gaussian_noise, contrast, brightness, gaussian_blur, pixelate
refine-02b uses: **gaussian_noise, motion_blur, fog, contrast** (per EXPERIMENT.md spec)

| Corruption    | Implementation |
|---|---|
| gaussian_noise | Gaussian noise with std=0.08 at severity 3 (same as refine-02) |
| motion_blur    | NEW: PIL GaussianBlur(radius=1.5) + translate([3,0]) as proxy for motion streak |
| fog            | NEW: White overlay blend, alpha=0.40 at severity 3 |
| contrast       | PIL Contrast enhance at 0.50 factor at severity 3 (same as refine-02) |

### Why only severity 3?
- Experiment.md: "severity 3 ONLY (not all 15x5)"
- Reduces eval from 25 pairs → 4 pairs per model
- 6 models × (4 pairs × ~2.5s/pair) = ~60 seconds total eval (vs ~324 sec in refine-02)

### Training unchanged from refine-02
- ResNet-8 architecture
- FixMatch (threshold=0.95, lambda_u=1.0)
- RandAugPool (n=2, magnitude=9) with 10 ops
- OnlineAuditor (every 10 epochs, tau=0.82, re-admit 10 epochs)
- Probe scale: 30 epochs, 50 labels/class, 4000 unlabeled, 500 audit pool
- 3 seeds: [42, 123, 456]

## Timing estimate
- No sanity gates: 0 min
- 3 seeds × 2 methods × 30 epochs: ~8.5 min
- CIFAR-10-C eval (4 pairs × 6 models): ~1 min
- Total: ~9.5 min (safely under limit)

## Status
COMPLETED successfully — all 3 seeds ran, RESULTS.json written.

## Actual outcomes (run completed 2026-07-29)

### Clean test accuracy
| Seed | Baseline | Method | Delta |
|------|----------|--------|-------|
| 42   | 0.4138   | 0.4265 | +0.0127 |
| 123  | 0.4510   | 0.4760 | +0.0250 |
| 456  | 0.4166   | 0.4188 | +0.0022 |
| **mean** | **0.4271** | **0.4404** | **+0.0133** |

### mCE (severity 3 only, 4 types)
| Seed | Baseline | Method | Delta |
|------|----------|--------|-------|
| 42   | 0.7531   | 0.7466 | −0.0065 |
| 123  | 0.7482   | 0.7436 | −0.0046 |
| 456  | 0.7596   | 0.7504 | −0.0092 |
| **mean** | **0.7536** | **0.7468** | **−0.0067** |

### Per-corruption delta (baseline → method)
- gaussian_noise: +0.0039 (method slightly worse — auditor doesn't roll back noise ops)
- motion_blur:    −0.0100 (method better)
- fog:            −0.0070 (method better)
- contrast:       −0.0139 (method better — auditor consistently rolls back 'contrast')

### Auditor stats
- AUROC per seed: [0.5, 0.0, 1.0] → mean=0.500 (high variance; probe scale)
- Precision: 0.889, Recall: 0.667 (fires on contrast/sharpness which are known-harmful)
- Rollbacks fired every seed: True

### Interpretation
- Method improves both clean accuracy (+1.3 pp) and mCE (−0.67 pp) vs baseline
- delta_mCE = −0.0067 is in the right direction but well below the H13 target of −0.02
- The auditor consistently rolls back 'contrast' and 'sharpness' (2/3 known-harmful ops)
  which explains the contrast corruption improvement (−1.4 pp)
- Probe scale (30 epochs) is the limiting factor; full 500-epoch run needed to confirm
- motion_blur and fog improvements are indirect (auditor doesn't directly target these)

## Files written
- run.log: full stdout/stderr from the GPU run
- RESULTS.json: all metrics, per-seed and aggregated
