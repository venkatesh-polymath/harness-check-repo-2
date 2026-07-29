# LOG — round refine-02 (probe)

## Experiment
Online Mid-Training Invariance Auditor via Unlabeled Feature-Tail Consistency Signal

## What this round does
**Critical fix**: main-01 trained both baseline and method for 30 epochs across 3 seeds
but SKIPPED the CIFAR-10-C mCE evaluation — the primary testable claim.
This round adds CIFAR-10-C mCE evaluation, keeping everything else identical to main-01.

## Decisions made

### What is reused from main-01 (unchanged)
- ResNet-8 architecture (same 8-layer design)
- FixMatch training loop (same hyperparameters: lr=0.03, wd=5e-4, cosine LR)
- RandAugPool with 10 operations and `KNOWN_HARMFUL = {contrast, color, sharpness}`
- OnlineAuditor (runs every 10 epochs, threshold τ=0.82, re-admittance after 10 epochs)
- Probe scale: 30 epochs, 50 labeled/class, 4000 unlabeled, 500 audit pool
- Same 3 seeds: [42, 123, 456]
- Same data split (seed=42, fixed across all model seeds)

### What is new in refine-02
**CIFAR-10-C evaluation** implemented from scratch (no external `imagecorruptions` package):

| Corruption | Motivation |
|---|---|
| `gaussian_noise` | General robustness to sensor noise |
| `contrast` | Directly maps to auditor's `contrast` rollback op |
| `brightness` | Directly maps to auditor's `brightness` rollback op |
| `gaussian_blur` | Tests inverse of `sharpness` augmentation |
| `pixelate` | General spatial distortion robustness |

Severities 1–5 for each type = 25 (corruption, severity) pairs per model.
**mCE** = raw mean error rate over all 25 pairs (not AlexNet-normalized; we report raw
for the probe since we have no AlexNet reference; labeled "mCE" following convention).

### CIFAR-10-C implementation
Each corruption function takes a PIL image and severity (1–5):
- `_corrupt_gaussian_noise`: adds Gaussian noise with std ∈ [0.04, 0.10]
- `_corrupt_brightness`: reduces brightness by factor ∈ [0.88, 0.40]
- `_corrupt_contrast`: reduces contrast by factor ∈ [0.80, 0.20]
- `_corrupt_gaussian_blur`: GaussianBlur radius ∈ [1.0, 3.0]
- `_corrupt_pixelate`: downscale to [28,24,20,16,12] then upscale to 32×32

Corruption applied to clean test images (raw PIL from CIFAR-10 test set).
Batched evaluation: 512 images per forward pass.

### Why keep probe scale (30 epochs)?
- main-01 already shows the pipeline works end-to-end at 30 epochs
- Hypothesis 13 (mCE gap ≥ 2 pp) is the claim to test — needs the METRIC first,
  not necessarily the full 500 epochs
- Full-scale run would follow after confirming the metric moves

### sanity gates
Re-run all 4 gates (take ~2-3 min); main-01 confirmed all passing but we re-run
for this round's integrity.

## Status
RUNNING (started after script written)

## Expected outcomes
Based on main-01 (+1.33 pp clean accuracy):
- mCE: TBD — at 30 epochs both models are still learning, corrupted accuracy will be poor
- Auditor should roll back contrast/sharpness ops (seen in main-01)
- Rollbacks on "contrast" aug → expect lower error on "contrast" corruption

## Timing estimate
- Sanity gates: ~2 min
- 3 seeds × 2 methods × 30 epochs: ~8 min
- CIFAR-10-C eval (25 (c,s) pairs × 6 models × 10k images): ~5 min
- Total: ~15 min
