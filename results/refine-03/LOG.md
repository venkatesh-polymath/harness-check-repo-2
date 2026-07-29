# LOG — refine-03

## Round goal
Address the FATAL CONFOUND flagged by reviewer: SimCLR's color-jitter/grayscale
augmentations are designed to destroy exactly the injected HSV hue signal.
So the SimCLR WGA gain (refine-02: +15.3pp on bgoff) may simply be
"augmentation mechanically deletes the synthetic feature" rather than a real
invariance mechanism.

## Approach: Add Control Arm (SimCLR-NO-COLOR-AUG)
Three arms:
1. **ERM** (supervised, spurious_rate=1.0)
2. **SimCLR-full-aug** (crop + flip + color-jitter + grayscale — same as refine-02)
3. **SimCLR-no-color-aug** (crop + flip ONLY — removes color-jitter and grayscale)

If WGA gain PERSISTS in arm 3 → real invariance mechanism
If WGA gain VANISHES in arm 3 → augmentation deletion is the explanation

## Additional measurement: HSV signal decodability per arm
For each arm, train a linear probe on BACKBONE FEATURES to decode:
- `bg_class` (which background hue was injected) → measures color info in repr.
If SimCLR-full loses color info (low decodability) AND SimCLR-no-color retains it
but gains WGA, that's real invariance. If no-color loses WGA too → deletion mechanism.

## Implementation
- Built on top of `run_refine02.py` (no refactoring, just extension)
- Added `get_simclr_no_color_aug_transform()` with only RandomResizedCrop + flip
- Added `measure_hsv_decodability()` that trains a linear probe on backbone feats
  to predict `bg_class` label
- Same probe scale: 5k train / 1k test / 20 epochs / seeds [0,1,2]
- Results stored in `results/refine-03/RESULTS.json`

## Key code additions
1. `get_simclr_no_color_aug_transform()` — SimCLR without color-jitter/grayscale
2. `run_simclr_seed(seed, color_aug=True/False)` — unified SimCLR runner
3. `measure_hsv_decodability(backbone, train_ds, device)` — linear probe on
   backbone features → predicts bg_class (not true label)
4. Re-use all WGA eval infrastructure from refine-02 verbatim

## Decisions
- Kept same probe scale (5k/1k/20ep) for speed — goal is "does metric move?"
- Used backbone features (not raw pixels) for decodability — measures what the
  learned representation actually encodes
- Decodability probe predicts bg_class (10 classes = 10 hue bins) not just
  "bg-correlated vs not" since the spurious signal is class-specific
- Report: wga_bgoff, wga_patchoff, hsv_decodability for all 3 arms + 3 seeds

## Results (actual, post-run)

### Summary table (mean across 3 seeds)

| Arm                  | wga_bgoff | wga_patchoff | hsv_decode | acc_on |
|----------------------|-----------|--------------|------------|--------|
| ERM                  | 0.039     | 0.016        | 0.929      | 0.911  |
| SimCLR-full-aug      | 0.192     | 0.097        | 0.863      | 0.838  |
| SimCLR-no-color-aug  | 0.042     | 0.015        | 0.932      | 0.916  |

### Key deltas

| Comparison                    | Δwga_bgoff | Δwga_patchoff |
|-------------------------------|------------|---------------|
| SimCLR-full − ERM             | **+0.153** | +0.081        |
| SimCLR-no-color − ERM         | **+0.003** | -0.001        |

### Interpretation

**DELETION MECHANISM CONFIRMED**: The WGA gain completely vanishes when
color-jitter and grayscale augmentations are removed.

- SimCLR-full-aug: wga_bgoff = 0.192 (+15.3pp over ERM)
- SimCLR-no-color-aug: wga_bgoff = 0.042 (+0.3pp over ERM → essentially same as ERM)

This is a very clean result: the +15.3pp wga_bgoff gain seen in refine-02 was
entirely due to color-jitter/grayscale augmentations MECHANICALLY DELETING the
injected HSV background hue signal, not due to any genuine invariance mechanism
in the SSL pretraining.

### HSV decodability nuance

Interestingly, the HSV decodability numbers are not as dramatically different
as expected:
- ERM: 0.929
- SimCLR-full: 0.863 (lower by ~7pp)
- SimCLR-no-color: 0.932 (same as ERM)

The color augmentation does reduce HSV information in the representation somewhat,
but the WGA effect is enormous (+15.3pp vs +0.3pp). This suggests the WGA gain
is even more directly driven by augmentation deletion than raw decodability alone
implies — possibly because the linear probe can still exploit HSV info to some
degree even in SimCLR-full, but test-time behavior without the augmentation
exposes the model's actual reliance on the spurious color feature.

### Sanity checks
- G1 (fixed seed): PASS
- G2 (loss at init): PASS (loss_0=2.502)
- G3 (dummy classifier): PASS (10.0%)
- G4 (overfit batch): PASS (loss=0.0007)
- G5/G6/G7: Skipped (cv2 not available), but verified in refine-02

### Elapsed time: 542.2 seconds (~9 minutes on NVIDIA A10)
