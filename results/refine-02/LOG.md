# LOG — refine-02

## Round goal
Fix degenerate WGA from main-01: groups 0-2 were NaN (only group 3 populated)
because `spurious_rate=1.0` makes ALL samples have both signals correlated →
`group_labels = bg_matches*2 + patch_matches = 3` always.

## Root cause analysis
In `src/dataset.py`, `group_labels = bg_matches * 2 + patch_matches` where
`bg_matches = (bg_class == label)` and `patch_class == label`. With
`spurious_rate=1.0`, both are always True → all examples group 3.

The test datasets were also created with `spurious_rate=1.0`, so the same
degeneracy applies there.

## Fix design
Add independent `bg_spurious_rate` and `patch_spurious_rate` parameters to
`TogglableCIFAR10`. For WGA evaluation test sets use rate=0.5 independently
for each signal → balanced 4-group structure (~25% each group).

- Training: keep `spurious_rate=1.0` (both signals always correlated → maximizes bias)
- WGA evaluation: `bg_spurious_rate=0.5, patch_spurious_rate=0.5` (independent)
  → all 4 groups populated

Group semantics for WGA:
- bgoff test (bg=False, patch=True): groups by patch_matches
  - Group 0 (bg_no, patch_no): no active shortcut → hard
  - Group 1 (bg_no, patch_yes): patch shortcut available → easy
  - Group 2 (bg_yes, patch_no): bg disabled, no patch → hard
  - Group 3 (bg_yes, patch_yes): patch shortcut, bg disabled → easy
  - WGA = min over all 4 non-empty groups = min(groups 0,2) = hard cases

- patchoff test (bg=True, patch=False): groups by bg_matches
  - Group 0 (bg_no, patch_no): bg anti-correlated, patch disabled → hard
  - Group 1 (bg_no, patch_yes): bg anti-correlated, patch disabled → hard
  - Group 2 (bg_yes, patch_no): bg correlated, patch disabled → easy
  - Group 3 (bg_yes, patch_yes): bg correlated, patch disabled → easy
  - WGA = min over all 4 non-empty groups = min(groups 0,1) = hard cases

## Implementation
1. Modified `src/dataset.py`: Added `bg_spurious_rate`/`patch_spurious_rate` params
   with independent RNG draws when provided. Backward-compatible (default=None
   falls back to `spurious_rate`).
2. New `src/run_refine02.py`: Fixed experiment runner using balanced WGA test sets.
   Reports: all 4 per-group accuracies, true 4-group WGA, binary WGA (by active
   signal), suppress-vs-shift deltas.

## Decisions
- Kept same probe scale (5k train / 1k test, 20 epochs) for speed
- Used 1000 test samples per eval split to keep WGA groups ~250 samples each
- Added group_sizes to evaluation output for transparency
- Sanity gates re-run to verify correctness (backward compat preserved)

## Results summary

### Bug fix confirmed
- G7 (new gate): All 4 groups populated with ~25% each (198-326 samples per group)
- Previous main-01 had only group 3 populated (100%)
- Now: groups {0: ~20%, 1: ~25%, 2: ~25%, 3: ~30%}

### TRUE WGA results (vs degenerate main-01)

| Metric | ERM (main-01 degenerate) | ERM (refine-02 TRUE) | SimCLR (main-01 degenerate) | SimCLR (refine-02 TRUE) |
|--------|--------------------------|----------------------|------------------------------|--------------------------|
| wga_bgoff  | 0.071 (group 3 only) | 0.039 ± 0.005 | 0.240 (group 3 only) | 0.192 ± 0.017 |
| wga_patchoff | 0.904 (group 3 only) | 0.016 ± 0.012 | 0.811 (group 3 only) | 0.097 ± 0.001 |

The main-01 "wga_bgoff" of 0.071 was actually just the accuracy of group 3 examples
in the bgoff test (which = acc_off since all test samples were group 3). Similarly
the "wga_patchoff" of 0.904 was acc_on for group 3 in the patchoff test.

### True 4-group per-group accuracies

**ERM bgoff (bg=False, patch=True, balanced groups):**
- Group 0 (patch anti-correlated, bg anti-correlated): 4.4% ← HARD
- Group 1 (patch correlated, bg anti-correlated): 7.1%
- Group 2 (patch anti-correlated, bg correlated): 5.9%
- Group 3 (patch correlated, bg correlated): 8.4%
→ All groups near chance, ERM barely uses patch when bg is removed

**ERM patchoff (bg=True, patch=False, balanced groups):**
- Group 0 (bg anti-correlated, patch anti-correlated): 1.7% ← HARD
- Group 1 (bg anti-correlated, patch correlated): 2.9% ← HARD
- Group 2 (bg correlated, patch anti-correlated): 91.1% ← EASY
- Group 3 (bg correlated, patch correlated): 90.2%
→ Massive gap: ERM uses bg exclusively, can't generalize without it

**SimCLR bgoff:**
- Group 0: 21.6%, Group 1: 25.7%, Group 2: 21.9%, Group 3: 22.1%
→ Roughly uniform across groups! SimCLR doesn't rely on patch much either,
  but it's more robust overall (~20% vs ~5% for ERM)

**SimCLR patchoff:**
- Group 0: 11.2%, Group 1: 9.7% ← hard (no bg)
- Group 2: 82.4%, Group 3: 80.8% ← easy (bg correlated)
→ Still relies on bg color, but less dramatically than ERM

### Key findings
- **Δwga_bgoff (SSL-ERM) = +15.3pp**: SimCLR dramatically better when bg removed
  (much larger than predicted +5-8pp)
- **Δwga_patchoff (SSL-ERM) = +8.1pp**: SimCLR also BETTER on patch WGA!
  (opposite of H10 prediction that ERM would be better by +3-7pp)
- H9, H10, H11 all FALSE at probe scale
- The suppress-vs-shift effect is NOT observed: SimCLR is uniformly more
  robust on BOTH signals' hard groups, not selectively better on one
- SimCLR's improvement is broader robustness, not a spurious shift

### Interpretation
The probe scale (20 epochs, small dataset) may not be sufficient to see the
full suppress-vs-shift effect. At this scale, SimCLR's color-jitter augmentation
provides general robustness improvements that dwarf any signal-specific shift.
The ERM model's almost complete failure on hard groups (1-4% accuracy) when
the dominant spurious signal is removed shows it learns essentially a single
shortcut. SimCLR learns a more distributed representation.

→ Full-scale experiment needed to test whether the predicted shift emerges
  with more training and larger datasets.

