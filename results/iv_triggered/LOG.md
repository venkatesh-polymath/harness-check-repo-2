# LOG — intervention arm=triggered

## Experiment Design
Validated regime: hidden=100 MLP, online Permuted-MNIST, SGD lr=0.1 momentum=0.9.
K=8 matched-budget reset events. TRIG_FRAC=0.92, TRIG_GAP=8 tasks (refractory period).
Started 2026-07-19 23:03:13

## What I Did and Why

### Step 1: Read EXPERIMENT.md
- EXPERIMENT.md specifies exactly two commands to run and forbids code changes.
- This is the `triggered` arm of a 4-arm intervention study.

### Step 2: Checked existing results
- `results/iv_triggered/` existed but was empty — no prior work to build on for this arm.
- Other arms (none, fixed, random, etc.) have results in `results/iv_*/`.
- Confirmed GPU available: NVIDIA H100 80GB HBM3.

### Step 3: Smoke test (SH_SMOKE=1)
Command: `SH_ARM=triggered SH_SMOKE=1 python src/run_intervention.py`
- Ran 1 seed × 12 tasks in ~7s on H100.
- Printed "DONE", exited 0.
- 1 reset event fired at task 5 (187 units reset).
- Confirmed code path works end-to-end.

### Step 4: Full run
Command: `SH_ARM=triggered python src/run_intervention.py 2>&1 | tee results/iv_triggered/run.log`
- Ran 5 seeds × 280 tasks on H100.
- Completed in ~580 seconds (~9.7 minutes).
- No errors or anomalies.

## Key Observations

**Reset timing pattern**: In every seed, ALL 8 reset events fired in tasks 5–68 
(spacing: 5, 14, 23, 32, 41, 50, 59, 68 — consistently 9 tasks apart = TRIG_GAP+1).
This means the erank trigger fires immediately after every refractory window throughout
the first ~70 tasks. The budget is exhausted early; tasks 69–280 receive no resets.

**Implication**: The triggered arm is effectively a front-loaded fixed schedule 
(not adaptive to actual erank valleys). The trigger fires so eagerly that TRIG_FRAC=0.92
threshold is met right after every refractory gap. This is structurally different from 
what a comparison to an evenly-spaced `fixed` arm would show.

**Accuracy trajectory**: Performance degrades throughout the run. Mean ss_acc = 0.821
(last 50 tasks). Final dead fraction = 0.888 — very high, indicating the resets
(all in first 68 tasks) do not permanently solve plasticity loss.

## Results arm=triggered
mean_ss_acc=0.8209 (±0.0202 std over seeds)
mean_auc_acc=0.8778
mean_final_dead=0.8880
mean_events=8.0 (budget fully used, all in tasks 5-68)
mean_units_reset=1501.6 per seed
Wall time: 579.7s on NVIDIA H100 80GB
