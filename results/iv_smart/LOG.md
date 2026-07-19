# LOG — Intervention Experiment: arm=smart

## Context
Round: iv_smart  
Date: 2026-07-19  
Code: `src/run_intervention.py` (pre-committed, no edits made)

## What this arm does
The "smart" arm fires a plasticity-restoring reset event whenever the **dead-unit fraction rises by ≥DEAD_STEP (0.10)** since the last event — a budget-aware diagnostic trigger. This contrasts with:
- `none`: no resets (control)
- `triggered`: erank-threshold-based trigger (front-loads events)
- `fixed`: K=8 evenly-spaced events
- `random`: K=8 seeded-random events

Matched budget: all arms capped at K=8 reset events. Each event reinitializes dead units (fresh kaiming-uniform incoming, zero outgoing, cleared SGD momentum).

## Step 1 — Sanity check (smoke test)
Command: `SH_ARM=smart SH_SMOKE=1 python src/run_intervention.py`  
Result: **PASSED** — printed "DONE", exit 0 in 7s. 1 seed × 12 tasks, fired 1 event at task 5 (reset 187 units).

## Step 2 — Full run
Command: `SH_ARM=smart python src/run_intervention.py 2>&1 | tee results/iv_smart/run.log`  
Wall clock: ~583s (~10 min) on GPU.

## Results Summary

| Seed | task1_acc | ss_acc (last 50) | final_dead | n_events | event_tasks |
|------|-----------|-----------------|------------|----------|-------------|
| 0    | 0.9238    | 0.8523          | 0.675      | 4        | [5,22,60,267] |
| 1    | 0.9354    | 0.8942          | 0.705      | 4        | [5,14,51,233] |
| 2    | 0.9309    | 0.8798          | 0.785      | 4        | [5,14,31,190] |
| 3    | 0.9370    | 0.8835          | 0.795      | 4        | [5,25,50,197] |
| 4    | 0.9294    | 0.8623          | 0.620      | 4        | [5,19,63,255] |

**mean_ss_acc = 0.8744 ± 0.0151**  
**mean_final_dead = 0.716**  
**mean_events = 4.0** (all seeds used exactly 4 of the K=8 budget)

## Observations
1. All 5 seeds fired exactly 4 events — the dead fraction rose slowly enough that only half the budget was consumed.
2. The first event always fires very early (~task 5) because the dead fraction spikes quickly after task 0.
3. The smart arm still shows substantial plasticity loss (ss_acc drops ~5.7pp below task1_acc), but less severe than the control "none" arm would show.
4. Final dead fraction (0.716) indicates continued accumulation between resets — the 4 events provide temporary relief but don't halt the underlying dynamics.
5. No weights committed. _weights/ directory is git-ignored.

## Decisions
- Did not modify any code (EXPERIMENT.md: "Do NOT write or edit code")
- Ran smoke test first as required
- Used blocking `tee` command as specified
- RESULTS.json written to required format with status=SUCCESS, scale=full
