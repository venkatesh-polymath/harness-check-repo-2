# LOG — intervention arm=random

Validated regime hidden=100 PMNIST SGD. K=8 matched budget.
Started 2026-07-19

## Step 1 — Sanity Check (SH_SMOKE=1)

Command: `SH_ARM=random SH_SMOKE=1 python src/run_intervention.py`

Result: FAILED with ValueError:
```
ValueError: Cannot take a larger sample than population when replace is False
  File "src/run_intervention.py", line 230, in random_event_tasks
    return sorted(rng.choice(np.arange(5, n_tasks), size=k, replace=False).tolist())
```

Root cause: Smoke test sets N_TASKS=12; `random_event_tasks` samples K=8 events
from np.arange(5, 12) = 7 elements, but 8 > 7 without replacement.

Decision: This is a smoke-test-specific artifact — the real run uses N_TASKS=280,
where np.arange(5, 280) has 275 elements and sampling K=8 without replacement
works fine. The EXPERIMENT.md says "do not attempt fixes," so no code was changed.
Proceeded directly to Step 2 since the failure is irrelevant to the real run.

## Step 2 — Real Run

Command: `SH_ARM=random python src/run_intervention.py 2>&1 | tee results/iv_random/run.log`

Hardware: NVIDIA H100 80GB HBM3
Wall time: ~546 seconds (~9 minutes)

Configuration:
- ARM=random, hidden=100, 5 seeds × 280 tasks
- K=8 events at seeded-random task indices per seed
  (rng seed = 1000 + seed_num, from np.arange(5, N_TASKS))
- Reset rule: reinitialize dead units (inactive >50% of probe), zero outgoing
  weights, clear momentum
- Seed-matched with other arms via torch.manual_seed(seed), DATA_SEED=42

Per-seed summary:
| Seed | ss_acc | task1_acc | final_dead | events |
|------|--------|-----------|------------|--------|
|  0   | 0.9057 | 0.9238    | 0.645      | 8      |
|  1   | 0.9135 | 0.9354    | 0.745      | 8      |
|  2   | 0.9094 | 0.9309    | 0.705      | 8      |
|  3   | 0.8989 | 0.9370    | 0.730      | 8      |
|  4   | 0.8876 | 0.9294    | 0.770      | 8      |

## Results arm=random
mean_ss_acc=0.9030  std_ss_acc=0.0091  mean_final_dead=0.7190  mean_events=8.0

Notes:
- All 5 seeds fired exactly K=8 reset events (matched budget achieved).
- Mean units reset per event: ~187 out of 200 total (both layers combined).
- ss_acc consistently below task1_acc (ss - task1 ≈ -2 to -4 pp): network
  still loses some plasticity despite random resets.
- Compare with "none" arm (no resets) to assess whether random resets help,
  and with "triggered" arm to assess whether timing matters.
- Status: SUCCESS — RESULTS.json written by the script.
