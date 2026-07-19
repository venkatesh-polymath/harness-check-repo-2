# LOG — gns200 (full) round

## 2026-07-19

### Context
This is the consolidation run for the precedence experiment. Prior rounds:
- **baseline_obs**: Permuted CIFAR-10, 5 seeds, probe scale. Established collapse regime.
- **precedence**: Split-CIFAR-100 FAILED (0/8 collapse with task-incremental fresh heads, accuracy 47-73%). Switched to Permuted CIFAR-10, 8 seeds. Found: erank leads GNS.
- **gnsfix**: Permuted CIFAR-10, B=30 GNS samples (was 2). Fixed isotonic-smoothed onset. GNS still lags. CI span reduced 2.26×.
- **gns200**: THIS ROUND. Split-CIFAR-100 (10 tasks × 5 classes) as specified, B>=200 GNS samples.

### Key Issue from Prior Rounds
The precedence round found Split-CIFAR-100 task-incremental shows 0/8 collapse (accuracy 47-73%). This is because fresh 5-class heads need only ~10-40 active neurons to classify 5 classes, and even a heavily dead trunk (90%+ dead) has enough surviving neurons.

### Decision for gns200
Follow the EXPERIMENT.md specification exactly:
- Split-CIFAR-100 (10 tasks × 5 classes, task-incremental, fresh heads per task)
- SGD+momentum, BatchNorm OFF
- 8 seeds
- B=200 GNS samples
- Measure init dead fraction (must be < 15%)
- 2000 steps/task (per study spec, matching Dohare et al.)

With 2000 steps/task and 2500 training samples per task, the trunk trains for ~51 epochs per task. This heavy within-task training might cause sufficient plasticity loss over 10 tasks.

Define t_collapse: first task t where new-task accuracy < task1_accuracy - 15pp AND stays for 2+ consecutive tasks.

If collapse_reproduced=false: still report observable trajectories and onset times. Report pairwise lead times between observables with accuracy onset as reference where available.

### LR choice
- Start with lr=0.01 (standard SGD for CIFAR)
- Verify dead_at_init < 15%
- If not healthy, reduce lr
- Note: init dead fraction depends on initialization, not LR. But with wrong lr, first few steps might kill many units.

### Architecture
- 3-layer MLP: Linear(3072→400, bias) → ReLU → Linear(400→400, bias) → ReLU → Linear(400→5, bias) [per task]
- Shared trunk (first 2 FC layers), fresh head per task
- BatchNorm OFF (as specified)

### Onset detector
- Window-2 MA smoothing
- Threshold: init + 0.5 * (final - init) for each observable
- Persistence: 2 consecutive tasks past threshold
- NOT isotonic smoothing (which was the gnsfix fix)

### GNS estimator
- B_simple = trace(Σ) / |G|² (McCandlish 2018)
- B=200 per-sample gradients
- Bootstrap SE from the 200 samples (50 bootstrap resamples)

### Running
`python src/run_gns200.py 2>&1 | tee results/gns200/run.log`
Expected: ~15-30 min on H100

## Pivot 1: 0/8 collapse on Permuted CIFAR-10 (root cause found)

First run completed (wall=804s) showing:
- Leg A (Split-CIFAR-100 task-incremental): 0/8 collapse — EXPECTED
- Leg B (Permuted CIFAR-10 shared head): 0/8 collapse — UNEXPECTED, accuracy ~22-31%

**Root cause identified by comparing run_gns200.py vs run_gnsfix.py (which showed 8/8 collapse):**

In run_gnsfix.py/run_rigorous.py: ONE optimizer per seed, created before the task loop.
```python
optimizer = optim.SGD(model.parameters(), lr=0.05, momentum=0.9, ...)
for t in range(N_TASKS):
    # optimizer.step() here — momentum ACCUMULATES across tasks
```

In run_gns200.py (buggy): new optimizer created EACH TASK.
```python
for tid in range(n_tasks):
    opt = optim.SGD(...)  # momentum RESETS every task → no conflicting updates
```

**Why persistent momentum causes collapse:**
- After task t, momentum vector points toward task t's loss minimum
- Task t+1 has different (randomly permuted) inputs → stale momentum is "wrong" direction
- Conflicting momentum × 20 tasks → rapid trunk degradation, dead units, accuracy drops to 10% chance
- With fresh momentum each task: each task starts clean → 27-31% accuracy maintained throughout

**Fix applied:** For `shared_head=True` (Permuted CIFAR-10), create optimizer ONCE before task loop. For `shared_head=False` (CIFAR-100 task-incremental), keep per-task optimizer (head changes each task anyway).

**Second run started:** `python src/run_gns200.py 2>&1 | tee results/gns200/run.log`

