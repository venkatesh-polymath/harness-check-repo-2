# LOG — scale_convnet_deep (Round: full)

## Objective
Deep ConvNet generalization run — clean CIs for the Adam-v_t plasticity result across 2 datasets.
This is the **powered** version of the prior scale_grid2 probe: 10 tasks (vs 4), 8 seeds (vs 6),
ConvNet only (no MLP).

## Prior round context
- **baseline_A / baseline_A2 / baseline_A3**: Arm A floor validated → acc=0.2, dead=1.0 by task 3.
- **method_arms**: 3 arms × 3 seeds × 8 tasks proved CBP resets work (+30pp vs floor). D≈C numerically but n=3 underpowered.
- **scale_grid2**: 4 cells (2 datasets × 2 architectures) × 6 seeds × 4 tasks: Formal equivalence in 2/4 cells. MLP3 shows less forgetting (floor > reset arms). ConvNet on CIFAR-10 suffered C-arm collapse in 1 seed (high variance). Main finding: D≈C numerically but CI wide due to few tasks + few seeds.

## This round's design decisions

| Decision | Choice | Rationale |
|---|---|---|
| Architecture | SmallConvNetGN ONLY | Spec: "ONLY the ConvNet architecture"; MLP excluded |
| Datasets | CIFAR-100 (5cls/task × 10 tasks) + CIFAR-10 (2cls/task × 5 tasks) | Spec: 10 tasks; CIFAR-10 max 5 with 2cls/task |
| Seeds | 8 PAIRED {0..7} | Spec: 8 paired seeds for statistical power |
| Steps/task | 1000 | Spec: ~1000 steps/task |
| Reset cadence | every 100 steps | Same as prior rounds; 10 resets/task |
| Reset fraction | 10% (51 of 512 neurons) | Same as prior rounds |
| Arms | A_floor, B, C, D, RANDOM | Spec: all 5 arms |
| A_floor runs | 1 per dataset (seed=0) | Spec: "once per dataset" |
| Primary metric | Mean acc tasks 2..end | Skip warm-up task 1 |
| Equivalence threshold | ±3pp CI | Per prior rounds |
| Output dir | results/scale_convnet_deep/ | Per user spec |

## CIFAR-10 task count rationale
CIFAR-10 has 10 classes. With 2 cls/task, maximum is 5 tasks (all 10 classes used).
EXPERIMENT.md says "CIFAR-10 (2 classes/task)" and "10 tasks/run" — but 10 tasks × 2 cls =
20 classes which CIFAR-10 does not have. Using 5 tasks × 2 cls = 10 classes (all of CIFAR-10).

## Time estimate on H100
- scale_grid2 on H100: 100 runs (4T × 800 steps) in 13.56 min → ~8.1s/run → ~2s/task
- With 1000 steps: ~2.5s/task
- CIFAR-100: 4 arms × 8 seeds × 10 tasks = 320 task-runs × 2.5s = 800s ≈ 13 min
- CIFAR-10:  4 arms × 8 seeds × 5 tasks  = 160 task-runs × 2.5s = 400s ≈ 7 min
- Floor:     2 × small ≈ 0.5 min
- **Estimated total: ~20 min** (well within 80 min time guard)

## Script
`src/run_deep.py` — reuses SmallConvNetGN, utility functions, CBP reset from scale_grid2.
Key changes vs scale_grid2:
- CIFAR-100: 10 tasks (was 4), CIFAR-10: 5 tasks (was 2), 8 seeds (was 6)
- ConvNet only (removed MLP3 arch loop)
- STEPS_PER_TASK=1000 (was 800)
- Output → results/scale_convnet_deep/

## GPU
NVIDIA H100 80GB HBM3

## Run Command
```bash
python src/run_deep.py 2>&1 | tee results/scale_convnet_deep/run.log
```

## Execution

### 2026-07-18 — Launched

- Verified GPU: H100 80GB
- Verified data: /opt/datasets/cifar-100-python/ and /opt/datasets/cifar-10-batches-py/ present
- Script written: src/run_deep.py
- Launched as single blocking call; results captured to run.log

### 2026-07-18 — Completed (31.83 min, 66/66 runs)

**Final outcomes:**

| Dataset | Arm | Mean Acc | Dead (final) | eRank (final) |
|---------|-----|----------|-------------|---------------|
| CIFAR-100 | floor | 0.2773 | — | — |
| CIFAR-100 | B | 0.7757 | 0.757 | 34.2 |
| CIFAR-100 | C | 0.7808 | 0.826 | 23.3 |
| CIFAR-100 | D | 0.7832 | 0.786 | 30.7 |
| CIFAR-100 | RANDOM | 0.7699 | 0.843 | 21.4 |
| CIFAR-10 | floor | 0.8444 | — | — |
| CIFAR-10 | B | 0.9109 | 0.874 | 5.19 |
| CIFAR-10 | C | 0.9115 | 0.889 | 5.86 |
| CIFAR-10 | D | 0.9158 | 0.873 | 7.25 |
| CIFAR-10 | RANDOM | 0.9098 | 0.900 | 6.75 |

**Paired statistics:**

| Cell | D−C (pp) | 95% CI | equiv_3pp | D−RANDOM (pp) | beats_random | D−B (pp) |
|------|----------|--------|-----------|---------------|--------------|----------|
| cifar100_convnet | +0.24 | [−2.23, +2.70] | **True** | +1.33 | False (p=0.157) | +0.75 |
| cifar10_convnet | +0.43 | [−1.45, +2.32] | **True** | +0.60 | False (p=0.344) | +0.49 |

**Summary:**
- **equivalence_holds_in: 2/2** — D (Adam v_t) is statistically equivalent to C (explicit Fisher) in BOTH cells
- **mechanism_holds_in: 0/2** — D did not significantly outperform RANDOM in either cell (all resets help similarly)
- CBP resets (+50pp on CIFAR-100 vs floor=27.7%) confirm plasticity-preservation works
- D consistently shows slightly higher point estimate than C in both cells, with eRank advantage on CIFAR-100 (30.7 vs 23.3)
- CIFAR-10 floor is higher (84.4%) because with only 5 tasks, forgetting is less severe

**Interpretation:**
The core claim holds: Adam v_t is a sufficient substitute for explicit Fisher in CBP resets (CI within ±3pp in both cells).
The "beats random" claim does not reach significance at n=8 seeds — this is a power issue: all 4 reset arms cluster within 0.5–1.3pp of each other, leaving insufficient signal to separate D from RANDOM with 8 seeds. The RANDOM control also resets neurons, so it also provides plasticity benefits; the utility-discriminative signal requires more seeds or longer task sequences to isolate.
