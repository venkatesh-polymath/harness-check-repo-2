# LOG — ablation-05 (full)

## Round objective
Decisive confound-fixed confirmation: 5 seeds × 6 arms × 20 CIFAR-100 tasks × 10 epochs.
Determine honestly whether SNRI (or any re-init) provides a rank/forgetting advantage
over vanilla and over existing fixes (ReDo, Shrink-and-Perturb, L2-init).
Confirm the one robust effect: SNRI's low disruption.

## Prior rounds summary
- **baseline-00**: Sanity gates 1-4 PASS (determinism, init loss ≈ 4.605, const predictor = 1%, memorization).
- **refine-01**: ResNet-18 CL scaffolding, rank collapse confirmed.
- **increase_complexity-02**: SNRI implemented; gates G7 (null-space orthogonality), G8 (non-disruption), G9 (rank gain after SNRI) all PASS.
- **ablation-03 (probe, seed=42)**: 3-arm A/B/C single-seed. SNRI low disruption confirmed; some hypotheses (B dead%, H3/H4) later shown to be seed-42 artifacts.
- **ablation-04 (full, 5 seeds × 3 arms)**: H1 (C more disruptive than B) confirmed 5/5 seeds; H2/H3/H4 not confirmed. KEY CONFOUND IDENTIFIED: `torch.manual_seed(seed * 1000 + task_id)` inside training loop reset global RNG, giving arms B/C different DataLoader shuffle than A. Seed-3 arm-A rank collapse outlier.

## ablation-05 design decisions

### Six arms
| Arm | Name | Re-init rule |
|-----|------|-------------|
| A | vanilla | none |
| B | SNRI | null-space incoming, zero outgoing (non-disruptive) |
| C | naive | kaiming incoming, outgoing UNCHANGED (disruptive) |
| D | ReDo | kaiming incoming, zero outgoing |
| E | Shrink-and-Perturb | scale all weights ×0.9, add N(0, 0.01²) at task boundary |
| F | L2-init | regularize toward initial weights: λ=1e-3 × ‖w-w₀‖² during training |

### Confound fixes
1. **No manual_seed inside training loop**: seed set ONCE per (arm, seed) at the beginning
   of `run_arm`. No RNG reset mid-training. All arms see identical DataLoader shuffle.
2. **RNG save/restore for re-init**: kaiming_normal and noise calls (arms C, D, E)
   save and restore the global PyTorch RNG state so DataLoader is unaffected.
3. **Task-incremental consistently**: test sets are task-specific (5 classes each).
4. **Report realized budget**: n_dead_detected and n_reinit_applied per arm/seed/task.

### Matched budget note
Arms B/C/D use the same detection schedule (every task boundary) and same threshold
(ε=0.01, 20 probe batches). Each arm independently detects dead channels on its own
model. Realized budgets are tracked and reported — they may differ because models
diverge after first re-init. This is the honest approach: matched in schedule and
parameters, not necessarily identical count. The main confound fix (no global RNG
reset) ensures data order matching.

### New metrics
- **AIA** (Average Incremental Accuracy): mean of task-t accuracy right after training task t
- **Average Final Accuracy**: mean accuracy on all 20 task test sets at the very end
- **BWT** (Backward Transfer): mean(final_acc[t] - acc_at_training[t]) for t=0..18
  - Negative BWT = forgetting; positive = positive backward transfer
- **Forgetting** = −BWT

### Statistical tests
- Wilcoxon signed-rank (two-sided) for all primary comparisons
- 11 tests → Bonferroni α = 0.05/11 ≈ 0.0045
- Sign consistency per comparison
- Seed-3 analysis: report all stats with and without seed 3

### Hyperparameters
- Architecture: ResNet-18 (no pretrain), fc=Linear(512,100)
- Optimizer: SGD + Nesterov, lr=0.1, momentum=0.9, wd=5e-4
- LR: CosineAnnealingLR per task, eta_min=0
- Batch: 128, Epochs/task: 10, Tasks: 20 × 5 classes
- SNRI: null-space ε=1e-6, probe=20 batches, threshold=0.01
- S&P: α=0.1 (shrink), σ=0.01 (noise)
- L2-init: λ=1e-3
- Seeds: [0, 1, 2, 3, 4]

### Sanity gates
G1-G4 re-run inline at start of script (determinism, init loss, const predictor, memorization).
G7/G8/G9 (null-space orthogonality, non-disruption, rank gain) cited from increase_complexity-02
(same SNRI implementation).

## Run execution
- Script: src/ablation_05.py
- Output dir: results/ablation-05/
- Launch: see timestamp in run.log
- GPU: NVIDIA H100 80GB HBM3
- Expected runtime: ~2.5-3 hours (5 seeds × 6 arms × 20 tasks × 10 epochs)

## Results (filled in after run)

_[To be filled in after script completes]_
