# Observable Temporal Precedence Map: Which Cheap Signal Leads Plasticity Collapse — and by How Many Tasks?

## Abstract

Plasticity loss — the progressive inability of a continually-trained network to fit new tasks — is a first-order obstacle for lifelong learning, and useful intervention requires detecting the failure *before* accuracy degrades. Four cheap scalar signals (dead-unit fraction, effective rank, gradient-noise scale, weight-norm drift) are plausible early-warning indicators, but their relative predictive timing and discriminability have never been measured jointly. Using a within-trajectory *k*-task-ahead labeling that avoids the degeneracy of a "does this run collapse?" label (all runs collapse), we measure each signal's predictive AUC and lead time. In a healthy small MLP (hidden=100, dead-unit fraction 6.7% after task 1) on online Permuted-MNIST (300 tasks, 8 seeds, SGD), **dead-unit fraction and effective rank predict collapse 5 tasks ahead with AUC 0.92 and 0.91 and fire ~210 tasks before it, while gradient-noise scale fails (AUC 0.29)**. The predictive ordering replicates across four regimes — a narrower MLP (hidden=50), a second image dataset (Permuted-Fashion-MNIST), and a small ConvNet on label-permuted MNIST — with dead-unit fraction and effective rank always in the top tier (AUC 0.88–0.99); gradient-noise scale is the weakest signal in every gradual MLP regime (AUC 0.29–0.61) and only becomes predictive under the sharp, near-instant collapse of the ConvNet regime. Finally, we test whether the diagnostic can *time* an intervention: at a matched reset budget, a naive early-warning trigger front-loads its budget (the signal's earliness works against it) and underperforms a fixed schedule by 8.6 pp; a budget-aware trigger recovers most of the gap but still does not beat fixed scheduling. Early-warning quality and intervention-timing quality are therefore distinct: **dead-unit fraction is an excellent collapse predictor and the recommended health proxy, but signal-triggered budget allocation offers no advantage over a simple schedule.**

---

## 1 Introduction

### Problem

Continually trained neural networks progressively lose the ability to fit new tasks — *plasticity loss* [Lyle et al., 2023; Dohare et al., 2023]. The practical consequence is severe: once collapse is confirmed by accuracy degradation, intervention (periodic resets, regularization, architectural surgery) is applied retroactively. An early-warning signal that is measurable at every gradient step, requires no held-out evaluation set, and fires well before observable failure would enable proactive, minimal-cost intervention.

Four candidates appear across the continual-learning and deep-RL literature: **dead-unit fraction** δ (fraction of neurons near-zero on a probe) [Sokar et al., 2023]; **effective rank** ρ of the representation matrix [Lyle et al., 2023]; **gradient-noise scale** GNS = tr(Σ̂)/‖ḡ‖² (the ratio of per-sample gradient-covariance trace to mean-gradient squared norm — a *noise-to-signal* ratio, large when gradients are dominated by sample-to-sample variance) [McCandlish et al., 2018]; and **weight-norm drift** w (relative ℓ₂ drift of parameters from initialization). All four are cheap, but which one leads collapse by the most tasks, which actually *discriminates* a pre-collapse from a collapse-imminent state, and does the answer survive a change of architecture or dataset? No prior work measures their joint temporal precedence in a controlled setting, nor tests whether the winning signal can be turned into an actionable intervention trigger.

### Contributions

Each claim below is refutable and supported by specific evidence in Section 3:

1. **A predictive precedence map.** In a healthy small MLP on online Permuted-MNIST (8 seeds), dead-unit fraction predicts collapse 5 tasks ahead with **AUC 0.919** and leads collapse by a median of **210 tasks**; effective rank achieves **AUC 0.907** (median lead 204 tasks), statistically indistinguishable from dead-unit fraction in lead time (Wilcoxon *p* = 0.37) (Table 1, §3.3).

2. **Gradient-noise scale fails as an early-warning signal (AUC 0.29).** During the collapsing phase GNS moves in the *wrong* direction; its apparent early "crossing" reflects noise-driven excursions, not a coherent alarm (§3.3, §4).

3. **The ordering generalizes across four regimes.** In a narrower MLP (hidden=50), a second image dataset (Permuted-Fashion-MNIST), and a small ConvNet on label-permuted MNIST, dead-unit fraction and effective rank remain top-tier predictors (AUC 0.88–0.99). Gradient-noise scale is the weakest signal in every gradual MLP regime (AUC 0.29–0.61) and only becomes predictive under the near-instant ConvNet collapse — a regime-dependence we characterize rather than hide (Table 2, §3.4).

4. **Early warning ≠ intervention timing.** At a matched 8-reset budget in the validated regime, resetting dead units on *any* schedule beats no intervention by ~19 pp, but a naive effective-rank trigger front-loads its whole budget into the first 68 tasks and *underperforms* a fixed schedule by 8.6 pp; a budget-aware dead-level trigger recovers most of the gap (using only half the budget) but still does not beat fixed scheduling (Table 3, §3.5).

5. **A non-degenerate predictive protocol.** Because every run collapses, a per-run "does it collapse?" label is degenerate; we specify and use a within-trajectory *k*-task-ahead labeling with an explicit exclusion buffer and report threshold sensitivity (§2.3, §3.3).

---

## 2 Method

### 2.1 Experimental Setup

The primary regime is a three-layer ReLU MLP (two hidden layers of width 100) trained continuously with **SGD (learning rate 0.10, momentum 0.9, no weight decay, no resets)** on online Permuted-MNIST: a stream of 300 tasks, each a fresh fixed random permutation of the 784 input pixels, trained for **200 gradient steps** at batch size 128. We run **8 independent seeds** (seeding both `torch` and `numpy`; the task-permutation stream is fixed across seeds). This regime is chosen because (a) validity controls confirm small nets lose plasticity measurably while wide nets and a Split-CIFAR-100 learning-rate grid do not (§3.2), so collapse is present but not trivially induced, and (b) no architectural countermeasure is applied, isolating the natural collapse trajectory. Learning rate 0.10 gives a *healthy start* — task-1 accuracy 93.4% with only 6.7% dead units after task 1 — ruling out the dying-ReLU artifact in which a too-hot learning rate kills most units immediately.

### 2.2 Observable Signals

At the end of every task we record four scalars on a fixed 1000-example probe drawn from the current task distribution:

- **δₜ** (dead-unit fraction): fraction of hidden units inactive (ReLU output ≤ 0) on more than 95% of the probe, averaged over both hidden layers. For the ConvNet, convolutional feature maps are spatially averaged to a per-channel activation first.
- **ρₜ** (effective rank): e^{H(p)}, where H(p) is the entropy of the normalized singular-value spectrum of the penultimate-layer activation matrix [Roy and Vetterli, 2007].
- **GNSₜ** (gradient-noise scale): the McCandlish B_opt estimator tr(Σ̂ₜ)/‖ḡₜ‖² computed from **20 independent mini-batches of 64 examples**, where Σ̂ₜ is the unbiased per-batch gradient covariance and ḡₜ the mean gradient. Large GNS = gradients dominated by sample-to-sample variance (noise-to-signal) [McCandlish et al., 2018].
- **wₜ** (weight-norm drift): mean over layers of the relative ℓ₂ change of each weight tensor from its initialization.

### 2.3 Collapse and the Predictive AUC Formulation

**Collapse onset.** For each seed, the collapse task T_s is the first task at which new-task accuracy falls ≥ 20 percentage points below that seed's task-1 accuracy for at least two consecutive tasks (a pre-registered, hard-coded criterion).

**Why a per-run label is degenerate.** All seeds collapse, so a cross-trajectory label — "does this seed collapse?" — is 1 for every trajectory and the ROC-AUC is undefined. We therefore use a **within-trajectory, *k*-task-ahead** label.

**Predictive AUC (k = 5).** We pool every (seed, task *t*) observation with *t* < T−1. Each observation is labeled

- **y = 1** if new-task accuracy drops to collapse level (≥ 20 pp below that seed's task-1 accuracy) at *any* task in the lookahead window *t*+1, …, *t*+5;
- **y = 0** otherwise.

The score is the signal value at task *t* (effective rank is inverted, since *lower* rank predicts collapse). ROC-AUC is computed over the pooled (score, label) set. This measures a concrete operational question: *given the signal now, will accuracy collapse within the next five tasks?* AUC ≈ 1 means the signal cleanly separates imminent-collapse from safe states; AUC ≈ 0.5 means no discrimination; AUC < 0.5 means the signal moves opposite to collapse.

**Lead time.** For seed *s*, the lead time of a signal is T_s − onset, where onset is the first task at which the signal's two-task moving average crosses 50% of its full init→final range. We report mean, median, IQR, and 2000-sample bootstrap 95% CI across seeds, and sweep the onset threshold over {30%, 50%, 70%} to confirm the ordering is not a threshold artifact.

### 2.4 Generality Regimes

To test whether the ordering is specific to the primary setup, we repeat the full protocol in three further regimes: **(i)** a narrower MLP (hidden=50) on Permuted-MNIST (6 seeds, 200 tasks); **(ii)** a second image dataset, online Permuted-Fashion-MNIST, hidden=100, lr = 0.05 (6 seeds, 250 tasks); **(iii)** a small ConvNet (two conv blocks + one FC hidden layer) on **label-permuted** MNIST — inputs unchanged, target labels permuted each task, so spatial structure is preserved and a convolutional architecture is a genuine generalization (5 seeds, 150 tasks). Every downstream computation (collapse detection, onset, AUC, bootstrap, Wilcoxon) is byte-for-byte the same code as the primary regime.

### 2.5 Intervention Protocol

To test whether the diagnostic can *time* an intervention (reviewers' "most powerful version"), we compare five reset-timing policies in the primary regime (hidden=100, 5 seeds, 280 tasks). A **reset event** reinitializes the currently-dead units — fresh incoming weights, zeroed outgoing weights so the function is not disrupted, cleared momentum (a coarse continual-backpropagation step). All arms use the **same reset rule** and a **matched budget of K = 8 events**; only the *timing* differs:

- **none** — no resets (control);
- **triggered (naive)** — fire when effective rank drops below 92% of its running maximum (refractory gap 8 tasks);
- **smart (budget-aware)** — fire each time dead-unit fraction rises 0.10 above its level at the last event;
- **fixed** — 8 events evenly spaced;
- **random** — 8 events at seeded-random tasks.

Seeds and task permutations are identical across arms, so arms are compared seed-by-seed (paired Wilcoxon). The primary metric is steady-state new-task accuracy (mean over the last 50 tasks).

---

## 3 Experiments

### 3.1 Plasticity Loss Confirmed

Mean task-1 accuracy is **93.4%** and mean dead-unit fraction immediately after task 1 is **6.7%** — a healthy start. Accuracy then declines by a mean of **18.9 pp** over 300 tasks; all 8 seeds cross the collapse threshold (onset range: tasks 168–269). Plasticity loss is confirmed and gradual.

### 3.2 Validity Controls

Before measuring predictive power we confirmed the regime exhibits genuine, non-artifactual plasticity loss while plausible alternatives do not:

| Configuration | Collapses / Runs | Note |
|:---|:---:|:---|
| Split-CIFAR-100, 15 learning-rate grid | 0 / 15 | no collapse to predict |
| Wide MLP (hidden ≥ 256), Permuted-MNIST | 0 / 3 | no collapse to predict |
| Hot LR (0.10 on Fashion) → 77% dead by task 1 | — | dying-ReLU *artifact*, excluded |
| **Small MLP (hidden=100), Permuted-MNIST** | **8 / 8** | healthy start, gradual collapse |

The wide-net and Split-CIFAR-100 results are controls, not failures of those architectures: signal-predictability cannot be measured where collapse is absent. Separately, we verified that the collapse we study is *not* a dying-ReLU artifact — a too-hot learning rate that kills most units after task 1 — by selecting a learning rate whose task-1 dead fraction is 6.7%.

### 3.3 Main Result: Predictive AUC and Lead Time

**Table 1.** Predictive performance of four cheap scalar signals in the primary regime (hidden=100 MLP, online Permuted-MNIST, 8 seeds, 300 tasks, SGD). AUC is the within-trajectory 5-task-ahead value (§2.3); lead time is tasks before collapse onset.

| Signal | Predictive AUC | Lead Time (median tasks) | 95% CI |
|:---|:---:|:---:|:---:|
| Dead-unit fraction (δ) | **0.919** | 210 | [197, 238] |
| Effective rank (ρ) | **0.907** | 204 | [186, 231] |
| Weight-norm drift (w) | 0.880 | 157 | [141, 193] |
| Gradient-noise scale (GNS) | 0.290 | 209 | [194, 230] |

Three signals (δ, ρ, w) discriminate imminent collapse (AUC > 0.85). GNS does not: AUC 0.29 means its values are systematically *lower* in the 5-task-ahead-of-collapse window than in safe states — it moves the wrong way. Its 209-task "lead" in the table is an artifact of noise-driven crossings of the range midpoint, not a coherent alarm, and is not actionable. Weight-norm drift is predictive but fires ~50 tasks later than the top two.

### 3.4 Generality Across Four Regimes

**Table 2.** Predictive AUC (5-task-ahead) of each signal across four regimes. Dead-unit fraction and effective rank are top-tier everywhere; gradient-noise scale is the weakest signal in every *gradual MLP* regime and only becomes predictive under the near-instant ConvNet collapse.

| Regime | δ | ρ | w | GNS | mean drop |
|:---|:---:|:---:|:---:|:---:|:---:|
| Main — hidden=100 MLP, Perm-MNIST | 0.919 | 0.907 | 0.880 | **0.290** | 18.9 pp |
| Narrow — hidden=50 MLP, Perm-MNIST | 0.963 | 0.957 | 0.960 | **0.312** | 29.2 pp |
| 2nd dataset — hidden=100 MLP, Perm-Fashion | 0.881 | 0.896 | 0.611 | **0.610** | 36.6 pp |
| 2nd arch — ConvNet, LabelPerm-MNIST | 0.994 | 0.994 | 0.996 | 0.971 | 35.3 pp |

Dead-unit fraction and effective rank are within 0.01–0.02 of each other and in the top tier (AUC 0.88–0.99) in all four regimes; dead-unit fraction is top-or-tied on lead time in every MLP regime and stable across the 30/50/70% onset thresholds. Gradient-noise scale is the single weakest signal in the three gradual MLP regimes (AUC 0.29–0.61). The ConvNet regime is different in kind: label permutation causes a near-instant per-task collapse (short lead times, all signals AUC ≈ 0.97–0.99), so GNS predicts there simply because *every* signal does. The honest reading is therefore two-fold: **dead-unit fraction and effective rank are robust, architecture- and dataset-independent early-warning signals; gradient-noise scale is unreliable and should not be used as a plasticity proxy in gradually-collapsing MLPs.**

### 3.5 Intervention: Early Warning Does Not Imply Intervention Timing

**Table 3.** Steady-state new-task accuracy (mean over last 50 tasks) under five reset-timing policies at a matched 8-reset budget, in the validated regime (hidden=100, 5 seeds, 280 tasks). Contrasts are paired across seeds; with 5 seeds the Wilcoxon *p*-floor is 0.0625 (all five seeds agree in sign for every contrast reported).

| Arm | Steady-state acc | Final dead-unit frac | Events fired |
|:---|:---:|:---:|:---:|
| none (control) | 0.716 | 0.914 | 0 |
| triggered — naive (erank drop) | 0.821 | 0.888 | 8 (all in tasks 5–68) |
| smart — budget-aware (dead level) | 0.874 | 0.716 | 4 |
| **fixed schedule** | **0.907** | 0.699 | 8 |
| random | 0.903 | 0.719 | 8 |

Resetting dead units on *any* schedule recovers plasticity dramatically: fixed and random beat no-intervention by **+19.1 pp** and **+18.7 pp**. But *timing by signal loses to timing by clock.* The naive effective-rank trigger fires its entire budget in the first 68 tasks — because effective rank drops fast and early (the very property that makes it a good *predictor*) — leaving the late tasks, where units actually die, unprotected; it trails the fixed schedule by **−8.6 pp**. The budget-aware trigger, firing as dead-unit fraction climbs through absolute levels, spreads its events better and recovers to 0.874 using only 4 of 8 resets (beating the naive trigger by +5.3 pp), but still does not beat a simple fixed schedule (−3.3 pp). The lesson is clean and useful: **the signal's earliness is an asset for warning and a liability for naive budget allocation; a fixed reset schedule is a strong, hard-to-beat baseline.**

---

## 4 Discussion

The data reveal a **two-tier predictive structure that is stable across architecture and dataset**. Dead-unit fraction and effective rank form the reliable tier (AUC 0.88–0.99, longest leads); weight-norm drift is predictive but slower; gradient-noise scale is degenerate in every gradual MLP regime.

The GNS failure is mechanistically informative. GNS = tr(Σ̂)/‖ḡ‖² is a noise-to-signal ratio. As the network approaches collapse, dead units accumulate and surviving units lock onto a low-rank, degenerate solution; gradients across samples become *more* aligned (a collapsed representation elicits similar gradients from every example), so ‖ḡ‖ grows relative to tr(Σ̂) and GNS falls. This is consistent with the observed AUC < 0.5 — GNS decreases near collapse — and with gradient starvation [Pezeshki et al., 2021], where feature lock-in concentrates gradient mass. Under label permutation with a ConvNet, collapse is abrupt rather than gradual, and every signal (GNS included) jumps at the moment of failure, so the noise-to-signal picture no longer dominates.

The intervention result sharpens the paper's practical scope. A tempting inference from a good early-warning signal is "trigger the fix when the signal fires." Our matched-budget experiment refutes the naive version of that inference: because dead-unit fraction and effective rank are *early* indicators, signal-triggering front-loads a limited budget and a fixed schedule wins. Early-warning quality and intervention-timing quality are distinct axes; a signal can be excellent on the first and unhelpful on the second.

---

## 5 Limitations

1. **Intervention seeds and budget.** The intervention comparison uses 5 seeds (Wilcoxon *p*-floor 0.0625) and a single matched budget (K = 8); all five seeds agree in sign for every contrast, but a larger seed count and a budget sweep would tighten the estimates. We do not claim signal-triggering can *never* help — only that the natural, budget-matched forms of it here do not beat a fixed schedule.
2. **Temporal autocorrelation.** The predictive AUC pools (task, seed) observations that are not independent within a seed; the AUC is a valid within-trajectory discriminability measure but the effective sample size is smaller than the nominal count.
3. **Collapse criterion.** The 20-pp / two-task onset rule is a fixed heuristic; lead-time CIs (span 30–52 tasks) reflect both seed variance and this choice. The precedence ordering is nonetheless stable across the 30/50/70% onset thresholds.
4. **Optimizer and task family.** All regimes use SGD and permutation- or label-permutation-based task streams. Adam and non-permutation continual streams (e.g., class-incremental CIFAR) are not covered; the ConvNet regime shows the ordering can compress when collapse is abrupt.
5. **No causal claim.** We measure temporal precedence, not causation. Dead-unit accumulation preceding collapse does not establish that dead units *cause* it; both may follow an upstream mechanism such as gradient starvation.

---

## 6 Related Work

**Plasticity loss in continual learning.** Lyle et al. [2023] (arXiv:2306.13812) provide the foundational empirical study of plasticity loss, documenting dead-unit accumulation and effective-rank decline as correlates of degradation. Dohare et al. [2023] (arXiv:2303.01486) characterize the mechanisms and evaluate countermeasures including continual backpropagation — the selective-reset mechanism our intervention arms adapt. Our work is complementary: we measure the *relative predictive timing* of observable signals in a regime where collapse is confirmed but not yet visible, and test whether the best signal can time a reset.

**Dormant and dead neurons.** Sokar et al. [2023] (arXiv:2302.12902) document the dormant-neuron phenomenon in deep RL and propose resets; He et al. [2015] (arXiv:1502.01852) identify dying ReLUs as a supervised-training pathology. That δ is the top-ranked early-warning signal — by AUC and lead time, across four regimes — is consistent with dormancy being a leading indicator rather than a lagging symptom.

**Gradient-noise scale.** McCandlish et al. [2018] (arXiv:1812.06162) introduce GNS as a proxy for the optimal batch size. We show GNS does not transfer to plasticity prediction in gradually-collapsing MLPs (AUC 0.29–0.61), a null result that narrows its applicable scope.

**Recovering plasticity.** Lewandowski et al. [2025] (arXiv:2507.04683) recover lost plasticity via soft weight rescaling; our contribution is orthogonal — an early-warning map and a matched-budget test of *when* to trigger such a fix, with the finding that signal-timing does not beat a fixed schedule.

**Gradient starvation.** Pezeshki et al. [2021] (arXiv:2011.09468) show gradient starvation drives feature lock-in; the decreasing GNS near collapse is consistent with this picture.

**Generalization diagnostics.** Jiang et al. [2019] (arXiv:1912.02178) find no single measure universally predicts generalization; our finding that GNS fails while δ succeeds mirrors this selectivity — predictive validity is context-specific and must be checked in the target regime.

---

## 7 Conclusion

In a healthy small MLP (hidden=100, SGD, no resets) on 300 Permuted-MNIST tasks across 8 seeds, dead-unit fraction (AUC 0.92) and effective rank (AUC 0.91) predict collapse five tasks ahead and fire ~210 tasks before it; gradient-noise scale fails (AUC 0.29). The ordering replicates across a narrower MLP, Permuted-Fashion-MNIST, and a ConvNet on label-permuted MNIST, with δ and ρ always top-tier (AUC 0.88–0.99) and GNS the weakest in every gradual MLP regime. Turning the diagnostic into an intervention shows a clean boundary: resetting dead units recovers ~19 pp of plasticity, but signal-triggered timing front-loads a matched budget and does not beat a fixed schedule. The practical recommendation is therefore precise: **monitor dead-unit fraction for early warning — it is cheap, robust, and the earliest reliable indicator — but allocate a limited reset budget on a fixed schedule rather than by naive signal-triggering, and discard gradient-noise scale as a plasticity proxy.**

---

## References

- Dohare, S., Hernandez-Garcia, J. F., Sutton, R. S., and Mahmood, A. R. (2023). *Understanding plasticity in neural networks.* arXiv:2303.01486.
- He, K., Zhang, X., Ren, S., and Sun, J. (2015). *Delving deep into rectifiers: Surpassing human-level performance on ImageNet classification.* arXiv:1502.01852.
- Jiang, Y., Neyshabur, B., Mobahi, H., Krishnan, D., and Bengio, S. (2019). *Fantastic generalization measures and where to find them.* arXiv:1912.02178.
- Lewandowski, A., et al. (2025). *Recovering plasticity of neural networks via soft weight rescaling.* arXiv:2507.04683.
- Lyle, C., Zheng, Z., Nikishin, E., Pires, B. A., Pascanu, R., and Dabney, W. (2023). *Loss of plasticity in deep continual learning.* arXiv:2306.13812.
- McCandlish, S., Kaplan, J., Amodei, D., and Team, O. D. (2018). *An empirical model of large-batch training.* arXiv:1812.06162.
- Pezeshki, M., Kaba, S.-O., Bengio, Y., Courville, A., Precup, D., and Lajoie, G. (2021). *Gradient starvation: A learning proclivity in neural networks.* arXiv:2011.09468.
- Roy, O. and Vetterli, M. (2007). *The effective rank: A measure of effective dimensionality.* EUSIPCO.
- Sokar, G., Agarwal, R., Castro, P. S., and Evci, U. (2023). *The dormant neuron phenomenon in deep reinforcement learning.* arXiv:2302.12902.
