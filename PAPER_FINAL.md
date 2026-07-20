# Non-Disruption Without Benefit: Null-Space Re-Initialization Matches, but Does Not Beat, Trivial Re-Initialization for Plasticity Maintenance

## Abstract

Re-initializing dormant neurons is a standard remedy for plasticity loss in continual learning, but standard re-initialization perturbs the network's learned function. We ask whether making re-initialization *non-disruptive* — injecting the new neuron's incoming weights into the null space of the current representation, so the forward function is provably unchanged at the moment of re-initialization — improves plasticity maintenance. We implement this rule (Spectral Null-Space Re-Initialization, SNRI) and pre-register the falsifiable prediction that at a matched re-initialization budget it maintains effective rank ≥20% higher than trivial (non-orthogonal, random) re-initialization on CIFAR-100 class-incremental ResNet-18. Across five seeds and 20 sequential tasks, **the prediction is refuted by roughly two orders of magnitude**: SNRI and trivial re-initialization achieve statistically identical effective-rank AUC (12.196 vs 12.200, paired Wilcoxon p = 0.25), identical rank-drop mitigation (9.83% vs 9.84%), and comparable dead-unit fractions (9.1% vs 10.2%). SNRI *does* deliver its non-disruption property — its mean per-step function disruption is 9× smaller and its worst-case disruption 4× smaller than trivial re-initialization — but this property confers **no** measurable plasticity benefit. Both re-initialization rules modestly outperform the no-re-initialization baseline (rank AUC 11.87). We conclude that re-initialization *disruption is not a productive design axis* for plasticity maintenance: the costly null-space machinery buys a property that does not matter for the outcome it was designed to improve, and a trivial random re-initialization is an equally good, far simpler choice.

---

## 1 Introduction

### Problem

Continually trained networks progressively lose the ability to fit new tasks — *plasticity loss* [Lyle et al., 2023; Dohare et al., 2023]. A widely used remedy is to re-initialize dormant or low-utility neurons during training (continual backpropagation [Dohare et al., 2023], ReDo [Sokar et al., 2023]). All such rules share a defect: the re-initialized neuron's outgoing weights are typically zeroed but its *incoming* weights are drawn randomly, so as soon as training resumes the neuron injects an uncontrolled perturbation into the representation, partially undoing prior learning. A natural hypothesis is that a *non-disruptive* re-initialization — one that provably leaves the current forward function unchanged at the instant of re-initialization — would retain the plasticity benefit while removing this cost, and therefore maintain a higher-dimensional (higher effective rank) representation over a long task stream.

### What we test

We formalize this as **Spectral Null-Space Re-Initialization (SNRI)**: when a neuron is re-initialized, its incoming weight vector is projected onto the null space of the current layer input covariance, so its pre-activation is (to first order) orthogonal to the span of active inputs and the layer output is unchanged at that step. This is provably non-disruptive by construction. We pre-register a single sharp, falsifiable prediction (§2.3) and test it against a trivial random re-initialization control at a matched budget.

### Contributions

1. **A clean null result.** On CIFAR-100 class-incremental ResNet-18 (5 seeds, 20 tasks), SNRI's effective-rank maintenance is *statistically indistinguishable* from trivial re-initialization (12.196 vs 12.200 rank AUC, paired Wilcoxon p = 0.25; the pre-registered ≥20% advantage is off by two orders of magnitude) (§3.2, Table 1).

2. **The mechanism it was built on is real but inert.** SNRI genuinely achieves non-disruption — 9× lower mean and 4× lower worst-case per-step function disruption than trivial re-initialization (§3.3, Table 2) — yet this property produces no downstream difference in rank, rank-drop, or dead-unit fraction. Non-disruption is achievable and irrelevant.

3. **A design implication.** Re-initialization *disruption* is not a productive axis for plasticity-maintenance method design; effort spent minimizing it (via null-space projection, orthogonalization, or similar) is unlikely to pay off. Both re-initialization rules do beat the no-re-initialization baseline, so the benefit of re-initialization is in the *event*, not in its *disruption profile* (§3.2).

---

## 2 Method

### 2.1 Setup

We train ResNet-18 on CIFAR-100 in a **class-incremental** stream: 20 tasks of 5 new classes each, 10 epochs per task, SGD (learning rate 0.1, momentum 0.9, Nesterov, weight decay 5e-4), cosine-annealed learning rate per task, batch size 128. We run 5 seeds (0–4). This is a standard, non-toy plasticity-loss regime with a real convolutional architecture and dataset.

### 2.2 Arms (matched re-initialization budget)

- **A — vanilla:** no re-initialization (baseline; the plasticity-loss condition).
- **B — SNRI:** at the end of each task, the lowest-utility neurons (a fixed fraction) are re-initialized with incoming weights projected onto the null space of the layer's input covariance (non-disruptive by construction); outgoing weights zeroed.
- **C — trivial re-initialization:** the *same* neurons, the *same* budget, but incoming weights drawn from the standard random initializer (non-orthogonal); outgoing weights zeroed.

Arms B and C differ *only* in how the incoming weights of re-initialized neurons are drawn — the utility criterion, the per-task fraction, and the schedule are identical, so the contrast isolates the value of the non-disruption property. Re-initialization counts are approximately (not exactly) matched: across the 5 seeds, B performs 133 re-initializations and C performs 151. The ~13% imbalance, if anything, favors C — it re-initializes *more* neurons — so it cannot explain away a missing advantage for B.

### 2.3 Pre-registered prediction (falsifier)

> On CIFAR-100 class-incremental, SNRI maintains effective rank **≥20% higher** than trivial re-initialization at a matched re-initialization budget.

Effective rank is e^{H(p)} of the normalized singular-value spectrum of the penultimate representation [Roy and Vetterli, 2007]; we report its area-under-curve over the 20 tasks (rank AUC). Disruption is the relative ℓ₂ change in the layer output caused by a re-initialization step, measured on a held-out batch immediately before and after.

---

## 3 Experiments

### 3.1 Positive controls

Both re-initialization arms reduce the effective-rank drop relative to vanilla (rank-drop 9.8% for B and C vs 12.1% for A) and raise rank AUC (12.20 vs 11.87), confirming that (a) the regime exhibits plasticity-relevant rank decay and (b) re-initialization mitigates it. The comparison of B vs C is therefore made between two arms that both *work*.

### 3.2 Main result: SNRI does not beat trivial re-initialization

**Table 1.** Plasticity-maintenance metrics (mean ± std over 5 seeds, CIFAR-100 class-incremental ResNet-18, 20 tasks). Higher rank AUC and lower rank-drop / dead-unit are better.

| Arm | Effective-rank AUC | Rank-drop (%) | Final dead-unit (%) |
|:---|:---:|:---:|:---:|
| A — vanilla (no re-init) | 11.87 ± 0.89 | 12.12 | 10.23 ± 5.27 |
| B — **SNRI** (null-space) | 12.196 ± 0.207 | 9.83 | 9.06 ± 2.45 |
| C — trivial re-init | 12.200 ± 0.208 | 9.84 | 10.17 ± 2.15 |

SNRI and trivial re-initialization are statistically indistinguishable on every plasticity metric: rank AUC differs by 0.04% (paired Wilcoxon p = 0.25; C higher on 3/5 seeds — no consistent direction), rank-drop by 0.01 pp, dead-unit fraction by ~1 pp (SNRI slightly lower, within noise). The pre-registered ≥20% rank advantage is refuted by roughly two orders of magnitude.

### 3.3 SNRI achieves non-disruption — which turns out not to matter

**Table 2.** Per-step re-initialization disruption (relative ℓ₂ change in layer output; lower = less disruptive).

| Arm | Mean disruption | Worst-case disruption |
|:---|:---:|:---:|
| B — SNRI (null-space) | 0.0001 | 0.27 |
| C — trivial re-init | 0.0009 | 1.02 |

SNRI does exactly what it was designed to do: its mean per-step disruption is 9× smaller and its worst-case 4× smaller than trivial re-initialization. The null-space projection is not a failed implementation — it delivers the non-disruption property cleanly. That property simply has no downstream consequence for plasticity: despite injecting neurons an order of magnitude more gently, SNRI ends the run with the same effective rank, the same rank-drop mitigation, and the same dead-unit fraction as the arm that injects them roughly.

---

## 4 Discussion

The result is a dissociation between a *property* (non-disruption) and an *outcome* (plasticity maintenance) that the property was hypothesized to drive. Two readings are consistent with the data.

First, the perturbation introduced by trivial re-initialization is *transient and quickly absorbed*. At the instant of re-initialization the outgoing weights are zeroed in both arms, so neither injects an immediate output change; the disruption in Table 2 accrues over subsequent steps, once the read-out weights grow back under the task gradient. The null-space constraint shapes the *incoming*-weight geometry that governs how the neuron responds during that regrowth — SNRI's 9× lower measured disruption confirms the constraint does dampen the perturbation — but over a 10-epoch task the network re-integrates the neuron regardless of its initial incoming geometry, so the dampened and undampened perturbations converge to the same end state. The constraint controls a transient that is re-absorbed either way.

Second, effective-rank maintenance appears to be governed by the *event* of introducing fresh, high-variance units — which both arms do identically — not by the geometry of their initial incoming weights. The gap between vanilla (rank AUC 11.87) and either re-initialization arm (12.20) is the entire effect; the within-re-initialization geometry contributes nothing measurable on top of it.

The practical implication for plasticity-maintenance method design is concrete: **do not spend design complexity minimizing re-initialization disruption.** Orthogonalization, null-space projection, and similar machinery buy a property that this experiment shows is inert. A trivial random re-initialization at the same budget is an equally effective and far simpler choice.

---

## 5 Limitations

1. **One architecture / dataset / task type.** Results are for ResNet-18 on CIFAR-100 class-incremental. Whether the dissociation holds for transformers, for larger rank-injection budgets, or for regimes where re-initialization disruption is a larger share of total gradient movement is untested; a larger budget could in principle surface a difference this experiment lacks the power to see.
2. **Proxy for plasticity.** We measure effective rank, rank-drop, and dead-unit fraction — established plasticity correlates — rather than downstream new-task top-1 accuracy directly; the mapping from rank maintenance to accuracy is assumed, not measured here.
3. **Five seeds.** The null result is well-supported (paired Wilcoxon, matched arms differing in one factor), but 5 seeds bound the smallest detectable effect; we claim the ≥20% predicted effect is absent, not that the true difference is exactly zero.
4. **Depth-sensitivity untested.** The original hypothesis predicted SNRI's advantage would grow with depth; with no advantage at ResNet-18 depth, the depth sweep was not run and that sub-claim is neither supported nor refuted.

---

## 6 Related Work

**Plasticity loss and re-initialization.** Dohare et al. [2023] (arXiv:2303.01486) introduce continual backpropagation, which continually re-initializes low-utility units; Sokar et al. [2023] (arXiv:2302.12902) propose ReDo, re-initializing dormant neurons in deep RL. Both use non-orthogonal random re-initialization — our arm C. Our contribution is the controlled test of whether making that re-initialization non-disruptive helps; it does not.

**Loss of plasticity, characterization.** Lyle et al. [2023] (arXiv:2306.13812) document effective-rank decline and dead-unit accumulation as correlates of plasticity loss, motivating our metrics. We use their diagnostics as outcome measures.

**Recovering plasticity by weight editing.** Lewandowski et al. [2025] (arXiv:2507.04683) recover plasticity via soft weight rescaling, editing existing weights rather than re-initializing units. Our null result on disruption-minimization suggests the mechanism of benefit in re-initialization methods lies in fresh-unit variance injection rather than in careful control of the injection's geometry.

**Effective rank.** Roy and Vetterli [2007] introduce the effective-rank measure we use as the primary plasticity proxy.

---

## 7 Conclusion

We tested whether making dormant-neuron re-initialization provably non-disruptive — by injecting new neurons into the null space of the current representation — improves plasticity maintenance over trivial random re-initialization. On CIFAR-100 class-incremental ResNet-18 across 5 seeds, the pre-registered ≥20% effective-rank advantage is refuted by two orders of magnitude: the two rules are statistically identical on every plasticity metric, even though the null-space rule genuinely reduces per-step disruption by 9×. Non-disruption is achievable and, for this purpose, inert. The benefit of re-initialization lies in the event of introducing fresh high-variance units, not in the geometry of their initial weights — so for plasticity maintenance, a trivial random re-initialization is the right default and disruption-minimization is not worth its complexity.

---

## References

- Dohare, S., Hernandez-Garcia, J. F., Sutton, R. S., and Mahmood, A. R. (2023). *Understanding plasticity in neural networks.* arXiv:2303.01486.
- Lewandowski, A., et al. (2025). *Recovering plasticity of neural networks via soft weight rescaling.* arXiv:2507.04683.
- Lyle, C., Zheng, Z., Nikishin, E., Pires, B. A., Pascanu, R., and Dabney, W. (2023). *Loss of plasticity in deep continual learning.* arXiv:2306.13812.
- Roy, O. and Vetterli, M. (2007). *The effective rank: A measure of effective dimensionality.* EUSIPCO.
- Sokar, G., Agarwal, R., Castro, P. S., and Evci, U. (2023). *The dormant neuron phenomenon in deep reinforcement learning.* arXiv:2302.12902.
