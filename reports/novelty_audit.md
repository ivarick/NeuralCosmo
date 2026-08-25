# Novelty Audit — Pass 1

Phase 0 deliverable (plan sections 68, 86).

**Date:** 2026-08-24
**Status:** first pass. Section 86 requires this to be rerun immediately before
submission, and again before the Phase 6 novelty gate.
**Verdict:** a defensible gap appears to exist, but it is narrower than the
plan assumed, and one of the plan's framings is already occupied.

---

## 1. What was searched

Section 68's required queries, plus follow-ups suggested by the results:

- CAMELS robustness / domain adaptation / domain generalization
- CAMELS simulator invariant, CAMELS MIEST
- CAMELS N-body hydrodynamic paired learning
- CAMELS multifidelity inference
- CAMELS contrastive representation, self-supervised cosmology
- baryonic invariant representation
- paired hydrodynamic/N-body contrastive learning, multifidelity cosmology
- asymmetric cross-simulation transfer, direction of transfer

---

## 2. Prior-work table

| Paper | Input | Sources | Target | Method | Uses target data? | Main limitation | Overlap with us |
|---|---|---|---|---|---|---|---|
| **MIEST** — Jo et al. 2025, [arXiv:2502.13239](https://arxiv.org/abs/2502.13239), [ApJ](https://doi.org/10.3847/1538-4357/adec78) | **HI maps only** (total matter *not* used) | TNG, SIMBA, Astrid, Swift-EAGLE, trained *jointly* | held-out suites | encoder + regressor + adversarial de-classification / information bottleneck | No — domain generalization | Single modality; joint multi-source training only; no single-source transfer directions reported | **High on framing, low on modality.** Occupies "remove simulator information adversarially for cross-suite robustness" |
| **DA-GNN** — Roncoli et al. 2023, [arXiv:2311.01588](https://arxiv.org/abs/2311.01588) | galaxy catalogues as graphs | CAMELS suites | other suite | GNN + MMD alignment | **Yes** — unlabelled target used | Domain *adaptation*, not generalization; different modality | Moderate. Occupies "MMD alignment on CAMELS" |
| **Multifidelity transfer** — [arXiv:2505.21215](https://arxiv.org/abs/2505.21215), MNRAS 542, 3231 | CMD **dark matter density maps**, N-body + hydro | one suite | — | **sequential pre-training** on N-body, then fine-tune on hydro | n/a | Goal is *data efficiency*, not robustness; no unseen-suite evaluation; matched initial conditions not exploited | **Highest on data usage.** Closest prior use of N-body alongside hydro in CMD |
| **One latent to fit them all** — [arXiv:2509.01881](https://arxiv.org/abs/2509.01881), ApJL | matter **power spectra** and fields; hydro paired with gravity-only | 4 models, 5072 sims | — | unified 2D latent of baryonic feedback | n/a | Characterises feedback; does not perform zero-shot cosmological regression on an unseen simulator | **Moderate.** Occupies "hydro/N-body pairs give a feedback latent" |
| CMD benchmark — Villaescusa-Navarro et al. 2022, [arXiv:2109.10915](https://arxiv.org/abs/2109.10915) | 2D maps incl. Mtot | one suite | — | CNN regression | n/a | Establishes the task and names cross-suite robustness as *open* | Baseline reference |
| Robust marginalization of baryonic effects — [arXiv:2109.10360](https://arxiv.org/abs/2109.10360) | field level | — | — | marginalisation | n/a | not representation learning | Low |
| Self-supervised compression — [arXiv:2308.09751](https://arxiv.org/abs/2308.09751) | CAMELS | — | — | SSL / data compression | n/a | not cross-simulator robustness | Low; **needs closer reading before Phase 6** |

---

## 3. What is occupied

**Adversarial de-classification for cross-simulator robustness is taken.** MIEST
does exactly this. The plan already anticipated it (section 3.3) and correctly
demoted it to a baseline. Confirmed.

**MMD alignment on CAMELS is taken.** DA-GNN. Also already a baseline in the plan.

**N-body-plus-hydro in CMD is no longer untouched.** This is the finding that
changes things. The multifidelity transfer-learning paper already uses CMD dark
matter maps together with hydrodynamic maps. The plan (section 4.1) implied the
paired hydro/N-body structure was an unexploited feature of the dataset. It is
not unexploited — but *how* it is exploited differs from PPIRL in three ways:

1. **Mechanism.** They pre-train sequentially, then fine-tune. PPIRL proposes a
   *simultaneous paired-view alignment loss*. These are different objectives.
2. **Pairing.** They do not appear to exploit matched initial conditions. PPIRL's
   whole premise is that a hydro map and its N-body counterpart are the *same
   region of the same universe*, which is what makes a consistency loss
   meaningful rather than a generic augmentation.
3. **Goal.** Theirs is data efficiency — fewer expensive simulations for the same
   posterior. Ours is zero-shot robustness to an unseen simulator. They do not
   evaluate on a held-out suite at all.

**Hydro/N-body pairs as a feedback representation is partly taken.** "One latent
to fit them all" builds a 2D latent of baryonic feedback from paired hydro and
gravity-only runs. But it works in power-spectrum space and characterises
feedback; it does not do zero-shot parameter inference on an unseen simulator.

---

## 4. What appears to remain open

Stated as one sentence, per section 68's exit criterion:

> **Use matched hydrodynamic / gravity-only map pairs as an explicit
> representation-consistency constraint during training, in order to improve
> zero-shot cosmological regression on a simulation suite never seen during
> training or model selection.**

No paper found does this. The nearest neighbours each miss on a different axis:
MIEST has the goal but not the mechanism; multifidelity transfer has the data but
not the goal; the feedback-latent paper has the pairing but neither the modality
nor the zero-shot evaluation.

**This gap is narrower than the plan assumed** and depends entirely on the paired
mechanism outperforming its own controls. Section 50's shuffled-pair control and
section 51's unpaired-extra-data control are now doubly important: if correct
pairing gives no advantage over simply adding N-body data, then what remains is
the multifidelity transfer paper's contribution with extra steps, and Kill 4
applies.

---

## 5. A second, unexpected candidate

Our own Phase 4 measurement may be a contribution independent of PPIRL.

We measured, on **total matter** maps, single-source transfer in both directions:

```
TNG   -> SIMBA    G = 2.22 (Omega_m), 1.57 (sigma_8)
SIMBA -> TNG      G = 1.17 (Omega_m), 0.92 (sigma_8)
```

Two observations that the literature does not appear to report:

1. **Directional asymmetry, quantified.** MIEST notes SIMBA is an outlier
   (pairwise AUC 0.83 against TNG on HI maps) but **trains jointly on multiple
   suites and does not report single-direction transfer experiments**. We could
   find no quantitative TNG→SIMBA versus SIMBA→TNG comparison for total matter.
2. **Cross-simulator error is systematic, not stochastic.** Averaging the 15 maps
   of a simulation removes 57% of in-domain error but only 15% of transfer error.
   This means simulator shift cannot be averaged away, and it is not visible in
   map-level metrics alone.

Both need seed replication before they are claims rather than observations.
Seeds 1 and 2 are running.

**Caution:** one search result asserted that "the SIMBA model performs poorly when
applied to all other simulation suites", which if read as *trained on SIMBA* would
contradict our measurement. The provenance of that statement could not be pinned
to a specific paper and its wording is ambiguous between "trained on SIMBA" and
"tested on SIMBA". **This must be resolved before any asymmetry claim is made.**

---

## 6. External sanity check against the benchmark

Section 27 wants our in-domain numbers compared to the published CMD benchmark.
Reported CMD CNN performance on total matter maps is R² ≈ 0.96 for Ω_m and ≈ 0.83
for σ_8. Ours, in-domain on held-out test simulations:

| | R² Ω_m | R² σ_8 |
|---|---:|---:|
| CMD benchmark (published) | ~0.96 | ~0.83 |
| E01 TNG, map level | 0.991 | 0.974 |
| E02 SIMBA, map level | 0.990 | 0.973 |

Ours are higher. That is **not** yet a claim of superiority: the split protocol,
training budget, augmentation and exact map subset may differ, and section 32
warns against claiming superiority when the comparison is not apples-to-apples.
The useful reading is the weaker one — our pipeline is not broken and is in the
right regime. Before any comparison appears in a paper, the benchmark's exact
protocol must be read and matched or the differences documented.

---

## 7. Consequences for the plan

1. **Keep MIEST and DA-GNN as baselines.** Confirmed occupied; already planned.
2. **Add the multifidelity transfer paper as a required baseline and citation.**
   It is the closest prior use of N-body data in CMD and was not in the plan's
   related work. An "N-body pre-training then fine-tune" arm should be added to
   the Phase 8 comparison, alongside hydro-only, unpaired-N-body, paired, and
   shuffled-pair.
3. **The paired mechanism must beat its controls or the contribution collapses.**
   Sections 50 and 51 are now the load-bearing experiments, not optional rigour.
4. **Consider the empirical asymmetry as a parallel contribution.** It requires no
   new method, only seeds and analysis, and it is defensible on its own.
5. **Read [arXiv:2308.09751](https://arxiv.org/abs/2308.09751) properly** before
   Phase 6 — self-supervised representation learning on CAMELS is close enough to
   the contrastive framing to matter.

---

## 8. Exit criterion

Section 68: *"the proposed contribution can be stated in one sentence that is not
already implemented by a prior paper."*

**Met, provisionally**, by the sentence in section 4 above — with the explicit
caveat that its novelty rests on the paired-alignment mechanism being distinct
from sequential multifidelity pre-training, and that this distinction only
survives if the Phase 8 controls show correct pairing beating unpaired N-body data.

**Not yet met** for any broader claim such as "using N-body counterparts to
improve cosmological inference", which is occupied.
