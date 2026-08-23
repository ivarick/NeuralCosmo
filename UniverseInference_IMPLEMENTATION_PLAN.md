# UniverseInference
## End-to-End Research and Implementation Plan

**Working research theme:** simulator-robust scientific inference from CAMELS cosmological maps  
**Primary ML focus:** domain generalization, representation learning, nuisance invariance, and robust regression  
**Scientific testbed:** CAMELS Multifield Dataset (CMD)  
**Initial targets:** \(\Omega_m\) and \(\sigma_8\)  
**Initial data modality:** 2D total-matter density maps (`Mtot`) at \(z=0\)  
**Primary compute target:** a single 12 GB consumer NVIDIA GPU  
**Research-audit cutoff for this plan:** 2026-08-23

---

# 0. Executive Summary

UniverseInference is **not** primarily a cosmology paper. Cosmology provides a difficult and scientifically meaningful environment in which to study a machine-learning problem:

> **How can a neural network preserve task-relevant information while discarding simulator-specific information, so that it generalizes to a data-generating simulator it never saw during training?**

The basic supervised task is simple:

```text
2D simulated matter map
        ↓
neural network
        ↓
Ωm and σ8
```

The hard part is that the same task exists in multiple CAMELS simulation suites—such as IllustrisTNG, SIMBA, and Astrid—but the suites are produced by different numerical codes and different astrophysical/subgrid models. A network can therefore obtain excellent in-distribution performance while partly exploiting simulator-specific artifacts and then fail when evaluated on another simulation family.

The first-stage experiment is:

```text
TRAIN
IllustrisTNG + SIMBA
        ↓
       model

TEST
Astrid — never used for training or hyperparameter selection
```

The research objective is to improve **true domain generalization**, not target-aware domain adaptation.

However, there is a crucial literature fact:

- The original CAMELS/CMD benchmark explicitly identified cross-suite robustness as an open challenge.
- Roncoli et al. (2023) already applied MMD-based domain adaptation to CAMELS galaxy graphs.
- Jo et al. (2025) introduced **MIEST**, which uses an information-bottleneck/adversarial de-classification strategy to remove simulation-model information while retaining cosmological information, training on multiple suites and evaluating on unseen suites.

Therefore, **“add a domain classifier and make simulator identity hard to predict” is not a novel paper contribution anymore.**

This implementation plan has two layers:

1. **Reproduce the problem rigorously and establish strong baselines.**
2. **Only after a formal novelty audit, implement a more specific ML contribution.**

The most promising provisional contribution in this plan is:

> **Paired-Physics Invariant Representation Learning (working name, not a novelty claim): use CAMELS' matched hydrodynamic/N-body total-matter maps as paired views of the same cosmology and initial conditions, using the gravity-only map as an invariant anchor against baryonic/simulator-specific effects.**

Conceptually:

```text
Hydrodynamic Mtot map ─┐
                       ├─ same region, same cosmology, same initial seed
Matched N-body map ────┘

              ↓ paired representation learning

retain:
    Ωm, σ8, large-scale cosmological structure

suppress:
    simulator-specific baryonic/subgrid signatures
```

This is attractive because the dataset itself provides a special supervision signal that ordinary domain-generalization datasets do not: **matched high-fidelity and gravity-only views**. CAMELS documentation states that total-matter hydrodynamic maps can be matched to N-body counterparts for the same region.

This paired-physics branch must still pass a dedicated literature audit before being presented as novel.

---

# 1. Research Goal

## 1.1 Primary ML question

> Can a representation-learning objective trained only on source simulation suites improve zero-shot cosmological regression on an unseen simulator without using target-domain maps, target-domain labels, or target-specific normalization?

This is a **domain-generalization** problem.

It is intentionally stricter than domain adaptation.

### Domain adaptation

The method can inspect target-domain examples during training:

```text
Source labeled data + target unlabeled data
                ↓
               model
```

### Domain generalization

The target domain is completely absent:

```text
Source A + Source B
        ↓
       model
        ↓
unseen Source C
```

UniverseInference should make its strongest claim under **domain generalization**.

---

## 1.2 Scientific task

For each map \(x\), predict:

\[
y =
\begin{bmatrix}
\Omega_m \\
\sigma_8
\end{bmatrix}
\]

where:

- \(\Omega_m\) is the matter-density parameter.
- \(\sigma_8\) controls the amplitude of matter fluctuations on a standard scale.

For the CAMELS LH set, the current documentation gives:

\[
\Omega_m \in [0.1, 0.5]
\]

\[
\sigma_8 \in [0.6, 1.0]
\]

The four standard astrophysical feedback parameters vary too, but they must initially be treated as **nuisance variables**, not cross-suite prediction targets.

This is important because the CAMELS documentation explicitly warns that the similarly named astrophysical parameters in different suites have different meanings/implementations. `A_SN1` in Astrid is not the same physical parameterization as `A_SN1` in IllustrisTNG or SIMBA.

---

# 2. Non-Goals

The first paper should **not** attempt all of the following simultaneously:

- discover new cosmological physics;
- infer all astrophysical feedback parameters;
- train on the full 325 GB 2D archive;
- use 3D 120 TB grids;
- use every CAMELS simulator;
- build a giant vision transformer;
- claim simulation-to-real observational generalization;
- solve posterior inference, OOD detection, calibration, multifield fusion, and domain generalization in one paper;
- beat every published CAMELS result;
- claim that a latent space is “physical” solely because UMAP looks mixed;
- claim novelty before checking 2023–2026 related work.

The first paper should be narrow enough to defend:

> **Robust regression of \(\Omega_m,\sigma_8\) across unseen simulation families, with a specific representation-learning contribution and rigorous leakage-free evaluation.**

---

# 3. Existing Work That Changes the Research Plan

This section is mandatory because it prevents us from accidentally reimplementing an existing paper.

## 3.1 CAMELS/CMD benchmark

The CAMELS Multifield Dataset authors explicitly list cross-suite robustness as a challenge:

> models trained on one suite can perform well in-domain but fail on another suite.

Official challenge page:

https://camels-multifield-dataset.readthedocs.io/en/latest/inference.html

---

## 3.2 Domain-Adaptive GNNs — 2023

Roncoli et al.:

**Domain Adaptive Graph Neural Networks for Constraining Cosmological Parameters Across Multiple Data Sets**

https://arxiv.org/abs/2311.01588

They use:

- CAMELS galaxy catalogs rather than the 2D maps planned here;
- Graph Neural Networks;
- Maximum Mean Discrepancy (MMD);
- cross-dataset/domain adaptation.

This means **“use MMD to align CAMELS representations” is already a baseline, not a sufficient contribution.**

---

## 3.3 MIEST — 2025

Jo et al.:

**Toward Robustness across Cosmological Simulation Models IllustrisTNG, SIMBA, Astrid, and Swift-Eagle**

arXiv:

https://arxiv.org/abs/2502.13239

Published DOI:

https://doi.org/10.3847/1538-4357/adec78

MIEST:

- predicts \(\Omega_m\) and \(\sigma_8\);
- operates on CAMELS simulation models;
- explicitly tries to remove model-specific information;
- uses a CNN/encoder/regressor/classifier framework;
- incorporates adversarial de-classification/information-bottleneck ideas;
- trains with multiple models;
- evaluates on unseen simulations.

Therefore:

> **A plain adversarial simulator classifier is not the novel method.**

MIEST must appear in the literature review and, if technically possible under the selected data modality, either be reproduced or represented by a faithful comparable baseline.

---

# 4. Revised Candidate Contribution

## 4.1 Working concept

**Paired-Physics Invariant Representation Learning (PPIRL)**

This is a working internal name only.

Do **not** write “we propose a novel method called PPIRL” until the literature audit confirms that the core idea is unoccupied.

The key CAMELS-specific structure is:

```text
Hydrodynamic simulation
    Mtot map
       │
       │ same cosmology + same initial random field + same spatial region
       │
N-body counterpart
    Mtot map
```

The total-matter field is special because CMD documents that hydrodynamic and N-body total-matter maps can be matched.

The gravity-only N-body counterpart lacks the hydrodynamic simulation's baryonic feedback implementation. It can therefore act as a **paired nuisance-reduced view**.

### Intended ML principle

For a matched pair:

\[
x_h = \text{hydrodynamic map}
\]

\[
x_n = \text{matched N-body map}
\]

we want:

\[
f(x_h) \approx f(x_n)
\]

for the part of the representation used to infer cosmology.

At the same time:

\[
g(f(x_h)) \rightarrow (\Omega_m,\sigma_8)
\]

must remain accurate.

The model should therefore learn a representation that is:

- predictive of cosmology;
- consistent across a paired change in physical fidelity;
- less sensitive to source simulator identity.

The general ML idea is broader than cosmology:

> use paired low-/high-fidelity simulator views to learn nuisance-invariant representations for scientific inference.

Potential later applications include:

- CFD solvers;
- climate simulators;
- robotics simulation;
- materials simulators;
- medical simulators.

---

# 5. Hypotheses

All hypotheses must be written **before** final runs.

## H0 — In-domain competence

A standard CNN trained and tested within a suite should recover useful information about \(\Omega_m\) and \(\sigma_8\).

If this fails, there is a data/preprocessing/model problem before any research question can be addressed.

---

## H1 — Simulator shift exists

A model trained on one simulator will have worse error on another simulator than on held-out maps from its own simulator.

Define a generalization ratio:

\[
G =
\frac{\mathrm{OOD\ Error}}
{\mathrm{ID\ Error}}
\]

A meaningful shift should produce \(G > 1\).

If Mtot produces almost no shift, the research question needs to be reconsidered or strengthened with more baryon-sensitive fields.

---

## H2 — Standard ERM encodes simulator identity

A frozen representation from a normal source-trained regressor should allow a probe classifier to predict simulator identity better than chance.

For two source simulators:

\[
P(\text{chance}) = 50\%
\]

For three:

\[
P(\text{chance}) \approx 33.3\%
\]

High probe accuracy indicates simulator information remains in the representation.

---

## H3 — Marginal alignment is not necessarily enough

Naively aligning entire source-domain feature distributions may remove task-relevant structure because \(\Omega_m,\sigma_8\) themselves alter the data distribution.

Therefore:

> lower simulator-probe accuracy is not automatically better.

The representation must reduce simulator information **while retaining target information**.

---

## H4 — Paired N-body anchoring improves transfer

A model trained with matched hydro/N-body representation consistency should outperform an otherwise identical hydro-only model on an unseen hydrodynamic simulator.

This is the central provisional hypothesis.

---

## H5 — Correct pairing matters

If the N-body pairing is shuffled while preserving marginal data distributions, performance should decline relative to correct pairing.

This is a critical control.

Without it, any gain could simply come from seeing more N-body data rather than exploiting pair structure.

---

## H6 — Improvement survives multiple seeds

Any claimed improvement must survive at least 3 independent seeds; 5 is preferred for final tables.

Do not publish a single favorable seed.

---

# 6. Dataset

## 6.1 Dataset family

**CAMELS Multifield Dataset (CMD)**

Official documentation:

https://camels-multifield-dataset.readthedocs.io/en/latest/

Official CAMELS documentation:

https://camels.readthedocs.io/en/latest/

Official CAMELS analysis repository:

https://github.com/franciscovillaescusa/CAMELS

CMD paper:

https://doi.org/10.3847/1538-4365/ac5ab0

---

## 6.2 Why use CMD rather than raw snapshots?

The raw CAMELS project is enormous.

Current CAMELS documentation states that the project contains more than 2 PB of data across 16,960 simulations as of June 2026.

CMD offers preconstructed arrays specifically intended for ML.

The CMD access documentation lists approximately:

- 325 GB total 2D maps;
- 120 TB 3D grids.

We need neither.

The initial experiment uses only a few 2D files.

---

# 7. Initial Data Manifest

## 7.1 Required Phase-A hydrodynamic files

Start with only the LH total-matter maps at \(z=0\):

### IllustrisTNG

```text
Maps_Mtot_IllustrisTNG_LH_z=0.00.npy
params_LH_IllustrisTNG.txt
```

### SIMBA

```text
Maps_Mtot_SIMBA_LH_z=0.00.npy
params_LH_SIMBA.txt
```

### Astrid

```text
Maps_Mtot_Astrid_LH_z=0.00.npy
params_LH_Astrid.txt
```

Each LH map file is documented as:

```text
15,000 maps
256 × 256 pixels
```

Each parameter file contains:

```text
1,000 rows
6 parameters per row
```

The map-to-simulation mapping is:

```python
simulation_id = map_index // 15
```

Thus:

```text
maps 0..14     -> simulation 0
maps 15..29    -> simulation 1
...
maps 14985..14999 -> simulation 999
```

The first two parameter columns are:

```text
0 -> Ωm
1 -> σ8
```

The remaining four are astrophysical parameters.

Official data-structure documentation:

https://camels-multifield-dataset.readthedocs.io/en/latest/data.html

---

## 7.2 Initial storage requirement

At the time this plan was prepared, the three selected Mtot LH files were approximately 3.66 GB each in the Flatiron data listing.

Therefore the initial hydrodynamic download is approximately:

```text
3 × ~3.66 GB ≈ 11 GB
```

plus negligible parameter text files.

Record the actual local sizes after download rather than treating this estimate as a checksum.

---

# 8. Data Acquisition

## 8.1 Preferred method

CMD officially recommends **Globus** for large transfers because URL downloads can be slow or unstable.

Official access instructions:

https://camels-multifield-dataset.readthedocs.io/en/latest/access.html

Use the official CAMELS/CMD Globus collection linked there.

---

## 8.2 URL method

The direct URL method is acceptable if Globus is inconvenient.

The Flatiron browser root used by CMD is available through the official access page.

Expected files should be selected by exact filename, not by manually downloading entire directories.

---

# 9. Local Storage Layout

Keep the repository on `C:` and the raw data on `G:`.

Recommended:

```text
C:\
└── projects\
    └── UniverseInference\
        ├── configs\
        ├── src\
        ├── tests\
        ├── scripts\
        ├── outputs\
        ├── reports\
        ├── paper\
        ├── pyproject.toml
        ├── README.md
        └── .gitignore

G:\
└── datasets\
    └── CAMELS_CMD\
        ├── IllustrisTNG\
        │   ├── Maps_Mtot_IllustrisTNG_LH_z=0.00.npy
        │   └── params_LH_IllustrisTNG.txt
        ├── SIMBA\
        │   ├── Maps_Mtot_SIMBA_LH_z=0.00.npy
        │   └── params_LH_SIMBA.txt
        ├── Astrid\
        │   ├── Maps_Mtot_Astrid_LH_z=0.00.npy
        │   └── params_LH_Astrid.txt
        └── manifests\
            └── local_manifest.json
```

The Git repository must never contain raw maps.

---

# 10. Environment Variable

Recommended Windows environment variable:

```powershell
$env:CAMELS_DATA_ROOT = "G:\datasets\CAMELS_CMD"
```

The project should resolve data paths from:

1. CLI/config override;
2. `CAMELS_DATA_ROOT`;
3. fail with an explicit error.

Do **not** silently assume a machine-specific absolute path.

---

# 11. Repository Structure

Recommended full structure:

```text
UniverseInference/
│
├── README.md
├── pyproject.toml
├── .gitignore
├── LICENSE
│
├── configs/
│   ├── data/
│   │   ├── mtot.yaml
│   │   └── mtot_nbody_pairs.yaml
│   │
│   ├── model/
│   │   ├── small_cnn.yaml
│   │   ├── resnet.yaml
│   │   ├── dann.yaml
│   │   └── ppirl.yaml
│   │
│   └── experiments/
│       ├── e00_smoke.yaml
│       ├── e01_tng_id.yaml
│       ├── e02_simba_id.yaml
│       ├── e10_tng_to_simba.yaml
│       ├── e11_simba_to_tng.yaml
│       ├── e20_multisource_erm.yaml
│       ├── e21_dann.yaml
│       ├── e22_coral.yaml
│       ├── e23_mmd.yaml
│       ├── e30_ppirl_pair.yaml
│       └── e40_final_astrid.yaml
│
├── src/
│   └── universe_inference/
│       ├── __init__.py
│       │
│       ├── data/
│       │   ├── manifest.py
│       │   ├── validate.py
│       │   ├── splits.py
│       │   ├── dataset.py
│       │   ├── paired_dataset.py
│       │   ├── transforms.py
│       │   └── statistics.py
│       │
│       ├── models/
│       │   ├── backbones/
│       │   │   ├── small_cnn.py
│       │   │   └── resnet.py
│       │   ├── heads.py
│       │   ├── erm.py
│       │   ├── dann.py
│       │   ├── coral.py
│       │   ├── mmd.py
│       │   └── ppirl.py
│       │
│       ├── losses/
│       │   ├── regression.py
│       │   ├── domain.py
│       │   ├── alignment.py
│       │   ├── paired.py
│       │   └── regularization.py
│       │
│       ├── training/
│       │   ├── trainer.py
│       │   ├── checkpoint.py
│       │   ├── seed.py
│       │   └── schedules.py
│       │
│       ├── evaluation/
│       │   ├── metrics.py
│       │   ├── bootstrap.py
│       │   ├── domain_probe.py
│       │   ├── representations.py
│       │   └── robustness.py
│       │
│       └── cli.py
│
├── scripts/
│   ├── validate_data.py
│   ├── build_splits.py
│   ├── compute_stats.py
│   ├── run_experiment.py
│   ├── evaluate_checkpoint.py
│   ├── run_domain_probe.py
│   └── make_paper_figures.py
│
├── tests/
│   ├── test_manifest.py
│   ├── test_map_sim_mapping.py
│   ├── test_split_integrity.py
│   ├── test_no_target_leakage.py
│   ├── test_normalization.py
│   ├── test_dataset.py
│   ├── test_paired_dataset.py
│   ├── test_losses.py
│   └── test_metrics.py
│
├── outputs/
│   ├── runs/
│   ├── checkpoints/
│   ├── predictions/
│   ├── embeddings/
│   └── metrics/
│
├── reports/
│   ├── data_report.md
│   ├── baseline_report.md
│   ├── novelty_audit.md
│   └── experiment_log.md
│
└── paper/
    ├── figures/
    ├── tables/
    ├── notes/
    └── manuscript/
```

---

# 12. Data Validation — Before Training Anything

Create a validation command whose only purpose is to reject bad data.

It should record:

```text
file
exists
byte size
shape
dtype
min
max
mean sample
NaN count
Inf count
parameter shape
parameter ranges
```

For every hydrodynamic LH suite assert:

```text
maps.shape == (15000, 256, 256)
params.shape == (1000, 6)
15000 == 1000 * 15
```

Then validate:

```text
Ωm min approximately 0.1
Ωm max approximately 0.5

σ8 min approximately 0.6
σ8 max approximately 1.0
```

Do not hard fail on tiny floating-point deviations around bounds.

---

# 13. Generate a Local Data Manifest

After download, create:

```text
G:\datasets\CAMELS_CMD\manifests\local_manifest.json
```

Example structure:

```json
{
  "dataset": "CAMELS CMD",
  "set": "LH",
  "redshift": 0.0,
  "field": "Mtot",
  "files": [
    {
      "suite": "IllustrisTNG",
      "map_file": "...",
      "parameter_file": "...",
      "shape": [15000, 256, 256],
      "dtype": "record actual value",
      "bytes": 0,
      "sha256_local": "compute after download"
    }
  ]
}
```

The SHA-256 is **your own local integrity record**.

Do not present it as an official CAMELS checksum unless CAMELS publishes a matching checksum.

---

# 14. Memory-Mapped Loading

Do not load a multi-gigabyte `.npy` file into RAM for every run.

Use NumPy memory mapping:

```python
maps = np.load(path, mmap_mode="r")
```

CMD documentation itself recommends memory mapping for large array products.

The dataset should read only requested maps.

---

# 15. Dataset Sample Definition

Each map sample should return at least:

```python
{
    "image": ...,
    "target": [omega_m, sigma_8],
    "suite_id": ...,
    "simulation_id": ...,
    "map_id": ...,
}
```

Do not discard `simulation_id`.

It is required for:

- leakage prevention;
- grouped evaluation;
- bootstrap confidence intervals;
- map aggregation.

---

# 16. Simulation-Level Splits

This is non-negotiable.

All 15 maps from one simulation share the same parameter vector and are statistically related.

Never randomly split maps.

The original CMD benchmark itself splits by simulation for exactly this reason.

## Recommended fixed base split per suite

Use:

```text
900 simulations -> train
50 simulations  -> validation
50 simulations  -> test
```

This matches the original CMD benchmark split proportions:

```text
13,500 train maps
750 validation maps
750 test maps
```

The specific simulation IDs should be generated once with a fixed seed and stored permanently:

```text
configs/splits/split_v1.json
```

Do not regenerate splits every run.

---

# 17. Split Integrity Tests

Automated tests must confirm:

```text
train_sim_ids ∩ val_sim_ids  = ∅
train_sim_ids ∩ test_sim_ids = ∅
val_sim_ids   ∩ test_sim_ids = ∅
```

and:

```text
all 15 maps from simulation X belong to exactly one split
```

A failed split-integrity test must stop training.

---

# 18. Strict Domain-Generalization Protocol

The most important design decision in the paper is preventing target leakage.

## Development domains

Use:

```text
IllustrisTNG
SIMBA
```

for method development.

Allowed during development:

```text
TNG -> TNG
SIMBA -> SIMBA
TNG -> SIMBA
SIMBA -> TNG
TNG + SIMBA source validation
```

## Final unseen domain

Use:

```text
Astrid
```

as the **sealed final target**.

The final paper configuration must be frozen before looking at final Astrid metrics.

---

# 19. What “Sealed Astrid” Means

During development:

- Astrid maps may be downloaded.
- Data-integrity scripts may confirm file shape and basic numerical validity.
- Do not use Astrid label-conditioned performance to select:
  - architecture;
  - augmentation;
  - loss weights;
  - learning rate;
  - early stopping;
  - representation dimension;
  - alignment strength.
- Do not compute target normalization statistics for the primary DG condition.
- Do not tune based on Astrid domain-probe plots.

Before the first final Astrid run:

1. commit code;
2. commit config;
3. record Git commit hash;
4. record experiment hash;
5. freeze all hyperparameters;
6. then run Astrid once per planned seed.

If the final result is poor, report it.

Do not silently return to tuning on Astrid and continue calling it “unseen.”

---

# 20. Preprocessing

## 20.1 Log transform

The original CMD parameter-inference paper reports taking the logarithm of pixel values before normalization for its fields.

For `Mtot`, first inspect the actual minimum.

### Required logic

```text
if all training-source Mtot values > 0:
    use log(value)
else:
    STOP
    inspect data semantics
    do not silently invent epsilon
```

Do not automatically use `log1p` unless justified.

The original paper specifically notes a special `log(1 + ρ*)` treatment for stellar density; do not transfer that choice to Mtot without reason.

---

## 20.2 Normalization

For the **strict DG condition**, compute normalization statistics from source-training data only.

Example when training TNG + SIMBA:

```text
normalization mean:
    source train pixels only

normalization std:
    source train pixels only
```

Then apply the same statistics to:

```text
source validation
source test
Astrid final target
```

Do **not** normalize Astrid using Astrid's own mean and standard deviation in the primary experiment.

Why?

Because target-specific statistics are target-domain information.

That may be legitimate under unsupervised adaptation, but it is no longer the cleanest domain-generalization protocol.

---

# 21. Normalization Conditions to Report

It is useful to explicitly compare:

### DG-Strict

```text
source-train normalization only
```

### Target-stat-aware

```text
each target normalized with its own unlabeled statistics
```

The second condition is allowed only as a **separate target-aware adaptation analysis**.

Never mix the two silently.

---

# 22. Target Normalization

For regression stability, transform the two targets using **source-training target statistics** or a fixed physical-range transform.

A simple fixed-range mapping is attractive because the LH ranges are known:

\[
\tilde{\Omega}_m =
\frac{\Omega_m - 0.1}{0.4}
\]

\[
\tilde{\sigma}_8 =
\frac{\sigma_8 - 0.6}{0.4}
\]

This maps both targets approximately into `[0,1]`.

Because these are dataset-design ranges rather than sample statistics, they do not leak target-suite empirical information.

Record predictions in physical units after inverse transformation.

---

# 23. Data Augmentation

Start conservative.

Potential transformations:

- horizontal flip;
- vertical flip;
- 90° rotations.

Do not introduce arbitrary natural-image augmentations such as:

- color jitter;
- JPEG compression;
- random grayscale;
- ImageNet normalization.

These are scalar physical fields, not photographs.

Before finalizing rotation/flip augmentation, verify the relevant statistical symmetries of the map construction and compare no-augmentation versus symmetry augmentation.

---

# 24. Baseline Phase

Do not implement the proposed method first.

The baseline phase determines whether a paper exists.

---

# 25. B0 — Summary-Statistic Baseline

Deep learning must beat something simple.

For each map calculate simple summaries such as:

- mean;
- standard deviation;
- selected quantiles;
- histogram bins;
- optionally a 2D power-spectrum summary if implemented carefully.

Feed summary features into:

- linear regression;
- small MLP.

Purpose:

> determine how much information can be recovered without spatial representation learning.

---

# 26. B1 — Small CNN

Implement a compact, transparent CNN.

Goals:

- verify data pipeline;
- establish an inexpensive baseline;
- make debugging fast;
- support repeated experiments on a 12 GB GPU.

The initial encoder should output a feature vector:

\[
z \in \mathbb{R}^{d}
\]

with a configurable dimension such as:

```text
128
256
512
```

Do not start with a massive architecture.

---

# 27. B2 — CMD Benchmark-Style CNN

The original CMD paper describes a convolutional architecture and publishes benchmark code/weights.

The project should either:

1. reproduce the official architecture as closely as practical; or
2. clearly document why our reimplementation differs.

Official benchmark documentation:

https://camels-multifield-dataset.readthedocs.io/en/latest/inference.html

This provides an important external sanity check.

---

# 28. B3 — ResNet-Style Baseline

Use a moderate residual CNN configured for:

```text
1 input channel
256 × 256
2 regression outputs
```

Do not use pretrained ImageNet weights in the main comparison unless explicitly studying transfer learning.

A from-scratch baseline is cleaner for scientific fields.

---

# 29. B4 — DANN-Style Baseline

Architecture:

```text
                ┌── regression head -> Ωm, σ8
map -> encoder ─┤
                └── domain head -> TNG / SIMBA
                         ↑
                gradient reversal
```

Purpose:

- provide a classical adversarial domain-invariance baseline;
- connect to the MIEST-type principle;
- measure whether generic de-classification helps.

This is a baseline, not the paper contribution.

---

# 30. B5 — CORAL Baseline

Align source feature covariance statistics across domains.

Purpose:

> test whether simple second-order feature alignment is sufficient.

---

# 31. B6 — MMD Baseline

Use Maximum Mean Discrepancy to align source feature distributions.

This is particularly important because MMD has prior CAMELS use in DA-GNN.

Again:

> baseline, not novelty.

---

# 32. B7 — MIEST-Comparable Baseline

A complete reproduction may be difficult if the original implementation/data modality differs from ours.

At minimum:

- understand MIEST's loss and training structure;
- document exact differences;
- reproduce its central adversarial de-classification principle on our Mtot map protocol if feasible;
- avoid claiming superiority if the comparison is not apples-to-apples.

Create:

```text
reports/miest_reproduction_notes.md
```

with:

- data differences;
- target differences;
- split differences;
- preprocessing differences;
- architecture differences;
- loss differences.

---

# 33. Baseline Experiment Matrix

Use stable experiment IDs.

```text
E00  data smoke test
E01  TNG -> TNG small CNN
E02  SIMBA -> SIMBA small CNN
E03  TNG -> TNG benchmark CNN
E04  SIMBA -> SIMBA benchmark CNN

E10  TNG -> SIMBA ERM
E11  SIMBA -> TNG ERM

E12  TNG -> SIMBA DANN
E13  SIMBA -> TNG DANN

E14  TNG -> SIMBA CORAL
E15  SIMBA -> TNG CORAL

E16  TNG -> SIMBA MMD
E17  SIMBA -> TNG MMD

E20  TNG+SIMBA multisource ERM
E21  TNG+SIMBA DANN
E22  TNG+SIMBA CORAL
E23  TNG+SIMBA MMD
```

Astrid does not appear here.

That is intentional.

---

# 34. Baseline Acceptance Gate

Do not proceed to the proposed method until:

### Gate B1

In-domain model clearly learns the task.

### Gate B2

Cross-suite performance is measurably worse than in-domain performance.

### Gate B3

At least one reasonable domain-invariance baseline is implemented correctly.

### Gate B4

Results reproduce across at least two seeds before expensive sweeps.

---

# 35. What if Mtot Has Too Little Domain Shift?

This is a real possibility.

Total matter may be more robust than strongly baryonic fields.

Do not force the paper if:

```text
OOD error ≈ ID error
```

across TNG and SIMBA.

Predefine a criterion such as:

> if the OOD/ID error ratio is consistently below approximately 1.2–1.3 across both targets and both transfer directions, the shift may be too weak to motivate the intended method.

This numerical threshold is a project decision, not a community standard.

If the shift is too weak:

### Option A

Add a more baryon-sensitive field:

- gas density;
- temperature;
- HI.

### Option B

Use multifield input:

```text
Mtot + Mgas
```

or:

```text
Mtot + Mgas + T
```

### Option C

Reformulate the paper around **which representations remain robust as baryonic sensitivity increases**.

Do not merely cherry-pick the field with the biggest apparent improvement after many trials without reporting the selection process.

---

# 36. Representation Diagnostics

A domain-generalization paper needs to inspect what the model encodes.

---

# 37. Domain Probe

Freeze the encoder.

Extract:

\[
z_i = f(x_i)
\]

Train a small classifier on source data to predict:

```text
TNG
SIMBA
```

Metrics:

- accuracy;
- balanced accuracy;
- AUROC if binary.

Interpretation:

```text
high probe score:
representation contains simulator information

near-chance probe:
simulator is difficult to recover linearly/nonlinearly
```

But:

> near-chance domain accuracy is not sufficient evidence of a good representation.

A collapsed representation would also hide the domain.

Therefore always report cosmological predictive performance alongside domain leakage.

---

# 38. Target Probe

Freeze the representation and fit simple probes for:

```text
Ωm
σ8
```

Compare:

- linear probe;
- small nonlinear probe.

This measures how accessible target information is in latent space.

---

# 39. Representation Visualization

Use:

- PCA;
- UMAP as qualitative visualization only.

Create plots colored by:

```text
suite
Ωm
σ8
```

Desired qualitative pattern:

```text
suite colors increasingly mixed
while
Ωm / σ8 gradients remain organized
```

Do not use UMAP alone as quantitative evidence.

---

# 40. Information Trade-off Plot

One of the strongest paper figures could be:

```text
x-axis: simulator probe accuracy
y-axis: cosmology OOD error
```

Each model is one point:

```text
ERM
DANN
CORAL
MMD
MIEST-like
PPIRL variants
```

Ideal models move toward:

```text
less simulator information
+
lower OOD error
```

This makes the invariance trade-off visible.

---

# 41. Proposed Paired-Physics Dataset

After the baseline and novelty gates, download matched N-body Mtot data for **source suites**.

CMD documentation states that total-matter hydrodynamic and N-body maps can be matched spatially.

The paired dataset should produce:

```python
{
    "hydro_image": x_h,
    "nbody_image": x_n,
    "target": y,
    "suite_id": d,
    "simulation_id": s,
    "map_id": m,
}
```

with strict assertions that the pair shares:

- simulation index;
- map index;
- cosmological parameters;
- matching region correspondence.

---

# 42. Pair Validation

Before training, randomly inspect at least 50 matched pairs.

For each pair verify:

```text
same simulation_id
same map_id
same Ωm
same σ8
```

Generate visual comparison grids:

```text
Hydro Mtot | N-body Mtot | difference
```

Store them in:

```text
reports/pair_validation/
```

If map correspondence is not exact for the selected files, stop and fix the manifest before training.

---

# 43. Candidate Model

## 43.1 Shared encoder

Let:

\[
z_h = f_\theta(x_h)
\]

\[
z_n = f_\theta(x_n)
\]

The same encoder is preferred initially because both inputs are total-matter maps.

---

## 43.2 Regression head

\[
\hat{y}_h = g_\phi(z_h)
\]

Optionally:

\[
\hat{y}_n = g_\phi(z_n)
\]

Because both views share the same cosmological target.

---

## 43.3 Projection head

A separate projection head can be used for pair alignment:

\[
p_h = q(z_h)
\]

\[
p_n = q(z_n)
\]

This avoids forcing the exact regression representation to satisfy every contrastive/alignment constraint.

---

# 44. Base Regression Loss

Initial deterministic version:

\[
\mathcal{L}_{reg}
=
\mathrm{MSE}(\hat{y}, y)
\]

Because both targets are range-normalized, equal weighting is reasonable as a starting point.

Report physical-unit errors after inverse scaling.

Later, a probabilistic head can be added as a secondary experiment.

---

# 45. Pair Consistency Loss

Simplest candidate:

\[
\mathcal{L}_{pair}
=
1 -
\cos(p_h,p_n)
\]

for correctly matched hydro/N-body pairs.

However, simple pair collapse is a risk.

Therefore the final method should include either:

- supervised target loss strong enough to retain information;
- variance/covariance regularization;
- contrastive negatives;
- or another explicitly collapse-resistant formulation.

Do not assume pair cosine loss alone is sufficient.

---

# 46. Optional Variance Regularization

For a batch representation \(Z\), encourage every latent dimension to retain nonzero variation.

A VICReg-style principle can be investigated:

```text
pair invariance
+
feature variance
+
feature covariance reduction
```

If used, cite the original method and clearly separate borrowed components from the new contribution.

---

# 47. Conditional Cross-Simulator Alignment

A second candidate component is **continuous-label-conditioned alignment**.

Problem with naive domain alignment:

```text
TNG Ωm=.11, σ8=.95
and
SIMBA Ωm=.48, σ8=.62
```

should not necessarily have identical representations just because they are from different suites.

Instead define closeness in target space:

\[
d_y(i,j)
=
\left\|
\tilde{y}_i-\tilde{y}_j
\right\|_2
\]

and a soft affinity:

\[
w_{ij}
=
\exp
\left(
-\frac{d_y(i,j)^2}{2\tau_y^2}
\right)
\]

Then cross-suite samples with similar cosmological parameters receive stronger alignment.

This is a **candidate mechanism**, not yet a novelty claim.

---

# 48. Provisional Full Objective

A candidate objective is:

\[
\mathcal{L}
=
\mathcal{L}_{reg,h}
+
\alpha \mathcal{L}_{reg,n}
+
\beta \mathcal{L}_{pair}
+
\gamma \mathcal{L}_{cond}
+
\delta \mathcal{L}_{var}
\]

where:

- \(\mathcal{L}_{reg,h}\): hydro cosmology regression;
- \(\mathcal{L}_{reg,n}\): optional N-body regression;
- \(\mathcal{L}_{pair}\): matched hydro/N-body consistency;
- \(\mathcal{L}_{cond}\): conditional cross-source alignment;
- \(\mathcal{L}_{var}\): anti-collapse regularization.

Do not launch a 4-dimensional loss-weight sweep immediately.

Ablate components incrementally.

---

# 49. Minimal Method Development Sequence

## M0

Hydro-only ERM.

## M1

Hydro + N-body samples, no pair information.

Purpose:

> Does simply adding N-body training data help?

## M2

Correct hydro/N-body pair alignment.

Purpose:

> Does pair structure add value?

## M3

Shuffled-pair control.

Purpose:

> Is correct correspondence actually important?

## M4

Conditional source-domain alignment only.

## M5

Pair alignment + conditional alignment.

Only proceed to more complex formulations if M2/M4 produce evidence worth combining.

---

# 50. Critical Negative Control — Shuffled Pairs

Create a training condition in which N-body maps are randomly reassigned within the source suite while preserving approximately similar target distributions.

Compare:

```text
correct pair
vs
shuffled pair
```

If they perform the same, the claimed pair mechanism is unsupported.

This control is extremely important for the paper.

---

# 51. Critical Negative Control — Extra Data

To separate:

```text
benefit from extra N-body examples
```

from:

```text
benefit from matched pair structure
```

include:

### Hydro only

### Hydro + unpaired N-body multitask

### Hydro + paired N-body alignment

Only the third uses correspondence.

---

# 52. Final Target Protocol

After choosing the method **without Astrid metrics**, train final models on:

```text
TNG train + SIMBA train
```

validate on:

```text
TNG val + SIMBA val
```

then freeze.

Evaluate:

```text
Astrid
```

No target fine-tuning.

No Astrid unlabeled adaptation.

No target-specific normalization in the primary result.

---

# 53. Secondary Leave-One-Suite-Out Evaluation

After the primary Astrid run, use the **same frozen method design and hyperparameters** for:

```text
TNG + SIMBA -> Astrid

TNG + Astrid -> SIMBA

SIMBA + Astrid -> TNG
```

These are secondary consistency experiments.

If a hyperparameter is changed per target suite, report that as a separate tuned condition.

---

# 54. Evaluation Granularity

Each simulation has 15 related maps.

Report two levels.

## Map-level

Every map receives a prediction.

Useful for comparison with map-based literature.

## Simulation-level

Average the 15 map predictions:

\[
\hat{y}^{(sim)}
=
\frac{1}{15}
\sum_{m=1}^{15}
\hat{y}^{(m)}
\]

This estimates what the model predicts when multiple views of the same simulation are available.

Do not mix map-level and simulation-level metrics in one table without labeling them.

---

# 55. Main Regression Metrics

For each target independently:

## MAE

\[
MAE
=
\frac1N
\sum_i
|y_i-\hat{y}_i|
\]

## RMSE

\[
RMSE
=
\sqrt{
\frac1N
\sum_i
(y_i-\hat{y}_i)^2
}
\]

## \(R^2\)

Report for compatibility, but do not use alone.

## Mean relative error

Can be reported because both targets remain safely away from zero.

---

# 56. Domain-Generalization Metrics

## Generalization ratio

\[
G =
\frac{E_{OOD}}{E_{ID}}
\]

where \(E\) can be MAE or RMSE.

## Absolute transfer penalty

\[
\Delta E
=
E_{OOD} - E_{ID}
\]

## Worst-domain error

For multi-target evaluation:

\[
E_{worst} = \max_d E_d
\]

This prevents a method from appearing robust by averaging one excellent and one disastrous domain.

---

# 57. Statistical Confidence

Because 15 maps from one simulation are correlated, **do not bootstrap individual maps as independent samples.**

Bootstrap by simulation ID.

Procedure:

1. sample simulation IDs with replacement;
2. include all associated map predictions;
3. recompute the metric;
4. repeat e.g. 2,000 bootstrap replicates;
5. report 95% confidence intervals.

The exact number of bootstrap replicates can be chosen based on convergence/cost; document it.

---

# 58. Multiple Random Seeds

Development:

```text
2 seeds for quick screening
```

Strong candidate:

```text
3 seeds
```

Final result:

```text
5 seeds preferred
```

Report:

```text
mean ± standard deviation across training seeds
```

plus simulation-level bootstrap intervals where appropriate.

These quantify different uncertainty sources:

- seed variance;
- test-simulation sampling variance.

---

# 59. Hyperparameter Policy

Avoid massive search.

Primary hyperparameters:

- learning rate;
- weight decay;
- batch size;
- encoder width;
- latent dimension;
- alignment coefficient(s).

Use source validation only.

Do not run hundreds of trials on the target domain.

A paper about robustness is weakened by hidden target-driven tuning.

---

# 60. Compute Strategy for a 12 GB GPU

Prioritize:

```text
small/medium CNN
batch-wise memory-mapped data
mixed precision where numerically stable
gradient accumulation if required
```

Do not start with:

```text
ViT-L
Swin-L
large diffusion encoder
```

The research value comes from experiment quality, not parameter count.

---

# 61. Batch Design for Multi-Source Training

A batch should be balanced by source suite.

Example:

```text
batch size 32

16 TNG
16 SIMBA
```

For paired training:

```text
16 TNG hydro/Nbody pairs
16 SIMBA hydro/Nbody pairs
```

This prevents one source from dominating the domain loss due only to batch composition.

---

# 62. Training Checkpoint Policy

Save:

```text
best source-validation checkpoint
last checkpoint
```

Selection metric should be fixed before training.

Potential selection metric:

\[
S =
\frac12
\left(
NRMSE_{\Omega_m}
+
NRMSE_{\sigma_8}
\right)
\]

on source validation.

Do not select checkpoints using target performance.

---

# 63. Early Stopping

Early stopping uses source validation only.

If training multiple source domains, validation metric should aggregate them with equal domain weight rather than proportional sample count.

Example:

\[
L_{val}
=
\frac12 L_{TNG}
+
\frac12 L_{SIMBA}
\]

---

# 64. Experiment Logging

Every run must save:

```text
run_id
timestamp
git commit
config snapshot
random seed
GPU name
software versions
data manifest hash
split file hash
training metrics
validation metrics
checkpoint path
```

Store machine-readable output in:

```text
outputs/runs/<run_id>/run.json
```

and:

```text
outputs/runs/<run_id>/metrics.json
```

---

# 65. Experiment Naming

Use deterministic names:

```text
E20_erm_tng-simba_seed00
E20_erm_tng-simba_seed01

E30_ppirl_pair_tng-simba_seed00
```

Never use:

```text
final2
final_best
final_really_best
test_new
```

---

# 66. Test Suite

## 66.1 Data tests

- file exists;
- expected map shape;
- expected parameter shape;
- no NaN/Inf;
- map-to-simulation mapping correct.

## 66.2 Split tests

- no simulation overlap;
- exactly 15 maps per selected simulation;
- deterministic split.

## 66.3 Leakage tests

- target suite absent from training manifest;
- target labels never passed to trainer;
- normalization stats contain only source training IDs.

## 66.4 Paired tests

- hydro and N-body pair IDs identical;
- cosmological labels identical;
- pair mapping one-to-one.

## 66.5 Model tests

- output shape `(batch, 2)`;
- finite loss;
- gradient reaches encoder;
- domain head gradient behavior correct for adversarial baseline.

## 66.6 Metrics tests

Test formulas against small arrays with hand-computed results.

---

# 67. Leakage Guardrails

Create a dedicated object:

```text
ExperimentProtocol
```

containing:

```text
source_suites
target_suites
allow_target_unlabeled
allow_target_statistics
allow_target_labels
```

For primary DG:

```text
allow_target_unlabeled = False
allow_target_statistics = False
allow_target_labels = False
```

The trainer should reject a dataset that violates the protocol.

This turns methodological discipline into executable code.

---

# 68. Phase-by-Phase Implementation

# Phase 0 — Literature Audit

Deliverable:

```text
reports/novelty_audit.md
```

Search at minimum:

```text
CAMELS robustness
CAMELS domain adaptation
CAMELS domain generalization
CAMELS simulator invariant
CAMELS MIEST
CAMELS N-body hydrodynamic paired learning
CAMELS multifidelity inference
CAMELS contrastive representation
CAMELS self-supervised cosmology
CAMELS baryonic invariant representation
```

Read:

- abstracts;
- methods;
- data modality;
- evaluation protocol;
- source/target domains;
- claimed novelty.

Create table:

| Paper | Input | Sources | Target | Method | Uses target data? | Main limitation | Overlap |
|---|---|---|---|---|---|---|---|

**Exit criterion:** the proposed contribution can be stated in one sentence that is not already implemented by a prior paper.

---

# Phase 1 — Data

Tasks:

1. download TNG Mtot + params;
2. download SIMBA Mtot + params;
3. download Astrid Mtot + params;
4. create manifest;
5. validate shape;
6. create simulation-level split;
7. visualize random samples;
8. calculate pixel distributions;
9. verify positivity before log transform.

Deliverable:

```text
reports/data_report.md
```

---

# Phase 2 — Loader and Smoke Model

Implement:

```text
CAMELSMapDataset
```

Run on:

```text
100 simulations or fewer
```

for a few epochs.

Purpose:

- verify GPU path;
- verify loss decreases;
- verify checkpointing;
- verify no RAM explosion;
- benchmark data throughput from G:.

Do not interpret smoke-run metrics scientifically.

---

# Phase 3 — Reproduce In-Domain Inference

Train baseline on:

```text
TNG train
```

evaluate:

```text
TNG test
```

Repeat SIMBA.

Compare general behavior against CMD literature.

If the model cannot infer the targets at all, debug:

- preprocessing;
- target mapping;
- normalization;
- split;
- optimizer;
- architecture.

---

# Phase 4 — Quantify Simulator Shift

Run:

```text
TNG -> SIMBA
SIMBA -> TNG
```

Generate:

```text
ID vs OOD error table
scatter predicted vs true
error vs Ωm
error vs σ8
residual histograms
domain probe accuracy
PCA/UMAP
```

Deliverable:

```text
reports/baseline_report.md
```

This phase determines whether Mtot is a strong enough testbed.

---

# Phase 5 — Strong Baselines

Implement and evaluate:

```text
ERM
DANN
CORAL
MMD
MIEST-comparable
```

Use exactly the same:

- backbone where possible;
- split;
- source data;
- evaluation metric;
- training budget.

This prevents architecture size from masquerading as method improvement.

---

# Phase 6 — Novelty Gate

Before PPIRL coding:

1. update literature search to current date;
2. search N-body/hydro paired representation learning;
3. inspect CAMELS publications page;
4. inspect citations of MIEST and DA-GNN;
5. search ML4PS / NeurIPS / ICLR / ICML / astro-ph;
6. decide whether the paired-physics formulation is still defensible.

Possible outcomes:

### A — gap confirmed

Proceed.

### B — close prior work exists

Modify contribution.

### C — method already exists

Do not build the same paper.

Return to a different hypothesis.

---

# Phase 7 — Download Source N-body Pairs

Download only the paired N-body total-matter data necessary for source suites.

Do not download N-body Astrid for the primary method unless an explicit experiment requires it.

Validate correspondence rigorously.

---

# Phase 8 — Pair-Only Model

Implement:

```text
hydro regression
+
paired hydro/Nbody representation consistency
```

No conditional alignment yet.

Compare:

```text
hydro only
hydro + unpaired Nbody
hydro + paired Nbody
hydro + shuffled Nbody pair
```

If correct pairing offers no advantage, stop before adding complexity.

---

# Phase 9 — Conditional Alignment

Only after pair-only evidence.

Implement soft target-conditioned source alignment.

Ablate:

```text
no conditional loss
fixed τy
multiple τy values
```

Use source validation/cross-source transfer only.

---

# Phase 10 — Full Candidate Method

Combine only components that individually earned inclusion.

Do not construct a six-loss “kitchen sink” model without evidence.

Run:

```text
3 seeds
TNG -> SIMBA
SIMBA -> TNG
TNG+SIMBA source-validation
```

Select final method.

---

# Phase 11 — Freeze

Create:

```text
reports/final_protocol.md
```

containing:

```text
git commit hash
data manifest hash
split hash
model config
training epochs
optimizer
learning rate
loss coefficients
normalization
seeds
metrics
plots to be generated
```

Commit it.

Then no more method changes before Astrid evaluation.

---

# Phase 12 — Final Astrid Evaluation

Train:

```text
TNG + SIMBA
```

Test:

```text
Astrid
```

for all final seeds.

Do not rerun with altered hyperparameters after seeing results unless the paper explicitly labels the next experiment as target-informed.

---

# Phase 13 — Secondary Rotated Evaluation

Use frozen settings:

```text
TNG + Astrid -> SIMBA
SIMBA + Astrid -> TNG
```

This checks whether the method only happened to favor Astrid.

---

# Phase 14 — Robustness

Only after the core claim is established.

Possible perturbations:

## Resolution

```text
256x256
128x128
64x64
```

## Additive noise

Use explicitly defined noise levels relative to source-training statistics.

## Blur

Use documented kernels.

## Masking

Mask spatial blocks or random pixels.

## Parameter extremes

Bin performance by:

```text
low / mid / high Ωm
low / mid / high σ8
```

Do not imply these synthetic perturbations exactly model an astronomical instrument unless they actually do.

---

# Phase 15 — Data-Efficiency Ablation

Train with:

```text
100 simulations/domain
250
500
900
```

Question:

> Does the invariant method help more when source coverage is limited?

This can strengthen the ML contribution.

---

# Phase 16 — Representation-Dimension Ablation

Test:

```text
64
128
256
512
```

Measure:

- ID error;
- OOD error;
- simulator probe;
- target probe.

This can reveal whether simulator information grows with excess representation capacity.

---

# Phase 17 — Loss Ablation Table

Required table:

| Variant | Hydro Reg | Nbody Reg | Pair | Conditional | Variance | OOD Ωm | OOD σ8 |
|---|---:|---:|---:|---:|---:|---:|---:|
| ERM | ✓ |  |  |  |  |  |  |
| + Nbody unpaired | ✓ | ✓ |  |  |  |  |  |
| + pair | ✓ | ✓ | ✓ |  |  |  |  |
| shuffled pair | ✓ | ✓ | ✓ shuffled |  |  |  |  |
| + conditional | ✓ |  |  | ✓ |  |  |  |
| full | ✓ | ✓ | ✓ | ✓ | ✓ |  |  |

Do not omit failed variants.

---

# 69. Paper Figures

Pre-plan the figures.

## Figure 1 — Problem

Representative maps:

```text
TNG
SIMBA
Astrid
```

with similar parameter values if possible.

Purpose:

> visually show domain variation.

---

## Figure 2 — Experimental Protocol

Diagram:

```text
TNG + SIMBA
    ↓
training
    ↓
frozen model
    ↓
ASTRID
```

No target adaptation.

---

## Figure 3 — Baseline Generalization Gap

Matrix:

```text
train suite × test suite
```

for ERM.

---

## Figure 4 — Method Diagram

```text
hydro Mtot ── encoder ── regression
     │            │
     │            └── representation
     │
paired Nbody ─ encoder
                  │
            pair consistency
```

---

## Figure 5 — ID/OOD Regression

Predicted versus true:

```text
Ωm
σ8
```

for ERM and proposed method.

---

## Figure 6 — Simulator Leakage

Domain-probe score:

```text
ERM
DANN
MMD
MIEST-like
Pair
Full
```

alongside target error.

---

## Figure 7 — Latent Geometry

PCA/UMAP:

- color by simulator;
- color by Ωm;
- color by σ8.

---

## Figure 8 — Pair Ablation

Correct pair vs shuffled pair vs unpaired N-body.

This may become one of the most important mechanistic figures.

---

# 70. Paper Tables

## Table 1

Dataset and split summary.

## Table 2

In-domain baseline results.

## Table 3

Cross-source transfer.

## Table 4

Final unseen Astrid performance.

## Table 5

Loss ablation.

## Table 6

Domain-probe/target-probe results.

## Table 7

Robustness/data-efficiency results if included.

---

# 71. Failure Analysis

For every final model, inspect errors against:

```text
Ωm
σ8
A_SN1
A_AGN1
A_SN2
A_AGN2
```

Even though the astrophysical parameters are not targets, they may reveal where the model fails.

For example:

> OOD errors may spike for extreme feedback settings.

This is useful because it connects a machine-learning failure mode to the nuisance variables that produce the shift.

Do not interpret astrophysical causality beyond what the analysis supports.

---

# 72. Error-Surface Analysis

Create 2D bins over:

```text
Ωm × σ8
```

Plot average residual.

Questions:

- Does the model regress toward the center?
- Are corners of parameter space harder?
- Does domain generalization fail asymmetrically?
- Does the proposed method improve only the center?

A single global RMSE can hide these effects.

---

# 73. Nuisance-Sensitivity Analysis

For fixed approximate cosmology, examine error versus feedback parameters.

The main ML question:

> Is the representation invariant to simulator/nuisance variation while remaining sensitive to cosmology?

This is more informative than domain label alone.

---

# 74. Optional Probabilistic Extension

Do **not** make this core until deterministic DG works.

A probabilistic head can predict:

```text
μΩ, log σ²Ω
μσ8, log σ²σ8
```

using Gaussian NLL.

Then evaluate:

- NLL;
- 68% interval coverage;
- 95% interval coverage;
- calibration versus simulator.

This could form a second paper or substantial extension:

> Does domain invariance improve predictive calibration under simulator shift?

---

# 75. Optional Multifield Extension

After Mtot:

```text
Mtot + Mgas
Mtot + temperature
Mtot + Mgas + temperature
```

CMD states fields at the same map index can correspond spatially, allowing channel stacking where supported.

The potential ML question becomes:

> Can a model use a robust gravity-dominated anchor while selectively exploiting simulator-sensitive baryonic channels?

Possible architecture:

```text
Mtot ── invariant encoder ─┐
Mgas ── field encoder ─────┼─ gated fusion -> regression
T    ── field encoder ─────┘
```

This is not part of minimum viable paper.

---

# 76. Minimum Viable Paper

A publishable first version should require only:

### Data

```text
TNG Mtot
SIMBA Mtot
Astrid Mtot
source N-body pair files if paired method survives audit
```

### Models

```text
ERM
DANN
MMD or CORAL
MIEST-comparable baseline
one proposed method
```

### Experiments

```text
in-domain
cross-source
unseen Astrid
domain probe
pair ablations
3–5 seeds
simulation-level confidence intervals
```

Anything beyond that is optional.

---

# 77. Kill Criteria

A research project needs explicit conditions under which we stop or pivot.

## Kill 1

Baseline cannot learn in-domain task after verified pipeline.

Interpretation:

> implementation issue or unsuitable representation.

Do not proceed.

---

## Kill 2

Mtot has negligible cross-suite shift.

Interpretation:

> the chosen data does not expose the intended ML weakness strongly enough.

Pivot to a more baryon-sensitive field or multifield setup.

---

## Kill 3

Strong published baseline already solves the selected setting.

Interpretation:

> novelty insufficient.

Change the research question before investing further.

---

## Kill 4

Paired N-body training gives identical results to shuffled/unpaired controls.

Interpretation:

> pairing mechanism unsupported.

Drop the paired claim.

---

## Kill 5

Method improves one seed or one transfer direction only.

Interpretation:

> likely instability/cherry-picking.

Do not present it as general improvement.

---

## Kill 6

Improvement requires target statistics or target unlabeled maps.

Interpretation:

> result is adaptation, not strict DG.

It can still be research, but must be relabeled correctly.

---

# 78. Research Integrity Rules

1. **No target-driven hyperparameter tuning** in primary DG.
2. **No map-level leakage** between splits.
3. **No best-seed reporting.**
4. **No hidden failed baselines.**
5. **No novelty claim before audit.**
6. **No physical claim from representation visualization alone.**
7. **No silent target normalization.**
8. **No changing primary metric after results are known.**
9. **No using Astrid repeatedly as a validation set while calling it unseen.**
10. **Archive exact final configs.**

---

# 79. Reproducibility Checklist

Before submission, a new machine should be able to:

1. clone repository;
2. create environment;
3. set `CAMELS_DATA_ROOT`;
4. validate data;
5. rebuild splits;
6. run a tiny smoke experiment;
7. train a baseline;
8. evaluate from checkpoint;
9. regenerate paper tables;
10. regenerate paper figures.

Raw data should not be required to run unit tests.

---

# 80. README Requirements

The public repository README should include:

- one-sentence research question;
- architecture/problem figure;
- key result table;
- exact dataset filenames;
- data download instructions;
- no raw data in repo;
- reproducibility commands;
- paper/preprint link when available;
- citation;
- limitations;
- hardware used.

Avoid a README that is mostly installation commands.

The first screen should communicate the research problem.

---

# 81. Proposed CLI Surface

These are project-defined interfaces to implement.

```powershell
python scripts/validate_data.py --config configs/data/mtot.yaml
```

```powershell
python scripts/build_splits.py --seed 42
```

```powershell
python scripts/run_experiment.py --config configs/experiments/e01_tng_id.yaml
```

```powershell
python scripts/evaluate_checkpoint.py --run E01_tng_id_seed00
```

```powershell
python scripts/run_domain_probe.py --run E20_erm_tng-simba_seed00
```

```powershell
python scripts/make_paper_figures.py --manifest reports/final_runs.json
```

The exact CLI implementation is ours; these commands are a design contract.

---

# 82. Configuration Example

```yaml
experiment:
  id: E20_erm_tng_simba
  seed: 42

data:
  field: Mtot
  set: LH
  redshift: 0.0
  source_suites:
    - IllustrisTNG
    - SIMBA
  target_suites:
    - Astrid
  maps_per_simulation: 15

protocol:
  use_target_for_training: false
  use_target_for_validation: false
  use_target_statistics: false

preprocessing:
  log_transform: true
  normalization: source_train_global

targets:
  names:
    - omega_m
    - sigma8
  scaling: fixed_lh_range

model:
  type: small_cnn
  latent_dim: 256

training:
  batch_size: 32
  epochs: 100
  optimizer: adamw
  learning_rate: 0.0003
  weight_decay: 0.0001

evaluation:
  map_level: true
  simulation_level: true
  bootstrap_by_simulation: true
```

The numeric hyperparameters above are **initial configuration examples**, not literature-backed optimum values.

They must be tuned on source validation.

---

# 83. Data Throughput Benchmark

Because data resides on `G:` while code is on `C:`, run an I/O benchmark.

Measure:

```text
maps/second with num_workers=0
maps/second with num_workers=2
maps/second with num_workers=4
GPU utilization
```

On Windows, start with conservative worker settings and increase only after confirming stable memory-mapped loading.

If `G:` is an HDD and GPU utilization is poor:

- use larger contiguous reads;
- cache a working subset on SSD if space permits;
- reduce random access;
- profile before rewriting loader.

Do not assume the GPU is the bottleneck.

---

# 84. Training-Time Budget

Before full experiments, measure one epoch.

Record:

```text
training seconds/epoch
validation seconds
peak VRAM
host RAM
```

Then estimate:

```text
baseline run hours
× seeds
× methods
```

This prevents designing an experiment matrix that takes months unexpectedly.

---

# 85. Experiment Prioritization

Order experiments by information value.

### High priority

```text
ERM
cross-suite gap
DANN/MMD
domain probe
paired-vs-shuffled
final Astrid
```

### Medium priority

```text
conditional alignment
data efficiency
latent size
robustness
```

### Low priority initially

```text
large architecture sweep
multifield
probabilistic posterior
many augmentations
3D data
```

---

# 86. Paper-Level Novelty Audit Checklist

Before naming the method, search:

## Exact conceptual terms

```text
paired simulator invariant representation learning
paired fidelity contrastive learning scientific simulation
low fidelity high fidelity contrastive learning
hydrodynamic n-body contrastive
hydro n-body representation alignment
CAMELS paired N-body neural inference
baryonic invariant representation CAMELS
cross-simulator cosmological inference
```

## Existing CAMELS papers

Use:

https://camels.readthedocs.io/en/latest/publications.html

## Conferences

Search:

- NeurIPS;
- ICML;
- ICLR;
- ML4PS;
- AI4Science;
- astro-ph.CO;
- cs.LG.

The novelty audit should be rerun immediately before paper submission.

---

# 87. What Counts as a Contribution?

A defensible paper contribution could be:

> We identify a failure mode in global simulator-invariance objectives for continuous scientific regression and introduce a paired-physics representation objective that uses matched low-/high-fidelity simulations to suppress nuisance-specific information while retaining target geometry.

This is stronger than:

> We apply contrastive learning to cosmology.

The final wording depends entirely on the evidence.

---

# 88. What Would Be a Strong Result?

Illustrative only:

| Model | ID RMSE | Astrid OOD RMSE | Domain Probe |
|---|---:|---:|---:|
| ERM | 0.020 | 0.070 | 96% |
| DANN | 0.024 | 0.055 | 67% |
| MMD | 0.023 | 0.052 | 61% |
| MIEST-like | 0.025 | 0.047 | 50% |
| Paired method | 0.021 | 0.036 | 44% |

These values are **fictional examples**.

A strong real result would show:

1. lower unseen-domain error;
2. no major ID collapse;
3. repeatability across seeds;
4. correct-pair advantage over shuffled pair;
5. reduced simulator leakage;
6. retained cosmology information.

---

# 89. What Would Be an Interesting Negative Result?

Negative results can still be useful if analyzed properly.

Example:

> Strong simulator de-classification reduces domain-probe accuracy to chance but worsens unseen-domain regression because the removed representation components also contain cosmological information.

That would demonstrate:

> **domain invariance and task invariance are not equivalent.**

Another:

> N-body pair alignment improves \(\Omega_m\) but harms \(\sigma_8\), indicating the two targets depend differently on baryonic small-scale structure.

That could motivate a target-specific representation decomposition.

---

# 90. Potential Follow-Up if Pairing Works

Split representation:

\[
z =
[z_{inv}, z_{res}]
\]

where:

- \(z_{inv}\): aligned with N-body pair;
- \(z_{res}\): allowed to capture hydro-specific residuals.

Regression can learn to use both with regularization.

Goal:

> avoid throwing away potentially useful baryonic information while isolating a robust cosmological core.

This is a stronger but more complex second-stage method.

---

# 91. Potential Follow-Up — Mixture of Experts

Use:

```text
invariant expert
+
source-specific experts
+
gating network
```

But be cautious:

A source-specific expert may not generalize to a new target.

This should only be attempted after the invariant core is understood.

---

# 92. Potential Follow-Up — Uncertainty

A model could output both:

```text
prediction
uncertainty
```

and ask:

> Does predictive uncertainty increase on unseen simulators?

If uncertainty remains confident while error explodes, the model is not trustworthy under shift.

This is a valuable future direction but not necessary for the first result.

---

# 93. Potential Follow-Up — Second-Generation CAMELS

Current CAMELS documentation reports a 2026 second-generation IllustrisTNG release with larger boxes and a 35-dimensional parameter space.

This could later provide an additional stress test:

```text
first-generation source
    ↓
second-generation TNG
```

But this should not complicate the initial paper unless the core method is already stable.

---

# 94. Minimal Cosmology Knowledge Required

You do not need graduate-level cosmology to implement the ML paper, but you must understand enough to avoid invalid experimental design.

Required:

- meaning of \(\Omega_m\);
- meaning of \(\sigma_8\);
- cosmic variance;
- hydrodynamic vs N-body simulation;
- baryonic feedback;
- why TNG/SIMBA/Astrid differ;
- what a projected matter-density map represents;
- why the four feedback parameter names are not cross-suite equivalent;
- why 15 maps from one simulation are correlated;
- why N-body counterparts can act as a physics-controlled view.

Do not make domain claims beyond this knowledge without expert review.

---

# 95. Recommended Collaboration Strategy

The ML contribution can remain yours while getting domain review.

Ideal workflow:

- you own hypothesis;
- you own implementation;
- you own ML experiments;
- you own ablations;
- you draft paper;
- an astrophysics/cosmology researcher reviews:
  - data interpretation;
  - physics assumptions;
  - evaluation validity;
  - scientific language.

This strengthens the paper without changing first-author ownership if you are the primary researcher.

---

# 96. Timeline

This is a planning estimate, not a promise.

## Weeks 1–2

```text
literature audit
data download
loader
validation
EDA
```

## Weeks 3–4

```text
in-domain baseline
cross-suite baseline
representation probes
```

## Weeks 5–6

```text
DANN
CORAL
MMD
MIEST-comparable study
```

## Week 7

```text
novelty gate
pair-data validation
```

## Weeks 8–9

```text
paired method
negative controls
```

## Weeks 10–11

```text
conditional alignment
ablation
3 seeds
```

## Week 12

```text
freeze protocol
Astrid final evaluation
```

## Weeks 13–14

```text
robustness
statistics
figures
```

## Weeks 15–16

```text
paper writing
code cleanup
reproducibility pass
```

If results do not support the hypothesis, extend the research period rather than manufacturing a positive conclusion.

---

# 97. Milestones

## M1 — Data Valid

Success:

```text
all 3 hydro files validated
loader reads via mmap
split tests pass
```

## M2 — Baseline Learns

Success:

```text
clear in-domain signal
```

## M3 — Shift Exists

Success:

```text
cross-suite degradation demonstrated
```

## M4 — Baselines Complete

Success:

```text
ERM + DANN + MMD/CORAL implemented
```

## M5 — Novelty Confirmed

Success:

```text
literature audit supports proposed gap
```

## M6 — Mechanism Supported

Success:

```text
correct paired training beats shuffled/unpaired control
```

## M7 — Final Target Frozen

Success:

```text
Astrid evaluated after configuration lock
```

## M8 — Reproducible Paper

Success:

```text
tables/figures regenerate from saved runs
```

---

# 98. Final Experiment Registry

Before paper submission create:

```text
reports/final_runs.json
```

with exact run IDs used for every table and figure.

Example:

```json
{
  "table_4": {
    "erm": [
      "E20_erm_seed00",
      "E20_erm_seed01",
      "E20_erm_seed02",
      "E20_erm_seed03",
      "E20_erm_seed04"
    ],
    "proposed": [
      "E30_ppirl_seed00",
      "E30_ppirl_seed01",
      "E30_ppirl_seed02",
      "E30_ppirl_seed03",
      "E30_ppirl_seed04"
    ]
  }
}
```

No manual copy-pasting results from random notebooks.

---

# 99. Paper Draft Structure

## Abstract

Four components:

1. problem;
2. method;
3. experimental protocol;
4. main quantitative result.

## 1. Introduction

- simulator shift;
- scientific ML;
- why source-specific shortcut learning is dangerous;
- contribution bullets.

## 2. Related Work

- CAMELS inference;
- domain adaptation;
- MIEST;
- representation invariance;
- multifidelity learning;
- paired-view learning.

## 3. Dataset

- suites;
- Mtot;
- targets;
- pairing;
- split protocol.

## 4. Method

- ERM;
- paired representation;
- optional conditional alignment;
- losses.

## 5. Experiments

- source domains;
- target domain;
- preprocessing;
- metrics;
- baselines;
- seeds.

## 6. Results

- ID;
- OOD;
- probes;
- ablations;
- final target.

## 7. Analysis

- representation;
- nuisance sensitivity;
- failure regions.

## 8. Limitations

Examples:

- simulated data only;
- limited fields;
- limited suites;
- no observational domain;
- only two cosmological targets;
- compute constraints;
- no theorem guaranteeing invariance.

## 9. Conclusion

State only supported claims.

---

# 100. Reproducibility Artifacts for Release

Release:

```text
source code
configs
split IDs
environment lock
small synthetic/smoke data generator
metric scripts
figure scripts
pretrained weights where practical
experiment manifest
```

Do not redistribute CAMELS data unless its license and distribution conditions explicitly permit the intended method; instead link to the official source.

---

# 101. Git Ignore

At minimum:

```gitignore
data/
*.npy
*.hdf5
*.pt
*.pth
outputs/checkpoints/
outputs/runs/
.env
__pycache__/
.pytest_cache/
```

If small final metrics/plots are intended for Git, whitelist only those deliberately.

---

# 102. Data Citation

The paper/repository must cite the CAMELS Multifield Dataset paper.

Official citation page:

https://camels.readthedocs.io/en/latest/citation.html

CMD paper DOI:

https://doi.org/10.3847/1538-4365/ac5ab0

Also cite relevant CAMELS simulation-suite papers according to the data actually used.

---

# 103. Research Notes

This implementation plan was constructed after checking the current CAMELS/CMD documentation and the closest obvious robustness work.

## Primary dataset/documentation sources

### CAMELS current documentation

https://camels.readthedocs.io/en/latest/

### CAMELS current parameters

https://camels.readthedocs.io/en/latest/parameters.html

### CAMELS organization/suites

https://camels.readthedocs.io/en/latest/suites_sets.html

### CAMELS current data access

https://camels.readthedocs.io/en/latest/data_access.html

### CAMELS GitHub

https://github.com/franciscovillaescusa/CAMELS

### CMD documentation

https://camels-multifield-dataset.readthedocs.io/en/latest/

### CMD data structure

https://camels-multifield-dataset.readthedocs.io/en/latest/data.html

### CMD access

https://camels-multifield-dataset.readthedocs.io/en/latest/access.html

### CMD parameter-inference challenge

https://camels-multifield-dataset.readthedocs.io/en/latest/inference.html

---

## Core papers

### CAMELS project

Villaescusa-Navarro et al.

https://arxiv.org/abs/2010.00619

### CAMELS Multifield Dataset

Villaescusa-Navarro et al., 2022.

DOI:

https://doi.org/10.3847/1538-4365/ac5ab0

Citation information:

https://camels.readthedocs.io/en/latest/citation.html

### Astrid/CAMELS expansion

Ni et al.

https://arxiv.org/abs/2304.02096

---

## Closest robustness/domain papers

### Domain Adaptive Graph Neural Networks

Roncoli et al., 2023.

https://arxiv.org/abs/2311.01588

### MIEST / robustness across simulation models

Jo et al., 2025.

https://arxiv.org/abs/2502.13239

https://doi.org/10.3847/1538-4357/adec78

These papers are essential because they mean a generic “domain-invariant CAMELS network” is no longer sufficient novelty.

---

# 104. Immediate Next Actions

Do these in order.

## Step 1

Create:

```text
C:\projects\UniverseInference
```

and:

```text
G:\datasets\CAMELS_CMD
```

## Step 2

Download only:

```text
TNG Mtot LH
SIMBA Mtot LH
Astrid Mtot LH
their params files
```

## Step 3

Implement:

```text
validate_data.py
```

before any model.

## Step 4

Implement simulation-level splits.

## Step 5

Implement memory-mapped dataset.

## Step 6

Train a small in-domain TNG baseline.

## Step 7

Run:

```text
TNG -> SIMBA
SIMBA -> TNG
```

## Step 8

Measure the actual generalization gap.

## Step 9

Complete the paper-level novelty audit, especially MIEST and paired N-body/multifidelity literature.

## Step 10

Only if the gap and novelty survive:

```text
download matched source N-body Mtot
implement paired-physics experiments
```

---

# 105. Bottom Line

The project should not be framed as:

> “AI predicts cosmological parameters from images.”

That is established work.

It should not even be framed only as:

> “AI learns simulator-invariant representations.”

That is now too close to existing MIEST and domain-adaptation work.

The strongest current formulation is:

> **Use the controlled structure of scientific simulations—especially matched hydrodynamic and gravity-only views—to develop and test a representation-learning method that isolates task-relevant information from simulator-specific nuisance variation, then evaluate it under strict zero-shot simulator shift.**

CAMELS is the benchmark.

The actual research contribution must be the **machine-learning mechanism and its evidence**.

The project succeeds only if the final method is:

1. more than an application;
2. properly compared to existing robustness methods;
3. tested without target leakage;
4. supported by ablations;
5. reproducible;
6. honest about negative results and limitations.

That is the standard to build toward for a credible first research paper.
