# NeuralCosmos — Progress Log

Running record of what has been built, what was measured, and why each
non-obvious decision was made. Plan section references point at
`UniverseInference_IMPLEMENTATION_PLAN.md`.

**Last updated:** 2026-08-24
**Repository:** <https://github.com/ivarick/NeuralCosmo>
**Current position:** Phases 0–4 complete. Phase 5 (strong baselines) not started.
**Tests:** 221 passing, none requiring the 11.8 GB archive.

---

## 1. Where we are

| Plan phase | Status |
|---|---|
| Phase 0 — literature audit | **complete** (pass 1; rerun required before submission) |
| Phase 1 — data | **complete** |
| Phase 2 — loader and smoke model | **complete** |
| Phase 3 — in-domain baselines | **complete** — gate B1 passed |
| Phase 4 — quantify simulator shift | **complete** — Kill 2 avoided, 3 seeds |
| Phase 5 — strong baselines (DANN/CORAL/MMD/MIEST-like) | not started |
| Phase 6 — novelty gate | partially informed by the Phase 0 audit |
| Phases 7–17 | not started |

Milestones (section 97): **M1 data valid** ✅ · **M2 baseline learns** ✅ ·
**M3 shift exists** ✅ · M4–M8 pending.

---

## 2. Findings so far

All errors are MAE in physical units on held-out **test** simulations, with
source-train-only normalization (DG-strict).

### 2.1 In-domain performance (gate B1)

| Suite | Level | MAE Ωm | MAE σ8 | R² Ωm | R² σ8 |
|---|---|---:|---:|---:|---:|
| IllustrisTNG | map | 0.0082 | 0.0137 | 0.991 | 0.974 |
| IllustrisTNG | simulation | 0.0035 | 0.0055 | 0.999 | 0.996 |
| SIMBA | map | 0.0091 | 0.0161 | 0.990 | 0.973 |
| SIMBA | simulation | 0.0040 | 0.0101 | 0.998 | 0.991 |

Published CMD benchmark performance on total matter is roughly R² 0.96 (Ωm) and
0.83 (σ8). Ours is higher, but the split protocol and training budget may differ,
and section 32 forbids claiming superiority on a comparison that is not
apples-to-apples. The defensible reading is only that the pipeline is in the
right regime.

### 2.2 The simulator shift is strongly asymmetric — 3 seeds

| Level | Target | TNG→SIMBA | SIMBA→TNG | Welch *t* |
|---|---|---:|---:|---:|
| map | Ωm | 2.22 ± 0.31 | 1.20 ± 0.03 | 5.6 |
| map | σ8 | 1.57 ± 0.02 | 0.98 ± 0.05 | 20.1 |
| simulation | Ωm | 4.25 ± 0.86 | 2.00 ± 0.20 | 4.4 |
| simulation | σ8 | 2.69 ± 0.13 | 1.00 ± 0.15 | 15.2 |

Training on SIMBA transfers to IllustrisTNG with little or no degradation;
training on IllustrisTNG does not transfer to SIMBA. σ8 in the reverse direction
is 1.00 ± 0.05 — no measurable loss at all.

Section 35's kill criterion required G to be low across **both** directions.
Largest map-level mean is 2.22, so **Kill 2 does not apply**.

### 2.3 Transfer error is systematic, not stochastic

Fraction of error surviving simulation-level averaging over 15 maps:

| Direction | Target | ID | OOD |
|---|---|---:|---:|
| TNG→SIMBA | Ωm | 44% | 84% |
| TNG→SIMBA | σ8 | 44% | 75% |
| SIMBA→TNG | Ωm | 44% | 73% |
| SIMBA→TNG | σ8 | 60% | 61% |

In-domain error sits near 44%, close to what independent noise across 15 views
would predict. Transfer error survives at 61–84%, so it is largely systematic
bias that observing the same region more times cannot remove. Map-level metrics
alone therefore understate the practical cost of simulator shift.

### 2.4 ERM encodes simulator identity (H2 supported)

Frozen encoders probed on TNG+SIMBA test maps, probe split by simulation:

| Encoder | domain probe (balanced) | AUROC | target R² Ωm | target R² σ8 |
|---|---:|---:|---:|---:|
| TNG-trained | 0.756 | 0.789 | 0.977 | 0.948 |
| SIMBA-trained | 0.726 | 0.783 | 0.983 | 0.960 |

Chance is 0.500. The representation carries simulator identity *and* cosmology
simultaneously — the entangled state the method is meant to separate, and the
ERM anchor for the trade-off plot of section 40.

The SIMBA-trained encoder leaks slightly less, pointing the same way as the
transfer asymmetry, but the gap is ~3 points with no uncertainty estimate and is
**not claimed**.

### 2.5 A falsified hypothesis

After measuring the asymmetry, the proposed explanation was that SIMBA's feedback
erases small-scale structure. That predicts a SIMBA/TNG power ratio falling
monotonically with k. Measured:

```
k = 0.25   1.041
k = 2.61   0.858    <- minimum
k = 10.7   0.974
k = 29.3   1.149    <- excess, not deficit
```

Suppression is at *intermediate* scales and reverses at the smallest. SIMBA has
about 15% **more** small-scale power than TNG. **The hypothesis is retracted and
no replacement is offered.** Testing why the asymmetry exists needs an
intervention — band-limiting the input and observing whether the asymmetry moves
— not another inference from a static statistic.

The estimator was validated against analytically known inputs first, including a
Gaussian-smoothing test confirming it *can* detect the predicted signature, so
the null result is a property of the data rather than a blind instrument.

### 2.6 Data facts

- All 2,949,120,000 pixels strictly positive (min 4.67e9), so `log10` is legal
  with no epsilon (section 20.1 resolved).
- Linear suite means agree to five significant figures — mass conservation plus a
  shared Latin-hypercube design over Ωm.
- Log-space means do **not**: TNG 11.1027, SIMBA 11.1413, a 0.08σ separation.
  The model sees log inputs, so DG-strict normalization leaves a systematic 0.08σ
  offset. A real, quantified cost of the protocol.

---

## 3. Novelty position (Phase 0 audit)

| Prior work | Overlap |
|---|---|
| MIEST (Jo 2025) | **HI maps only — total matter not used.** Trains jointly; reports no single-direction transfer. Occupies adversarial de-classification |
| DA-GNN (Roncoli 2023) | galaxy graphs, MMD, domain *adaptation* (uses target data) |
| Multifidelity transfer (arXiv:2505.21215) | **Closest.** Already uses CMD dark-matter + hydro maps, but as sequential pre-training for *data efficiency*, with no unseen-suite evaluation |
| One latent to fit them all (arXiv:2509.01881) | hydro/N-body pairs → feedback latent, in power-spectrum space, no zero-shot inference |

The plan assumed CMD's paired hydro/N-body structure was unexploited. **It is
not.** PPIRL survives on three distinctions — simultaneous paired alignment vs
sequential pre-training, explicit use of matched initial conditions, and a
zero-shot robustness goal rather than data efficiency.

**Consequence:** sections 50 and 51 (shuffled-pair and unpaired-extra-data
controls) are now load-bearing. If correct pairing does not beat merely adding
N-body data, what remains is the multifidelity paper's contribution with extra
steps, and Kill 4 applies. An N-body-pretrain-then-finetune arm has been added as
a required Phase 8 baseline.

---

## 4. Infrastructure

```
src/neuralcosmos/
├── paths.py            CAMELS_DATA_ROOT resolution; fails loudly, never guesses
├── protocol.py         ExperimentProtocol — leakage rules as executable code
├── data/
│   ├── manifest.py     what is on disk + content hashing
│   ├── validate.py     the section 12 rejection gate
│   ├── splits.py       simulation-level partitions, hash-locked
│   ├── targets.py      fixed-design-range target scaling
│   ├── statistics.py   source-train-only normalization
│   ├── cache.py        uint16 log-quantised cache, disk and RAM
│   ├── dataset.py      memory-mapped map dataset
│   └── builders.py     assembly point where protocol checks are enforced
├── models/             small_cnn, heads, erm
├── training/           trainer, balanced sampler, seeding
└── evaluation/         metrics, spectra, representations
```

Scripts: `validate_data`, `build_splits`, `compute_stats`, `build_cache`,
`benchmark_io`, `run_experiment`, `evaluate_checkpoint`, `run_domain_probe`,
`analyze_spectra`, `aggregate_seeds`, `make_data_report`.

Reports generated from run JSON rather than typed by hand (section 98):
`data_report.md`, `baseline_report.md`, `novelty_audit.md`.

---

## 5. Leakage defences in force

| Defence | Where |
|---|---|
| Splits by simulation, never by map | `splits.py`, tested at map *and* simulation level |
| Split file hash-checked on every load | `load_split_file` |
| Split regeneration refused without `--force` | `build_splits.py` |
| Normalization from source *train* only | `statistics.py`, `stats_sources` |
| Normalizer provenance scanned for target names | `protocol.check_normalizer` |
| Target blocked from training sets | `protocol.check_training_suites` |
| Target blocked from validation / model selection | `protocol.check_validation_suites` |
| Same suite as source *and* target rejected | `ExperimentProtocol.__post_init__` |
| Probe train/test split by simulation | `representations.simulation_level_split` |
| Target scaling from design ranges, not sample statistics | `targets.py` |

`tests/test_no_target_leakage.py` attempts each leak and asserts refusal.

---

## 6. Decisions worth remembering

- **Root-anchored `.gitignore`.** `data/` matched `configs/data/` and
  `src/neuralcosmos/data/`, silently excluding them from the first commit.
- **Per-suite split seeds** from `sha256(master_seed:suite)`, so adding a fourth
  suite cannot reshuffle existing partitions.
- **Val/test from the front of the permutation**, so the data-efficiency ablation
  cannot change what is held out.
- **GroupNorm, never BatchNorm.** BatchNorm would mix suite statistics inside the
  normalisation layer and turn target evaluation into covert adaptation.
- **Unbounded regression head.** A squashed output cannot express a confident
  error, hiding the regress-to-the-mean behaviour section 72 exists to detect.
- **uint16 over float16** for the cache: same 2 bytes, ~70× finer resolution over
  this dynamic range.
- **Per-source normalizers.** E01 originally used the combined TNG+SIMBA
  normalizer — not a leak, but a confound, since the transfer run cannot use
  SIMBA statistics. Restarted with matched preprocessing.

---

## 7. Hardware and throughput

RTX 3060 12 GB, PyTorch 2.4.1+cu124, 32 GB RAM. Archive on an external USB disk.

| Data source | maps/s | ratio to model | verdict |
|---|---:|---:|---|
| USB, cold, 4 workers | 74.3 | 0.33 | I/O bound, GPU idle 67% |
| **RAM cache** | **4516.4** | **21.8** | compute bound |

61×, for 3.66 GiB resident and a ~217 s one-off load. Training is ~67 s/epoch on
13,500 maps. An on-disk SSD cache was built and then abandoned: `C:` is 96% full
and 3.66 GiB left the system drive under 7 GB.

---

## 8. Open risks

| Risk | Status |
|---|---|
| No novel method implemented yet | Phases 7–10 |
| PPIRL novelty rests entirely on beating its own controls | Kill 4 live |
| Why the asymmetry is directional | **unexplained**; first hypothesis falsified |
| An unresolved literature claim about SIMBA transfer direction | blocks any asymmetry claim until traced |
| Astrid power spectrum was computed pre-freeze | disclosed and quarantined; must not inform design |
| Phase 5 baselines absent | any improvement claim needs them |
| `C:` at 95% full | monitored |

---

## 9. Next steps

1. **Phase 5 baselines** — DANN, CORAL, MMD, MIEST-comparable, all on the same
   backbone, split, budget and metric so architecture cannot masquerade as method.
2. **B0 summary-statistic baseline** (section 25) — deep learning must beat
   something simple, and given the suites differ mainly in distribution *shape*,
   a global-moment baseline should be weak. Worth confirming rather than assuming.
3. **Bootstrap by simulation** (section 57) — would also settle whether the
   3-point domain-probe gap is real.
4. **Latent geometry** (section 39) — PCA/UMAP coloured by suite, Ωm, σ8.
5. Only then Phase 6's novelty gate and the paired-physics work.

Astrid remains sealed for training, validation, normalization and model
selection.
