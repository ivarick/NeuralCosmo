# NeuralCosmos — Progress Log

Running record of what has been built, what was measured, and why each
non-obvious decision was made. Plan section references point at
`UniverseInference_IMPLEMENTATION_PLAN.md`.

**Last updated:** 2026-08-24
**Repository:** <https://github.com/ivarick/NeuralCosmo>
**Current phase:** Phase 2 (loader and smoke model) — nearly complete
**Scientific results so far:** none. Nothing here is a finding.

---

## 1. Where we are

| Plan phase | Status |
|---|---|
| Phase 0 — literature audit | not started |
| Phase 1 — data | **complete** |
| Phase 2 — loader and smoke model | in progress |
| Phase 3 — in-domain baselines | not started |
| Phase 4 — quantify simulator shift | not started |
| Phases 5–17 | not started |

Milestones from plan section 97:

| | Milestone | Status |
|---|---|---|
| M1 | Data valid | **done** |
| M2 | Baseline learns | pending |
| M3 | Shift exists | pending |
| M4 | Baselines complete | pending |
| M5 | Novelty confirmed | pending |
| M6 | Mechanism supported | pending |
| M7 | Final target frozen | pending |
| M8 | Reproducible paper | pending |

---

## 2. Environment

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 3060, 12 GB (matches the plan's section 60 target) |
| PyTorch | 2.4.1+cu124, CUDA available |
| Python | 3.10.11 |
| OS | Windows 11 |
| Archive | `E:\universeinfrencedata` — **external USB disk**, ~33 MB/s |
| Repo | `C:\Users\Ivarick\Desktop\universeinfrence` (ADATA SU630 SATA SSD) |

Deviations from the plan's assumed layout, all handled rather than fought:

- Plan assumes data on `G:` and code at `C:\projects\UniverseInference`. Paths
  are resolved from `CAMELS_DATA_ROOT` instead (section 10), never hardcoded.
- Parameter files are named `IllustrisTNG LH parameters.txt`, not
  `params_LH_IllustrisTNG.txt`. The config records the real names rather than
  renaming a 11.8 GB archive to match a document.

---

## 3. Commits

| Commit | Content |
|---|---|
| `42c837c` | Project scaffold, `paths.py`, data config |
| `80deb28` | Data validation, manifest, Phase 1 report |
| `cbb1670` | Simulation-level splits and leakage tests |
| `a339ddf` | Dataset, target scaling, executable leakage guardrails |
| `cfff6ea` | Compact CNN, heads, throughput benchmark |

---

## 4. Measured findings

### 4.1 Archive is complete and valid

39/39 checks pass across all three suites. Every file is exactly
`3,932,160,128` bytes, which is `15000 × 256 × 256 × 4 + 128`, so nothing is
truncated. Manifest content hash `d8068b38f75e0a17…`.

### 4.2 The log transform is legal (section 20.1 resolved)

Scanned all **2,949,120,000** pixels:

| Suite | min | max | mean | std | ≤0 | NaN/Inf |
|---|---:|---:|---:|---:|---:|---:|
| IllustrisTNG | 4.6650e+09 | 2.8518e+15 | 4.1632e+11 | 3.6415e+12 | 0 | 0 |
| SIMBA | 5.2276e+09 | 4.2038e+15 | 4.1632e+11 | 3.6380e+12 | 0 | 0 |
| Astrid | 5.4158e+09 | 2.8644e+15 | 4.1631e+11 | 3.6788e+12 | 0 | 0 |

Section 20.1 posed a hard fork: if every value is positive use `log(x)`,
otherwise **stop** and inspect rather than inventing an epsilon. Zero
non-positive values in ~3 × 10⁹, so we are cleanly on the first branch. No
epsilon, no `log1p`. The ~6-decade dynamic range makes the transform necessary
rather than cosmetic.

### 4.3 The domain shift is not in the global statistics

The three suite means agree to **five significant figures**. That is mass
conservation plus a shared Latin-hypercube design: mean surface density is set
by Ωm, all suites sample Ωm ∈ [0.1, 0.5], so all three average to Ωm ≈ 0.3.
Baryonic feedback redistributes matter, it does not create or destroy it.

Two consequences carried forward:

1. DG-strict normalization (section 20.2) costs almost nothing here, since the
   global statistics barely differ between suites. The strict choice is cheap.
2. Any real cross-suite shift lives in **spatial structure and tails**, not
   global moments. The B0 summary-statistic baseline (section 25) should
   therefore be expected to show weak suite separation. Phase 4 must quantify
   the shift with per-map distributions and power spectra.

Explicitly **not** concluded: SIMBA's maximum is 47% higher than the others.
A maximum over 10⁹ pixels is a single-cell order statistic, far too fragile to
support a claim about feedback physics.

### 4.4 Normalization statistics (section 20.2)

Computed over source-training simulations only — 1,769,472,000 values:

```
mean 11.122040   std 0.483565   range 9.668856 … 15.623643   (log10)
provenance: log10|train|IllustrisTNG+SIMBA|split_v1:f567a2877cdb|n=1769472000
```

The provenance string is load-bearing: the protocol guard substring-scans it
for target-suite names, so a normalizer that ever touched Astrid makes every
dataset built with it raise on construction.

Incidental result: grouping the 13,500 scattered training indices into
contiguous runs yielded only **92 runs**, not the ~900 predicted. Because 900
of 1000 simulations are in train, chosen simulations are mostly adjacent and
merge into long runs. The 7 GB read completed in 203.7 s.

### 4.5 Throughput — we are I/O bound (section 83)

```
DATA      workers=0    62.7 maps/s
          workers=2   130.8 maps/s   ← best
          workers=4   116.9 maps/s
MODEL                 222.7 maps/s   (AMP, 2904 MB peak of 12288 MB)

ratio 0.59 → GPU idles ~41%
```

Four workers being *slower* than two is the signature of a USB or mechanical
disk thrashing on concurrent random reads, not a CPU limit. 130.8 maps/s ×
256 KB = 32.7 MB/s, matching the sequential rate — the disk is saturated.

Cost: 206 s/epoch, ≈5.7 h per 100-epoch run. Decision taken to build a uint16
SSD cache (section 6.6 below).

---

## 5. Architecture of the codebase

```
src/neuralcosmos/
├── paths.py           CAMELS_DATA_ROOT resolution; fails loudly, never guesses
├── protocol.py        ExperimentProtocol — leakage rules as executable code
├── data/
│   ├── manifest.py    what is on disk + content hashing
│   ├── validate.py    the section 12 rejection gate
│   ├── splits.py      simulation-level partitions, hash-locked
│   ├── targets.py     fixed-design-range target scaling
│   ├── statistics.py  source-train-only normalization
│   ├── dataset.py     memory-mapped map dataset
│   └── builders.py    assembly point where protocol checks are enforced
└── models/
    ├── backbones/small_cnn.py
    ├── heads.py       regression / projection / domain
    └── erm.py
```

Scripts are thin CLIs over these modules: `validate_data.py`,
`build_splits.py`, `compute_stats.py`, `make_data_report.py`,
`benchmark_io.py`.

---

## 6. Decisions and their reasons

### 6.1 `.gitignore` patterns are root-anchored

`data/` matches a directory named `data` at any depth, so it silently excluded
`configs/data/` and `src/neuralcosmos/data/` from the first commit. Fixed to
`/data/`. Caught by diffing the staged file list against what was written.

### 6.2 Per-suite split seeds, not one shared RNG stream

Each suite's split derives from `sha256(master_seed:suite_name)`. With a single
shared stream, adding a fourth suite later would shift every subsequent draw
and silently reshuffle existing partitions, invalidating results computed
before it was added. A test adds Astrid and asserts TNG and SIMBA come back
byte-identical.

### 6.3 Validation and test taken from the front of the permutation

The data-efficiency ablation (section 15) trains on 100/250/500/900
simulations. If train were the front slice, shrinking it would change which
simulations are held out, and the ablation would compare different test sets.

### 6.4 GroupNorm, never BatchNorm

BatchNorm would mix suite statistics inside the normalization layer, entangling
what the method is trying to separate (section 61), and its running statistics
— estimated on source data — act as covert adaptation at target-evaluation
time. A test asserts no BatchNorm module exists anywhere in the model.

### 6.5 Average pooling, unbounded regression output

Average pooling because the meaningful aggregation of a 2×2 patch of a density
field is its mean surface density. Unbounded output because a squashed head
cannot express a confident error, which would hide the regress-to-the-mean
behaviour that section 72's residual analysis exists to detect.

### 6.6 uint16 log-quantised SSD cache

Chosen after measurement, per section 83's instruction to profile first.
Stores `log10(x)` quantised over a fixed documented range into 2 bytes per
pixel: 1.83 GiB per suite instead of 3.66 GiB. Source suites only; Astrid
stays on USB since it is read only at the end.

uint16 over float16 for the same 2 bytes: float16 spacing near log10 ≈ 15.6 is
0.0078 dex, whereas uint16 over a 7-dex window gives 1.07e-4 dex — about 70×
finer. Quantisation noise is ~6e-5 of one pixel standard deviation.

The cache deliberately stores **un-normalised** log values, so it does not bake
in a normalization choice and stays valid when the source-suite set changes for
the leave-one-suite-out experiments of section 53.

### 6.7 Windows-specific hazards handled

- `np.memmap` cannot be pickled and Windows *spawns* DataLoader workers rather
  than forking. Handles are opened lazily inside whichever process reads them,
  cached by PID, and dropped in `__getstate__`.
- `np.rot90` / `np.fliplr` return negative-stride views that torch refuses;
  augmentation forces `ascontiguousarray`.
- Git for Windows ships its own MSYS `ssh.exe` which could not reach port 22 on
  this network, while Windows OpenSSH could. `core.sshCommand` points at the
  latter.

---

## 7. Leakage defences currently in force

| Defence | Where |
|---|---|
| Splits by simulation, never by map | `splits.py`, tested at map *and* simulation level |
| Split file hash-checked on every load | `load_split_file` |
| Split regeneration refused without `--force` | `build_splits.py` |
| Normalization from source *train* only | `statistics.py`, `stats_sources` |
| Normalizer provenance scanned for target names | `protocol.check_normalizer` |
| Target suite blocked from training sets | `protocol.check_training_suites` |
| Target suite blocked from validation/model selection | `protocol.check_validation_suites` |
| Same suite as both source and target rejected | `ExperimentProtocol.__post_init__` |
| Target scaling uses design ranges, not sample statistics | `targets.py` |

`tests/test_no_target_leakage.py` attempts each leak and asserts refusal.

---

## 8. Test suite

**141 tests, all passing, none requiring the 11.8 GB archive** (section 79).
A synthetic mini-archive fixture reproduces the structural invariants at tiny
scale, so the leakage and mapping tests are meaningful without real data.

```
tests/
├── conftest.py                 synthetic archive fixture
├── test_paths.py               fail-loudly path resolution
├── test_manifest.py            manifest + content hashing
├── test_validate.py            corruption must be rejected, not just passed
├── test_map_sim_mapping.py     the //15 contract, hand-computed boundaries
├── test_split_integrity.py     disjointness, determinism, tamper detection
├── test_dataset.py             sample contract, scaling, augmentation
├── test_no_target_leakage.py   protocol enforcement
└── test_models.py              shapes, gradients, no BatchNorm
```

---

## 9. Next steps

1. **Build the uint16 SSD cache** and re-run the benchmark to confirm we become
   compute-bound.
2. **Trainer** — checkpointing, early stopping on source validation only
   (sections 62–63), run metadata capture (section 64).
3. **Smoke run** on ≤100 simulations to verify the GPU path end to end
   (Phase 2). Smoke metrics are not to be interpreted scientifically.
4. **Phase 3** — in-domain baselines: TNG→TNG and SIMBA→SIMBA. This is
   milestone M2 and gate B1: if the model cannot learn in-domain, nothing else
   matters.
5. **Phase 4** — TNG→SIMBA and SIMBA→TNG. This decides whether the project has
   a testbed at all. Section 35 sets the kill criterion: if the OOD/ID error
   ratio is consistently below ~1.2–1.3, Mtot is too weak and we pivot to a
   more baryon-sensitive field.

Phase 0 (literature audit) has not been started and is a prerequisite for any
novelty claim, but it does not block Phases 3–4.

---

## 10. Open risks

| Risk | Status |
|---|---|
| Mtot may show too little cross-suite shift (Kill 2) | unknown until Phase 4 |
| `C:` has ~13 GB free; cache will use 3.66 GiB | acceptable, monitored |
| Novelty may be occupied by MIEST / DA-GNN | audit not yet run |
| N-body paired data not yet downloaded | not needed until Phase 7 |
