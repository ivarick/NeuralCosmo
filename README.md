# NeuralCosmos

**Can a neural network infer cosmology from a simulated universe it has never seen the physics of?**

Neural networks recover the cosmological parameters $\Omega_m$ and $\sigma_8$ from simulated
matter-density maps with impressive accuracy — as long as you test them on the same simulator
they were trained on. Change the simulation code, and accuracy can collapse. The network has
learned the cosmology *and* the simulator's fingerprint, and it cannot tell you which is which.

NeuralCosmos studies how to separate the two.

```
              TRAIN                                  TEST
    ┌───────────────────────┐              ┌────────────────────┐
    │  IllustrisTNG   SIMBA │  ─────────▶  │       Astrid       │
    │   (2 simulators)      │   frozen     │  never seen during │
    └───────────────────────┘   model      │  training or tuning│
                                           └────────────────────┘

    256×256 matter-density map  ──▶  encoder  ──▶  Ωm , σ8
```

This is **domain generalization**, not domain adaptation: no target maps, no target labels, and
no target normalization statistics are used at any point before the configuration is frozen.

---

## Status

> **Phases 0–4 of 17 complete.** In-domain baselines and the cross-simulator
> measurement are done and replicated across three seeds. **No novel method has
> been implemented yet**, and nothing here has been peer reviewed. Astrid remains
> sealed: it has never been used for training, validation, normalization or model
> selection.

### What has been measured

In-domain, held-out test simulations (map level):

| Suite | MAE Ωm | MAE σ8 | R² Ωm | R² σ8 |
|---|---:|---:|---:|---:|
| IllustrisTNG | 0.0082 | 0.0137 | 0.991 | 0.974 |
| SIMBA | 0.0091 | 0.0161 | 0.990 | 0.973 |

Cross-simulator transfer, `G = OOD error / ID error`, mean ± sd over 3 seeds:

| Level | Target | TNG → SIMBA | SIMBA → TNG |
|---|---|---:|---:|
| map | Ωm | **2.22 ± 0.31** | 1.20 ± 0.03 |
| map | σ8 | **1.57 ± 0.02** | 0.98 ± 0.05 |
| simulation | Ωm | **4.25 ± 0.86** | 2.00 ± 0.20 |
| simulation | σ8 | **2.69 ± 0.13** | 1.00 ± 0.15 |

Three things follow, and they are the current state of the science here:

1. **The shift is strongly directional.** Training on SIMBA transfers to
   IllustrisTNG with little or no loss; the reverse fails. σ8 in the reverse
   direction degrades by nothing measurable at all.
2. **Transfer error is systematic, not stochastic.** Averaging the 15 maps of a
   simulation removes 56% of in-domain error but only 16–39% of transfer error.
   Simulator shift cannot be averaged away by observing a region more times.
3. **Ordinary training entangles the two signals.** A frozen ERM representation
   yields 0.75 balanced accuracy on a simulator-identity probe (chance 0.50)
   while still supporting R² ≈ 0.98 on cosmology.

Why the shift is *directional* remains **unexplained**. The first hypothesis —
that SIMBA's feedback erases small-scale structure — was tested against measured
power spectra and falsified; the SIMBA/TNG ratio dips at intermediate k and rises
*above* 1 at small scales. It is recorded as a negative result rather than
replaced with a story that fits the data after the fact.

Full detail: [`docs/PROGRESS.md`](docs/PROGRESS.md),
[`reports/baseline_report.md`](reports/baseline_report.md),
[`reports/novelty_audit.md`](reports/novelty_audit.md).

---

## The research question

> Can a representation-learning objective trained **only** on source simulation suites improve
> zero-shot cosmological regression on an unseen simulator, without using target-domain maps,
> target-domain labels, or target-specific normalization?

Two published results define the floor this project has to clear. Roncoli et al. (2023) applied
MMD-based domain adaptation to CAMELS galaxy graphs, and Jo et al. (2025) introduced **MIEST**,
which removes simulation-model information via adversarial de-classification and evaluates on
unseen suites. Consequently *"add a domain classifier so simulator identity is hard to predict"*
is treated here as a **baseline, not a contribution**.

The contribution under investigation exploits a structure specific to CAMELS: total-matter maps
exist in **matched hydrodynamic and gravity-only (N-body) versions** of the same region, same
cosmology, same initial conditions.

```
    hydro Mtot  ──┐
                  ├── same cosmology, same seed, same region
    N-body Mtot ──┘
                          │
                          ▼
        the gravity-only view contains no baryonic feedback,
        so it can act as an invariance anchor against
        simulator-specific subgrid physics
```

Whether this survives a formal novelty audit (`reports/novelty_audit.md`) and its own negative
controls is an open question. It is **not** claimed as novel in this README.

---

## Why this is hard to do honestly

Most of the engineering in this repository exists to stop the project from fooling itself:

- **Splits are by simulation, never by map.** Each CAMELS simulation produces 15 correlated maps
  sharing one parameter vector. A random map-level split leaks labels across train and test.
- **Astrid is sealed.** It is never used to choose an architecture, a loss weight, a learning
  rate, an early-stopping point, or a normalization statistic. The configuration is committed and
  hash-recorded *before* the first Astrid number is ever computed.
- **Protocol violations are executable errors,** not conventions. An `ExperimentProtocol` object
  declares whether target data, target statistics, and target labels are permitted, and the
  trainer refuses datasets that violate it.
- **Confidence intervals bootstrap over simulations, not maps,** because 15 maps from one
  simulation are not 15 independent samples.
- **Kill criteria are written down in advance.** If total-matter maps turn out to show almost no
  cross-simulator shift, the testbed is declared too weak and the project pivots rather than
  manufacturing a result.

---

## Data

This repository contains **no data**. The CAMELS Multifield Dataset is obtained from its official
source and is never committed or redistributed.

Phase-A files (2D total-matter maps, Latin-Hypercube set, $z=0$):

| Suite | Map file | Parameter file |
|---|---|---|
| IllustrisTNG | `Maps_Mtot_IllustrisTNG_LH_z=0.00.npy` | `IllustrisTNG LH parameters.txt` |
| SIMBA | `Maps_Mtot_SIMBA_LH_z=0.00.npy` | `SIMBA LH parameters.txt` |
| Astrid | `Maps_Mtot_Astrid_LH_z=0.00.npy` | `Astrid LH parameters.txt` |

Each map file is `(15000, 256, 256)` float32, about 3.9 GB. Each parameter file is
`(1000, 6)`, where columns 0 and 1 are $\Omega_m$ and $\sigma_8$; the remaining four are
astrophysical feedback parameters treated here as **nuisance variables**. Maps map to
simulations by `simulation_id = map_index // 15`.

Download instructions: <https://camels-multifield-dataset.readthedocs.io/en/latest/access.html>
(Globus is recommended by CMD for transfers of this size.)

Point the project at your local copy — no path is ever hardcoded:

```powershell
$env:CAMELS_DATA_ROOT = "E:\universeinfrencedata"
```

---

## Reproducing

```bash
pip install -e ".[dev]"
```

Validate the archive before anything else — this is the gate that refuses bad
data, and it verifies positivity across all 2.9 billion pixels:

```bash
python scripts/validate_data.py --config configs/data/mtot.yaml --manifest
```

Build the permanent simulation-level split, then the source-train-only
normalization statistics:

```bash
python scripts/build_splits.py --seed 42
```

```bash
python scripts/compute_stats.py --source IllustrisTNG
```

Train an in-domain baseline, then evaluate the frozen checkpoint on the transfer
suite without retraining:

```bash
python scripts/run_experiment.py --config configs/experiments/e01_tng_id.yaml --seed 0
```

```bash
python scripts/evaluate_checkpoint.py --run E01_tng_id_seed00 --suite IllustrisTNG --suite SIMBA
```

Diagnostics and reports — every table is regenerated from run JSON, never typed
by hand:

```bash
python scripts/run_domain_probe.py --run E01_tng_id_seed00
```

```bash
python scripts/aggregate_seeds.py
```

If the archive lives on a slow disk, hold the quantised maps in RAM instead
(3.66 GiB for two suites, measured 61× faster here). It is on by default in the
experiment configs; `scripts/benchmark_io.py` reports whether it is needed.

The unit-test suite runs **without** the CAMELS archive present — 221 tests
against a synthetic mini-archive that reproduces the structural invariants at
tiny scale:

```bash
pytest
```

---

## Hardware

Developed and run on a single NVIDIA GeForce RTX 3060 (12 GB), PyTorch 2.4.1 + CUDA 12.4,
Python 3.10, Windows 11. The architecture budget is deliberately constrained to what this GPU
can train repeatedly across multiple seeds — the scientific value here comes from experimental
rigour, not parameter count.

---

## Limitations

- Simulated data only; no observational domain is tested.
- A single field (total matter) and a single redshift ($z=0$) in the initial phase.
- Three simulation suites, one held out.
- Two cosmological targets; astrophysical parameters are nuisances, not predictions.
- No theoretical guarantee of invariance is offered — the claims are empirical.

---

## Citation

This project uses the CAMELS Multifield Dataset. If you use this code, please cite CMD:

- Villaescusa-Navarro et al., *The CAMELS Multifield Dataset*, ApJS (2022).
  DOI: [10.3847/1538-4365/ac5ab0](https://doi.org/10.3847/1538-4365/ac5ab0)
- CAMELS citation guidance: <https://camels.readthedocs.io/en/latest/citation.html>

Closest prior work on cross-simulator robustness:

- Roncoli et al. (2023), *Domain Adaptive Graph Neural Networks…*, [arXiv:2311.01588](https://arxiv.org/abs/2311.01588)
- Jo et al. (2025), *Toward Robustness across Cosmological Simulation Models*, [arXiv:2502.13239](https://arxiv.org/abs/2502.13239)

---

## License

MIT for the code in this repository. The CAMELS data is governed by its own terms and is not
redistributed here. See [LICENSE](LICENSE).
