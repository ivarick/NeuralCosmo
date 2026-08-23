# Data Report

Phase 1 deliverable (plan section 68). **Generated file - do not edit by hand.**
Regenerate with `python scripts/make_data_report.py`.

- Generated: `2026-08-23T23:46:59+00:00`
- Validation mode: `FULL pixel scan`
- Validation elapsed: `338.9 s`
- Manifest content hash: `d8068b38f75e0a1790c2ece9e44b31d7c12363334d2aa84c6802783e6054c793`
- Overall verdict: **PASS**

## 1. Files

| Suite | Map file | Bytes | GiB | Shape | dtype | Params |
|---|---|---:|---:|---|---|---|
| IllustrisTNG | `Maps_Mtot_IllustrisTNG_LH_z=0.00.npy` | 3,932,160,128 | 3.66 | (15000, 256, 256) | float32 | (1000, 6) |
| SIMBA | `Maps_Mtot_SIMBA_LH_z=0.00.npy` | 3,932,160,128 | 3.66 | (15000, 256, 256) | float32 | (1000, 6) |
| Astrid | `Maps_Mtot_Astrid_LH_z=0.00.npy` | 3,932,160,128 | 3.66 | (15000, 256, 256) | float32 | (1000, 6) |

## 2. Validation checks

39/39 checks passed across 3 suite(s).

### IllustrisTNG - OK

| Check | Result | Detail |
|---|---|---|
| `map_file_exists` | PASS | E:\universeinfrencedata\Maps_Mtot_IllustrisTNG_LH_z=0.00.npy |
| `param_file_exists` | PASS | E:\universeinfrencedata\IllustrisTNG LH parameters.txt |
| `map_byte_size` | PASS | 3,932,160,128 bytes (expected 3,932,160,128) |
| `maps_shape` | PASS | (15000, 256, 256) (expected (15000, 256, 256)) |
| `maps_dtype` | PASS | float32 (expected float32) |
| `params_shape` | PASS | (1000, 6) (expected (1000, 6)) |
| `params_finite` | PASS | all parameter values finite |
| `map_simulation_mapping` | PASS | 15000 maps == 1000 simulations x 15 maps/sim |
| `range_omega_m` | PASS | [0.10020, 0.49980] within [0.1, 0.5] +/- 0.01 |
| `range_sigma8` | PASS | [0.60020, 0.99980] within [0.6, 1.0] +/- 0.01 |
| `no_nan` | PASS | 0 NaN |
| `no_inf` | PASS | 0 Inf |
| `strictly_positive` | PASS | min = 4.66504e+09; log(x) is safe |

### SIMBA - OK

| Check | Result | Detail |
|---|---|---|
| `map_file_exists` | PASS | E:\universeinfrencedata\Maps_Mtot_SIMBA_LH_z=0.00.npy |
| `param_file_exists` | PASS | E:\universeinfrencedata\SIMBA LH parameters.txt |
| `map_byte_size` | PASS | 3,932,160,128 bytes (expected 3,932,160,128) |
| `maps_shape` | PASS | (15000, 256, 256) (expected (15000, 256, 256)) |
| `maps_dtype` | PASS | float32 (expected float32) |
| `params_shape` | PASS | (1000, 6) (expected (1000, 6)) |
| `params_finite` | PASS | all parameter values finite |
| `map_simulation_mapping` | PASS | 15000 maps == 1000 simulations x 15 maps/sim |
| `range_omega_m` | PASS | [0.10020, 0.49980] within [0.1, 0.5] +/- 0.01 |
| `range_sigma8` | PASS | [0.60020, 0.99980] within [0.6, 1.0] +/- 0.01 |
| `no_nan` | PASS | 0 NaN |
| `no_inf` | PASS | 0 Inf |
| `strictly_positive` | PASS | min = 5.22761e+09; log(x) is safe |

### Astrid - OK

| Check | Result | Detail |
|---|---|---|
| `map_file_exists` | PASS | E:\universeinfrencedata\Maps_Mtot_Astrid_LH_z=0.00.npy |
| `param_file_exists` | PASS | E:\universeinfrencedata\Astrid LH parameters.txt |
| `map_byte_size` | PASS | 3,932,160,128 bytes (expected 3,932,160,128) |
| `maps_shape` | PASS | (15000, 256, 256) (expected (15000, 256, 256)) |
| `maps_dtype` | PASS | float32 (expected float32) |
| `params_shape` | PASS | (1000, 6) (expected (1000, 6)) |
| `params_finite` | PASS | all parameter values finite |
| `map_simulation_mapping` | PASS | 15000 maps == 1000 simulations x 15 maps/sim |
| `range_omega_m` | PASS | [0.10020, 0.49980] within [0.1, 0.5] +/- 0.01 |
| `range_sigma8` | PASS | [0.60020, 0.99980] within [0.6, 1.0] +/- 0.01 |
| `no_nan` | PASS | 0 NaN |
| `no_inf` | PASS | 0 Inf |
| `strictly_positive` | PASS | min = 5.41584e+09; log(x) is safe |

## 3. Pixel statistics

Computed over every pixel of every map. These are **integrity statistics only**.
They must never be used for normalization: section 20.2 requires normalization
statistics computed from source-training simulations alone.

| Suite | N pixels | min | max | mean | std | NaN | Inf | <=0 | ==0 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| IllustrisTNG | 983,040,000 | 4.6650e+09 | 2.8518e+15 | 4.1632e+11 | 3.6415e+12 | 0 | 0 | 0 | 0 |
| SIMBA | 983,040,000 | 5.2276e+09 | 4.2038e+15 | 4.1632e+11 | 3.6380e+12 | 0 | 0 | 0 | 0 |
| Astrid | 983,040,000 | 5.4158e+09 | 2.8644e+15 | 4.1631e+11 | 3.6788e+12 | 0 | 0 | 0 | 0 |

### 3.1 Log-transform decision (section 20.1)

All pixel values are strictly positive (smallest observed: 4.6650e+09).

Section 20.1 permits `log(x)` in this case. No epsilon is introduced and
`log1p` is **not** used: the plan warns that the `log(1 + rho*)` treatment
documented for stellar density must not be transferred to Mtot without reason.

The dynamic range spans about **6.0 decades** (4.6650e+09 to 4.2038e+15), which is why the transform matters.

### 3.2 Cross-suite comparison

- Relative spread of the global mean across suites: **4.39e-05**
- Relative spread of the global std across suites:  **1.12e-02**

A negligible spread in the mean is expected rather than surprising: the mean
surface density of total matter is fixed by Omega_m and the box geometry, and
all suites sample the same Latin-hypercube design over Omega_m. Baryonic
feedback redistributes matter; it does not create or destroy it.

Two consequences follow:

1. The DG-strict normalization of section 20.2 (source statistics applied to
   the target) costs almost nothing here, because the global statistics barely
   differ between suites. The methodologically strict choice is also cheap.
2. Any cross-suite domain shift must live in **spatial structure and tails**,
   not in global moments. The B0 summary-statistic baseline (section 25) should
   therefore be expected to show weak suite separation, and quantifying the
   shift properly requires the per-map analysis of Phase 4.

Extreme order statistics (the per-suite maximum) are **not** interpreted here.
A maximum over ~10^9 pixels is set by a single densest cell and is far too
fragile to support a claim about feedback physics.

## 4. Integrity record

Local digests. Section 13: these are **our own** integrity record. CAMELS does
not publish matching checksums, so they must never be presented as official.
The map digest samples the file rather than reading it whole; it is designed to
catch truncation and accidental modification, not adversarial tampering.

| Suite | sampled map sha256 | full params sha256 |
|---|---|---|
| IllustrisTNG | `c47cc50ec4cbb717...` | `11ed02f023d1859b...` |
| SIMBA | `69b1515bad304820...` | `7828cd88d4193fc5...` |
| Astrid | `ed1dc6cba0918333...` | `0640a8b86daaebfd...` |

## 5. Reproduce

```bash
python scripts/validate_data.py --config configs/data/mtot.yaml --manifest
python scripts/make_data_report.py
```
