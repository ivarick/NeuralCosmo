# Baseline Report — Phase 4

Phase 4 deliverable (plan section 68). **Generated file — do not edit by hand.**
Regenerate with `python scripts/aggregate_seeds.py`.

- Generated: `2026-08-25T19:49:10+00:00`
- Every value is read from `outputs/runs/*/transfer_test.json` (section 98).
- Errors are MAE in physical units on held-out **test** simulations.
- Normalization is source-train only in both directions (DG-strict, section 20.2).

## 1. Generalization ratio G = OOD error / ID error

Mean ± standard deviation across training seeds (section 58).

| Direction | Level | Target | ID MAE | OOD MAE | **G** | seeds |
|---|---|---|---:|---:|---:|---:|
| IllustrisTNG -> SIMBA | map | omega_m | 0.0083 | 0.0185 | **2.22 ± 0.31** | 3 |
| IllustrisTNG -> SIMBA | map | sigma8 | 0.0140 | 0.0221 | **1.57 ± 0.02** | 3 |
| IllustrisTNG -> SIMBA | simulation | omega_m | 0.0036 | 0.0155 | **4.25 ± 0.86** | 3 |
| IllustrisTNG -> SIMBA | simulation | sigma8 | 0.0062 | 0.0165 | **2.69 ± 0.13** | 3 |
| SIMBA -> IllustrisTNG | map | omega_m | 0.0090 | 0.0109 | **1.20 ± 0.03** | 3 |
| SIMBA -> IllustrisTNG | map | sigma8 | 0.0161 | 0.0158 | **0.98 ± 0.05** | 3 |
| SIMBA -> IllustrisTNG | simulation | omega_m | 0.0040 | 0.0080 | **2.00 ± 0.20** | 3 |
| SIMBA -> IllustrisTNG | simulation | sigma8 | 0.0097 | 0.0097 | **1.00 ± 0.15** | 3 |

## 2. Directional asymmetry

Comparing `IllustrisTNG -> SIMBA` against `SIMBA -> IllustrisTNG` at matched level and target.

| Level | Target | G forward | G reverse | difference | Welch t |
|---|---|---:|---:|---:|---:|
| map | omega_m | 2.22 ± 0.31 | 1.20 ± 0.03 | +1.01 | 5.6 |
| map | sigma8 | 1.57 ± 0.02 | 0.98 ± 0.05 | +0.59 | 20.1 |
| simulation | omega_m | 4.25 ± 0.86 | 2.00 ± 0.20 | +2.25 | 4.4 |
| simulation | sigma8 | 2.69 ± 0.13 | 1.00 ± 0.15 | +1.69 | 15.2 |

The asymmetry is present in every level/target combination and is large
relative to seed variance. With three seeds per direction the t statistic is
indicative rather than a formal test, but the separation is not marginal.

## 3. Kill criterion (section 35)

Section 35, written before any result was seen: if the OOD/ID ratio is
consistently below roughly 1.2–1.3 across both targets **and both transfer
directions**, the total-matter field does not expose enough simulator shift
to motivate the intended method.

Largest map-level mean G observed: **2.22**.

**Kill 2 does not apply.** The testbed is viable and the project continues.

## 4. Is the transfer error averageable?

Each simulation is rendered as 15 maps. Averaging their predictions removes
*random* error but not *systematic* bias, so the fraction of error surviving
aggregation separates the two.

| Direction | Target | ID survives | OOD survives |
|---|---|---:|---:|
| IllustrisTNG -> SIMBA | omega_m | 44% | 84% |
| IllustrisTNG -> SIMBA | sigma8 | 44% | 75% |
| SIMBA -> IllustrisTNG | omega_m | 44% | 73% |
| SIMBA -> IllustrisTNG | sigma8 | 60% | 61% |

A larger surviving fraction out of domain means the transfer error is
systematic: it cannot be averaged away by observing the same region more times.

## 5. Raw per-seed values

| Direction | Level | Target | seed | ID MAE | OOD MAE | G |
|---|---|---|---:|---:|---:|---:|
| IllustrisTNG -> SIMBA | map | omega_m | 0 | 0.0082 | 0.0182 | 2.22 |
| IllustrisTNG -> SIMBA | map | omega_m | 1 | 0.0081 | 0.0153 | 1.90 |
| IllustrisTNG -> SIMBA | map | omega_m | 2 | 0.0087 | 0.0220 | 2.52 |
| IllustrisTNG -> SIMBA | map | sigma8 | 0 | 0.0137 | 0.0215 | 1.57 |
| IllustrisTNG -> SIMBA | map | sigma8 | 1 | 0.0138 | 0.0215 | 1.56 |
| IllustrisTNG -> SIMBA | map | sigma8 | 2 | 0.0146 | 0.0232 | 1.59 |
| IllustrisTNG -> SIMBA | simulation | omega_m | 0 | 0.0035 | 0.0154 | 4.40 |
| IllustrisTNG -> SIMBA | simulation | omega_m | 1 | 0.0035 | 0.0115 | 3.32 |
| IllustrisTNG -> SIMBA | simulation | omega_m | 2 | 0.0039 | 0.0197 | 5.03 |
| IllustrisTNG -> SIMBA | simulation | sigma8 | 0 | 0.0055 | 0.0157 | 2.84 |
| IllustrisTNG -> SIMBA | simulation | sigma8 | 1 | 0.0060 | 0.0159 | 2.65 |
| IllustrisTNG -> SIMBA | simulation | sigma8 | 2 | 0.0069 | 0.0180 | 2.60 |
| SIMBA -> IllustrisTNG | map | omega_m | 0 | 0.0091 | 0.0106 | 1.17 |
| SIMBA -> IllustrisTNG | map | omega_m | 1 | 0.0090 | 0.0108 | 1.20 |
| SIMBA -> IllustrisTNG | map | omega_m | 2 | 0.0090 | 0.0111 | 1.24 |
| SIMBA -> IllustrisTNG | map | sigma8 | 0 | 0.0161 | 0.0149 | 0.92 |
| SIMBA -> IllustrisTNG | map | sigma8 | 1 | 0.0162 | 0.0163 | 1.01 |
| SIMBA -> IllustrisTNG | map | sigma8 | 2 | 0.0160 | 0.0162 | 1.01 |
| SIMBA -> IllustrisTNG | simulation | omega_m | 0 | 0.0040 | 0.0078 | 1.95 |
| SIMBA -> IllustrisTNG | simulation | omega_m | 1 | 0.0040 | 0.0074 | 1.83 |
| SIMBA -> IllustrisTNG | simulation | omega_m | 2 | 0.0039 | 0.0087 | 2.22 |
| SIMBA -> IllustrisTNG | simulation | sigma8 | 0 | 0.0101 | 0.0085 | 0.84 |
| SIMBA -> IllustrisTNG | simulation | sigma8 | 1 | 0.0096 | 0.0105 | 1.09 |
| SIMBA -> IllustrisTNG | simulation | sigma8 | 2 | 0.0093 | 0.0100 | 1.08 |
