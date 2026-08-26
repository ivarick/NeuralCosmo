# Phase 5 Report — Domain-Generalization Baselines

Generated file — do not edit by hand. `python scripts/phase5_report.py`.

- Generated: `2026-08-26T03:15:33+00:00`
- Sources: IllustrisTNG + SIMBA. Astrid sealed and untouched.
- Every method shares the same backbone, split, budget and metric (section 5).

**None of these methods is a contribution.** MIEST already applies adversarial
de-classification to CAMELS and DA-GNN already applies MMD; sections 29 and 31
classify both as baselines. They exist to establish what the proposed method
must beat.

## 1. What Phase 5 can and cannot show

With only two development suites and Astrid sealed, the out-of-distribution
benefit of a multi-source method **cannot be measured before the freeze**.
What follows is therefore source-side evidence only:

- source-test error — does the method damage the task?
- domain probe (section 37) — does it actually remove simulator information?
- the trade-off of section 40 — both together, which is the only honest reading

Section 52 freezes the design on this evidence and evaluates Astrid once.

## 2. Source performance

| Method | best epoch | selection score | source MAE omega_m | source MAE sigma8 |
|---|---:|---:|---:|---:|
| ERM (no invariance) | 19 | 0.07298 | 0.0170 | 0.0249 |
| DANN (adversarial) | 19 | 0.22954 | 0.0732 | 0.0862 |
| CORAL (2nd-order) | 19 | 0.07354 | 0.0175 | 0.0252 |
| MMD (distribution) | 19 | 0.24147 | 0.0765 | 0.0962 |
| MIEST-comparable | 12 | 0.22439 | 0.0748 | 0.0892 |

## 3. The section 40 trade-off

Domain probe chance level is 0.500. Lower probe accuracy means less
simulator information survives; lower source error means less task damage.

| Method | domain probe | vs chance | source MAE (mean) | target probe R² |
|---|---:|---:|---:|---:|
| ERM (no invariance) | 0.602 | +0.102 | 0.0210 | 0.932 |
| DANN (adversarial) | 0.656 | +0.156 | 0.0797 | 0.370 |
| CORAL (2nd-order) | 0.588 | +0.088 | 0.0214 | 0.930 |
| MMD (distribution) | 0.570 | +0.070 | 0.0863 | 0.318 |
| MIEST-comparable | 0.620 | +0.120 | 0.0820 | 0.408 |

### Reading

- **DANN (adversarial)**: probe +0.054, source error +280.2% — did NOT reduce simulator information
- **CORAL (2nd-order)**: probe -0.014, source error +1.9% — less simulator information at no task cost
- **MMD (distribution)**: probe -0.032, source error +312.0% — removed simulator information but cost +312.0% source error
- **MIEST-comparable**: probe +0.018, source error +291.3% — did NOT reduce simulator information

Section 37 is explicit that a lower probe score is not automatically
better: a collapsed representation hides the simulator perfectly while
being useless. Any method that reduced the probe while raising source
error has bought invariance with task information, and the target-probe
column is where that shows up.

## 4. Per-run detail

| Run | method | seed | epochs | train maps |
|---|---|---:|---:|---:|
| `E20_erm_multisource_seed00` | erm | 0 | 20 | 6,000 |
| `E21_dann_seed00` | dann | 0 | 20 | 6,000 |
| `E22_coral_seed00` | coral | 0 | 20 | 6,000 |
| `E23_mmd_seed00` | mmd | 0 | 20 | 6,000 |
| `E24_miest_like_seed00` | miest_like | 0 | 20 | 6,000 |
