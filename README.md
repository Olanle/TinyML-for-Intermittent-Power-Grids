# TinyML for Intermittent Power Grids
### Missing-Data-Aware Energy Forecasting and Edge Model Compression for Nigerian Households

A research project studying household energy forecasting under intermittent-grid conditions, combining **data engineering** (real-data-calibrated feature simulation, missingness-aware imputation) with **TinyML** (edge-deployable model compression and benchmarking).

---

## Overview

Public energy-forecasting datasets (e.g. UCI Individual Household Electric Power Consumption) assume continuous, stable grid power. This assumption does not hold in Nigeria, where households alternate between grid, solar, and generator power on an unpredictable schedule.

This project:
1. Calibrates Nigeria-specific power-source features against real, cited data (rather than arbitrary fixed schedules)
2. Tests whether outage-aware missing-data imputation outperforms naive forward-fill under realistic conditions
3. Compares XGBoost and a compact GRU on accuracy, model size, and inference latency — for deployment on constrained edge hardware

---

## Research Questions

- **RQ1:** Does calibrating a foreign proxy dataset against real Nigerian survey and irradiance data change what the model learns?
- **RQ2:** Does treating outage-correlated missingness differently from random missingness improve imputation accuracy?
- **RQ3:** How do XGBoost and a compact GRU trade off accuracy, size, and latency for edge deployment?

---

## Data

- **Base dataset:** [UCI Individual Household Electric Power Consumption](https://archive.ics.uci.edu/ml/datasets/individual+household+electric+power+consumption) — 2,075,259 minute-level readings, Sceaux, France (Dec 2006–Nov 2010), resampled to hourly (34,589 rows). Used as a structural proxy; the underlying consumption behavior reflects a French household, not Nigerian usage — this is an explicitly stated limitation, not resolved by calibration.
- **Grid availability:** calibrated to 6.6 hrs/day average supply (Pelz et al., *Scientific Data*, 2023 — PeopleSuN survey, 3,599 Nigerian households).
- **Solar availability:** modeled from a 10-year (2015–2024) [NASA POWER](https://power.larc.nasa.gov/) irradiance climatology for Lagos, mapped by (month, hour) to capture harmattan/rainy-season seasonality.
- **Generator activation:** calibrated to ~6.8 hrs/day national self-supply estimate (industry report), reconciled against a weekday/weekend "patience window" behavioral model.

---

## Methodology

| Stage | Description |
|---|---|
| **Part A** | UCI baseline pipeline — load, datetime index, missing-value handling, hourly resampling, temporal features |
| **Part B** | Nigeria-specific calibration — NASA POWER solar climatology, PeopleSuN-calibrated grid availability, solved-probability generator model |
| **Part C** | Feature finalization — lag/rolling features, target creation, redundancy cleanup (dropped features found to be near-duplicates via correlation check) |
| **Part D** | XGBoost training & evaluation |
| **Part E** | GRU training & evaluation (raw windowed sequences, no hand-engineered lag features) |
| **Part F** | Compression & edge benchmarking — XGBoost tree/depth sweep, GRU TFLite quantization, relative inference latency |
| **Part G** | Outage-aware imputation ablation — controlled MCAR vs. MNAR synthetic missingness study |

---

## Key Results

**Forecasting (test set)**

| Model | MAE (kW) | RMSE (kW) | MAPE |
|---|---|---|---|
| Naive persistence | 0.361 | 0.566 | 43.80% |
| **XGBoost (final)** | **0.304** | **0.439** | 43.38% |
| GRU (compact) | 0.390 | 0.547 | 61.43% |

**Edge deployment (compression sweep)**

| Config | Size | MAE | Latency* |
|---|---|---|---|
| XGBoost — 500 trees (original) | 2236.2 KB | 0.304 | 16.2 ms |
| **XGBoost — 100 trees (best)** | **479.9 KB** | **0.302** | **7.7 ms** |
| XGBoost — 50 trees, depth 4 | 95.5 KB | 0.338 | 7.8 ms |
| GRU — native Keras | 89.9 KB | 0.390 | 97.4 ms |

*Colab-CPU relative timing — not representative of physical microcontroller latency (no hardware access; noted as a limitation).*

**Missing-data ablation (imputation RMSE)**

| | Naive forward-fill | Type-aware imputation |
|---|---|---|
| MCAR (random) | **0.684** | 0.698 |
| MNAR (outage-correlated) | 0.692 | **0.680** |

Effect direction flips exactly as hypothesized: naive fill wins under random missingness; outage-aware imputation wins under outage-correlated missingness — the realistic Nigerian scenario.

---

## Key Findings

- Calibrating grid/generator features against real data collapsed their feature importance from a combined 23.9% to 1.6% — revealing the original fixed-schedule features' high importance was a deterministic-encoding artifact, not genuine signal.
- A 100-tree XGBoost model outperforms the original 500-tree model in accuracy at less than a quarter of the file size.
- At matched model size (~90–95 KB), XGBoost outperforms the GRU on every metric — gradient-boosted trees remain more size-efficient for this tabular time-series task even under tight compression.
- The GRU is ~12x slower per inference despite being the smallest model on disk — a framework-overhead effect, not a parameter-count effect.

---

## Limitations

- Target variable (consumption behavior) is still a French-household proxy; calibration addresses feature realism, not label realism.
- Nigerian calibration sources are survey-level (PeopleSuN) or industry-report-level (generator estimate), not high-frequency metered ground truth.
- Latency benchmarking is Colab-CPU based; no physical microcontroller validation performed.
- GRU results exhibit minor run-to-run variance due to non-deterministic GPU/cuDNN operations.

---

## Future Work

- Physical hardware deployment (ESP32) for real latency and energy-draw measurement
- Real metered pilot data from Nigerian households to replace the proxy target variable
- On-device continual learning for household/feeder-specific adaptation
- SHAP-based per-prediction interpretability for deployed forecasting decisions

---

## References

- Hébrail, G. & Bérard, A. (2012). Individual Household Electric Power Consumption Dataset. UCI Machine Learning Repository.
- Chen, T. & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. *KDD 2016*.
- Pelz, S., Chinichian, N., Neyrand, C. & Blechinger, P. (2023). Electricity supply quality and use among rural and peri-urban households and small firms in Nigeria. *Scientific Data*, 10, 273.
- NASA POWER Project, NASA Langley Research Center.
- Ezennaya, S. et al. Data on the daily electricity load profile and solar PV system components for residential buildings in Lagos, Nigeria. *Data in Brief*.
- Tekedia (2022). 40% Of Nigerian Households Use Generators, Spend $14bn On Fuel.
- ACM Computing Surveys (2025). From Tiny Machine Learning to Tiny Deep Learning: A Survey.

---
