# TinyML Energy Forecasting for Unreliable Grids (Nigeria Case Study)

Forecasting household energy consumption in contexts with unreliable grid power — and testing whether that forecast can run on cheap, constrained (TinyML/edge) hardware — using Nigeria as the motivating case study.

## Motivation

Public household energy forecasting datasets (e.g. UCI Individual Household Electric Power Consumption) assume continuous grid supply. That assumption doesn't hold in places like Nigeria, where households average **6.6 hrs/day** of grid power (PeopleSuN, *Scientific Data*, 2023) and rely heavily on generators for the rest. Deployment hardware also needs to survive the same outages it's forecasting around. This project sits at the intersection of three underexplored areas: energy forecasting research (grid-stable assumptions), Nigeria-specific energy research (survey-level, not time-series), and TinyML research (mostly vision/audio, not tabular forecasting).

## Research Questions

1. Can Nigeria-specific grid/solar/generator availability be credibly simulated on top of a Global North dataset (UCI), calibrated against real, cited statistics?
2. Can a compressed/edge-friendly model match or beat a full-size model on next-hour consumption forecasting?
3. How does forecast accuracy degrade under different types of missing data (random vs. outage-correlated), and can outage-aware imputation help?

## Methodology

1. **Data Engineering** — Load and clean the UCI household power dataset; resample to hourly.
2. **Nigeria Calibration** — Pull real Lagos solar irradiance from NASA POWER; simulate `grid_available` and `gen_active` schedules calibrated against real-world statistics (grid: 6.6 hrs/day, generator: 6.8 hrs/day) rather than assumed fixed schedules. Simulation lands within 0.01–0.05 hrs of both targets.
3. **Feature Engineering** — Lag features (1h/24h/168h), rolling mean/std (24h), target = next-hour consumption.
4. **Modeling** — Train and compare:
   - **XGBoost** (with hand-engineered lag/rolling features)
   - **GRU** (learns temporal structure from raw 24h sequences)
   - **Naive persistence baseline** (predict next hour = current hour)
5. **TinyML Compression** — Convert GRU to TFLite (fp32); reduce XGBoost tree count/depth; benchmark size and CPU inference latency.
6. **Missing-Data Ablation** — Inject synthetic MCAR (random) and MNAR (outage-correlated) missingness; compare naive forward-fill vs. an outage-aware imputation strategy.

## Key Results

- **Forecasting**: XGBoost beats the naive persistence baseline by ~22% on RMSE. The GRU underperforms both — expected, since it lacks hand-engineered features and uses a deliberately small parameter count for edge deployment (tree ensembles generally outperform RNNs on tabular time series).
- **Compression**: A 100-tree XGBoost model beats the original 500-tree model on accuracy at under a quarter of the size. At matched size, XGBoost still beats the GRU on every metric. The GRU is ~12x slower per prediction despite being the smallest file (framework overhead, not parameter count).
- **Missing-data ablation**: Naive forward-fill wins under random missingness; an outage-aware imputation method wins under outage-correlated missingness (the realistic Nigerian scenario) — a modest but directionally correct effect (~0.01–0.02 RMSE).

## Limitations

- **Simulated, not real, Nigerian data.** No public Nigerian smart-meter time series exists at this resolution (PeopleSuN is survey-level). This work uses UCI as a structural proxy and calibrates the *feature* side (grid/solar/generator availability) against real statistics — it does **not** correct the *label* side (the underlying consumption behavior is still from the original UCI households). A real metered pilot is the direct next step.
- **Latency benchmarks were measured on Colab CPU**, not physical microcontroller hardware. They're presented explicitly as a relative comparison between models, not a deployment measurement.
- The generator-hours figure (6.8 hrs/day) comes from a secondary source (industry news); tracing it to a primary source is a follow-up item. The grid-hours figure (6.6 hrs/day) is from peer-reviewed data (PeopleSuN, *Scientific Data*, 2023).

## Repository Structure

```
main.ipynb   — Full pipeline: data engineering → Nigeria calibration → feature
               engineering → XGBoost/GRU training → TinyML compression →
               missing-data ablation
```

## Requirements

- Python 3, pandas, numpy, scikit-learn, xgboost, tensorflow, matplotlib, joblib, requests
- Designed to run in Google Colab (mounts Google Drive for data access)

## Future Work

- Physical deployment/validation on real microcontroller hardware (e.g. via TFLite Micro)
- Real metered pilot data collection in Nigeria to validate the label side, not just the feature side
- Tracing the generator-hours statistic to its primary source

## Data Sources

- UCI Individual Household Electric Power Consumption dataset
- NASA POWER (solar irradiance, Lagos, Nigeria)
- PeopleSuN grid availability statistics (*Scientific Data*, 2023)
```
