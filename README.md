# Premier League Card Prediction

## Overview
This project predicts the number of yellow and red cards in Premier League football matches using statistical modeling. The goal is to accurately forecast match disciplinary cards using time-weighted team profiles and walk-forward validation.

## Tools Used
* Python (Pandas, NumPy, SciPy)
* Matplotlib
* Jupyter Notebook / VS Code

## Key Steps
* Data cleaning and standardization (team/referee names)
* Time-weighted profile construction (exponential decay)
* Learned mixing weights (correlation-based optimization)
* Walk-forward validation (no data leakage)
* Poisson probability framework
* Model evaluation and performance tracking

## Models
* **Prototype (Phase 1)**: Season-level averages with fixed 50/50 mixing weights
* **Production (Phase 2)**: Match-level data with learned mixing weights
* **Side-piece: Poisson regression (GLM)**: benchmarks the production model against a textbook MLE-fitted Poisson GLM using the same underlying features. See [Poisson Regression Comparison](#poisson-regression-comparison) below and `03_poisson_regression_results.md` for the full writeup.

## Key Results
* MAE: 1.67 cards | RMSE: 2.13 cards | Log-Loss: 2.16 cards
* 66.5% of predictions within 2 cards of actual outcome
* Discovered away teams are more context-dependent (55% influenced by home opponent vs 45% own tendency)
* Removing referee scaling improved all metrics — team tendencies are learned from real historical matches (which already include various referees), so the learned mix weights implicitly capture referee effects; explicit scaling was redundant and conflicted with the learned weights

## Poisson Regression Comparison
The production model is a heuristic: lambda is built from correlation-weighted team profiles, with Poisson applied only afterward to turn lambda into a scorable distribution. As a side-piece, a textbook Poisson regression (GLM, fitted by maximum likelihood with a log link) was benchmarked against it using the exact same features, plus one the production model excludes (referee tendency).

| Model | MAE | RMSE | Log-Loss | Within 2 cards |
|---|---|---|---|---|
| Existing (correlation-weighted) | 1.69 | 2.14 | 2.162 | 65.7% |
| Poisson regression (GLM) | 1.70 | 2.14 | 2.162 | 65.1% |

They came out statistically indistinguishable — the heuristic edges it out slightly across every metric, and is cheaper to compute. Adding the referee feature to the GLM didn't help either, corroborating the earlier finding that team tendencies already implicitly capture referee effects. Full writeup, including the worked maths behind the GLM, in `03_poisson_regression_results.md`.

## Data
* Data files are not included in this repository. The model uses match-level data (5 seasons, 2021-2026), team statistics, and referee data sourced from football-data.co.uk, thestatsdontlie.com, and whoscored.com.

## Automation
* Weekly data collection and prediction generation is automated via weekly_update.py. The script downloads new match data from football-data.co.uk, appends it to the dataset, runs the prediction model, and logs performance metrics. Can be scheduled via Windows Task Scheduler to run automatically every Monday.

## Future Work
* Parameter tuning: Grid search for optimal decay rates (alpha values).
* Feature engineering: Derby/rivalry indicators.
* Alternative distributions: Negative binomial for overdispersion.
