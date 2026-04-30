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

## Key Results
* MAE: 1.67 cards | RMSE: 2.13 cards | Log-Loss: 2.16 cards
* 66.5% of predictions within 2 cards of actual outcome
* Discovered away teams are more context-dependent (55% influenced by home opponent vs 45% own tendency)
* Removing referee scaling improved all metrics — team tendencies are learned from real historical matches (which already include various referees), so the learned mix weights implicitly capture referee effects; explicit scaling was redundant and conflicted with the learned weights

## Data
* Data files are not included in this repository. The model uses match-level data (5 seasons, 2021-2026), team statistics, and referee data sourced from football-data.co.uk, thestatsdontlie.com, and whoscored.com.

## Automation
* Weekly data collection and prediction generation is automated via weekly_update.py. The script downloads new match data from football-data.co.uk, appends it to the dataset, runs the prediction model, and logs performance metrics. Can be scheduled via Windows Task Scheduler to run automatically every Monday.

## Future Work
* Parameter tuning: Grid search for optimal decay rates (alpha values).
* Feature engineering: Derby/rivalry indicators.
* Alternative distributions: Negative binomial for overdispersion.
