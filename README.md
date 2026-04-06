# 🏆 NTU Module 3 Kaggle Moneyball Project

This project is part of the NTU SCTP Module 3, focused on building a machine learning solution for a Kaggle Moneyball dataset to predict the number of games a Major League Baseball team will win in a season based on their performance statistics. This is a regression problem where historical MLB team data will be used to build a model that can accurately forecast seasonal win totals.

## The Data
The dataset contains comprehensive team statistics from the 2016 Lahman Baseball Database, including:

Batting statistics: Runs, hits, home runs, strikeouts, etc.
Pitching statistics: Earned run average, saves, strikeouts, etc.
Fielding statistics: Errors, double plays, fielding percentage
Team information: Year, team name, franchise ID
Game outcomes: Wins, losses, championships

## Evaluation
The objective is to minimize Mean Absolute Error (MAE), which measures the average absolute difference between the predicted wins and actual wins. Lower scores indicate better performance, with a perfect score being 0. The MAE is calculated as the mean of the absolute values of the differences between predicted and actual wins across all teams.


##  🚀 Project Overview

This repository implements a modular and extensible ML pipeline for tabular data:
- Feature engineering (offensive, defensive, pitching, Sabermetrics)
- Model Experimentation with Ensemble stacking
- Config-driven model training and hyperparameter tuning
- Experiment tracking

## 1. Feature Engineering
- Normalizing statistics on per-game basis e.g. <br>
9 offensive stats per game <br>
10 pitching stats per game <br>
2 defensive stats per game <br>
<br>
- Compute metrics from Sabermetrics e.g. <br>
 WHIP <br>
 Slugging Percentage (SLG) <br>
 On-Base Percentage (OBP) <br>
 On-Base Plus Slugging (OPS) <br>
 WHIP x FP -> Interaction between pitching and defense<br>
 OPS x offense_index -> Interaction between offense and era-normalization<br>


## 2. Model Experimentation
Supported base models:
- Linear Regression
- Ridge Regression
- Lasso Regression
- ElasticNet Regression
- LightGBM (with optional PCA)

Supported Ensemble method:
- Stacking with ElasticNet/Ridge as the meta-model
- Base models are exported as pickle files and loaded into StackingRegressor model

## 3. Config-Driven Design
All models, parameters, and tuning strategies are defined in [config.py](/src/config/config.py) e.g.

```python
"models": {
        "linear": {
            "params": {}
        },
        "lasso": {
            "params": {
                "max_iter": 10000,
                "random_state": 42
            },
            "param_grid": {
                "model__alpha": [0.01, 0.1, 1.0],
                "model__max_iter": [5000, 10000, 15000, 20000],
                "model__tol": [1e-4, 1e-3, 1e-2],
                "model__selection": ["cyclic", "random"]
            }
        },
        "ridge": {
            "params": {
                "max_iter": None,
                "random_state": 42
            },
            "param_grid": {
                "model__alpha": [1e-3, 1e-2, 0.1, 1.0, 10.0],
                "model__solver": ["auto", "lsqr", "saga"],
                "model__max_iter": [None, 1000, 5000, 10000],
                "model__tol": [1e-3, 1e-2]
            }
        }
}
```

### Hyperparameter Tuning
Supports:
- GridSearchCV
- RandomizedSearchCV (faster, recommended)

Key model evaluation features:
- Multi-metric scoring (MAE, RMSE, R²)
- Custom refit metric (MAE)

## 4. Experiment Tracking
All results are logged via [tracker.py](/src/experiments/tracker.py)

Tracks:
- MAE, RMSE, R², Adjusted R²
- Model configurations

## Project Structure
```bash
.
├── data/                      # Train / test datasets
├── experiments/
│   └── results.csv            # Experiment results
├── models/                    # Saved models (pickle)
├── notebooks/                 # EDA and models experiments
├── scripts/
│   └── run_training.py        # Main training pipeline        
├── src/
│   └── config.py              # Model + hyperparameter tuning + ensemble stacking configuration
│   ├── experiments/
│       └── tracker.py         # Experiment logging                  
│   ├── models/                # Train, prediction, ensemble, evaluate metrics, save/load models
│   ├── pipelines/             # Preprocessing, pipelines
│   ├── tuning/                # GridSearchCV, RandomSearchCV
│   ├── submission/            # Kaggle submission files based on models
```