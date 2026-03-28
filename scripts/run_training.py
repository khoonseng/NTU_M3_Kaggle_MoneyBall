import pandas as pd
import numpy as np
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso

from src.pipelines.preprocessing import build_preprocessor
from src.pipelines.pipeline_builder import build_pipeline
from src.pipelines.model_factory import get_model
from src.models.evaluate import evaluate
from src.tuning.tuning import run_grid_search, extract_best_per_metric
from src.experiments.tracker import log_experiment
from src.config.config import CONFIG

def main():
    # -------------------------
    # 1. Load data
    # -------------------------
    data_df = pd.read_csv("data/data.csv")
    predict_df = pd.read_csv("data/predict.csv")

    # Display basic information about the datasets
    print(f"Data set shape: {data_df.shape}")
    print(f"Predict set shape: {predict_df.shape}")

    # Select only the default features from DATA_DESCRIPTION.md
    default_features = [
        # Basic Statistics
        'G', 'R', 'AB', 'H', '2B', '3B', 'HR', 'BB', 'SO', 'SB', 'CS', 'HBP', 'SF',
        'RA', 'ER', 'ERA', 'CG', 'SHO', 'SV', 'IPouts', 'HA', 'HRA', 'BBA', 'SOA',
        'E', 'DP', 'FP', 'attendance', 'BPF', 'PPF',
        
        # Derived Features
        'R_per_game', 'RA_per_game', 'mlb_rpg',
        
        # Era Indicators
        'era_1', 'era_2', 'era_3', 'era_4', 'era_5', 'era_6', 'era_7', 'era_8',
        
        # Decade Indicators
        'decade_1910', 'decade_1920', 'decade_1930', 'decade_1940', 'decade_1950',
        'decade_1960', 'decade_1970', 'decade_1980', 'decade_1990', 'decade_2000', 'decade_2010'
    ]

    # Filter features that exist in both datasets
    available_features = [col for col in default_features if col in data_df.columns and col in predict_df.columns]
    print(f"Number of available default features: {len(available_features)}")

    # Separate features and target variable
    X = data_df[available_features]
    y = data_df['W']
    print(X.dtypes)

    # Perform the split (adjust test_size / random_state as needed)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,      # 20% for testing
        random_state=42    # ensures reproducibility
    )

    # -------------------------
    # 2. Preprocessing
    # -------------------------
    # Scale features
    # Identify columns to exclude from scaling (one-hot encoded and label columns)
    one_hot_cols = [col for col in X_train.columns if col.startswith(('era_', 'decade_'))]
    other_cols = [col for col in X_train.columns if col not in one_hot_cols]

    preprocessor = build_preprocessor(other_cols, None)

    # -------------------------
    # 3. Baseline: Linear Regression
    # -------------------------
    # lr_model = LinearRegression()
    model_name = "linear"
    params = CONFIG["models"][model_name]
    lr_model = get_model(model_name, params)
    lr_pipeline = build_pipeline(preprocessor, lr_model)

    lr_pipeline.fit(X_train, y_train)
    y_pred_lr = lr_pipeline.predict(X_test)

    lr_results = evaluate(y_test, y_pred_lr)
    print("\nLinear Regression Results:", lr_results)

    # Feature importance from Linear Regression
    print(lr_pipeline)
    lr_pipeline_model = lr_pipeline.named_steps["model"]
    lr_pipeline_preprocessor = lr_pipeline.named_steps["preprocessor"]
    lr_pipeline_feature_names = lr_pipeline_preprocessor.get_feature_names_out()
    lr_pipeline_coefficients = lr_pipeline_model.coef_

    feature_importance = pd.DataFrame({
        'Feature': lr_pipeline_feature_names,
        'Coefficient': lr_pipeline_coefficients
    }).sort_values('Coefficient', key=abs, ascending=False)

    print("\nTop 10 Most Important Features:")
    print(feature_importance.head(10))

    log_experiment(lr_results, {}, "linear")

    # -------------------------
    # 4. Lasso with GridSearchCV
    # -------------------------
    # lasso_model = Lasso(max_iter=10000)
    # model_name = "lasso"
    # params = CONFIG["models"][model_name]
    # lasso_model = get_model(model_name, params)
    # lasso_pipeline = build_pipeline(preprocessor, lasso_model)

    # param_grid = CONFIG["models"]["lasso"]["param_grid"]
    # scoring = CONFIG["scoring"]
    # refit_metric = CONFIG["refit_metric"]

    # grid = run_grid_search(
    #     lasso_pipeline,
    #     param_grid,
    #     X_train,
    #     y_train,
    #     scoring,
    #     refit_metric
    # )

    # # check best params
    # best_params = grid.best_params_
    # best_score = grid.best_score_
    # print(f"Best Lasso Parameters: {best_params}")
    # print(f"Best Lasso Score: {best_score:.4f}")

    # best_model = grid.best_estimator_
    # y_pred_lasso = best_model.predict(X_test)

    # lasso_results = evaluate(y_test, y_pred_lasso)

    # print("\nBest Lasso Params (refit metric):", grid.best_params_)
    # print("Lasso Results:", lasso_results)

    # # Extract best per metric
    # best_per_metric = extract_best_per_metric(grid)

    # print("\nBest Params Per Metric:")
    # for metric, info in best_per_metric.items():
    #     print(metric, "->", info)

    # # Prepare CV summary
    # cv_summary = {
    #     metric: info["score"]
    #     for metric, info in best_per_metric.items()
    # }

    # log_experiment(
    #     lasso_results,
    #     grid.best_params_,
    #     "lasso",
    #     cv_results=cv_summary
    # )

    # Prepare Submission
    predict_X = predict_df[available_features].copy()
    predict_preds = lr_pipeline.predict(predict_X)

    # Build submission in the same format as submission.csv
    submission_df = pd.DataFrame({
        'ID': predict_df['ID'],
        'W': np.round(predict_preds).astype(int)
    })

    PROJECT_ROOT = Path(os.environ.get("LOCAL_DATA_DIR", Path.cwd())).resolve()    
    LOCAL_SUBMISSION_FILE = PROJECT_ROOT.home().name + "_submission_predict.csv"
    submission_path = PROJECT_ROOT / "submission" / LOCAL_SUBMISSION_FILE
    submission_df.to_csv(submission_path, index=False)
    print(f"Kaggle submission saved to {submission_path}")

if __name__ == "__main__":
    main()