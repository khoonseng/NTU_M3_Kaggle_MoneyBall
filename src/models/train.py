import pandas as pd

from src.config.config import CONFIG
from src.pipelines.model_factory import get_model
from src.pipelines.pipeline_builder import build_pipeline
from src.models.evaluate import evaluate
from src.models.prediction import run_prediction
from src.experiments.tracker import log_experiment
from src.tuning.tuning import run_grid_search, extract_best_per_metric


def train_model(model_name, preprocessor, X_train, y_train, X_test, y_test, use_GridSearch):
    params = CONFIG["models"][model_name]["params"]
    model = get_model(model_name, params)
    pipeline = build_pipeline(preprocessor, model)
    pipeline.fit(X_train, y_train)

    if model_name == 'linear':
        y_pred = pipeline.predict(X_test)
        n_features = pipeline.named_steps["preprocessor"].get_feature_names_out().shape[0]
        results = evaluate(y_test, y_pred, n_features)
        print("\nLinear Regression Results:", results)

        # Feature importance from Linear Regression
        print(pipeline)
        pipeline_model = pipeline.named_steps["model"]
        pipeline_preprocessor = pipeline.named_steps["preprocessor"]
        pipeline_feature_names = pipeline_preprocessor.get_feature_names_out()
        pipeline_coefficients = pipeline_model.coef_

        feature_importance = pd.DataFrame({
            'Feature': pipeline_feature_names,
            'Coefficient': pipeline_coefficients
        }).sort_values('Coefficient', key=abs, ascending=False)

        print("\nTop 20 Most Important Features:")
        print(feature_importance.head(20))
    
        log_experiment(results, {}, model_name)

    if use_GridSearch:
        param_grid = CONFIG["models"][model_name]["param_grid"]
        scoring = CONFIG["scoring"]
        refit_metric = CONFIG["refit_metric"]

        grid = run_grid_search(
            pipeline,
            param_grid,
            X_train,
            y_train,
            scoring,
            refit_metric
        )

        # check best params
        best_params = grid.best_params_
        best_score = grid.best_score_
        print(f"Best {model_name} Parameters: {best_params}")
        print(f"Best {model_name} Score: {best_score:.4f}")

        best_model = grid.best_estimator_
        y_pred = best_model.predict(X_test)
        n_features = best_model.named_steps["preprocessor"].get_feature_names_out().shape[0]
        results = evaluate(y_test, y_pred, n_features)

        print(f"\nBest {model_name} Params (refit metric):", grid.best_params_)
        print(f"{model_name} Results:", results)

        # Extract best per metric
        best_per_metric = extract_best_per_metric(grid)

        print("\nBest Params Per Metric:")
        for metric, info in best_per_metric.items():
            print(metric, "->", info)

        # Prepare CV summary
        cv_summary = {
            metric: info["score"]
            for metric, info in best_per_metric.items()
        }              

        log_experiment(
            results,
            grid.best_params_,
            model_name,
            # cv_results=cv_summary
        )

    return pipeline

def run_all_models(preprocessor, X_train, y_train, X_test, y_test, predict_df, available_features):
    trained_pipelines = {}

    for model_cfg in CONFIG["models_to_run"]:
        model_name = model_cfg["name"]
        use_grid = model_cfg["gridsearch"]

        print(f"\n===== Training {model_name.upper()} (GridSearch={use_grid}) =====")

        pipeline = train_model(
            model_name=model_name,
            preprocessor=preprocessor,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            use_GridSearch=use_grid
        )

        trained_pipelines[model_name] = pipeline

        # Run prediction
        run_prediction(predict_df, available_features, pipeline)

    return trained_pipelines