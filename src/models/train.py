import pandas as pd
import os

from src.config.config import CONFIG
from src.pipelines.model_factory import get_model
from src.pipelines.pipeline_builder import build_pipeline
from src.models.evaluate import evaluate
from src.models.prediction import run_prediction
from src.models.model_util import save_model
from src.experiments.tracker import log_experiment
from src.tuning.tuning import run_grid_search, extract_best_per_metric
from src.models.ensemble import build_stacking_model
from src.models.model_util import load_model


def train_model(model_name, preprocessor, X_train, y_train, X_test, y_test, use_GridSearch):
    ensemble_config = CONFIG["ensemble"]
    model_path = os.path.join(ensemble_config["model_dir"], f"{model_name}.pkl")

    if ensemble_config["use_saved_models"] and os.path.exists(model_path):
        print(f"Loading existing model: {model_name}")
        return load_model(model_name, ensemble_config["model_dir"])

    model_type = CONFIG["model_type"][model_name]
    model_config = CONFIG["models"][model_name]
    model = get_model(model_name, model_config["params"])
    pipeline = build_pipeline(preprocessor, model)
    pipeline.fit(X_train, y_train)

    if use_GridSearch:
        search_config = CONFIG[model_type +"_search"]
        search = run_grid_search(
            pipeline,
            model_config,
            search_config,
            X_train,
            y_train        
        )

        # check best params
        best_params = search.best_params_
        best_score = search.best_score_
        print(f"Best {model_name} Parameters: {best_params}")
        print(f"Best {model_name} Score: {best_score:.4f}")

        best_model = search.best_estimator_
        y_pred = best_model.predict(X_test)
        n_features = best_model.named_steps["preprocessor"].get_feature_names_out().shape[0]
        results = evaluate(y_test, y_pred, n_features)

        print(f"\nBest {model_name} Params (refit metric):", search.best_params_)
        print(f"{model_name} Results:", results)

        # Extract best per metric
        best_per_metric = extract_best_per_metric(search)

        print("\nBest Params Per Metric:")
        for metric, info in best_per_metric.items():
            print(metric, "->", info)

        # Prepare CV summary
        cv_summary = {
            metric: info["score"]
            for metric, info in best_per_metric.items()
        }

        # Top 20 Features
        display_top_features(best_model=best_model, no_of_features=20)

        log_experiment(
            results,
            search.best_params_,
            model_name,
            cv_results=cv_summary
        )
    else:
        y_pred = pipeline.predict(X_test)

        if model_name == "lightgbm":
            n_features = X_train.shape[1]
        elif model_name == "lightgbm_pca":
            pca = pipeline.named_steps["preprocessor"].named_transformers_["num"].named_steps["pca"]
            n_features = pca.n_components_
        else:
            n_features = pipeline.named_steps["preprocessor"].get_feature_names_out().shape[0]

        results = evaluate(y_test, y_pred, n_features)
        print(f"\n{model_name} Regression Results:", results)

        # Top 20 Features
        display_top_features(best_model=pipeline, no_of_features=20)
    
        log_experiment(results, {}, model_name)

    # Save model
    if ensemble_config["use_saved_models"]:
        save_model(pipeline, model_name, ensemble_config["model_dir"])

    return pipeline

def run_all_models(preprocessors, X_train, y_train, X_test, y_test, predict_df, available_features):
    trained_pipelines = {}

    for models in CONFIG["models_to_run"]:
        model_name = models["name"]
        model_type = CONFIG["model_type"][model_name]
        use_grid = models["gridsearch"]
        preprocessor = preprocessors[model_type]

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

def display_top_features(best_model, no_of_features: int):
    print(f"Best model:\n", best_model)

    pipeline_model = best_model.named_steps["model"]
    pipeline_preprocessor = best_model.named_steps["preprocessor"]
    pipeline_feature_names = pipeline_preprocessor.get_feature_names_out()

    
    if "pca" in pipeline_preprocessor.named_transformers_["num"].named_steps:
        print("PCA applied — feature importance not directly interpretable")
        return

    if hasattr(pipeline_model, "coef_"):
        # Linear models
        importances = pipeline_model.coef_
    elif hasattr(pipeline_model, "feature_importances_"):
        # Tree models (LightGBM)
        importances = pipeline_model.feature_importances_
    else:
        print("Model does not support feature importance")
        return

    feature_importance = sorted(
        zip(pipeline_feature_names, importances),
        key=lambda x: abs(x[1]),
        reverse=True
    )

    print(f"\nTop {no_of_features} Features:")
    for name, value in feature_importance[:no_of_features]:
        print(f"{name}: {value}")


    # pipeline_model = best_model.named_steps["model"]
    # pipeline_preprocessor = best_model.named_steps["preprocessor"]
    # pipeline_feature_names = pipeline_preprocessor.get_feature_names_out()
    # pipeline_coefficients = pipeline_model.coef_

    # feature_importance = pd.DataFrame({
    #     'Feature': pipeline_feature_names,
    #     'Coefficient': pipeline_coefficients
    # }).sort_values('Coefficient', key=abs, ascending=False)

    # print(f"\nTop {no_of_features} Most Important Features:")
    # print(feature_importance.head(no_of_features))

def run_ensemble(X_train, y_train, X_test, y_test, config, n_features):
    stacking_model = build_stacking_model(config=config)

    stacking_model.fit(X_train, y_train)
    y_pred = stacking_model.predict(X_test)
    results = evaluate(y_test, y_pred, n_features)

    log_experiment(
        model_name="stacking",
        params=config,
        results=results
    )

    return stacking_model
