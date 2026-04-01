from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

def run_grid_search(pipeline, model_config, search_config, X_train, y_train):
    if search_config["method"] == "grid":
        search = GridSearchCV(
            estimator=pipeline,
            param_grid=model_config["param_grid"],
            scoring=search_config["scoring"],
            refit=search_config["refit_metric"],  # key decision
            cv=search_config["cv"],
            n_jobs=search_config["n_jobs"],
            return_train_score=True
        )

    elif search_config["method"] == "random":
        search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=model_config["param_dist"],
            n_iter=search_config["n_iter"],
            scoring=search_config["scoring"],
            refit=search_config["refit_metric"],  # key decision
            cv=search_config["cv"],
            n_jobs=search_config["n_jobs"],
            return_train_score=True,
            random_state=42
        )

    search.fit(X_train, y_train)
    return search

def extract_best_per_metric(search):
    """
    Extract best params for EACH metric (not just refit metric)
    """
    results = search.cv_results_
    metrics = [k.replace("mean_test_", "") for k in results if k.startswith("mean_test_")]

    best_per_metric = {}

    for metric in metrics:
        scores = results[f"mean_test_{metric}"]

        # For neg metrics → higher is better
        best_idx = scores.argmax()

        best_per_metric[metric] = {
            "score": scores[best_idx],
            "params": results["params"][best_idx]
        }

    return best_per_metric