from sklearn.model_selection import GridSearchCV

def run_grid_search(pipeline, param_grid, X_train, y_train, scoring, refit_metric):
    """
    scoring: dict of metrics
    refit_metric: which metric to select best model
    """

    grid = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        scoring=scoring,
        refit=refit_metric,  # key decision
        cv=5,
        n_jobs=-1,
        return_train_score=True
    )

    grid.fit(X_train, y_train)
    return grid

def extract_best_per_metric(grid):
    """
    Extract best params for EACH metric (not just refit metric)
    """
    results = grid.cv_results_
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