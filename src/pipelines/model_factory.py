from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet

def get_model(model_name, params=None):
    params = params or {}

    models = {
        "linear": LinearRegression(),
        "ridge": Ridge(**params),
        "lasso": Lasso(**params),
        "elasticnet": ElasticNet(**params),
    }

    return models[model_name]