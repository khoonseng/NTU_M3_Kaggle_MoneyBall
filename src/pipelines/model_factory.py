from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet

MODEL_REGISTRY = {
    "linear": LinearRegression,
    "ridge": Ridge,
    "lasso": Lasso,
    "elasticnet": ElasticNet,
}

def get_model(model_name, params=None):
    params = params or {}

    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Model '{model_name}' not supported")
    
    # if model_name == "lasso":
    #     return Lasso(max_iter=10000, **params)

    return MODEL_REGISTRY[model_name](**params)