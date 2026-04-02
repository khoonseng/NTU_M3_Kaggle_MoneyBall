from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from lightgbm import LGBMRegressor
# from sklearn.decomposition import PCA

MODEL_REGISTRY = {
    "linear": LinearRegression,
    "ridge": Ridge,
    "lasso": Lasso,
    "elasticnet": ElasticNet,
    "lightgbm": LGBMRegressor,
    "lightgbm_pca": LGBMRegressor
}

def get_model(model_name, params=None):
    params = params or {}

    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Model '{model_name}' not supported")

    return MODEL_REGISTRY[model_name](**params)