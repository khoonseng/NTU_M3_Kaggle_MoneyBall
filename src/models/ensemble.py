from sklearn.ensemble import StackingRegressor
from src.models.model_util import load_model
from src.pipelines.model_factory import get_model

def build_stacking_model(config):
    base_models = []
    
    for name in config["base_models"]:
        if config["use_saved_models"]:
            model = load_model(name, config["model_dir"])
        else:
            model = get_model(name)

        base_models.append((name, model))

    final_model = get_model(config["final_model"])

    stacking_model = StackingRegressor(
        estimators=base_models,
        final_estimator=final_model,
        n_jobs=-1
    )

    return stacking_model