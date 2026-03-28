from sklearn.pipeline import Pipeline

def build_pipeline(preprocessor, model):
    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])