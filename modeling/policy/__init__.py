from .denoise_actor_3d import DenoiseActor


def fetch_model_class(model_type):
    if model_type != "denoise3d":
        raise ValueError(f"Unknown model: {model_type}")
    return DenoiseActor
