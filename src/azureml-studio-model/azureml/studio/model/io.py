from .model_wrapper import ModelWrapper

def save_generic_model(model,
                       path: str ="./AzureMLModel",
                       overwrite_if_exists: bool = True
                       ):
    ModelWrapper.save(model, path, overwrite_if_exists)