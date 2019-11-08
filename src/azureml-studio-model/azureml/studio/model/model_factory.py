import importlib

from .generic_model import GenericModel
from .logger import get_logger

logger = get_logger(__name__)

class ModelFactory(object):

    @classmethod
    def load_model(cls, model_path: str, config: dict) -> GenericModel:
        flavor = config["flavor"]
        flavor_name = flavor["name"].lower()
        if flavor_name == "custom":
            module_path = flavor["module"]
            class_name = flavor["class"]
            module = importlib.import_module(module_path)
            return getattr(module, class_name).load(model_path)
        if flavor_name == "pytorch":
            if flavor["serializtion_method"] == "cloudpickle":
                from .pytorch.cloudpickle import PytorchCloudPickleModel
                return PytorchCloudPickleModel.load(model_path, config)