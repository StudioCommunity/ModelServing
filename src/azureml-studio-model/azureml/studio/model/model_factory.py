import importlib

from .logger import get_logger

logger = get_logger(__name__)

class ModelFactory(object):

    @classmethod
    def get_model_class(cls, flavor) -> type:
        flavor_name = flavor["name"].lower()
        if flavor_name == "custom":
            module_path = flavor["module"]
            class_name = flavor["class"]
            module = importlib.import_module(module_path)
            return getattr(module, class_name)
        if flavor_name == "pytorch":
            if flavor["serialization_method"] == "cloudpickle":
                from .builtin_models.pytorch.cloudpickle import PytorchCloudPickleModel
                return PytorchCloudPickleModel
