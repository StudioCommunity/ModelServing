import importlib
import re
import inspect

from abc import ABCMeta

from azureml.designer.model.constants import ModelSpecConstants
from azureml.designer.model.logger import get_logger

logger = get_logger(__name__)


def _get_default_flavor_name(cls):
    """
    Get flavor name from class
    :param cls: class
    :return: The module name right after "builtin_models" module path
    >>> cls1 = azureml.designer.model.builtin_models.sklearn.SklearnModel
    >>> _get_default_flavor_name(cls1)
    SklearnModel
    >>> cls2 = azureml.designer.model.builtin_models.pytorch.cloudpickle
    >>> _get_default_flavor_name(cls2)
    pytorch
    """
    module_name = cls.__module__
    root_module_name = "azureml.designer.model.builtin_models."
    flavor = module_name.replace(root_module_name, "")
    if "." in flavor:
        flavor = re.sub(r"\..+", "", flavor)
    return flavor


def _get_flavor_key(flavor):
    key = f'{flavor[ModelSpecConstants.FLAVOR_NAME_KEY]},{flavor[ModelSpecConstants.SERIALIZATION_METHOD_KEY]}'
    return key


class BuiltinModelMeta(ABCMeta):
    def __init__(cls, name, bases, attr_dict):
        super().__init__(name, bases, attr_dict)
        if inspect.isabstract(cls):
            return
        flavor = cls.flavor
        if not flavor[ModelSpecConstants.FLAVOR_NAME_KEY]:
            flavor[ModelSpecConstants.FLAVOR_NAME_KEY] = _get_default_flavor_name(cls)
        if cls.serialization_method:
            flavor[ModelSpecConstants.SERIALIZATION_METHOD_KEY] = cls.serialization_method

        if flavor[ModelSpecConstants.FLAVOR_NAME_KEY] is None:
            raise TypeError(f"Builtin model {cls} should be have a flavor name")

        key = _get_flavor_key(flavor)
        if key in FlavorRegistry.flavors:
            raise TypeError(f"{key} in {cls} is not a unique flavor name")

        logger.info(f"register {key} to flavor registry")
        FlavorRegistry.flavors[key] = cls


class FlavorRegistry(object):
    # flavor_key -> cls
    flavors = dict()

    @classmethod
    def get_flavor(cls, flavor_name, serialization_method=None):
        key = f"{flavor_name},{serialization_method}"
        if key not in FlavorRegistry.flavors:
            module_name = f"{__package__}.builtin_models.{flavor_name}.{serialization_method}"
            try:
                importlib.import_module(module_name)
            except ModuleNotFoundError:
                logger.warning(f"Not found module: {module_name}")

        if key not in FlavorRegistry.flavors:
            return None
        return FlavorRegistry.flavors[key]

    @classmethod
    def supported_flavors(cls, flavor_name=None):
        if not flavor_name:
            return FlavorRegistry.flavors.items()
        else:
            return [FlavorRegistry.flavors[key] for key in FlavorRegistry.flavors if
                    FlavorRegistry.flavors[key].flavor[ModelSpecConstants.FLAVOR_NAME_KEY] == flavor_name]


class ModelFactory(object):

    @classmethod
    def get_model_class(cls, flavor) -> type:
        flavor_name = flavor[ModelSpecConstants.FLAVOR_NAME_KEY].lower()
        if flavor_name == ModelSpecConstants.CUSTOM_MODEL_FLAVOR_NAME:
            module_path = flavor[ModelSpecConstants.MODEL_MODULE_KEY]
            class_name = flavor[ModelSpecConstants.MODEL_CLASS_KEY]
            module = importlib.import_module(module_path)
            return getattr(module, class_name)

        flavor_cls = FlavorRegistry.get_flavor(flavor_name, flavor[ModelSpecConstants.SERIALIZATION_METHOD_KEY])
        return flavor_cls
