import sys

from abc import abstractmethod

from ..constants import ModelSpecConstants
from ..core_model import CoreModel
from ..model_factory import BuiltinModelMeta
from ..model_spec.task_type import TaskType

PYTHON_VERSION = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"


class BuiltinModel(CoreModel, metaclass=BuiltinModelMeta):
    serialization_method = None
    task_type = TaskType.Regression

    flavor = {
        ModelSpecConstants.FLAVOR_NAME_KEY: None,
        ModelSpecConstants.SERIALIZATION_METHOD_KEY: None
    }

    default_conda = {
        "name": ModelSpecConstants.CONDA_ENV_NAME,
        'channels': ['defaults'],
        "dependencies": [f"python={PYTHON_VERSION}"]
    }

    # Load without flavor method shouldn't be called by Builtin models, instantiated to be placeholder
    @classmethod
    def load(cls, load_from):
        raise Exception("Load without flavor method shouldn't be called for Builtin models.")

    @classmethod
    @abstractmethod
    def load_with_flavor(cls, load_from: str, flavor: dict):
        pass

    @abstractmethod
    def get_default_feature_columns(self):
        pass
