import sys

from abc import abstractmethod

from .. import constants
from ..core_model import CoreModel
from ..model_factory import BuiltinModelMeta
from ..model_spec.task_type import TaskType

PYTHON_VERSION = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"


class BuiltinModel(CoreModel, metaclass=BuiltinModelMeta):
    serialization_method = None
    task_type = TaskType.Regression

    flavor = {
        "name": None,
        "serialization_method": None
    }

    default_conda = {
        "name": constants.CONDA_ENV_NAME,
        'channels': ['defaults'],
        "dependencies": [f"python={PYTHON_VERSION}"]
    }

    # Builtin models don't instantiate load without flavor method
    @classmethod
    def load(cls, load_from):
        pass

    @classmethod
    @abstractmethod
    def load_with_flavor(cls, load_from: str, flavor: dict):
        pass

    @abstractmethod
    def get_default_feature_columns(self):
        pass
