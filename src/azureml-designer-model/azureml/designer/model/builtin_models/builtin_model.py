import sys

from abc import abstractmethod

from .. import constants
from ..core_model import CoreModel
from ..model_factory import BuiltinModelMeta

PYTHON_VERSION = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"


class BuiltinModel(CoreModel, metaclass=BuiltinModelMeta):
    serialization_method = None

    flavor = {
        "name": None,
        "serialization_method": None
    }

    default_conda = {
        "name": constants.CONDA_ENV_NAME,
        'channels': ['defaults'],
        "dependencies": [f"python={PYTHON_VERSION}"]
    }

    @classmethod
    @abstractmethod
    def load_with_modelspec(cls, load_from: str, model_spec: dict):
        pass
