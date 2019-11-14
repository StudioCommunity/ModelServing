from abc import abstractmethod
from ..core_model import CoreModel
from ..model_factory import BuiltinModelMeta


class BuiltinModel(CoreModel, metaclass=BuiltinModelMeta):
    serialization_method = None

    flavor = {
        "name": None,
        "serialization_method": None
    }

    @abstractmethod
    def config(self, model_spec):
        pass
