from abc import abstractmethod
from .core_model import CoreModel

class BuiltinModel(CoreModel):

    @abstractmethod
    def config(self, model_spec):
        pass