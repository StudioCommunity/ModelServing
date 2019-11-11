from abc import ABC, abstractmethod, abstractclassmethod


class CoreModel(ABC):
    flavor = None

    @abstractmethod
    def save(self, save_to, overwrite_if_exists=True):
        pass
    
    @abstractclassmethod
    def load(cls, load_from):
        pass

    @abstractmethod
    def predict(self, *args, **kwargs):
        pass