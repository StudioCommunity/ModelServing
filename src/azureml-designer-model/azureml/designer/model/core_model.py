from abc import ABC, abstractmethod


class CoreModel(ABC):
    flavor = None
    _conda = None

    @abstractmethod
    def save(self, save_to: str, overwrite_if_exists=True):
        """Save CoreModel object to path save_to
        
        Arguments:
            save_to {str} -- path to save, can be file path or directory path
        
        Keyword Arguments:
            overwrite_if_exists {bool} -- Overwrite exist files if true, throw exeption otherwise (default: {True})
        """
        pass

    @classmethod
    @abstractmethod
    def load(cls, load_from: str):
        """ Load a CoreModel object from path load_from
        
        Arguments:
            load_from {str} -- path to file or directory
        """
        pass

    # TODO: support non-dataframe data structure
    @abstractmethod
    def predict(self, *args, **kwargs):
        pass

    @property
    def conda(self):
        return self._conda

    @conda.setter
    def conda(self, conda):
        self._conda = conda
