from abc import ABC, abstractmethod


class CoreModel(ABC):
    """
    CoreModel deals with flavor-specified behaviors.
    1. Save/Load raw_model
    2. Preprocess data in flavor-specified manner. e.g. torchvision.transform.
    3. Postprocess data according to task_type in flavor-specified manner. e.g. torch.nn.Softmax
    4. Provide default conda dependency
    """
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

    @abstractmethod
    def predict(self, *args, **kwargs):
        pass

    @property
    def conda(self):
        return self._conda

    @conda.setter
    def conda(self, conda):
        self._conda = conda
