from abc import abstractmethod, abstractclassmethod


class GenericModel(object):
    """Interface class to be inherited
    """

    _conda = None
    _local_dependencies = None
    _inputs = None
    _outputs = None
    _serving_config = None

    @abstractmethod
    def save(self, save_to, overwrite_if_exists=True):
        pass

    @abstractclassmethod
    def load(cls, load_from):
        pass

    @abstractmethod
    def predict(self, *args, **kwargs):
        pass

    @property
    @abstractmethod
    def conda(self):
        return self._conda

    @conda.setter
    @abstractmethod
    def conda(self, conda):
        self._conda = conda

    @property
    @abstractmethod
    def local_dependencies(self):
        return self._local_dependencies

    @local_dependencies.setter
    @abstractmethod
    def local_dependencies(self, local_dependencies):
        self._local_dependencies = local_dependencies

    @property
    @abstractmethod
    def inputs(self):
        return self._inputs
    
    @inputs.setter
    @abstractmethod
    def inputs(self, inputs):
        self._inputs = inputs

    @property
    @abstractmethod
    def outputs(self):
        return self._outputs

    @outputs.setter
    @abstractmethod
    def outputs(self, outputs):
        self._outputs = outputs

    @property
    @abstractmethod
    def serving_config(self):
        return self._serving_config
    
    @serving_config.setter
    @abstractmethod
    def serving_config(self, serving_config):
        self._serving_config = serving_config

    @abstractmethod
    def init_properties(self, conda, local_dependencies, inputs, outputs, serving_config):
        self._conda = conda
        self._local_dependencies = local_dependencies
        self._inputs = inputs
        self._outputs = outputs
        self._serving_config = serving_config
