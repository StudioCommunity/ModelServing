from .logger import get_logger

logger = get_logger(__name__)

class ResourceConfig(object):

    # TODO: Set the object variables to be properties and verify on set method
    def __init__(
        self,
        gpu_support: bool = False,
        cpu_core_num: float = 1.0,
        memory_in_GB: float = 0.5):
        self.gpu_support = gpu_support
        self.cpu_core_num = cpu_core_num
        self.memory_in_GB = memory_in_GB

    def to_dict(self):
        return self.__dict__