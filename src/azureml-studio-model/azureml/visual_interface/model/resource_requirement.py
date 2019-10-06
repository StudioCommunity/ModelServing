import logging

logger = logging.getLogger(__name__)

class ResourceRequirement(object):

    # TODO: Set the object variables to be properties and verify on set method
    def __init__(
        self,
        gpu_support: bool = False,
        cpu_core_num: int = 2,
        memory_in_MB: int = 512):
        self.gpu_support = gpu_support
        self.cpu_core_num = cpu_core_num
        self.memory_in_MB = memory_in_MB

    def to_dict(self):
        return self.__dict__