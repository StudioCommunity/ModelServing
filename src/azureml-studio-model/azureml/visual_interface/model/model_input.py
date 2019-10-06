import logging

logger = logging.getLogger(__name__)

class ModelInput(object):

    def __init__(
        self,
        name: str,
        value_type: str,
        default = None,
        description: str = None,
        optional: bool = False):
        self.name = name
        self.value_type = value_type
        self.default = default
        self.description = description
        self.optional = optional

    def to_dict(self):
        return self.__dict__