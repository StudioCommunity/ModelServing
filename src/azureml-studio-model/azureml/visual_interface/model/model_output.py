import logging

logger = logging.getLogger(__name__)

class ModelOutput(object):

    def __init__(
        self,
        name: str,
        value_type: str,
        description: str = None):
        self.name = name
        self.value_type = value_type
        self.description = description

    def to_dict(self):
        return self.__dict__