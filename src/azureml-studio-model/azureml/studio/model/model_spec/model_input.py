from ..logger import get_logger

logger = get_logger(__name__)


class ModelInput(object):

    def __init__(
        self,
        name: str,
        value_type: str,
        default = None,
        description: str = None,
        optional: bool = False
    ):
        self.name = name
        self.value_type = value_type
        self.default = default
        self.description = description
        self.optional = optional

    def to_dict(self):
        return self.__dict__
    
    @classmethod
    def from_dict(cls, value_dict):
        return cls(
            name=value_dict.get("name", None),
            value_type=value_dict.get("value_type", None),
            default=value_dict.get("value_dict", None),
            description=value_dict.get("description", None),
            optional=value_dict.get("optional", False)
        )
