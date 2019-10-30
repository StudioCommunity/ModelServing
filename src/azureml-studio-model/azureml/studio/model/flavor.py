from .logger import get_logger

logger = get_logger(__name__)

class Flavor(object):

    def to_dict(self):
        return self.__dict__ 