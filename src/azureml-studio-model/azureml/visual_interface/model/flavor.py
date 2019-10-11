import logging

logger = logging.getLogger(__name__)

class Flavor(object):

    def to_dict(self):
        return self.__dict__ 