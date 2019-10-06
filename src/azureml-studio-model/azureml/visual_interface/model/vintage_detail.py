import logging

logger = logging.getLogger(__name__)

class VintageDetail(object):

    def to_dict(self):
        return self.__dict__ 