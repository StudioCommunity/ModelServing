import sys
import logging

def get_logger(name):
    def eprint(*args, **kwargs):
        print(*args, file=sys.stderr, **kwargs)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.log = print
    logger.info = print
    logger.warning = eprint
    logger.error = eprint
    return logger