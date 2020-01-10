import logging


# TODO: Enrich this
def get_logger(name):
    logging.basicConfig(format="%(asctime)s %(name)-20s %(levelname)-10s %(message)s")
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    return logger
