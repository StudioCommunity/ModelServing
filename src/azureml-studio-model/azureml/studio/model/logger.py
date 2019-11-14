import logging

logging.basicConfig(format="%(asctime)s %(name)-20s %(levelname)-10s %(message)s",
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)


# TODO: Enrich this
def get_logger(name):
    return logging.getLogger(name)
