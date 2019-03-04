import logging

logger = logging.getLogger("pyRevolve")
logger.setLevel(logging.DEBUG)

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
# create formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s') # noqa

ch.setFormatter(formatter)
# add the handlers to the logger
logger.addHandler(ch)
