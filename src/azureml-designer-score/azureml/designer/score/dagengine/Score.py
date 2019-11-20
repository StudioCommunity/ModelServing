import sys
import copy

from .Processor import load_graph, handle_request, handle_not_supported, handle_empty
from .Processor import pip_install, enable_rawhttp

import logging


def eprint(*args, **kwargs): print(*args, file=sys.stderr, **kwargs)


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.log = print
logger.info = print
logger.warning = eprint
logger.error = eprint


def init():
    pip_install('azureml.contrib.services')
    global graph
    graph = load_graph()
    enable_rawhttp()


def run(request):
    if request.method == 'POST':
        try:
            dag = copy.deepcopy(graph)
        except Exception as ex:
            # TODO: temporary fix
            logger.error(f'Error while deepcopying: {ex}')
            dag = copy.copy(graph)
        return handle_request(dag, request.get_data(), request.args)
    else:
        return handle_empty(request)
