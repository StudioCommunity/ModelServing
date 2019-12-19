import copy

from .processor import load_graph, handle_request, handle_empty
from .processor import pip_install, enable_rawhttp

from .logger import get_logger

logger = get_logger(__name__)


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
