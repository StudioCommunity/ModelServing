import os
import json
import math
import numpy as np
import pandas as pd
from collections import defaultdict
import importlib
import traceback

from azureml.core.model import Model

from .score_exceptions import InputDataError, ModuleError
from .dag_execution_engine import DagGraph
from .utils import ModelZip, PerformanceCounter

from .logger import get_logger

logger = get_logger(__name__)


def pip_install(package):
    if importlib.util.find_spec(package):
        logger.info(f'{package} has been installed already.')
    else:
        os.system(f'pip install -q {package}')
        logger.info(f'{package} is installed on the fly.')


def enable_rawhttp():
    global is_rawhttp
    if importlib.util.find_spec('azureml.contrib.services'):
        from azureml.contrib.services.aml_request import rawhttp
        rawhttp(None)
        is_rawhttp = True
    else:
        is_rawhttp = False
        logger.warning('RAWHTTP is not enabled!')


def construct_errmsg(code, message, details):
    return {'error': {'code': code, 'message': message, 'details': details}}


def construct_response(message, code, json_str):
    ret = None
    if is_rawhttp:
        from azureml.contrib.services.aml_response import AMLResponse
        ret = AMLResponse(message, code, json_str=json_str)
        ret.headers['Access-Control-Allow-Origin'] = '*'
    else:
        ret = message
    return ret


def get_modelpackage_path():
    root_path = os.environ.get('DSPATH', '')
    config_file = os.path.join(root_path, 'configuration.json')
    with open(config_file) as fp:
        config = json.load(fp)
        model = config['model']
        if ':' in model:
            model_name, version = model.rsplit(':', 1)
            version = int(version) if version.isdigit() else None
        else:
            model_name, version = model, None
        logger.info(f'Model: name={model_name}, version={version}')
    if 'MODELPATH' in os.environ: # for local debug only
        return os.path.join(os.environ['MODELPATH'], model)
    else:
        return Model.get_model_path(model_name, version)


def escape_nan(val):
    if val == {'isNan': True}:
        val = np.nan
    return val


def transform_nan(val):
    if isinstance(val, float) and math.isnan(val):
        val = {'isNan': True}
    return val


def load_graph():
    try:
        path = get_modelpackage_path()
        modelzip = ModelZip(path)
        json_string = modelzip.get_graph_json()
        graph = DagGraph(json_string, modelzip.get_dir())
        logger.info('Init: Graph has been loaded')
    except Exception as ex:
        logger.error(f'Init: Service init failed: {ex}')
        raise ex
    return graph

class BaseProcessor(object):
    def run(self, raw_data):
        with PerformanceCounter(logger, 'Preprocessing'):
            try:
                webservice_input, global_parameters = self.preprocess(raw_data)
            except Exception as ex:
                logger.error(f'Run: Preprocessing error: {ex}')
                raise InputDataError(self.dag.input_name2schema, raw_data)

        with PerformanceCounter(logger, 'Processing'):
            webservice_output, name2schema = self.dag.execute(webservice_input, global_parameters)

        response_json = self.postprocess(webservice_output, name2schema)

        return response_json

    def preprocess(self, raw_data):
        raise NotImplementedError

    def postprocess(self, output_name2data, name2schema):
        raise NotImplementedError

class NewProcessor(BaseProcessor):
    def __init__(self, dag):
        self.dag = dag

    def preprocess(self, raw_data):
        json_data = json.loads(raw_data)
        all_inputs = json_data['Inputs']
        webservice_input = {}
        for input_name, input_data in all_inputs.items():
            input_entry = defaultdict(list)
            for row in input_data:
                for key, val in row.items():
                    input_entry[key].append(escape_nan(val))
            webservice_input[input_name] = input_entry
        global_parameters = json_data.get('GlobalParameters', {})
        if not global_parameters: global_parameters = {}
        return webservice_input, global_parameters

    def postprocess(self, output_name2data, name2schema):
        execution_outputs = {}
        for output_name, data in output_name2data.items():
            is_list = data and isinstance(data, dict) and all([isinstance(entry, list) for entry in data.values()])
            if is_list:
                amount = min([len(entry) for entry in data.values()])
                output_entry = []
                for i in range(amount):
                    item = dict(zip([key for key in data.keys()],
                                    [transform_nan(array[i]) for array in data.values()]))
                    output_entry.append(item)
            else:
                output_entry = [data]
            execution_outputs[output_name] = output_entry
        response_json = {'Results': execution_outputs}
        return response_json

class ClassicProcessor(BaseProcessor):
    def __init__(self, dag, with_details=True):
        self.dag = dag
        self.with_details = with_details

    def preprocess(self, raw_data):
        json_data = json.loads(raw_data)
        all_inputs = json_data['Inputs']
        webservice_input = {}
        for input_name, input_data in all_inputs.items():
            columns = input_data['ColumnNames']
            values = input_data['Values']

            input_entry = defaultdict(list)
            for i in range(len(values)):
                for idx, col in enumerate(columns):
                    input_entry[col].append(escape_nan(values[i][idx]))
            webservice_input[input_name] = input_entry
        global_parameters = json_data.get('GlobalParameters', {})
        if not global_parameters: global_parameters = {}
        return webservice_input, global_parameters

    def postprocess(self, output_name2data, name2schema):
        response_json = {}
        for output_name, data in output_name2data.items():
            is_list = data and isinstance(data, dict) and all([isinstance(entry, list) for entry in data.values()])
            output_schema = self.name2schema[output_name]
            values = []

            if is_list:
                amount = min([len(entry) for entry in data.values()])
                if output_schema and output_schema.get_column_names():
                    values = [[data[col][i] for col in output_schema.get_column_names()]
                              for i in range(amount)]
                    column_names = output_schema.get_column_names()
                    column_types = output_schema.get_column_types()
                else:
                    values = [[data[col][i] for col in data.keys()]
                              for i in range(amount)]
                    column_names = list(data.keys())
                    column_types = []
            else:
                if output_schema and output_schema.get_column_names():
                    values = [[data[col]
                               for col in output_schema.get_column_names()]]
                    column_names = output_schema.get_column_names()
                    column_types = output_schema.get_column_types()
                else:
                    values = [[data[col] for col in data.keys()]]
                    column_names = list(data.keys())
                    column_types = []
            if self.with_details:
                response_json[output_name] = {
                    'type': 'DataTable',
                    'value': {
                        'ColumnNames': column_names,
                        'ColumnTypes': column_types,
                        'Values': values}}
            else:
                response_json[output_name] = {
                    'type': 'DataTable',
                    'value': {
                        'ColumnNames': column_names,
                        'Values': values}}
        return response_json

def handle_request(dag, raw_data, args):
    verbose = False
    ret = None
    try:
        with PerformanceCounter(logger, 'handling http request'):
            format = args.get('format', 'swagger')
            is_classic = format != 'swagger'
            with_details = args.get('details', 'false').lower() == 'true'
            verbose = args.get('verbose', 'false').lower() == 'true'
            logger.info(f'Run: is_classic = {is_classic}, with_details = {with_details}, verbose = {verbose}')

            if is_classic:
                processor = ClassicProcessor(dag, with_details)
            else:
                processor = NewProcessor(dag)
            response = processor.run(raw_data)
        response_string = json.dumps(response, cls=NpEncoder)
        logger.info(f'Run: output data(raw) = {response_string}')
        ret = construct_response(response_string, 200, json_str=False)

    except InputDataError as ex:
        error = str(ex)
        logger.error(error)
        errmsg = construct_errmsg(400, 'Input Data Error', error)
        ret = construct_response(errmsg, 400, json_str=True)

    except ModuleError as ex:
        error = (f'Run: User input error is from {dag.error_module} : {ex}')
        logger.error(error)
        errmsg = construct_errmsg(400, 'Module Executing Error', error)
        ret = construct_response(errmsg, 400, json_str=True)

    except Exception as ex:
        error = 'Run: Server internal error'
        if dag.error_module:
            error += f' is from Module {dag.error_module}'
        logger.error(f'{error} : {ex}')
        if verbose:
            error += f' : {ex}\n'
            error += traceback.format_exc()
        errmsg = construct_errmsg(500, 'Internal Server Error', error)
        ret = construct_response(errmsg, 500, json_str=True)

    return ret


def handle_not_supported(request):
    error = f'"{request.method}" is not supported'
    logger.error(error)
    errmsg = construct_errmsg(400, 'HTTP Method Not Supported', error)
    ret = construct_response(errmsg, 400, json_str=True)
    return ret


def handle_empty(request):
    warning = f'"{request.method}" is not supported, returning empty response'
    logger.warning(warning)
    ret = construct_response('', 200)
    return ret


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.Timestamp):
            return str(obj)
        else:
            return super(NpEncoder, self).default(obj)
