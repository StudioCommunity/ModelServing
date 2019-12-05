import os
import sys
import json
import math
import numpy as np
import pandas as pd
from collections import defaultdict
import importlib

from azureml.core.model import Model

from .score_exceptions import InputDataError, ModuleError
from .dag_execution_engine import DagGraph
from .common import set_global_setting, ModelZip, PerformanceCounter

import logging


def eprint(*args, **kwargs): print(*args, file=sys.stderr, **kwargs)


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.log = print  # temporary fix for app insight telemetry
logger.info = print
logger.warning = eprint
logger.error = eprint


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
    with open('configuration.json') as fp:
        config = json.load(fp)
        model = config['model']
        if ':' in model:
            model_name, version = model.rsplit(':', 1)
            version = int(version) if version.isdigit() else None
        else:
            model_name, version = model, None
        logger.info(f'Model: name={model_name}, version={version}')
    if 'MODELPATH' in os.environ:
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
        set_global_setting('AZUREML_DESIGNER_DS_PATH', modelzip.get_dir())
        json_string = modelzip.get_graph_json()
        graph = DagGraph(json_string)
        logger.info('Init: Graph has been loaded')
    except Exception as ex:
        logger.error(f'Init: Service init failed: {ex}')
        raise ex
    return graph


def preprocess_new(raw_data):
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


def preprocess_classic(raw_data):
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


def postprocess_new(output_name2data):
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


def postprocess_classic(output_name2data, name2schema, with_details=True):
    response_json = {}
    for output_name, data in output_name2data.items():
        is_list = data and isinstance(data, dict) and all([isinstance(entry, list) for entry in data.values()])
        output_schema = name2schema[output_name]
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
        if with_details:
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


def process(dag, raw_data, is_classic=True, with_details=True):
    logger.info('Run: Begin preprocessing')
    try:
        if is_classic:
            webservice_input, global_parameters = preprocess_classic(raw_data)
        else:
            webservice_input, global_parameters = preprocess_new(raw_data)
    except Exception as ex:
        logger.error(f'Run: Preprocessing error: {ex}')
        raise InputDataError(dag.input_name2schema, raw_data)
    logger.info('Run: End preprocessing successfully')

    logger.info('Run: Begin processing')
    webservice_output, name2schema = dag.execute(webservice_input, global_parameters)
    logger.info('Run: End processing successfully')

    logger.info('Run: Begin postprocessing')
    if is_classic:
        response_json = postprocess_classic(webservice_output, name2schema, with_details)
    else:
        response_json = postprocess_new(webservice_output)
    logger.info('Run: End postprocessing successfully')

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

            logger.info(f'Run: input data(raw) = {raw_data}')
            response = process(dag, raw_data, is_classic, with_details)
            response = json.dumps(response, cls=NpEncoder)
            logger.info(f'Run: output data(raw) = {response}')
        ret = construct_response(response, 200, json_str=False)
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
            error += f' : {ex}'
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
