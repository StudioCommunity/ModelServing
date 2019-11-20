import sys
import importlib
import os
import os.path
import json
import collections
import zipfile
from time import perf_counter
from pandas import DataFrame
import pandas as pd

from azureml.studio.common.datatable.data_table import DataTable
from azureml.studio.common.datatable.data_table_constructor import DataTableConstructor
from azureml.studio.common.datatable.data_type_conversion import convert_column_by_element_type

from azureml.studio.common.datatypes import DataTypes
from azureml.studio.core.io.data_frame_directory import DataFrameDirectory
from azureml.studio.core.data_frame_schema import DataFrameSchema

from azureml.studio.modulehost.deployment_service_module_host import DeploymentServiceModuleHost
from azureml.studio.modulehost.handler.data_handler import ZipHandler
from azureml.studio.modulehost.handler.port_io_handler import InputHandler
from azureml.studio.modulehost.module_reflector import ModuleEntry
from .ScoreExceptions import InputDataError, ResourceLoadingError

import logging


def eprint(*args, **kwargs): print(*args, file=sys.stderr, **kwargs)


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.log = print
logger.info = print
logger.warning = eprint
logger.error = eprint


def create_dfd_from_dict(json_data, schema_data):
    schema = DataFrameSchema.from_dict(schema_data)
    if set(json_data.keys()) != set(schema.column_attributes.names):
        different_names = set(schema.column_attributes.names).difference(set(json_data.keys()))
        raise ValueError(f'Input json_data must have the same column names as the meta data. '
                         f'Different columns are: {different_names}')
    df = pd.DataFrame()
    for column_name in schema.column_attributes.names:
        column = pd.Series(json_data[column_name])
        target_type = schema.column_attributes[column_name].element_type
        converted_column = convert_column_by_element_type(column, target_type)
        df[column_name] = converted_column
    return DataFrameDirectory.create(data=df, schema=schema_data)


def to_dfd(data):
    ret = data
    if isinstance(data, DataFrameDirectory):
        ret = data
    elif isinstance(data, DataFrame):
        ret = DataFrameDirectory.create(data=data)
    elif isinstance(data, dict):
        ret = DataFrameDirectory.create(data=DataFrame(data))
    elif isinstance(data, str):
        ret = DataFrameDirectory.create(data=DataFrame({'text': [data]}))
    else:
        logger.info(f'pass through the value of type {type(data)}')
    return ret


def to_dataframe(data):
    ret = data
    if isinstance(data, DataFrame):
        ret = data
    elif isinstance(data, DataFrameDirectory):
        ret = data.data
    elif isinstance(data, dict):
        ret = DataFrame(data)
    elif isinstance(data, str):
        ret = DataFrame({'text': [data]})
    else:
        logger.info(f'pass through the value of type {type(data)}')
    return ret


def to_datatable(data):
    ret = data
    if isinstance(data, DataTable):
        ret = data
    elif isinstance(data, DataFrame):
        ret = DataTable(data)
    elif isinstance(data, ZipHandler):
        ret = data
    elif isinstance(data, dict) or isinstance(data, str):
        ret = DataTable(to_dataframe(data))
    else:
        logger.info(f'pass through the value of type {type(data)}')
    return ret


def to_dict(data):
    ret = None
    if isinstance(data, DataFrameDirectory):
        ret = data.data.to_dict(orient='list')
    elif isinstance(data, DataFrame):
        ret = data.to_dict(orient='list')
    elif isinstance(data, dict):
        ret = data
    else:
        raise NotImplementedError
    return ret


class PerformanceCounter(object):
    def __init__(self, logger, name):
        self.name = name
        self.logger = logger
        self.start = 0

    def __enter__(self):
        self.logger.info(f'Start {self.name}.')
        self.start = perf_counter()

    def __exit__(self, *args):
        duration = (perf_counter() - self.start) * 1000
        self.logger.info(f'End {self.name}. Duration: {duration} milliseconds.')


def port2param(port_name):
    if ':' not in port_name:
        return port_name
    return port_name.split(':')[1]


class InputOutputSchema(object):
    def __init__(self, schema_string):
        self.json = None
        self.names = []
        self.types = []
        self.name2type = {}
        if schema_string:
            self._load(schema_string)

    def _load(self, json_string):
        self.json = json.loads(json_string)
        columns = self.json['columnAttributes']
        self.names = [col['name'] for col in columns]
        self.types = [col['type'] for col in columns]
        self.name2type = dict(zip(self.names, self.types))

    def get_column_names(self):
        return self.names

    def get_column_types(self):
        return self.types

    def get_name2type(self):
        return self.name2type

    def __str__(self):
        return str(self.name2type)

    def __repr__(self):
        return str(self.name2type)


class ModelPackage(object):
    def __init__(self, path, target_dir='./studiomodelpackage', graph_file='modelpackage.json'):
        self.path = path
        self.target_dir = target_dir
        self.graph_file = graph_file
        self._extractall()

    def get_graph_json(self):
        return self.get_resource_data(self.graph_file)

    def get_resource_data(self, name):
        fn = self.get_fullname(name)
        with open(fn) as fp:
            data = fp.read()
        return data

    def get_fullname(self, name):
        return os.path.join(self.target_dir, name)

    def _extractall(self):
        with zipfile.ZipFile(self.path) as zf:
            zf.extractall(self.target_dir)


def set_global_modelpackage(path):
    global modelpackage
    modelpackage = ModelPackage(path)
    return modelpackage


def get_global_modelpackage():
    return modelpackage


class IModule(object):
    def set_resource(self, resource):
        raise NotImplementedError

    def set_params(self, params):
        raise NotImplementedError

    def execute(self, input_data, params):
        raise NotImplementedError


class OfficialModule(IModule):
    def __init__(self, mid, module_name, class_name, method_name):
        self.id = mid
        self.module_name = module_name
        self.class_name = class_name
        self.method_name = method_name
        self.module_host = self.init_modulehost()

    def init_modulehost(self):
        module_entry = ModuleEntry(
            self.module_name,
            self.class_name,
            self.method_name)
        module_host = DeploymentServiceModuleHost(module_entry)
        module_host.resources_dict = {}
        module_host.parameters_dict = {}
        return module_host

    def set_params(self, params):
        self.module_host.parameters_dict.update(params)

    def set_resource(self, resource):
        self.module_host.resources_dict.update(resource)

    def execute(self, input_data, global_params={}):
        for key, val in input_data.items():
            input_data[key] = to_dfd(val)
        return self.module_host.execute(input_data, global_params)


class CustomModule(IModule):
    def __init__(self, mid, module_name, class_name, method_name):
        self.id = mid
        self.module_name = module_name
        self.class_name = class_name
        self.method_name = method_name
        self.module_host = None
        self.resources = []
        self.params = {}
        self.module_host = None

    def get_modulehost(self):
        if not self.module_host:
            module_host = CustomModuleHost(self.module_name, self.class_name, self.method_name)
            if self.params:
                module_host.init(*self.resources, self.params)
            else:
                module_host.init(*self.resources)
            self.module_host = module_host
        return self.module_host

    def set_params(self, params):
        self.params.update(params)

    def set_resource(self, resource):
        if isinstance(resource, list):
            self.resources.extend(resource)
        elif isinstance(resource, dict):
            self.resources.extend(resource.values())
        else:
            self.resources.append(resource)

    def execute(self, input_data, global_params={}):
        module_host = self.get_modulehost()
        inputs = []
        for data in input_data.values():
            inputs.append(to_dataframe(data))
        if global_params:
            inputs.append(global_params)
        return module_host.execute(*inputs)


class CustomModuleHost(object):
    def __init__(self, package, class_name, method_name):
        self.module = importlib.import_module(package)
        self.method_name = method_name
        self.class_ = getattr(self.module, class_name)
        self._run = None
        self.instance = None

    def init(self, *args):
        self.instance = self.class_(*args)
        self._run = getattr(self.instance, self.method_name)

    def execute(self, *args):
        ret = self._run(*args)
        if not isinstance(ret, tuple) and not isinstance(ret, list):
            ret = (ret,)
        return ret


class ModuleFactory(object):
    @staticmethod
    def get_module(mid, module_name, class_name, method_name):
        # TODO: temp solution, as some custom module named 'azureml.studio.score.xxx', below code needs change after module team provide a signal to indicate offical module 
        if module_name.startswith('azureml.studio') and not module_name.startswith('azureml.studio.score.'):
            return OfficialModule(mid, module_name, class_name, method_name)
        else:
            method_name = method_name or 'run'
            return CustomModule(mid, module_name, class_name, method_name)


class ResourceLoader(object):
    typename2datatype = {
        'TrainedModel': DataTypes.LEARNER,
        'TransformModule': DataTypes.TRANSFORM,
        'FilterModule': DataTypes.FILTER,
        'ClusterModule': DataTypes.CLUSTER,
        'DataSource': None
    }
    typeid2postfix = {
        'IClusterDotNet': 'data.icluster',
        'ITransformDotNet': 'data.itransform',
        'TransformationDirectory': 'data.itransform'
    }

    # 'ModelDirectory' 'TransformDirectory' 'AnyDirectory' 'DataFramDirectory' 'AnyFile'
    @classmethod
    def load_static_source(cls, static_source):
        logger.info(
            f'Loading static source {static_source.model_name}, {static_source.type_id}, {static_source.type_name}')
        try:
            is_not_datasource = static_source.type_name != 'DataSource'
            is_path = static_source.type_id == 'GenericFolder' or 'Directory' in static_source.type_id or static_source.type_id == "AnyFile"
            if static_source.type_id:
                if is_path:
                    data_type = None
                else:
                    data_type = DataTypes.from_name(static_source.type_id)
            else:
                data_type = cls.typename2datatype[static_source.type_name]
                logger.warning(f'StaticSource({static_source.model_name}) has no type_id')

            path = get_global_modelpackage().get_fullname(static_source.model_name)
            logger.info(f'Invoking handle_input_from_file_name({path}, {data_type})')

            if static_source.type_id in ('ModelDirectory', 'TransformDirectory') and os.path.isdir(
                    path):  # TODO remove this hardcode
                resource = InputHandler.handle_input_directory(path)
                is_not_datasource = True
            elif static_source.type_id == 'DataFrameDirectory' and os.path.isdir(path):
                resource = InputHandler.handle_input_directory(path)
                is_not_datasource = False
            elif static_source.type_id in cls.typeid2postfix.keys():
                if os.path.isdir(path):
                    path = os.path.join(path, cls.typeid2postfix[static_source.type_id])
                if not os.path.exists(path):
                    raise ResourceLoadingError(static_source.model_name, static_source.type_id)
                resource = InputHandler.handle_input_from_file_name(path, data_type)
            elif static_source.type_name == 'TrainedModel' and static_source.type_id == "ILearnerDotNet" and os.path.isfile(
                    path):
                resource = InputHandler.handle_input_from_file_name(path, DataTypes.LEARNER)
            elif static_source.type_name == 'TrainedModel' and os.path.isdir(path):
                official_ilearner = os.path.join(path, 'data.ilearner')
                official_metadata = os.path.join(path, 'data.metadata')
                if (os.path.exists(official_ilearner) and os.path.exists(official_metadata)):
                    resource = InputHandler.handle_input_from_file_name(official_ilearner, DataTypes.LEARNER)
                else:
                    resource = path
            elif is_path and os.path.isdir(path):
                resource = path
                is_not_datasource = True
            else:
                resource = InputHandler.handle_input_from_file_name(path, data_type)
        except ResourceLoadingError:
            raise
        except Exception as ex:
            logger.error(f'Error while loading {static_source.model_name}: {ex}')
            raise ResourceLoadingError(static_source.model_name, static_source.type_name)
        logger.info(f'Loaded static source {static_source.model_name}')
        return resource, is_not_datasource


class StaticSource(object):
    def __init__(self, sid, port_id, type_name, model_name, type_id):
        self.id = sid
        self.port_id = port_id
        self.type_name = type_name
        self.model_name = model_name
        self.type_id = type_id


class DAGNode(object):
    def __init__(self, module, params, input_port_mapping, output_ports):
        self.module = module
        self.params = params
        self.sources = None
        self.input_port_mapping = input_port_mapping
        self.output_ports = list(output_ports)
        self.output_port_mapping = collections.defaultdict(list)
        self.end_ports = []
        self.input_set = set()
        self.input_param2data = {}
        self.output_port2data = {}
        self.resource_loader = ResourceLoader()
        self.module.set_params(self.params)

    def is_ready(self):
        return self.input_set == set(self.input_port_mapping.keys())

    def execute(self, global_params={}):
        results = self.module.execute(self.input_param2data, global_params)
        for index, output_port in enumerate(self.output_ports):
            self.output_port2data[output_port] = results[index]

    def input2port(self, input_data, port_name):
        self.input_set.add(port_name)
        param_name = port2param(port_name)
        if isinstance(input_data, StaticSource):
            static_source, is_not_datasource = ResourceLoader.load_static_source(input_data)
            if is_not_datasource:
                self.module.set_resource({param_name: static_source})
            else:
                self.input_param2data[param_name] = static_source
        else:
            self.input_param2data[param_name] = input_data

    def get_output_port2data(self):
        return self.output_port2data


class DAGraph(object):
    def __init__(self):
        self.id = None
        self.json = None
        self.nodes = []

        self.input_ports = []
        self.input_name2port = {}
        self.input_name2schema = {}
        self.input_port2type = {}
        self.input_port2data = {}

        self.output_ports = []
        self.output_name2port = {}
        self.output_name2schema = {}
        self.output_port2type = {}
        self.output_port2data = {}

        self.static_sources = {}
        self.static_source_ports = {}
        self.version = None
        self.context = None
        self.global_params = None
        self.encryption_settings = None

        self.entry_nodes = {}
        self.ready_nodes = []
        self.running_nodes = []
        self.module_to_globalparams = collections.defaultdict(dict)

        self.error_module = ''

    def load(self, json_string):
        logger.info('Start loading DAGraph')
        self.json = json.loads(json_string)
        self.id = self.json.get('Id', {})
        nodes = self.json.get('Nodes', {})
        modules = self.json.get('Modules', {})
        sources = self.json.get('StaticSources', {})
        source_ports = self.json.get('StaticSourcePorts', {})
        inputs = self.json.get('Inputs', [])
        outputs = self.json.get('Outputs', [])
        self.version = self.json.get('RunTimeVersion', {})
        self.global_params = self.json.get('GlobalParameters', {})
        self.context = self.json.get('ModelContext', {})
        self.encryption_settings = self.json.get('EncryptionSettings', {})

        for entry in inputs:
            idx, input_type, input_name, data_type = entry['InputPort'].split(':')
            input_port = ':'.join([idx, input_type])
            input_schema = entry['InputSchema']
            self.input_ports.append(input_port)
            self.input_name2port[input_name] = input_port
            self.input_name2schema[input_name] = InputOutputSchema(input_schema)
            self.input_port2type[input_port] = data_type

        for entry in outputs:
            idx, output_type, output_name, data_type = entry['OutputPort'].split(':')
            output_port = ':'.join([idx, output_type])
            output_schema = entry['OutputSchema']
            self.output_ports.append(output_port)
            self.output_name2port[output_name] = output_port
            self.output_name2schema[output_name] = InputOutputSchema(output_schema)
            self.output_port2type[output_port] = data_type

        port2id = {}
        for key, val in source_ports.items():
            port = str(val)
            self.static_source_ports[key] = port
            port2id[port] = key
        for key, val in sources.items():
            port_id = port2id[key]
            source = StaticSource(
                val['Id'],
                port_id,
                val['Type'],
                val['ModelName'],
                val.get('DataTypeId')
            )
            self.static_sources[key] = source

        output_port2node = {}
        for key, val in nodes.items():
            module_id = val['ModuleId']
            meta = modules[module_id]
            module = ModuleFactory.get_module(
                module_id,
                meta['ModuleName'],
                meta['ClassName'],
                meta['MethodName'])
            params = val['Parameters']
            input_port_mapping = val['InputPortMappings']
            output_ports = val['OutputPorts']
            global_param_mappings = val.get('GlobalParameterMappings', {})
            for key, value in global_param_mappings.items():
                self.module_to_globalparams[module_id][key] = value
            node = DAGNode(module, params, input_port_mapping, output_ports)
            self.nodes.append(node)
            for output_port in output_ports:
                output_port2node[output_port] = node
                if output_port in self.output_ports:
                    node.end_ports.append(output_port)

        for node in self.nodes:
            for port_name, port_id in node.input_port_mapping.items():
                if port_id in self.input_ports:  # input entry
                    self.entry_nodes[port_id] = (node, port_name)
                elif port_id in self.static_source_ports:  # the input is ILearner/ITransform
                    static_source = self.static_sources[self.static_source_ports[port_id]]
                    node.input2port(static_source, port_name)
                elif port_id in output_port2node:  # the input is from other node's output
                    source_node = output_port2node[port_id]
                    source_node.output_port_mapping[port_id].append(
                        (node, port_name))
                else:
                    pass
        # self._run_ready_nodes()
        logger.info('DAGraph loaded')

    def execute(self, input_name2data, global_parameters):
        if set(input_name2data.keys()) != set(self.input_name2port.keys()):
            raise InputDataError(self.input_name2schema, input_name2data)

        for input_name, input_raw in input_name2data.items():
            input_port = self.input_name2port[input_name]
            schema = self.input_name2schema[input_name]
            with PerformanceCounter(logger, 'loading input to datatable'):
                try:
                    if schema.json:
                        input_data = create_dfd_from_dict(input_raw, schema.json)
                    else:
                        input_data = DataFrameDirectory.create(data=DataFrame(input_raw))
                except BaseException:
                    raise InputDataError(schema.json, input_raw)
            self.input_port2data[input_port] = input_data
            node, port = self.entry_nodes[input_port]
            node.input2port(input_data, port)
        self._run_ready_nodes(global_parameters)

    def _run_ready_nodes(self, global_parameters):
        for node in self.nodes:
            if node.is_ready() and node not in self.ready_nodes:
                self.ready_nodes.append(node)

        while self.ready_nodes:
            node = self.ready_nodes.pop()
            try:
                global_params = {}
                name_dict = self.module_to_globalparams.get(node.module.id, {})
                names = set(name_dict.keys()).intersection(set(global_parameters.keys()))
                for name in names:
                    param_name = name_dict[name]
                    global_params[param_name] = global_parameters[name]
                with PerformanceCounter(logger,
                                        f'executing {node.module.module_name} with global_params={global_params}'):
                    node.execute(global_params)
            except Exception as ex:
                self.error_module = node.module.module_name
                raise ex
            results = node.get_output_port2data()
            for output_port, targets in node.output_port_mapping.items():
                result = results[output_port]
                for target in targets:
                    node_target, port_name = target
                    node_target.input2port(result, port_name)
                    if node_target.is_ready():
                        self.ready_nodes.append(node_target)
            for end_port in node.end_ports:
                self.output_port2data[end_port] = to_dict(results[end_port])

    def get_output_name2data(self):
        ret = {}
        for output_name, output_port in self.output_name2port.items():
            output_data = self.output_port2data[output_port]
            ret[output_name] = output_data
        return ret

    def get_output_name2schema(self):
        return self.output_name2schema


class DAGExecutionEngine(object):
    @staticmethod
    def execute(graph, input_name2data, global_parameters):
        graph.execute(input_name2data, global_parameters)
