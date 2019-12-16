import os
import json
import importlib
import collections

from azureml.studio.modulehost.handler.port_io_handler import InputHandler
from azureml.studio.modulehost.deployment_service_module_host import DeploymentServiceModuleHost
from azureml.studio.modulehost.module_reflector import ModuleEntry
from azureml.studio.common.datatypes import DataTypes
from .modelpackage import ModelPackageDecoder, MPStaticSource
from .converter import create_dfd_from_dict, to_dfd, to_dict
from .utils import PerformanceCounter
from .score_exceptions import InputDataError, ResourceLoadingError

from .logger import get_logger

logger = get_logger(__name__)

class DagModule(object):
    def __init__(self, mp_module):
        self.mp_module = mp_module
        self._module_host = None
        self.params = {}
        self.resources = []

    def execute(self, input_data, global_params={}):
        raise NotImplementedError

    @property
    def module_host(self):
        raise NotImplementedError

    def set_params(self, params):
        raise NotImplementedError

    def set_resource(self, resource):
        raise NotImplementedError

class OfficialModule(DagModule):
    def execute(self, input_data, global_params={}):
        for key, val in input_data.items():
            input_data[key] = to_dfd(val)
        module_host = self.module_host
        return module_host.execute(input_data, global_params)

    @property
    def module_host(self):
        if not self._module_host:
            module_entry = ModuleEntry(
                self.mp_module.module_name,
                self.mp_module.class_name,
                self.mp_module.method_name)
            self._module_host = DeploymentServiceModuleHost(module_entry)
        return self._module_host

    def set_params(self, params):
        module_host = self.module_host
        module_host.parameters_dict.update(params)

    def set_resource(self, resource):
        module_host = self.module_host
        module_host.resources_dict.update(resource)

class CustomModule(DagModule):
    def execute(self, input_data, global_params={}):
        module_host = self.module_host
        inputs = []
        for data in input_data.values():
            inputs.append(data)
        if global_params:
            inputs.append(global_params)
        return module_host.execute(*inputs)

    @property
    def module_host(self):
        if not self._module_host:
            self.module_host = CustomModuleHost(
                self.mp_module.module_name,
                self.mp_module.class_name,
                self.mp_module.method_name)
        if self.params:
            self._module_host.init(*self.resources, self.params)
        else:
            self._module_host.init(*self.resources)
        return self._module_host

    def set_params(self, params):
        self.params = params

    def set_resource(self, resource):
        if isinstance(resource, list):
            self.resources.extend(resource)
        elif isinstance(resource, dict):
            self.resources.extend(resource.values())
        else:
            self.resources.append(resource)

class CustomModuleHost(object):
    def __init__(self, package, class_name, method_name):
        self.module = importlib.import_module(package)
        self.method_name = method_name
        self.module_class = getattr(self.module, class_name)
        self.module_run = None
        self.module_instance = None

    def init(self, *args):
        self.module_instance = self.module_class(*args)
        self.module_run = getattr(self.module_instance, self.method_name)

    def execute(self, *args):
        ret = self.module_run(*args)
        if not isinstance(ret, tuple) and not isinstance(ret, list):
            ret = (ret,)
        return ret

class DagModuleFactory(object):
    @staticmethod
    def get_module(mp_module):
        # TODO: temp solution, as some custom module named 'azureml.studio.score.xxx', below code needs change after module team provide a signal to indicate offical module
        is_official = (mp_module.module_name.startswith('azureml.studio.') and not mp_module.module_name.startswith('azureml.studio.score.')) \
                      or (mp_module.module_name.startswith('azureml.designer.') and not mp_module.module_name.startswith('azureml.designer.score.'))
        if is_official:
            return OfficialModule(mp_module)
        else:
            return CustomModule(mp_module)


class DagResourceLoader(object):
    def __init__(self, root_path=''):
        self.root_path = root_path
        self.typename2datatype = {
            'TrainedModel': DataTypes.LEARNER,
            'TransformModule': DataTypes.TRANSFORM,
            'FilterModule': DataTypes.FILTER,
            'ClusterModule': DataTypes.CLUSTER,
            'DataSource': None
        }
        self.typeid2postfix = {
            'IClusterDotNet': 'data.icluster',
            'ITransformDotNet': 'data.itransform',
            'TransformationDirectory': 'data.itransform'
        }

    def from_name(self, name):
        """Given a name of DataType, find a corresponding item in DataTypes enum.

        :param name: The name of DataType.
        :return: The corresponding item in DataTypes enum.
        """
        for e in DataTypes:
            if e.value.ws20_name == name:
                return e
        else:
            raise ValueError(f"Failed to load instance of DataTypes from dict")

    # 'ModelDirectory' 'TransformDirectory' 'AnyDirectory' 'DataFramDirectory' 'AnyFile'

    def load_static_source(self, static_source):
        logger.info(f'Loading static source {static_source}')
        try:
            is_not_datasource = static_source.type != 'DataSource'
            is_path = static_source.datatype_id == 'GenericFolder' or 'Directory' in static_source.datatype_id or static_source.datatype_id == "AnyFile"
            if static_source.datatype_id:
                if is_path:
                    data_type = None
                else:
                    data_type = self.from_name(static_source.datatype_id)
            else:
                data_type = self.typename2datatype[static_source.type]
                logger.warning(f'StaticSource({static_source}) has no type_id')

            path = os.path.join(self.root_path, static_source.model_name)
            logger.info(f'Invoking handle_input_from_file_name({path}, {data_type})')

            if static_source.datatype_id in ('ModelDirectory', 'TransformDirectory') and os.path.isdir(
                    path):  # TODO remove this hardcode
                resource = InputHandler.handle_input_directory(path)
                is_not_datasource = True
            elif static_source.datatype_id == 'DataFrameDirectory' and os.path.isdir(path):
                resource = InputHandler.handle_input_directory(path)
                is_not_datasource = False
            elif static_source.datatype_id in self.typeid2postfix.keys():
                if os.path.isdir(path):
                    path = os.path.join(path, self.typeid2postfix[static_source.datatype_id])
                if not os.path.isfile(path):
                    raise ResourceLoadingError(static_source.model_name, static_source.datatype_id)
                resource = InputHandler.handle_input_from_file_name(path, data_type)
            elif static_source.type == 'TrainedModel' and static_source.datatype_id == "ILearnerDotNet" and os.path.isfile(
                    path):
                resource = InputHandler.handle_input_from_file_name(path, DataTypes.LEARNER)
            elif static_source.type == 'TrainedModel' and os.path.isdir(path):
                official_ilearner = os.path.join(path, 'data.ilearner')
                official_metadata = os.path.join(path, 'data.metadata')
                if os.path.exists(official_ilearner) and os.path.exists(official_metadata):
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
            logger.error(f'Error while loading {static_source}: {ex}')
            raise ResourceLoadingError(static_source.model_name, static_source.datatype_id)
        logger.info(f'Loaded static source {static_source}')
        return resource, is_not_datasource

class DagNode(object):
    def __init__(self, mp_node, dag_module, root_path=''):
        self.mp_node = mp_node
        self.module = dag_module
        self.input_set = set()
        self.input_param2data = {}
        self.output_port2data = {}
        self.end_ports = []
        self.outputport_mapping = collections.defaultdict(list)
        self.module.set_params(self.mp_node.parameters)
        self.resource_loader = DagResourceLoader(root_path)

    def is_ready(self):
        return self.input_set == set(self.mp_node.inputport_mappings.keys())

    def execute(self, global_params={}):
        results = self.module.execute(self.input_param2data, global_params)
        for index, output_port in enumerate(self.mp_node.outputports):
            self.output_port2data[output_port] = results[index]
        return self.output_port2data

    def input2port(self, input_data, port_name):
        self.input_set.add(port_name)
        param_name = port_name.split(':')[-1]
        if isinstance(input_data, MPStaticSource):
            static_source, is_not_datasource = self.resource_loader.load_static_source(input_data)
            if is_not_datasource:
                self.module.set_resource({param_name: static_source})
            else:
                self.input_param2data[param_name] = static_source
        else:
            self.input_param2data[param_name] = input_data

class DagGraph(object):
    def __init__(self, json_string, root_path=''):
        self.root_path = root_path
        self.nodes = []
        self.entry_nodes = {}
        self.ready_nodes = []
        self.running_nodes = []
        self.modelpackage = self.load(json_string)
        self.module2globalparams = collections.defaultdict(dict)
        self.output_name2port = {}
        self.output_port2data = {}
        self.error_module = ''

    def load(self, json_string):
        logger.info('DagGraph loading...')
        modelpackage = json.loads(json_string, cls=ModelPackageDecoder)

        modules = {}
        for mp_module in modelpackage.modules.values():
            modules[mp_module.module_id] = DagModuleFactory.get_module(mp_module)
        
        output_port2node = {}
        for mp_node in modelpackage.nodes.values():
            for key, value in mp_node.global_parameter_mappings.items():
                self.module2globalparams[mp_node.module_id][key] = value
            node = DagNode(mp_node, modules[mp_node.module_id], self.root_path)
            self.nodes.append(node)
            for output_port in mp_node.outputports:
                output_port2node[output_port] = node
                if output_port in modelpackage.outputs:
                    node.end_ports.append(output_port)

        for node in self.nodes:
            for port_name, port_id in node.mp_node.inputport_mappings.items():
                if port_id in modelpackage.inputs.keys():
                    self.entry_nodes[port_id] = (node, port_name)
                elif port_id in modelpackage.static_source_ports:
                    static_source = modelpackage.static_sources[modelpackage.static_source_ports[port_id]]
                    node.input2port(static_source, port_name)
                elif port_id in output_port2node:  # the input is from other node's output
                    source_node = output_port2node[port_id]
                    source_node.outputport_mapping[port_id].append((node, port_name))
                else:
                    pass
        logger.info('DagGraph loaded')
        return modelpackage

    def execute(self, input_name2data, global_parameters):
        if set(input_name2data.keys()) != set(self.modelpackage.input_name2port.keys()):
            raise InputDataError(self.modelpackage.inputs, input_name2data)

        for input_name, input_raw in input_name2data.items():
            input_port = self.modelpackage.input_name2port[input_name]
            schema = self.modelpackage.input_name2schema[input_name]
            with PerformanceCounter(logger, 'loading input to datatable'):
                try:
                    input_data = create_dfd_from_dict(input_raw, schema)
                except Exception:
                    raise InputDataError(schema, input_raw)
            node, port = self.entry_nodes[input_port]
            node.input2port(input_data, port)
        return self.run_ready_nodes(global_parameters)

    def run_ready_nodes(self, global_parameters):
        for node in self.nodes:
            if node.is_ready() and node not in self.ready_nodes:
                self.ready_nodes.append(node)

        while self.ready_nodes:
            node = self.ready_nodes.pop()
            try:
                global_params = {}
                name_dict = self.module2globalparams.get(node.mp_node.module_id, {})
                names = set(name_dict.keys()).intersection(set(global_parameters.keys()))
                for name in names:
                    param_name = name_dict[name]
                    global_params[param_name] = global_parameters[name]
                with PerformanceCounter(logger,f'executing {node.module.mp_module} with global_params={global_params}'):
                    results = node.execute(global_params)
            except Exception as ex:
                self.error_module = node.module.mp_module.module_name
                raise ex
            for output_port, targets in node.outputport_mapping.items():
                result = results[output_port]
                for target in targets:
                    node_target, port_name = target
                    node_target.input2port(result, port_name)
                    if node_target.is_ready():
                        self.ready_nodes.append(node_target)
            for end_port in node.end_ports:
                self.output_port2data[end_port] = to_dict(results[end_port])
        return self.get_output_name2data(), self.modelpackage.output_name2schema

    def get_output_name2data(self):
        ret = {}
        for output_name, output_port in self.modelpackage.output_name2port.items():
            output_data = self.output_port2data[output_port]
            ret[output_name] = output_data
        return ret