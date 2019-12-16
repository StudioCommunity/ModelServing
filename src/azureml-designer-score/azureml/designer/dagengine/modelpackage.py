import json

class ModelPackageStructure(object):
    def __str__(self):
        return str(self.__dict__)
    def __repr__(self):
        return f'{self.__class__}({self.__dict__})'

class MPDependencies(ModelPackageStructure):
    def __init__(self, dictionary):
        self.conda_channels = dictionary['CondaChannels']
        self.conda_packages = dictionary['CondaPackages']
        self.pip_options = dictionary['PipOptions']
        self.pip_packages = dictionary['PipPackages']

    def generate_yaml(self):
        pass

class MPModuleInput(ModelPackageStructure):
    def __init__(self, dictionary):
        self.name = dictionary['Name']
        self.label = dictionary['Label']
        self.datatype_ids = dictionary['DataTypeIdsList']
        self.is_optional = dictionary['IsOptional']
        self.is_resource = dictionary['IsResource']
        self.datastore_mode = dictionary['DataStoreMode']
        self.data_reference_name = dictionary['DataReferenceName']
        self.dataset_types = dictionary['DatasetTypes']

class MPNode(ModelPackageStructure):
    def __init__(self, dictionary):
        self.module_id = dictionary['ModuleId']
        self.inputport_mappings = dictionary['InputPortMappings']
        self.outputports = dictionary['OutputPorts']
        self.parameters = dictionary['Parameters']
        self.global_parameter_mappings = dictionary['GlobalParameterMappings']
        self.dependencies = MPDependencies(dictionary['Dependencies'])

class MPModule(ModelPackageStructure):
    def __init__(self, dictionary):
        self.name = dictionary['Name']
        self.module_id = dictionary['ModuleId']
        self.module_name = dictionary['ModuleName']
        self.class_name = dictionary['ClassName']
        self.method_name = dictionary['MethodName']
        self.pip_requirement = dictionary['PipRequirement']
        self.inputs = [MPModuleInput(input_dict) for input_dict in dictionary['Inputs']]
        self.module_version = dictionary['ModuleVersion']

class MPStaticSource(ModelPackageStructure):
    def __init__(self, dictionary):
        self.id = dictionary['Id']
        self.type = dictionary['Type']
        self.datatype_id = dictionary['DataTypeId']
        self.file_path = dictionary['FilePath']
        self.model_name = dictionary['ModelName']
        self.dataset_type = dictionary['DatasetType']

class MPInput(ModelPackageStructure):
    def __init__(self, dictionary):
        self.inputport = dictionary['InputPort']
        self.input_schema = json.loads(dictionary['InputSchema'])
        self.port_name, self.input_name, self.data_type = self.parse_port(self.inputport)

    def parse_port(self, port_string):
        idx, port_type, input_name, data_type = port_string.split(':')
        port_name = ':'.join([idx, port_type])
        return port_name, input_name, data_type 

class MPOutput(ModelPackageStructure):
    def __init__(self, dictionary):
        self.outputport = dictionary['OutputPort']
        self.output_schema = json.loads(dictionary['OutputSchema'])
        self.port_name, self.output_name, self.data_type = self.parse_port(self.outputport)

    def parse_port(self, port_string):
        idx, port_type, output_name, data_type = port_string.split(':')
        port_name = ':'.join([idx, port_type])
        return port_name, output_name, data_type   

class ModelPackage(ModelPackageStructure):
    def __init__(self, dictionary):
        self.id = dictionary['Id']
        self.nodes = self.get_nodes(dictionary['Nodes'])
        self.modules = self.get_modules(dictionary['Modules'])
        self.static_sources = self.get_static_sources(dictionary['StaticSources'])
        self.static_source_ports = self.get_static_source_ports(dictionary['StaticSourcePorts'])
        self.inputs, self.input_name2port, self.input_name2schema = self.get_inputs(dictionary['Inputs'])
        self.outputs, self.output_name2port, self.output_name2schema = self.get_outputs(dictionary['Outputs'])
        
    def get_nodes(self, dictionary):
        nodes = {}
        for key, val in dictionary.items():
            nodes[key] = MPNode(val)
        return nodes

    def get_modules(self, dictionary):
        modules = {}
        for key, val in dictionary.items():
            modules[key] = MPModule(val)
        return modules

    def get_static_sources(self, dictionary):
        static_sources = {}
        for key, val in dictionary.items():
            static_sources[key] = MPStaticSource(val)
        return static_sources

    def get_static_source_ports(self, dictionary):
        static_source_ports = {}
        for key, val in dictionary.items():
            static_source_ports[key] = str(val)
        return static_source_ports
    
    def get_inputs(self, input_list):
        inputs = {}
        input_name2port = {}
        input_name2schema = {}
        for data in input_list:
            mp_input = MPInput(data)
            inputs[mp_input.port_name] = mp_input
            input_name2port[mp_input.input_name] = mp_input.port_name
            input_name2schema[mp_input.input_name] = mp_input.input_schema
        return inputs, input_name2port, input_name2schema
    
    def get_outputs(self, output_list):
        outputs = {}
        output_name2port = {}
        output_name2schema = {}
        for data in output_list:
            mp_output = MPOutput(data)
            outputs[mp_output.port_name] = mp_output
            output_name2port[mp_output.output_name] = mp_output.port_name
            output_name2schema[mp_output.output_name] = mp_output.output_schema
        return outputs, output_name2port, output_name2schema

class ModelPackageDecoder(json.JSONDecoder):
    def __init__(self, *args, **kwargs):
        json.JSONDecoder.__init__(self, object_hook=self.object_hook, *args, **kwargs)
    def object_hook(self, dictionary):
        ret = dictionary
        candidates = set(['Id', 'Nodes', 'Modules', 'StaticSources', 'StaticSourcePorts', 'Inputs', 'Outputs'])
        if candidates.issubset(set(dictionary.keys())):
            ret = ModelPackage(dictionary)
        return ret
