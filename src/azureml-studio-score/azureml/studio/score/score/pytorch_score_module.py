import torch
from torch.autograd import Variable
import importlib
import imp
import numpy as np
import pandas as pd
import sys
import pickle
import os
import ast
import cloudpickle
import logging

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class PytorchScoreModule(object):
    def __init__(self, model_path, config):
        pt_config = config['pytorch']
        self.model_file = pt_config['model_file_path']
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        serializer = pt_config.get('serialization_format', 'cloudpickle')
        if serializer == 'cloudpickle':
            model = self.load_from_cloudpickle(model_path, pt_config)
        elif serializer == 'savedmodel':
            model = self.load_from_savedmodel(model_path, pt_config)
        elif serializer == 'saveddict':
            model = self.load_from_saveddict(model_path, pt_config)
        else:
            raise Exception(f"Unrecognized serializtion format {serializer}")
        print('Load model success.')
        is_gpu = torch.cuda.is_available()   
        model = model.to(self.device)
        print(f'DEVICE: {self.device}')
        input_args = config.get('inputs', None)
        self.wrapper = PytorchWrapper(model, is_gpu, input_args)


    def run(self, df):
        return self.wrapper.predict(df)

    def load_scripts(self, model_path):
        modules = {}
        with os.scandir(model_path) as files_and_dirs:
            for entry in files_and_dirs:
                if entry.is_file() and entry.name.endswith('.py'):
                    name = entry.name[:-len('.py')]
                    #modules[name] = importlib.import_module(entry.path)
                    modules[name] = imp.load_source(name, entry.path)
        return modules 

    def load_from_cloudpickle(self, model_path, pt_config):
        modules = self.load_scripts(model_path)
        model_file_path = os.path.join(model_path, pt_config['model_file_path'])
        model = None
        retry = True
        while retry:  
            try:
                with open(model_file_path, 'rb') as fp:
                    model = cloudpickle.load(fp)
                retry = False
            except ModuleNotFoundError as ex:
                name = ex.name.rpartition('.')[-1]
                if name in modules:
                    sys.modules[ex.name] = modules[name]
                    retry = True
                else:
                    raise ex
        return model    

    def load_from_savedmodel(self, model_path, pt_config):
        modules = self.load_scripts(model_path)
        model_file_path = os.path.join(model_path, pt_config['model_file_path'])
        model_class_package = pt_config['model_class_package']
        module = importlib.import_module(model_class_package)
        #self._set_sys_modules(model_class_package, module)
        #return torch.load(model_path)
        model = None
        retry = True
        while retry:
            try:
                model = torch.load(model_file_path, map_location=self.device)
                retry = False
            except ModuleNotFoundError as ex:
                name = ex.name.rpartition('.')[-1]
                if name in modules:
                    sys.modules[ex.name] = modules[name]
                    retry = True
                else:
                    raise ex
        return model


    def load_from_saveddict(self, model_path, pt_config):
        model_class_package = pt_config['model_class_package']
        model_class_name = pt_config['model_class_name']
        model_class_init_args = os.path.join(model_path, pt_config['model_class_init_args'])
        model_file_path = os.path.join(model_path, pt_config['model_file_path'])
        module = importlib.import_module(model_class_package)
        model_class = getattr(module, model_class_name)
        print(f'model_class_init_args = {model_class_init_args}')
        with open(model_class_init_args, 'rb') as fp:
            args = pickle.load(fp)
        model = model_class(*args)
        model.load_state_dict(torch.load(model_file_path, map_location=self.device))
        return model
        

    def _set_sys_modules(self, package, module):
        entries = package.split('.')
        for i in range(len(entries)):
            sys.modules['.'.join(entries[i:])] = module
            print(f"{'.'.join(entries[i:])} : {sys.modules['.'.join(entries[i:])]}")


class PytorchWrapper(object):
    def __init__(self, model, is_gpu, input_args):
        self.model = model
        # self.model.eval()
        self.device = 'cuda' if is_gpu else 'cpu'
        self.input_args = input_args
    
    def predict(self, df):
        print(f"Device type = {self.device}, input_args = {self.input_args}")
        output = []

        def to_tensor(entry):
            if type(entry) == str:
                entry = ast.literal_eval(entry)
            return torch.Tensor(list(entry)).to(self.device)

        with torch.no_grad():
            print(f"predict df = \n {df}")
            for _, row in df.iterrows():
                input_params = []
                print(f"ROW = \n {row}")
                print(f"self.input_args = {self.input_args}")
                input_params = list(map(to_tensor, row[self.input_args]))
                print(f"FEATURES: {input_params}")
                print(f"input_params[0].size() = {input_params[0].size()}")
                print(f"input_params[1].size() = {input_params[1].size()}")
                predicted = self.model(*input_params)
                print(f"predicted = {predicted}")
                output.append(predicted.tolist())

        output_df = pd.DataFrame(output)
        output_df.columns = [f"Score_{i}" for i in range(0, output_df.shape[1])]
        print(f"output_df:\n{output_df}")
        return output_df
