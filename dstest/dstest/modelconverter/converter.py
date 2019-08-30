import os
import os.path
import sys
import logging
import urllib.request
import ast
import json
import importlib
import imp

from pip._internal import main as pipmain
pipmain(["install", "click"])
import click

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logging.info(f"loader echo")
logger = logging.getLogger(__name__)

@click.command()
@click.option('--flavor')
@click.option('--model_url')
@click.option('--serialization')
@click.option('--model_class_url', default='')
@click.option('--init_args', default='{}')
@click.option('--input_args', default='{}')
@click.option('--out_model_path', default='model')
def run_pipeline(flavor, model_url, serialization, model_class_url, init_args, input_args, out_model_path):
    cwd = os.getcwd()
    sys.path.append(cwd)
    print(f'flavor={flavor}, serialziation={serialization}, out_model_path={out_model_path}')
    print(f'PATH: {os.environ}')
    print(f'CWD: {cwd}')
    print(f'INIT_ARGS: {init_args}')
    model_file = extract_name(model_url)
    urllib.request.urlretrieve(model_url, model_file)
    print(f'DOWNLOAD to {model_file}')
    model_class_file = None
    if model_class_url:
        model_class_file = extract_name(model_class_url)
        urllib.request.urlretrieve(model_class_url, model_class_file)
        print(f'DOWNLOAD to {model_class_file}')
    init_py = '__init__.py'
    if not os.path.exists(init_py):
        open(init_py, 'wb').close()

    print(f'ALLFILES: {os.listdir(".")}')

    if flavor == 'pytorch':
        load_pytorch(model_file, serialization, out_model_path, model_class_file, init_args)
    elif flavor == 'keras':
        load_keras(model_file, serialization, out_model_path)
    elif flavor == 'tensorflow':
        load_tensorflow(model_file, serialization, out_model_path)
    elif flavor == 'sklearn':
        load_sklearn(model_file, serialization, out_model_path)
    else:
        raise NotImplementedError()

def load_module(path):
    module_path = path.replace('.\\', '').replace('./', '').replace('\\', '.').replace('/', '.')
    if module_path.endswith('.py'):
        module_path = module_path[:-len('.py')]
    print(f'LOADMODULE: {module_path} from {path}, {os.path.exists(path)}')
    return importlib.import_module(module_path)

def extract_name(url):
    return url.partition('?')[0].rpartition('/')[-1]

def parse_init(init_args):
    if not init_args:
        return '', None
    #args = ast.literal_eval(init_args)
    init_args = init_args.replace("'", '"').replace(";",",")
    print(f'INIT_ARGS2: {init_args}')
    args = json.loads(init_args)
    class_name = args.get('class', '')
    if class_name:
        args.pop('class')
    return class_name, args

def parse_input(input_args):
    pass

def load_scripts(model_path):
        modules = {}
        with os.scandir(model_path) as files_and_dirs:
            for entry in files_and_dirs:
                if entry.is_file() and entry.name.endswith('.py') and 'setup.py' not in entry.name:
                    name = entry.name[:-len('.py')]
                    print(f'LOADSCRIPT: {name} from {entry.path}')
                    #modules[name] = load_module(entry.path)
                    modules[name] = imp.load_source(name, entry.path)
        return modules  

def load_pytorch(model_file, serialization, out_model_path, model_class_file, init_args=None):
    import torch
    from builtin_models.pytorch import save_model
    dependencies = []
    modules = {}
    if model_class_file:
        dependencies.append(model_class_file)
        basepath = os.path.dirname(model_class_file) or '.'
        modules = load_scripts(basepath)
    class_name, init_args = parse_init(init_args)
    model = None
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'DEVICE={device}')
    if serialization == 'cloudpickle':
        print(f'model loading(cloudpickle): {model_file} to {out_model_path}')
        import cloudpickle
        retry = True
        while retry:
            try:
                with open(model_file, 'rb') as fp:
                    model = cloudpickle.load(fp)
                retry = False
            except ModuleNotFoundError as ex:
                name = ex.name.rpartition('.')[-1]
                if name in modules:
                    sys.modules[ex.name] = modules[name]
                    retry = True
                else:
                    raise ex      
    elif serialization == 'savedmodel':
        print(f'model loading(savedmodel): {model_file} to {out_model_path}')
        retry = True
        while retry:   
            try:
                with open(model_file, 'rb') as fp:
                    model = torch.load(model_file, map_location=device)
                retry = False
            except ModuleNotFoundError as ex:
                name = ex.name.rpartition('.')[-1]
                if name in modules:
                    sys.modules[ex.name] = modules[name]
                    retry = True
                else:
                    raise ex
    elif serialization == 'statedict':
        print(f'model loading(statedict): {model_file} to {out_model_path}')
        model_class = None
        for module in modules.values():
            model_class = getattr(module, class_name)
            if model_class:
                break
        if not model_class:
            raise NotImplementedError

        print(f'init_args = {init_args}')
        if init_args:
            model = model_class(**init_args)
        else:
            model = model_class()
        print(f'MODEL1 = {model}')
        model.load_state_dict(torch.load(model_file, map_location=device))
        print(f'MODEL2 = {model}')
    else:
        raise NotImplementedError

    print(f'model loaded: {out_model_path}')
    print(f'model={model}, dependencies={dependencies}')
    save_model(model, out_model_path, dependencies=dependencies)
    print(f'MODEL_FOLDER: {os.listdir(out_model_path)}')

def load_keras(model_file, serialization, out_model_path):
    import keras
    from builtin_models.keras import save_model, load_model_from_local_file
    model = load_model_from_local_file(model_file)
    path = './model'
    save_model(model, path)

def load_tensorflow(model_file, serialization, out_model_path):
    if serialization == 'saved_model':
        pass
    elif serialization == 'saver':
        pass
    else:
        pass

def load_sklearn(model_file, serialization, out_model_path):
    if serialization == 'pickle':
        pass
    else:
        pass

if __name__ == '__main__':
    run_pipeline()
