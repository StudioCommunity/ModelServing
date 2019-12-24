import fire
import sys
import os
import argparse
import shutil
import json
import zipfile
from azureml.core import Workspace
from azureml.core.webservice import Webservice

ROOT_DIR = os.path.dirname(__file__)

DOCKFILE_TEMPLATE = '''FROM {base_image}
ARG conda_env=project_environment
ARG work_dir=/app
WORKDIR $work_dir
ENV PATH /opt/miniconda/envs/$conda_env/bin:/opt/conda/envs/$conda_env/bin:$PATH
ENV CONDA_DEFAULT_ENV $conda_env
ADD conda.yaml $work_dir/conda.yaml
ADD configuration.json $work_dir
RUN conda env create -f $work_dir/conda.yaml && \
    pip install azureml-contrib-server && \
    pip install azureml-contrib-services
COPY *.py $work_dir/
'''


def create_from_params(subscription, resourcegroup, workspace, deployment=None, realtimeendpoint=None):
    if realtimeendpoint:
        deployment = realtimeendpoint
    args = {'subscription': subscription, 'resourcegroup': resourcegroup, 'workspace': workspace,
            'deployment': deployment}
    for key, value in args.items():
        if not value:
            print(f'Argument {key} is required')
            exit(1)
    ws = Workspace(subscription, resourcegroup, workspace)

    base_dir, model_dir, input_dir, output_dir, unzip_dir = f'./debug-{deployment}', 'data', 'input', 'output', 'studiomodelpackage'

    create_dirs(base_dir, [model_dir, input_dir, output_dir])

    copy_scripts(base_dir, True)

    modelname, base_image = download_model(ws, deployment, os.path.join(base_dir, model_dir))

    unzip(os.path.join(base_dir, model_dir, modelname), os.path.join(base_dir, unzip_dir))

    write_sample(os.path.join(base_dir, unzip_dir), os.path.join(base_dir, input_dir))

    # generate conda.yaml for docker build
    write_conda(os.path.join(base_dir, unzip_dir), base_dir)

    # generate Dockerfile for docker build
    write_dockerfile(base_image, base_dir)

    write_configuration(base_dir, modelname, model_dir, input_dir, output_dir)

    readme = write_readme(base_dir, input_dir, output_dir)

    print(readme)


# https://mlworkspacecanary.azure.ai/portal/subscriptions/ee85ed72-2b26-48f6-a0e8-cb5bcf98fbd9/resourceGroups/MT/providers/Microsoft.MachineLearningServices/workspaces/zhanxiaAML/deployments/sample1-xin7
# "https://ml.azure.com/endpoints/lists/realtimeendpoints/amlstudio-5ba5cfd7cf214ccea71438/detail?&wsid=/subscriptions/74eccef0-4b8d-4f83-b5f9-fa100d155b22/resourcegroups/AmlStudioV2DRI/workspaces/StudioV2DRI_EUS"
def create_from_url(deployment_url):
    deployment_url = deployment_url.replace('&', '/').replace('?', '/').replace('%2F', '/')
    entries = deployment_url.split('/')
    args = {'subscription': '', 'resourcegroup': '', 'workspace': '', 'deployment': '', 'realtimeendpoint': ''}
    for i, entry in enumerate(entries):
        key = entry[:-1].lower()
        if key in args.keys() and i < len(entries) - 1:
            args[key] = entries[i + 1]
    create_from_params(**args)


def unzip(model_zip, target_dir):
    with zipfile.ZipFile(model_zip) as zf:
        zf.extractall(target_dir)


def create_dirs(base_dir, directories):
    for directory in directories:
        directory = os.path.join(base_dir, directory)
        os.makedirs(directory, exist_ok=True)


def copy_scripts(target_dir, overwrite=True):
    candidates = ['main.py', 'rundocker.py']
    for candidate in candidates:
        target_file = os.path.join(target_dir, candidate)
        source_file = os.path.join(ROOT_DIR, candidate)
        if os.path.exists(target_file) and not overwrite:
            raise FileExistsError(target_file)
        shutil.copy(source_file, target_file)


def download_model(workspace, deployment, target_dir):
    webservices = Webservice.list(workspace)
    webservice = [
        webservice for webservice in webservices if webservice.name == deployment]
    if not webservice:
        raise Exception(
            f'Deployment {deployment} is not found in workspace {workspace}')

    if webservice[0].image:
        models = webservice[0].image.models
        base_image = 'mcr.microsoft.com/azureml/o16n-base/python-slim:latest.eastus'
    else:
        models = webservice[0].models
        base_image = webservice[0].environment.docker.base_image

    models = webservice[0].image.models if webservice[0].image else webservice[0].models
    if not models or len(models) > 1:
        raise Exception(
            f'Error in getting associated model for Deployment {deployment}')
    model = models[0]
    model.download(target_dir, True)

    return model.name, base_image


def write_configuration(base_dir, modelname, model_dir, input_dir, output_dir):
    target_file = os.path.join(base_dir, 'configuration.json')
    configuration = {"model": modelname,
                     "model_dir": model_dir,
                     "input_dir": input_dir,
                     "output_dir": output_dir}
    with open(target_file, 'w') as fp:
        json.dump(configuration, fp)


def write_readme(base_dir, input_dir, output_dir):
    input_dir = os.path.join(base_dir, input_dir)
    output_dir = os.path.join(base_dir, output_dir)
    content = f'''0. Setup your local alghost environment with specific version. For instance, "pip install alghost==0.0.83 -i https://pypi.org/simple --extra-index https://test.pypi.org/simple"
1. cd {base_dir}
2. Put your input json files in folder "{input_dir}". The sample input is available in API Doc
3. Run "python main.py"(local debug mode) or "python rundocker.py"(docker run mode)
4. Check your output json files in folder "{output_dir}"
    '''
    target_file = os.path.join(base_dir, 'Readme.txt')
    with open(target_file, 'w') as fp:
        fp.write(content)
    return content


def get_modelpackage(unzip_dir):
    modelpackage_file = os.path.join(unzip_dir, 'modelpackage.json')
    with open(modelpackage_file, 'r') as fp:
        modelpackage = json.load(fp)
    return modelpackage


def write_sample(unzip_dir, input_dir, sample_name='sample_input.json'):
    modelpackage = get_modelpackage(unzip_dir)
    sample = modelpackage.get('ExampleRequest', None)
    if not sample:
        return
    sample_inputs = sample['Inputs']
    schemas = modelpackage['Inputs']
    for input_key, input_values in sample_inputs.items():
        input_schema = [schema['InputSchema'] for schema in schemas if input_key in schema['InputPort'].split(':')]
        input_schema = json.loads(input_schema[0]) if input_schema else None
        names = [entry['name'] for entry in input_schema['columnAttributes']]
        input_dict = dict(zip(names, input_values[0]))
        sample_inputs[input_key] = [input_dict]
    target_file = os.path.join(input_dir, sample_name)
    with open(target_file, 'w') as fp:
        json.dump(sample, fp)


def write_conda(unzip_dir, target_dir):
    modelpackage = get_modelpackage(unzip_dir)
    conda = modelpackage['CondaYaml']
    conda_yaml = os.path.join(target_dir, 'conda.yaml')
    with open(conda_yaml, 'w') as fp:
        fp.write(conda)


def write_dockerfile(base_image, target_dir):
    dockerfile = DOCKFILE_TEMPLATE.format(base_image=base_image)
    with open(os.path.join(target_dir, 'Dockerfile'), 'w') as fp:
        fp.write(dockerfile)

# python -m azureml.designer.dagengine.debug.create_debug_env create_from_params subscription resourcegroup workspace deployment realtimeendpoint
# python -m azureml.designer.dagengine.debug.create_debug_env create_from_params 74eccef0-4b8d-4f83-b5f9-fa100d155b22 clwantest clwantest "" s7-test2
# python -m azureml.designer.dagengine.debug.create_debug_url create_from_url "https://master.ml.azure.com/endpoints/lists/realtimeendpoints/test-20191217-1509/detail?wsid=/subscriptions/e9b2ec51-5c94-4fa8-809a-dc1e695e4896/resourcegroups/ModuleX-WS2-rg/workspaces/ModuleX-PPE&tid=72f988bf-86f1-41af-91ab-2d7cd011db47"
if __name__ == '__main__':
    fire.Fire({'create_from_params': create_from_params, 'create_from_url': create_from_url})
