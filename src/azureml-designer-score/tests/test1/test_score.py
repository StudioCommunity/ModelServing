import os
import json
import sys
import os.path
DS_PATH = os.path.dirname(__file__)
ROOT_PATH = os.path.join(DS_PATH, '../..')
sys.path.append(ROOT_PATH)
import azureml.designer.dagengine.score_implementation as score_implementation


class MockRequest(object):
    def __init__(self, method, args, data):
        self.method = method
        self.args = args
        self.data = data

    def get_data(self):
        return self.data

def test_run():
    response = None
    input_dir, output_dir = init()
    for input_file in os.listdir(input_dir):
        input_file = os.path.join(input_dir, input_file)
        if os.path.isfile(input_file):
            response = run(input_file, output_dir)
            assert response.status_code == 200 
    print(f'Output files are available in folder "{output_dir}"')
    return response

def init():
    os.environ['DSPATH'] = DS_PATH
    config_file = os.path.join(DS_PATH, 'configuration.json')
    with open(config_file, 'r') as fp:
        configuration = json.load(fp)
    os.environ['MODELPATH'] = os.path.join(DS_PATH, configuration['model_dir'])
    input_dir = os.path.join(DS_PATH, configuration['input_dir'])
    output_dir = os.path.join(DS_PATH, configuration['output_dir'])
    score_implementation.init()
    return input_dir, output_dir

def run(input_file, output_dir):
    basename = os.path.basename(input_file)
    output_file = os.path.join(output_dir, 'out.' + basename)
    with open(input_file, 'r') as fp:
        data = fp.read()
    method = 'POST'
    args = {'format': 'swagger', 'details': 'false', 'verbose': 'true'}
    request = MockRequest(method, args, data)

    response = score_implementation.run(request)
    output_data = response.get_data()
    with open(output_file, 'wb') as fp:
        fp.write(output_data)
    return response

if __name__ == "__main__":
    test_run()
