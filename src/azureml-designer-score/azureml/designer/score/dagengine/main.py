import os
import json
import Score


class MockRequest(object):
    def __init__(self, method, args, data):
        self.method = method
        self.args = args
        self.data = data

    def get_data(self):
        return self.data


def testrun(input_file, output_dir):
    basename = os.path.basename(input_file)
    output_file = os.path.join(output_dir, 'out.' + basename)
    with open(input_file, 'r') as fp:
        data = fp.read()
    method = 'POST'
    args = {'format': 'swagger', 'details': 'false', 'verbose': 'true'}
    request = MockRequest(method, args, data)

    response = Score.run(request)
    if isinstance(response, str):
        output_json = response
        with open(output_file, 'w') as fp:
            fp.write(output_json)
    else:
        if response.is_json:
            output_json = response.get_json()
            with open(output_file, 'w') as fp:
                json.dump(output_json, fp)
        else:
            output_json = response.data
            with open(output_file, 'wb') as fp:
                fp.write(output_json)



if __name__ == "__main__":
    with open('configuration.json', 'r') as fp:
        configuration = json.load(fp)
    os.environ['MODELPATH'] = configuration['model_dir']
    input_dir = configuration['input_dir']
    output_dir = configuration['output_dir']

    Score.init()
    for input_file in os.listdir(input_dir):
        input_file = os.path.join(input_dir, input_file)
        if os.path.isfile(input_file):
            testrun(input_file, output_dir)
    print(f'Output files are available in folder "{output_dir}"')
