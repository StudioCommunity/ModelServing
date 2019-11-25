class InputDataError(Exception):
    def __init__(self, schema, data):
        super().__init__(f'Input data are inconsistent with schema.\nSchema: {str(schema)}\nData: {str(data)}')


class ResourceLoadingError(Exception):
    def __init__(self, name, type_name):
        super().__init__(f'Failed to load {type_name} {name} from Model')
