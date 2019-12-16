import traceback
from azureml.studio.common.error import ModuleError

class InputDataError(Exception):
    def __init__(self, schema, data, with_traceback = False):
        errmsg = f'Input data are inconsistent with schema.\nSchema: {str(schema)[:256]}\nData: {str(data)[:256]}\n'
        if with_traceback:
            errmsg += traceback.format_exc()
        super().__init__(errmsg)


class ResourceLoadingError(Exception):
    def __init__(self, name, type_name, with_traceback = False):
        errmsg = f'Failed to load {type_name} {name} from Model'
        if with_traceback:
            errmsg += traceback.format_exc()
        super().__init__(errmsg)
