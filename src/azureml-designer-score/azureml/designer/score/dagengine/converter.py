from pandas import DataFrame
import pandas as pd
from azureml.studio.common.datatable.data_table import DataTable
from azureml.studio.common.datatable.data_type_conversion import convert_column_by_element_type
from azureml.studio.core.io.data_frame_directory import DataFrameDirectory
from azureml.studio.core.data_frame_schema import DataFrameSchema
from azureml.studio.modulehost.handler.data_handler import ZipHandler

import logging

def eprint(*args, **kwargs): print(*args, file=sys.stderr, **kwargs)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.log = print
logger.info = print
logger.warning = eprint
logger.error = eprint

def create_dfd_from_dict(json_data, schema_data):
    if schema_data:
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
        ret = DataFrameDirectory.create(data=df, schema=schema_data)
    else:
        ret = DataFrameDirectory.create(data=DataFrame(json_data))
    return ret


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