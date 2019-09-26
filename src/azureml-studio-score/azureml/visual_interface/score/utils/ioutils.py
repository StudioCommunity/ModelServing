import logging
import os
import json
import pandas as pd
import numpy as np

logging.info(f"in {__file__}")
logger = logging.getLogger(__name__)

def read_parquet(data_path):
    """
    :param file_name: str,
    :return: pandas.DataFrame
    """
    logger.info("start reading parquet.")
    df = pd.read_parquet(os.path.join(data_path, 'data.dataset.parquet'), engine='pyarrow')
    logger.info("parquet read completed.")
    return df

def save_parquet1(df, output_path, writeCsv= False):
  from azureml.studio.modulehost.handler.port_io_handler import OutputHandler
  from azureml.studio.common.datatypes import DataTypes
  from azureml.studio.common.datatable.data_table import DataTable
  os.makedirs(output_path, exist_ok=True)
  #requires alghost 70
  OutputHandler.handle_output(DataTable(df), output_path, 'data.dataset.parquet', DataTypes.DATASET)
  save_datatype(output_path)
  logger.info(f"saved parquet to {output_path}, columns {df.columns}")

def transform_to_list(root):
    if isinstance(root, np.ndarray):
        root = root.tolist()
    if isinstance(root, list):
        root = [transform_to_list(child) for child in root]
    return root

def transform_ndarraycol_to_list(df):
    df.columns = df.columns.astype(str)
    if df.shape[0] > 0:
        for col in df.columns:
            if type(df.loc[0][col]) == np.ndarray:
                df[col] = df[col].transform(transform_to_list)
                logger.info(f"transformed ndarray column '{col}' to list")
    return df
   
def save_as_datatable(df, output_path):
    from azureml.studio.common.datatypes import DataTypes
    from azureml.studio.common.datatable.data_table import DataTable
    from azureml.studio.modulehost.handler.port_io_handler import OutputHandler
    
    os.makedirs(output_path, exist_ok=True)
    df = transform_ndarraycol_to_list(df)
    OutputHandler.handle_output(
        data=DataTable(df),
        file_path=output_path,
        file_name='data.dataset.parquet',
        data_type=DataTypes.DATASET,
    )
    logger.info(f"saved data to {output_path}, columns {df.columns}")

def save_datatype(output_path):
    dct = {
        "Id": "Dataset",
        "Name": "Dataset .NET file",
        "ShortName": "Dataset",
        "Description": "A serialized DataTable supporting partial reads and writes",
        "IsDirectory": False,
        "Owner": "Microsoft Corporation",
        "FileExtension": "dataset.parquet",
        "ContentType": "application/octet-stream",
        "AllowUpload": False,
        "AllowPromotion": True,
        "AllowModelPromotion": False,
        "AuxiliaryFileExtension": None,
        "AuxiliaryContentType": None
    }
    with open(os.path.join(output_path, 'data_type.json'), 'w') as f:
        json.dump(dct, f)

  
def save_parquet(df, output_path, writeCsv= False):
    os.makedirs(output_path, exist_ok=True)
    if(writeCsv):
        df.to_csv(os.path.join(output_path, "data.csv"))
    df = transform_ndarraycol_to_list(df)
    df.to_parquet(fname=os.path.join(output_path, "data.dataset.parquet"), engine='pyarrow')

    # Dump data_type.json as a work around until SMT deploys
    save_datatype(output_path)
    logger.info(f"saved parquet to {output_path}, columns {df.columns}")

def from_df_column_to_array(col):
    if(len(col)==0):
        return []
    
    if(col.dtype == 'O'):
        shape = []
        shape.append(len(col))
        shape.append(len(col[0]))
        values = np.zeros(shape)
        for i in range(len(col)):
            values[i] = col[i]
        return values
    else:
        return col.values