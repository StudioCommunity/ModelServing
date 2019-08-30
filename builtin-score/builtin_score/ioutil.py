import logging
import os
import json
import pandas as pd
import numpy as np

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
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

def ensure_folder_exists(output_path):
  if not os.path.exists(output_path):
    os.makedirs(output_path)
    logger.info(f"{output_path} not exists, created")

def save_parquet(df, output_path, writeCsv= False):
  from azureml.studio.modulehost.handler.port_io_handler import OutputHandler
  from azureml.studio.common.datatypes import DataTypes
  from azureml.studio.common.datatable.data_table import DataTable
  ensure_folder_exists(output_path)
  #requires alghost 70
  OutputHandler.handle_output(DataTable(df), output_path, 'data.dataset.parquet', DataTypes.DATASET)
  save_datatype(output_path)
  logger.info(f"saved parquet to {output_path}, columns {df.columns}")

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

def transform_to_list(root):
  if type(root) == np.ndarray:
    root = root.tolist()
  if type(root) == list:
    for i, child in enumerate(root):
      root[i] = transform_to_list(child)
  return root
  
def save_parquet1(df, output_path, writeCsv= False):
  ensure_folder_exists(output_path)
  if(writeCsv):
    df.to_csv(os.path.join(output_path, "data.csv"))
  # convert df columns type, as parquet must have string column names
  df.columns = df.columns.astype(str)
  if df.shape[0] > 0:
    for col in df.columns:
      if type(df.loc[0][col]) == np.ndarray:
        df[col] = df[col].transform(transform_to_list)
        logger.info(f"transformed ndarray column '{col}' to list")
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
  