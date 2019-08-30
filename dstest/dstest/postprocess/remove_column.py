import logging
import click
import pandas as pd
from builtin_score import ioutil
import math
import numpy as np
import base64

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logging.info(f"in {__file__} v1")
logger = logging.getLogger(__name__)

class Process:
  def __init__(self, meta: dict = {}):
    self.remove_columns = str(meta.get('Remove Columns', '')).split(',')

  def run(self, input_df: pd.DataFrame, meta: dict = None):
    print(input_df.columns)
    df = input_df.copy()
    for col in self.remove_columns:
      if col in df.columns:
        print(f"deleting column {col}")
        del df[col]
        print(f"deleted column {col}")
      else:
        logger.info(f"skip column {col}")

    print("input_df:", input_df.columns)
    print("df:", df.columns)

    return df

@click.command()
@click.option('--input_path', default="datas/mnist")
@click.option('--output_path', default="outputs/mnist")
@click.option('--remvoe_columns', default="")
def run(input_path, output_path, remvoe_columns):
  """
  This functions removes specified column from input
  """
  meta = {
    "Remove Columns": remvoe_columns
  }
  proccesor = Process(meta)
  df = ioutil.read_parquet(input_path)
  result = proccesor.run(df)
  ioutil.save_parquet(result, output_path)

# python -m dstest.postprocess.remove_column  --input_path datas/mnist --output_path outputs/mnist --remvoe_columns=x
if __name__ == '__main__':
  run()
