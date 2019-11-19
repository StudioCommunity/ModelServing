import os
import json
import yaml
import click
import pandas as pd
from os import walk
import base64
import pyarrow.parquet as pq

from ..utils import ioutils, datauri_utils
from ..logger import get_logger

logger = get_logger(__name__)
logger.info(f"in {__file__}")
logger.info(f"Load pyarrow.parquet explicitly: {pq}")

@click.command()
@click.option('--input_path', default="inputs/mnist")
@click.option('--output_path', default="datas/mnist")
def run(input_path, output_path):
  """
  This function reads images in an folder and encode it ans base64. Then save it as csv in output_path.
  """
  import glob
  print(f'INPUT_PATH({input_path}) : {os.listdir(input_path)}')
  types = ('**.jpg', '**.png', '**.jfif') # the tuple of file types
  files_grabbed = []
  for files in types:
    pattern = os.path.join(input_path,files)
    files_grabbed.extend(glob.glob(pattern))
  
  print(f"Got {len(files_grabbed)} files in folder {input_path}")
  print(files_grabbed)

  df = pd.DataFrame(columns=["image"])
  for i in range(len(files_grabbed)):
    filename = files_grabbed[i]
    image_64_encode = datauri_utils.imgfile_to_datauri(filename)
    df.loc[i] = image_64_encode

  ioutils.save_as_datatable(df, output_path)
  print(f"df =\n{df}")
  print(f'OUTPUT_PATH({output_path}) : {os.listdir(output_path)}')

# python -m dstest.preprocess.import_image  --input_path inputs/mnist --output_path pip/mnist
# python -m dstest.preprocess.import_image  --input_path inputs/imagenet --output_path datas/imagenet
if __name__ == '__main__':
    run()
