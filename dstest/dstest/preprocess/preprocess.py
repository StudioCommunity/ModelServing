import logging
import click
import pandas as pd
import cv2
from builtin_score import ioutil
from . import datauri_util
from . import mnist
from . import imagenet
from . import stargan

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logging.info(f"in {__file__} v1")
logger = logging.getLogger(__name__)

def add_data_to_dataframe(input_df, results, target_column):
  if target_column in input_df.columns:
    logger.info(f"writing to column {target_column}")
    input_df[target_column] = results
  else:
    logger.info(f"append new column {target_column}")
    input_df.insert(len(input_df.columns), target_column, results, True)

class PreProcess:
  def __init__(self, meta: dict = {}):
    self.image_column = str(meta.get('Image Column', 'image'))
    self.target_column = str(meta.get('Target Column', ''))
    self.target_data_column = str(meta.get('Target DataURI Column', ''))

    try:
      image_size = str(meta.get('Target Image Size', '')).strip()
      self.target_image_size = tuple(map(int, image_size.split('x')))
    except:
      raise Exception('Invalid [Target Image Size] Parameter:{image_size}')
    
    if not self.target_column:
      self.target_column = self.image_column

    if not self.target_data_column:
      self.target_data_column = f"{self.target_column}_data"

    print(self.image_column, self.target_column)

  def run(self, input_df: pd.DataFrame, meta: dict = None):
    results = []
    datauris = []
    
    for index, row in input_df.iterrows():
      if self.target_image_size == (256, 256):
        img = datauri_util.base64str_to_image(row[self.image_column])
        img = stargan.transform_image_stargan(img)
         # append datauris
        if self.target_data_column:
          datauri = datauri_util.tensor_to_datauri(img)
          datauris.append(datauri)
        results.append(img.tolist())
      else:
        #print(row['label'])
        img = datauri_util.base64str_to_ndarray(row[self.image_column])
        if self.target_image_size == (28,28):
          #logging.info(f"### mnist")
          img = mnist.transform_image_mnist(img)
        elif self.target_image_size == (224,224):
          #logging.info(f"### imagenet")
          img = imagenet.transform_image_imagenet(img)
        else:
          raise Exception("Not Implemented")
        
        # append datauris
        if self.target_data_column:
          datauri = datauri_util.img_to_datauri(img)
          datauris.append(datauri)
        
        # save the processed images
        cv2.imwrite("outputs/image_"+str(index)+".png", img)
        
        # Convert to 0-1 based range, so we can save it in dataframe
        flatten = img.flatten() / 255.0
        results.append(flatten)

    add_data_to_dataframe(input_df, results, self.target_column)
    if self.target_data_column:
      add_data_to_dataframe(input_df, datauris, self.target_data_column)
    logger.info(f"input_df.columns = {input_df.columns}")
    logger.info(f"input_df = \n {input_df}")

    return input_df

@click.command()
@click.option('--input_path', default="datas/mnist")
@click.option('--output_path', default="outputs/mnist")
@click.option('--image_column', default="image")
@click.option('--target_column', default="x")
@click.option('--target_datauri_column', default="")
@click.option('--target_image_size', default="")
def run(input_path, output_path, image_column, target_column, target_datauri_column, target_image_size):
  """
  This functions read base64 encoded images from df. Transform to format required by model input.
  """
  meta = {
    "Image Column": image_column,
    "Target Column": target_column,
    "Target DataURI Column": target_datauri_column,
    "Target Image Size": target_image_size
  }
  proccesor = PreProcess(meta)

  df = ioutil.read_parquet(input_path)
  result = proccesor.run(df)
  ioutil.save_parquet(result, output_path)

# mnist: python -m dstest.preprocess.preprocess  --input_path datas/mnist --output_path outputs/mnist --image_column=image --target_column=x --target_datauri_column=x.data --target_image_size=28x28
# imagenet: python -m dstest.preprocess.preprocess  --input_path datas/imagenet --output_path outputs/imagenet --image_column=image --target_column=import/images --target_datauri_column=import/images.data --target_image_size=224x224
# stargan: python -m dstest.preprocess.preprocess  --input_path inputs/stargan --output_path outputs/stargan --image_column=image --target_column=import/images --target_datauri_column=import/images.data --target_image_size=256x256
if __name__ == '__main__':
  run()
