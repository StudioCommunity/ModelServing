import os
import logging
import click
import pandas as pd
import json
from ..utils import ioutils
import numpy as np

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logging.info(f"in {__file__} v1")
logger = logging.getLogger(__name__)

def get_category(prob, categories):
  pred = np.argsort(prob)[::-1] ##[::-1] inverse order
  # Get top1 label
  top1 = categories[pred[0]]
  # Get top5 label
  #top5 = [synset[pred[i]] for i in range(5)]
  return top1

def get_categories(prob, categories):
  result=[]
  for i in range (len(prob)): 
    category = get_category(prob[i], categories)
    result.append(category)
  return result

def save_categories_to_file(categories, file_name):
  with open(file_name, 'w') as fp:
    json.dump(categories, fp)

def read_categories_from_file(file_name):
  categories = []
  if(file_name.endswith(".json")):
    with open(file_name) as f:
        categories = json.load(f)
  else:
    categories = [l.strip() for l in open(file_name).readlines()]
  return categories


APPEND_CATEGORY_COLUMN_TO_OUTPUT_KEY = "Append category column to output"
CATEGORY_FILE_NAME_KEY = "Category File Name"
PROBABILITY_COLUMN_NAME_KEY = "Probability Column Name"
CATEGORY_COL_NAME = "Category"

class Process:
  def __init__(self, meta_path, meta: dict = {}):
    append_score_column_to_output_value_str = meta.get(APPEND_CATEGORY_COLUMN_TO_OUTPUT_KEY, None)
    self.append_score_column_to_output = isinstance(append_score_column_to_output_value_str, str) and\
        append_score_column_to_output_value_str.lower() == "true"
    print(f"self.append_score_column_to_output = {self.append_score_column_to_output}")

    self.prob_col = str(meta.get(PROBABILITY_COLUMN_NAME_KEY, ''))
    file_name = str(meta.get(CATEGORY_FILE_NAME_KEY, ''))
    self.file_name = os.path.join(meta_path, file_name)
    logger.info(f"reading from {self.file_name}")
    self.categories = read_categories_from_file(self.file_name)

  def run(self, input_df: pd.DataFrame, meta: dict = None):
    print(input_df.columns)
    result = get_categories(input_df[self.prob_col], self.categories)
    if(self.append_score_column_to_output):
      input_df.insert(len(input_df.columns), CATEGORY_COL_NAME, result, True)
      return input_df
    else:
      df = pd.DataFrame({'Category': result})
      return df

@click.command()
@click.option('--input_path', default="datas/mnist")
@click.option('--meta_path', default="model/vgg")
@click.option('--output_path', default="outputs/mnist")
@click.option('--file_name', default="")
@click.option('--prob_col', default="")
@click.option('--append_category_column_to_output', default="True")
def run(input_path, meta_path, output_path, file_name, prob_col, append_category_column_to_output):
  """
  read
  """
  
  meta = {
    CATEGORY_FILE_NAME_KEY: file_name,
    PROBABILITY_COLUMN_NAME_KEY: prob_col,
    APPEND_CATEGORY_COLUMN_TO_OUTPUT_KEY: append_category_column_to_output
  }

  proccesor = Process(meta_path, meta)
  df = ioutils.read_parquet(input_path)
  result = proccesor.run(df)
  print(result)
  ioutils.save_parquet(result, output_path, True)

# python -m azureml.visual_interface.score.postprocess.prob_to_category --meta_path outputs/densenet/Label_Map_Path --input_path outputs/densenet/Scored_dataset --output_path outputs/densenet/output --file_name=index_to_label.json --prob_col=Score_0 --append_category_column_to_output True
if __name__ == '__main__':
  #categories = read_categories_from_file("model/vgg/synset.txt")
  #save_categories_to_file(categories, "model/vgg/synset.json")
  run()
