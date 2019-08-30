import logging
import click
import pandas as pd
import torch
from torchvision import transforms as T
from builtin_score import ioutil
from dstest.preprocess.datauri_util import tensor_to_datauri
from dstest.preprocess.preprocess import add_data_to_dataframe

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logging.info(f"in {__file__} v1")
logger = logging.getLogger(__name__)

class Process:
  def __init__(self, meta: dict = {}):
    self.tensor_column = meta.get("Tensor Column", "0")
    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  def run(self, input_df: pd.DataFrame, meta: dict = None):
    logger.info(f"input_df =\n{input_df}")
    output = []
    for _, row in input_df.iterrows():
      # ugly workaround to make demo work
      try:
        entry = row[self.tensor_column]
      except KeyError as e:
        entry = row[0]
      tensor = torch.Tensor(list(entry)).to(self.device)
      logger.info(f"tensor.size() = {tensor.size()}")
      tensor.squeeze_(0)
      tensor = self._denorm(tensor)
      output.append(tensor_to_datauri(tensor.cpu()))

    add_data_to_dataframe(input_df, output, "Result")
    return input_df

  def _denorm(self, x):
    # Convert the range from [-1, 1] to [0, 1]
    out = (x + 1) / 2
    return out.clamp_(0, 1)

@click.command()
@click.option('--input_path', default="datas/mnist")
@click.option('--output_path', default="outputs/mnist")
@click.option('--tensor_column', default="0")
def run(input_path, output_path, tensor_column):
  """
  This functions removes specified column from input
  """
  meta = {
    "Tensor Column": tensor_column
  }
  proccesor = Process(meta)
  df = ioutil.read_parquet(input_path)
  result = proccesor.run(df)
  #result = result[['image','Result']]
  # ioutil.save_parquet(result, output_path, True)
  ioutil.save_parquet1(result, output_path, True)

# python -m dstest.postprocess.tensor_to_image --input_path inputs/tensor_to_image --output_path outputs/tensor_to_image --tensor_column=0
if __name__ == '__main__':
  run()
