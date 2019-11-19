import click
import pandas as pd
import torch
from torchvision import transforms as T

from ..utils import ioutils, datauri_utils, dfutils
from ..logger import get_logger

logger = get_logger(__name__)
logger.info(f"in {__file__} v1")


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
      output.append(datauri_utils.tensor_to_datauri(tensor.cpu()))

    dfutils.add_column_to_dataframe(input_df, output, "Result")
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
  df = ioutils.read_parquet(input_path)
  result = proccesor.run(df)
  #result = result[['image','Result']]
  ioutils.save_dfd(result, output_path)

# python -m dstest.postprocess.tensor_to_image --input_path inputs/tensor_to_image --output_path outputs/tensor_to_image --tensor_column=0
if __name__ == '__main__':
  run()
