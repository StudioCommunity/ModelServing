import os
import logging
import argparse

import pyarrow.parquet as pq # imported explicitly to avoid known issue of pd.read_parquet
import pandas as pd
import click

from . import constants
from .builtin_score_module import BuiltinScoreModule
from ..utils import ioutils

# python -m azureml.studio.score.module_invoker --trained-model ../dstest/model/tensorflow-minist/ --dataset ../dstest/outputs/mnist/ --scored-dataset ../dstest/outputs/mnist/ouput --append-score-columns-to-output True
# python -m azureml.studio.score.module_invoker --trained-model ../dstest/model/vgg/ --dataset ../dstest/outputs/imagenet/ --scored-dataset ../dstest/outputs/imagenet/ouput --append-score-columns-to-output True
# python -m azureml.visual_interface.score.score.module_invoker --trained-model ./azureml/visual_interface/score/score/tests/pytorch/InputPort1 --dataset ./azureml/visual_interface/score/score/tests/pytorch/InputPort2 --scored-dataset ./azureml/visual_interface/score/score/tests/pytorch/OutputPort --append-score-columns-to-output True

logger = logging.getLogger(__name__)

INPUT_FILE_NAME = "data.dataset.parquet" # hard coded, to be replaced, and we presume the data is DataFrame inside parquet

@click.command()
@click.option("--trained-model", help="Path to ModelDirectory")
@click.option("--dataset", help="Path to DFD")
@click.option("--append-score-columns-to-output", default="true", help="Preserve all columns from input dataframe or not")
def entrance(trained_model: str, dataset: str, scored_dataset: str, append_score_columns_to_output: str = "true"):
    logger.info(f"append_score_columns_to_output = {append_score_columns_to_output}")
    params = {
        constants.APPEND_SCORE_COLUMNS_TO_OUTPUT_KEY: append_score_columns_to_output
    }
    score_module = BuiltinScoreModule(trained_model, params)
    input_df = pd.read_parquet(os.path.join(dataset, INPUT_FILE_NAME), engine="pyarrow")
    output_df = score_module.run(input_df)

    logger.info(f"input_df =\n{input_df}")
    logger.info(f"output_df =\n{output_df}")
    logger.info(f"trying to save_parquet1(output_df, {scored_dataset})")
    ioutils.save_parquet1(output_df, scored_dataset)


if __name__ == "__main__":
    entrance()
    