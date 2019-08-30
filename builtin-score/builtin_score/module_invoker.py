import os
import argparse

import pyarrow.parquet as pq # imported explicitly to avoid known issue of pd.read_parquet
import pandas as pd

from . import constants
from .builtin_score_module import BuiltinScoreModule
from . import ioutil

# python -m builtin_score.module_invoker --trained-model ../dstest/model/tensorflow-minist/ --dataset ../dstest/outputs/mnist/ --scored-dataset ../dstest/outputs/mnist/ouput --append-score-columns-to-output true
# python -m builtin_score.module_invoker --trained-model ../dstest/model/vgg/ --dataset ../dstest/outputs/imagenet/ --scored-dataset ../dstest/outputs/imagenet/ouput --append-score-columns-to-output true
# python -m builtin_score.module_invoker --trained-model test/TestInputPort1 --dataset test/TestInputPort2 --scored-dataset test/TestOutputFolder --append-score-columns-to-output true

INPUT_FILE_NAME = "data.dataset.parquet" # hard coded, to be replaced, and we presume the data is DataFrame inside parquet
OUTPUT_FILE_NAME = "output.csv"

parser = argparse.ArgumentParser()
parser.add_argument("--trained-model", type=str, help="model path")
parser.add_argument("--dataset", type=str, help="dataset")
parser.add_argument('--append-score-columns-to-output', choices=('true', 'false'))
parser.add_argument("--scored-dataset", type=str, help="scored dataset path")

args, _ = parser.parse_known_args()
params = {
    constants.APPEND_SCORE_COLUMNS_TO_OUTPUT_KEY: args.append_score_columns_to_output
}
score_module = BuiltinScoreModule(args.trained_model, params)
input_df = pd.read_parquet(os.path.join(args.dataset, INPUT_FILE_NAME), engine="pyarrow")
output_df = score_module.run(input_df)

print(f"input_df =\n{input_df}")
print(f"output_df =\n{output_df}")
print(f"trying to save_parquet1(output_df, {args.scored_dataset})")
# ioutil.save_parquet(output_df, args.scored_dataset)
ioutil.save_parquet1(output_df, args.scored_dataset)