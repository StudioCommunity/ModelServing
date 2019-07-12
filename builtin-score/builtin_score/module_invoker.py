import pyarrow.parquet as pq # imported explicitly to avoid known issue of pd.read_parquet
import argparse
import pandas as pd
import os
from .builtin_score_module import BuiltinScoreModule

INPUT_FILE_NAME = "data.dataset.parquet" # hard coded, to be replaced, and we presume the data is DataFrame inside parquet
OUTPUT_FILE_NAME = "output.csv"

parser = argparse.ArgumentParser()
parser.add_argument("--trained-model", type=str, help="model path")
parser.add_argument("--dataset", type=str, help="dataset")
parser.add_argument('--append-score-columns-to-output', choices=('True', 'False'))
parser.add_argument("--scored-dataset", type=str, help="scored dataset path")

args, _ = parser.parse_known_args()
score_module = BuiltinScoreModule(args.trained_model)
input_df = pd.read_parquet(os.path.join(args.dataset, INPUT_FILE_NAME), engine="pyarrow")
output_df = score_module.run(input_df)

if not os.path.exists(args.scored_dataset):
    os.makedirs(args.scored_dataset)
output_df.to_csv(os.path.join(args.scored_dataset, OUTPUT_FILE_NAME))
