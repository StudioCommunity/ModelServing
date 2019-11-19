import os
import json
import tensorflow as tf
import numpy as np
import pandas as pd
from azureml.designer.score.builtin_score_module import *
from azureml.designer.score.tensorflow_score_module import *
from azureml.designer.score.utils import ioutils

def test_builtin(model_path, df):
    module = BuiltinScoreModule(model_path, {"Append score columns to output": "True"})
    result = module.run(df)
    print(result)
    return result

def prepare_input():
    df = ioutils.read_parquet("../dstest/outputs/stargan/")
    results = []
    for index in range(len(df)):
      results.append([[0, 0, 0, 1, 0]])
    df.insert(len(df.columns), "c", results, True)
    print(df.columns)
    ioutils.save_parquet(df, "outputs/stargan/model_input", True)
    return df


def test(model_path):
    df = prepare_input()
    out = test_builtin(model_path, df)
    ioutils.save_parquet(out, "outputs/stargan/model_output", True)
    print(out.columns)
    print(out)
    print(out["0"].shape)

# python -m dstest.pytorch.stargan
if __name__ == '__main__':
    # model_path = "model/stargan/"
    # test(model_path)

    out = ioutils.read_parquet("outputs/stargan/model_output")
    print(out.columns)
    #print(out["0"][0])