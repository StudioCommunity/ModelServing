from __future__ import print_function

import os
import sys
import pandas as pd
import numpy as np

# This is a placeholder for a Google-internal import.
import tensorflow as tf
from builtin_score.builtin_score_module import BuiltinScoreModule
from sklearn import datasets


def load_model_then_predict(model_path = "./model/sklearn/"):
  iris = datasets.load_iris()
  x, y = iris.data, iris.target
  x_test= x[:8] # only pick 8 test data
  df = pd.DataFrame(data=x_test, columns=['input']*x.shape[1], dtype=np.float64)
  #df.to_csv('iris_sklearn_test_data.csv')

  module = BuiltinScoreModule(model_path)
  result = module.run(df)
  print('=====buildinScoreModule=======')
  print(result)

# python -m dstest.sklearn.load_saved_model_predict_test --model_path model/sklearn
if __name__ == '__main__':
  load_model_then_predict()
