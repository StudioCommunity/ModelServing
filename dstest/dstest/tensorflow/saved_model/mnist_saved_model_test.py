from __future__ import print_function

import os
import sys
import pandas as pd
import numpy as np
import json

# This is a placeholder for a Google-internal import.
import tensorflow as tf
from builtin_score.builtin_score_module import *
from builtin_score.tensorflow_score_module import _TFSavedModelWrapper, TensorflowScoreModule
from builtin_score import ioutil

def test_TFSavedWrapper():
  export_dir = 'model/tensorflow-minist-saved-model/mnist'
  tf_meta_graph_tags = ['serve']
  tf_signature_def_key = 'predict_images'

  wrapper = _TFSavedModelWrapper(export_dir, tf_meta_graph_tags, tf_signature_def_key)

  from tensorflow.examples.tutorials.mnist import input_data
  mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)
  batch_xs, batch_ys = mnist.train.next_batch(2)
  df = pd.DataFrame(data=batch_xs, columns=['images']*784, dtype=np.float64)
  out = wrapper.predict(df)
  print(out)


# python -m dstest.tensorflow.saved_model.mnist_saved_model_test
if __name__ == '__main__':
  df = ioutil.read_parquet("../dstest/outputs/mnist/")
  #df = df.rename(columns={"x": "images"})
  df = df.rename(columns={"x": "x:0"})

  with open("model/tensorflow-minist-saved-model/model_spec.yml") as fp:
      config = yaml.safe_load(fp)

  model_path = "./model/tensorflow-minist-saved-model/"
  tfmodule = TensorflowScoreModule(model_path, config)
  
  schema = tfmodule.get_schema()
  print('#################')
  print(schema)

  with open(os.path.join(model_path, 'contract.json'), 'w') as f:
    json.dump(schema, f)

  result = tfmodule.run(df)
  print(result)

  model_path = "./model/tensorflow-minist-saved-model/"
  module = BuiltinScoreModule(model_path)
  result = module.run(df)
  print(result)
