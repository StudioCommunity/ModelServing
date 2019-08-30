import os
import json
import tensorflow as tf
import numpy as np
import pandas as pd
from builtin_score.builtin_score_module import *
from builtin_score.python_score_module import *
from builtin_score import ioutil
import builtin_models.python

class DummyPythonModel(object):
    """
    Represents a generic Python model that evaluates inputs and produces API-compatible outputs.
    By subclassing :class:`~PythonModel`, users can create customized models.
    """
    def __init__(self):
        self.a = 1
        self.b = 2
        self.c = 3

    def predict(self, x, y):
        """
        Evaluates a pyfunc-compatible input and produces a pyfunc-compatible output.

        :param model_input: dataframe .

        """
        print("input:")
        print(x)
        print(y)
        result = self.a *x + self.b * y + self.c
        return result

# python -m dstest.python.dummy
if __name__ == '__main__':
    x = np.array([1, 2, 3])
    y = np.array([1, 2, 3])
    d = {'x': x, 'y': y}
    model = DummyPythonModel()
    model.b = 2
    result = model.predict(x, y)
    print(f"result: {result}")

    model_path = "model/python/dummy/"
    builtin_models.python.save_model(model, model_path)

    model1 = builtin_models.python.load_model(model_path)
    result = model1.predict(x, y)
    print(f"result: {result}")
    
    df = pd.DataFrame(data=d)
    # test_tensor(model_path, df)
    module = BuiltinScoreModule(model_path, {"Append score columns to output": "True"})
    result = module.run(df)
    print(result.columns)
    print(f"result: {result}")