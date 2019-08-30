import os
import json
import tensorflow as tf
import numpy as np
import pandas as pd
from builtin_score.builtin_score_module import *
from builtin_score.tensorflow_score_module import *
from builtin_score import ioutil

def test_tensor(model_path, df):
    with open(model_path + "model_spec.yml") as fp:
        config = yaml.safe_load(fp)

    tfmodule = TensorflowScoreModule(model_path, config)
    schema = tfmodule.get_schema()
    print(schema)
    with open(os.path.join(model_path, 'contract.json'), 'w') as f:
        json.dump(schema, f)
    
    result = tfmodule.run(df)
    print(result)

def test_builtin(model_path, df):
    module = BuiltinScoreModule(model_path, {"Append score columns to output": "True"})
    result = module.run(df)
    #print(result)
    return result

def prepare_input():
    df = ioutil.read_parquet("../dstest/outputs/mnist/")
    print(df.columns)
    return df

def prepare_input1():
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)
    batch_xs, batch_ys = mnist.train.next_batch(2)
    print(batch_xs)
    #df = pd.DataFrame()
    #df.insert(len(df.columns), 'x', batch_xs.tolist(), True)
    
    columns = [f"x.{i}" for i in range(784)]
    #columns = ['x']*784
    df = pd.DataFrame(data=batch_xs, columns=columns, dtype=np.float64)

    names = ["fixed acidity","volatile acidity","citric acid","residual sugar","chlorides","free sulfur dioxide","total sulfur dioxide","density","pH","sulphates","alcohol"]
    data = [[7,0.27,0.36,20.7,0.045,45,170,1.001,3,0.45,8.8]]
    df1 = pd.DataFrame(data=data, columns=names)
    
    df = pd.concat([df, df1], axis=1)    
    #df.to_parquet("test.parquet")
    #df.to_csv("test.csv")
    return df

def test(model_path, xname = "images", col1 = None, col2 = None):
    df = prepare_input()
    print(f"||| {df.columns}")
    if(xname != 'x'):
        if xname in df.columns:
            del df[xname]
        df = df.rename(columns={"x": xname})
    test_tensor(model_path, df)
    out = test_builtin(model_path, df)
    print(out.columns)
    print(out)
    _evaluate(out, col1, col2)

def _evaluate(df, col1, col2):
    if(col1 == None or col2 == None):
        return
    count = 0
    for index in range(len(df)):
        a = str(df[col1][index])
        b = str(df[col2][index])
        if a == b:
            count+=1
        else:
            print(f"## Failed: {index} {a} {b} {df['filename'][index]}")
    print(f"{count} correct in {len(df)} examples")

# python -m dstest.tensorflow.mnist_test
if __name__ == '__main__':
    model_path = "model/tensorflow-minist/"
    test(model_path, "x", "label", "y_label")
    
    model_path = "model/tensorflow-minist-saved-model/"
    test(model_path, "images")

    #saved_model_cli show --dir model/tensorflow-mnist-cnn-estimator/1565246816 --all
    model_path = "model/tensorflow-mnist-cnn-estimator/"
    test(model_path, "image", "label", "classes")