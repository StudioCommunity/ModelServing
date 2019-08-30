import os

import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import load_model
from keras import backend as K
from . import constants


class KerasScoreModule(object):
    
    def __init__(self, model_path, config):
        keras_conf = config["keras"]
        self.model = load_model(os.path.join(model_path, keras_conf[constants.MODEL_FILE_PATH_KEY]))
        print(f"Successfully loaded model from {model_path}")


    def run(self, df):
        df_output = pd.DataFrame([])
        for _, row in df.iterrows():
            input_params = []
            #print(f"Row = \n {row}")
            if(self.is_image(row)):
                tensor_row = tf.convert_to_tensor(row)
                input_row = K.reshape(tensor_row,(1, -1))
                input_params.append(input_row)
            else:
                for input_arg in row:
                    tensor_arg = tf.convert_to_tensor(input_arg)
                    input_params.append(tensor_arg)
            y_output = self.model.predict(input_params, steps = 1)
            tensor_row_output = y_output.reshape(1, -1)
            df_output = df_output.append(pd.DataFrame(tensor_row_output), ignore_index=True)

        return df_output


    def is_image(self, row):
        # TO DO:
        if(len(row)>100):
            return True
        else:
            return False
