import os
import numpy as np
import pandas as pd
import sklearn

from sklearn.externals import joblib
from . import constants


class SklearnScoreModule(object):
    
    def __init__(self, model_path, config):
        sklearn_conf = config["sklearn"]
        model_file_path = os.path.join(model_path, sklearn_conf[constants.MODEL_FILE_PATH_KEY])
        
        serialization_method = sklearn_conf.get(constants.SERIALIZATION_METHOD_KEY, 'pickle') # try to get gpu model firstly
        if serialization_method == 'pickle':
            import pickle
            with open(model_file_path, "rb") as fp:
                self.model = pickle.load(fp)
        elif serialization_method == 'joblib':
            from sklearn.externals import joblib
            self.model = joblib.load(model_file_path)
        else:
            raise Exception(f"Unrecognized serializtion format {serialization_method}")

    
    def run(self, df):
        y = self.model.predict(df)
        return y
