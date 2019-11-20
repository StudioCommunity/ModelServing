import os

import pandas as pd
import pickle

from azureml.designer.model.core_model import CoreModel


class MyCustomModel(CoreModel):
    
    def __init__(self, model):
        self.model = model
        
    def save(self, save_to, overwrite_if_exists=True):
        os.makedirs(save_to, exist_ok=overwrite_if_exists)
        model_path = os.path.join(save_to, "data.ilearner")
        with open(model_path, "wb") as fp:
            pickle.dump(self.model, fp)
    
    @classmethod
    def load(cls, load_from):
        model_path = os.path.join(load_from, "data.ilearner")
        with open(model_path, "rb") as fp:
            model = pickle.load(fp)
        return cls(model)

    def predict(self, df):
        result = self.model.predict(df.to_numpy())
        result_df = pd.DataFrame(data=result, columns=["score"])
        result_df.columns = result_df.columns.astype(str)
        return result_df
