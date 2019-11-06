import pickle
import os

from azureml.studio.model.generic import GenericModel


class BayesianModel(GenericModel):
    
    def __init__(self, model):
        self.model = model
        
    def save(self, save_to):
        model_path = os.path.join(save_to, "data.ilearner")
        with open(model_path, "wb") as fp:
            pickle.dump(self.model, fp)
    
    @classmethod
    def load(cls, load_from):
        model_path = os.path.join(load_from, "data.ilearner")
        with open(model_path, "rb") as fp:
            model = pickle.load(fp)
        return cls(model)