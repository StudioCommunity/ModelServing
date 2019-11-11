import os
from os.path import dirname, abspath

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import BayesianRidge

from my_custom_model import MyCustomModel

from azureml.studio.model.io import save_generic_model, load_generic_model


def test_save_load():
    np.random.seed(0)
    n_samples, n_features = 10, 2
    X = np.random.randn(n_samples, n_features) 
    lambda_ = 4.
    w = np.zeros(n_features)
    relevant_features = np.random.randint(0, n_features, 10)
    for i in relevant_features:
        w[i] = stats.norm.rvs(loc=0, scale=1. / np.sqrt(lambda_))
    alpha_ = 50.
    noise = stats.norm.rvs(loc=0, scale=1. / np.sqrt(alpha_), size=n_samples)
    y = np.dot(X, w) + noise

    clf = BayesianRidge(compute_score=True)
    clf.fit(X, y)
    y_hat = clf.predict(X)

    model = MyCustomModel(clf)
    model.conda = {
        "name": "test",
        "channels": "defaults",
        "dependencies": [{"pip": ["scipy", "sklearn"]}]
    }

    model_save_path = os.path.join(dirname(abspath(__file__)), "AzureMLModel")

    save_generic_model(model, path=model_save_path)

    df = pd.DataFrame(data=X)
    
    loaded_generic_model = load_generic_model(model_save_path)
    result_df = loaded_generic_model.predict(df)
    assert (result_df.to_numpy() == y_hat.reshape(-1, 1)).all()