import os
from os.path import dirname, abspath

import numpy as np
import pandas as pd
import pyarrow.parquet
from scipy import stats
from sklearn.linear_model import BayesianRidge

from my_custom_model import MyCustomModel

from azureml.designer.model.io import save_generic_model, load_generic_model


def get_traininig_data():
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
    return X, y


def test_save_load():
    clf = BayesianRidge(compute_score=True)
    X, y = get_traininig_data()
    clf.fit(X, y)
    y_hat = clf.predict(X)

    model = MyCustomModel(clf)
    conda = {
        "name": "test",
        "channels": "defaults",
        "dependencies": [{"pip": ["scipy", "sklearn"]}]
    }

    model_save_path = os.path.join(dirname(abspath(__file__)), "AzureMLModel")
    local_dependencies = [dirname(abspath(__file__))]

    save_generic_model(model, path=model_save_path, conda=conda, local_dependencies=local_dependencies)

    df = pd.DataFrame(data=X)
    df.columns = df.columns.astype(str)
    
    loaded_generic_model = load_generic_model(model_save_path)
    result_df = loaded_generic_model.predict(df)
    assert (result_df.to_numpy() == y_hat.reshape(-1, 1)).all()

    data_save_path = os.path.join(dirname(abspath(__file__)), "data.dataset.parquet")
    df.to_parquet(data_save_path, engine="pyarrow")
