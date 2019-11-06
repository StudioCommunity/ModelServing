import os

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import BayesianRidge

from bayesian_model import BayesianModel

# from azureml.studio.model.package_info import PROJECT_ROOT_PATH
from azureml.studio.model.io import save_generic_model


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

    # score_test_path = os.path.join(PROJECT_ROOT_PATH, "azureml-studio-score/azureml/studio/score/score/tests/official")
    # model_save_path = os.path.join(score_test_path, "InputPort1")
    # dataset_save_path = os.path.join(score_test_path, "InputPort2", "data.dataset.parquet")

    model = BayesianModel(clf)

    # save_generic_model(model, path=model_save_path)
    save_generic_model(model)

    # df = pd.DataFrame(data=X)
    # if os.path.exists(dataset_save_path):
    #     os.remove(dataset_save_path)
    # df.columns = df.columns.astype(str)
    # df.to_parquet(dataset_save_path, engine="pyarrow")
