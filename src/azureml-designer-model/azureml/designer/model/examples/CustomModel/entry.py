import os

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import BayesianRidge

from my_custom_model import MyCustomModel

from azureml.designer.model.io import save_generic_model, load_generic_model

def get_training_data():
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

def main():
    X, y = get_training_data()
    clf = BayesianRidge(compute_score=True)
    clf.fit(X, y)
    y_hat = clf.predict(X)

    model = MyCustomModel(clf)
    model.conda = {
        "name": "test",
        "channels": ["defaults"],
        "dependencies": [{"pip": ["scipy", "sklearn"]}]
    }

    save_generic_model(model, path="./AzureMLModel")
    loaded_generic_model = load_generic_model(path="./AzureMLModel", install_dependencies=False)

    df = pd.DataFrame(data=X)
    result_df = loaded_generic_model.predict(df)
    print(f"result_df = {result_df}")
    assert (result_df.to_numpy() == y_hat.reshape(-1, 1)).all()

if __name__ == "__main__":
    main()
    