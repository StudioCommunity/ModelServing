from sklearn import svm
from sklearn import datasets
import pickle
import os
from builtin_models.sklearn import save_model


# Test dynamic install package
from pip._internal import main as pipmain
pipmain(["install", "click"])
import click


@click.command()
@click.option('--model_path', default="./model/")
def run_pipeline(
    model_path
    ):
    clf = svm.SVC(gamma='scale')
    iris = datasets.load_iris()
    X, y = iris.data, iris.target
    # X: array([[5.1, 3.5, 1.4, 0.2],..., [5.9, 3. , 5.1, 1.8]])
    # # y: array([0,...,2])
    clf.fit(X, y)
    
    save_model(clf, model_path)

# python -m dstest.sklearn.trainer  --model_path model/sklearn
if __name__ == '__main__':
    run_pipeline()
    
