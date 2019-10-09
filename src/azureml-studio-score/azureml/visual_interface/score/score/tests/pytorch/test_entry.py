import os
import sys
from os.path import dirname, abspath

import pytest
import numpy as np
import pandas as pd

# TODO: Substitude visual_interface when Module solves conflict issue in azureml.studio.__init__.py
import azureml.visual_interface.model.pytorch
import azureml.visual_interface.model.generic
from azureml.visual_interface.score.score import module_invoker


def test_score():
    base_path = dirname(abspath(__file__))
    trained_model_path = os.path.join(base_path, "InputPort1")
    dataset_path = os.path.join(base_path, "InputPort2")
    scored_dataset_path = os.path.join(base_path, "OutputPort")

    os.makedirs(dataset_path, exist_ok=True)
    os.makedirs(scored_dataset_path, exist_ok=True)

    module_invoker.entrance(trained_model_path, dataset_path, scored_dataset_path, True)