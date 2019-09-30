import os
from os.path import dirname, abspath
import pytest

from ... import module_invoker


def test_score():
    base_path = dirname(abspath(__file__))
    trained_model_path = os.path.join(base_path, "InputPort1")
    dataset_path = os.path.join(base_path, "InputPort2")
    scored_dataset_path = os.path.join(base_path, "OutputPort")

    module_invoker.entrance(trained_model_path, dataset_path, scored_dataset_path, True)