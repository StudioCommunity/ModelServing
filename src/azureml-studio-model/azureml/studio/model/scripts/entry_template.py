import os
import sys
from os.path import dirname, abspath

import click
import pytest
import numpy as np
import pandas as pd
from click.testing import CliRunner

from azureml.studio.score.score import module_invoker


def test_score():
    base_path = dirname(abspath(__file__))
    trained_model_path = os.path.join(base_path, "InputPort1")
    dataset_path = os.path.join(base_path, "InputPort2")
    scored_dataset_path = os.path.join(base_path, "OutputPort")

    os.makedirs(dataset_path, exist_ok=True)
    os.makedirs(scored_dataset_path, exist_ok=True)

    runner = CliRunner()
    result = runner.invoke(
        module_invoker.entrance,
        ["--trained-model", trained_model_path,
        "--dataset", dataset_path,
        "--scored-dataset", scored_dataset_path,
        "--append-score-columns-to-output", "true"])
    assert result.exit_code == 0