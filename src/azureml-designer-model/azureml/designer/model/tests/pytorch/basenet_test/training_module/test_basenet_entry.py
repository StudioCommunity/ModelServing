import os
from os.path import dirname, abspath

import torch
from PIL import Image
from azureml.designer.model.io import save_pytorch_state_dict_model, load_generic_model
from azureml.designer.model.model_spec.task_type import TaskType

from .densenet import DenseNet


def mock_image_directory_iterator(directory_path):
    for root, _, files in os.walk(directory_path):
        for filename in files:
            if filename == "_meta.yaml":
                continue
            file_path = os.path.join(root, filename)
            yield Image.open(file_path).convert('RGB'), 0, filename


def test_save_load():
    init_params = {
        "model_type": "densenet201",
        "pretrained": True,
        "memory_efficient": False,
        "num_classes": 3
    }
    model = DenseNet(**init_params)

    model_save_path = os.path.join(dirname(dirname(abspath(__file__))), "AzureMLModel")
    local_dependencies = [dirname(dirname(abspath(__file__)))]
    # Also support list and csv_file
    index_to_label = {
        0: "056.dog",
        1: "060.duc",
        2: "080.frog"
    }
    
    save_pytorch_state_dict_model(
        model,
        init_params=init_params,
        path=model_save_path,
        task_type=TaskType.MultiClassification,
        label_map=index_to_label,
        local_dependencies=local_dependencies
    )
    loaded_generic_model = load_generic_model(model_save_path)
    image_directory = os.path.join(dirname(dirname(abspath(__file__))), "images")
    image_iterator = mock_image_directory_iterator(image_directory)
    predict_result = loaded_generic_model.predict(image_iterator)
    print(f"predict_result =\n{predict_result}")

    loaded_pytorch_model = loaded_generic_model.raw_model
    assert isinstance(loaded_pytorch_model, torch.nn.Module)


if __name__ == "__main__":
    test_save_load()

