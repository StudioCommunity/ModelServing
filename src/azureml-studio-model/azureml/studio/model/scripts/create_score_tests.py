import os
from os.path import dirname, abspath
import shutil

PROJECT_ROOT_PATH = dirname(dirname(dirname(dirname(dirname(dirname(abspath(__file__)))))))
MODEL_FOLDER_NAME = "AzureMLModel"
DATA_FILE_NAME = "data.dataset.parquet"


def test_create_score_test_cases():
    src_tests_root_dir = os.path.join(PROJECT_ROOT_PATH, "azureml-studio-model/azureml/studio/model/tests")
    dst_tests_root_dir = os.path.join(PROJECT_ROOT_PATH, "azureml-studio-score/azureml/studio/score/score/tests")
    template_file_path = os.path.join(dirname(abspath(__file__)), "entry_template.py")

    for root, dirs, files in os.walk(src_tests_root_dir):
        if MODEL_FOLDER_NAME in dirs and DATA_FILE_NAME in files:
            model_dir = os.path.join(root, MODEL_FOLDER_NAME)
            data_file = os.path.join(root, DATA_FILE_NAME)

            relative_to_tests_root = os.path.relpath(root, src_tests_root_dir)
            dst_test_path = os.path.join(dst_tests_root_dir, relative_to_tests_root)
            input_port1_path = os.path.join(dst_test_path, "InputPort1")
            input_port2_path = os.path.join(dst_test_path, "InputPort2")

            os.makedirs(input_port2_path, exist_ok=True)
            shutil.copytree(model_dir, input_port1_path)
            shutil.copy(data_file, os.path.join(input_port2_path, DATA_FILE_NAME))

            entry_file_name = f"test_{'_'.join(os.path.split(relative_to_tests_root))}.py"
            shutil.copy(template_file_path, os.path.join(dst_test_path, entry_file_name))


if __name__ == "__main__":
    test_create_score_test_cases()
