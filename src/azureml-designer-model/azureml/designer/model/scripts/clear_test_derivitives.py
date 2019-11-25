import os
from os.path import dirname, abspath
import shutil

PROJECT_ROOT_PATH = dirname(dirname(dirname(dirname(dirname(dirname(abspath(__file__)))))))
MODEL_FOLDER_NAME = "AzureMLModel"
DFD_DIR_NAME = "dfd"
IMAGE_DIR_NAME = "images"


def delete_test_derivitives():
    src_tests_root_dir = os.path.join(PROJECT_ROOT_PATH, "azureml-designer-model/azureml/designer/model/tests")
    dst_tests_root_dir = os.path.join(PROJECT_ROOT_PATH, "azureml-designer-score/azureml/designer/score/score/tests")
    template_file_path = os.path.join(dirname(abspath(__file__)), "entry_template.py")

    for root, dirs, files in os.walk(src_tests_root_dir):
        if MODEL_FOLDER_NAME in dirs:
            shutil.rmtree(os.path.join(root, MODEL_FOLDER_NAME))
            if DFD_DIR_NAME in dirs:
                shutil.rmtree(os.path.join(root, DFD_DIR_NAME))
            print(f"Deleted test derivitives in {root}")


if __name__ == "__main__":
    delete_test_derivitives()
