import os
from os.path import dirname, abspath
import shutil

PROJECT_ROOT_PATH = dirname(dirname(dirname(dirname(dirname(dirname(abspath(__file__)))))))
MODEL_FOLDER_NAME = "AzureMLModel"
DFD_DIR_NAME = "dfd"
IMAGE_DIR_NAME = "images"


def test_create_score_test_cases():
    src_tests_root_dir = os.path.join(PROJECT_ROOT_PATH, "azureml-designer-model/azureml/designer/model/tests")
    dst_tests_root_dir = os.path.join(PROJECT_ROOT_PATH, "azureml-designer-score/azureml/designer/score/score/tests")
    template_file_path = os.path.join(dirname(abspath(__file__)), "entry_template.py")

    for root, dirs, files in os.walk(src_tests_root_dir):
        if MODEL_FOLDER_NAME in dirs:
            if not DFD_DIR_NAME in dirs and not IMAGE_DIR_NAME in dirs:
                continue
            model_dir = os.path.join(root, MODEL_FOLDER_NAME)
            relative_to_tests_root = os.path.relpath(root, src_tests_root_dir)
            dst_test_path = os.path.join(dst_tests_root_dir, relative_to_tests_root)
            input_port1_path = os.path.join(dst_test_path, "InputPort1")
            input_port2_path = os.path.join(dst_test_path, "InputPort2")
            shutil.copytree(model_dir, input_port1_path)
            if DFD_DIR_NAME in dirs:
                src_dfd = os.path.join(root, DFD_DIR_NAME)
                dst_dfd = input_port2_path
                shutil.copytree(src_dfd, dst_dfd)
            elif IMAGE_DIR_NAME in dirs:
                src_image_dir = os.path.join(root, IMAGE_DIR_NAME)
                dst_image_dir = input_port2_path
                shutil.copytree(src_image_dir, dst_image_dir)
            entry_file_name = f"test_{'_'.join(os.path.split(relative_to_tests_root))}.py"
            shutil.copy(template_file_path, os.path.join(dst_test_path, entry_file_name))
            print(f"Copied test case in {root}")


if __name__ == "__main__":
    test_create_score_test_cases()
