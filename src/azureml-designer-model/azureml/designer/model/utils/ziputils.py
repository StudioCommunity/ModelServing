import os
import zipfile

from azureml.designer.model.utils.ioutils import validate_overwrite
from azureml.designer.model.logger import get_logger

logger = get_logger(__name__)


def zip_dir(src_dir_path, zip_file_path, overwrite_if_exists=True):
    validate_overwrite(zip_file_path, overwrite_if_exists)
    with zipfile.ZipFile(zip_file_path, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for root, _, file_names in os.walk(src_dir_path):
            for file_name in file_names:
                file_path = os.path.join(root, file_name)
                rel_path = os.path.relpath(file_path, src_dir_path)
                zip_file.write(file_path, rel_path)


def unzip_dir(zip_file_path, dst_dir_path, overwrite_if_exists=True):
    validate_overwrite(dst_dir_path, overwrite_if_exists)
    with zipfile.ZipFile(zip_file_path, 'r') as zip_file:
        zip_file.extractall(dst_dir_path)
