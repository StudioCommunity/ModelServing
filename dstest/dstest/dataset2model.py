import os

import argparse
from zipfile import ZipFile


parser = argparse.ArgumentParser()

parser.add_argument("--zipped_model_path",
                    default=None,
                    type=str,
                    required=True,
                    help="The input zipped model path.")
# Required parameters
parser.add_argument("--unzipped_model_path",
                    default=None,
                    type=str,
                    required=True,
                    help="The output unzipped model path.")
args, _ = parser.parse_known_args()

def main():
    src = args.zipped_model_path
    dst = args.unzipped_model_path
    src_files = os.listdir(src)
    print(f"src_files = {src_files}")
    for file_name in src_files:
        if file_name.endswith(".zip"):
            full_file_name = os.path.join(src, file_name)
            zf = ZipFile(full_file_name)
            zf.extractall(dst)



#python -m dstest.dataset2model --zipped_model_path ./input --unzipped_model_path ./output
main()
