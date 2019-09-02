from __future__ import absolute_import, division, print_function
import os
import argparse
import shutil

parser = argparse.ArgumentParser()

parser.add_argument("--input_model_path",
                    default=None,
                    type=str,
                    required=True,
                    help="The input model path.")
# Required parameters
parser.add_argument("--out_model_path",
                    default=None,
                    type=str,
                    required=True,
                    help="The out model path.")
args, _ = parser.parse_known_args()

def main():
    src = args.input_model_path
    dst = args.out_model_path
    if not os.path.exists(dst):
        os.makedirs(dst)
    src_files = os.listdir(src)
    for file_name in src_files:
        full_file_name = os.path.join(src, file_name)
        if os.path.isfile(full_file_name):
            shutil.copy(full_file_name, dst)

    with open(os.path.join(dst, "model_spec.yml"), 'w') as fp:
        fp.write("model_file_path: ./data.csv\nflavor:\n  framework: Pytorch\n  env: conda.yaml")


#python -m dstest.dummy_train  --out_model_path ../output
#python dummy_train.py --out_model_path ../output
main()
