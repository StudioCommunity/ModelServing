# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

from setuptools import setup, find_packages
import io
import os
import sys
import shutil

with io.open('../.inlinelicense', 'r', encoding='utf-8') as f:
    inline_license = f.read()

exclude_list = ["*.tests", "azureml/studio/tests", "tests", "examples*"]
packages = find_packages(exclude=exclude_list)

print("installing... ", packages)

# python setup.py install
setup(
    name="azureml-studio-score",
    version="0.0.1",
    description="",
    packages=packages,
    install_requires=[
          "cloudpickle",
          "PyYAML"
      ],
    author='Microsoft Corp',
    license=inline_license,
    include_package_data=True,
)