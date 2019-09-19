# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

from setuptools import setup, find_packages
import io
import os
import sys
import shutil
import re

BUILD_NUMBER_FILE = '../build.version'
VERSION_FILE = '../major.version'


def get_package_version():
    with open(VERSION_FILE) as version_file:
        version = version_file.read().strip()
    version_pattern = r'^\d+.\d+$'
    assert re.match(version_pattern, version), f'Invalid version number: {version}'
    with open(BUILD_NUMBER_FILE) as build_number_file:
        build_number = build_number_file.read()
    assert re.match(version_pattern, build_number), f'Invalid build number: {build_number}'
    return '.'.join([version, build_number])


with io.open('../.inlinelicense', 'r', encoding='utf-8') as f:
    inline_license = f.read()

exclude_list = ["*.tests", "azureml/studio/tests", "tests", "examples*"]
packages = find_packages(exclude=exclude_list)

print("installing... ", packages)

# python setup.py install
setup(
    name="azureml-studio-score",
    version=get_package_version(),
    description="",
    packages=packages,
    install_requires=[
          "cloudpickle",
          "PyYAML",
          "pandas",
          "pyarrow"
      ],
    author='Microsoft Corp',
    license=inline_license,
    include_package_data=True,
)