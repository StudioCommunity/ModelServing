# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

from setuptools import setup, find_packages
import io
import os
import sys
import shutil
import re

BUILD_FILE = '../../BUILD_NUMBER'
VERSION_FILE = '_version.py'


def generate_version_number():
    version = {}
    version_key = '__version__'
    with open(VERSION_FILE) as fp:
        exec(fp.read(), version)
    assert version_key in version, 'Version file is empty.'
    version_pattern = r'^\d+.\d$'
    assert re.match(version_pattern, version[version_key]), f'Invalid version number: {version[version_key]}'
    with open(BUILD_FILE) as fp:
        build_number = fp.read()
    assert re.match(version_pattern, build_number), f'Invalid build number: {build_number}'
    return '.'.join([version[version_key], build_number])


with io.open('../.inlinelicense', 'r', encoding='utf-8') as f:
    inline_license = f.read()

exclude_list = ["*.tests", "azureml/studio/tests", "tests", "examples*"]
packages = find_packages(exclude=exclude_list)

print("installing... ", packages)

# python setup.py install
setup(
    name="azureml-studio-score",
    version=generate_version_number(),
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