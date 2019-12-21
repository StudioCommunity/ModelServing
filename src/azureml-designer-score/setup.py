
# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

from setuptools import setup, find_packages
import io
import os
import sys
import shutil
import re
try:  # for pip >= 10
    from pip._internal.req import parse_requirements
except ImportError:  # for pip <= 9.0.3
    from pip.req import parse_requirements

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

exclude_list = ["*.tests", "azureml/designer/tests", "tests", "azureml.designer.score.preprocess", "azureml.designer.score.postprocess", "azureml.designer.score.modelconverter", "examples.*"]
packages = find_packages(exclude=exclude_list)
print(f"packages = {packages}")

print("installing... ", packages)
print("installing... ", inline_license)


def get_requirements() -> list:
    """Designed to install packages in build/release pipeline, but exclude them when installing from PyPI
    Returns:
        list -- list of requirements
    """
    exclude = ('pytest', 'pylint', 'torch', 'torchvision', 'pyarrow', 'fire', 'azureml.core', 'azureml-designer-internal', 'azureml-designer-classic-modules', 'azureml.contrib.services', 'azureml-designer-model', 'azureml-dataprep[pandas,fuse]', 'azureml-designer-core', 'git+https://github.com/chjinche/CustomModules-1.git@master#subdirectory=azureml-custom-module-examples/image-classification')
    install_reqs = parse_requirements('requirements.txt', session='hack')
    return [str(ir.req) for ir in install_reqs if ir.name not in exclude and ir.req is not None]

# python setup.py install
setup(
    name="azureml-designer-score",
    version=get_package_version(),
    description="",
    packages=packages,
    install_requires=get_requirements(),
    dependency_links=["https://pypi.org/simple"],
    author='Microsoft Corp',
    license=inline_license,
    include_package_data=True,
)