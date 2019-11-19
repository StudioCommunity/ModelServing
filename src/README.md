# README

## Introduction

This repository contains some built-in custom modules for azureml designer.

## Content

- [Introduction](#introduction)
- [Content](#content)
- [Repository Structure](#repository-structure)

## Repository Structure

Python packages are folders that contain python code or other packages and have an \_\_init\_\_.py.

Wheel files are how we distribute python packages. They can be managed through pip and conda. Wheels can contain multiple packages, and specify dependencies on other wheel files.

Python namespace packages are a special construct that allow subpackages to be sourced from multiple wheel files, and combined together seamlessly in the python environment they're installed into.

For example, consider the hypothetical azureml-A and azureml-B wheels. Each defines a single package: azureml.A and azureml.B, respectively. The "azureml" namespace package is what allows those packages to be installed together under the common "azureml" namespace without overwriting each other, even though they both define different views of the "azureml" package.

```
root/
    README.md                       [you are here]
    scripts/                        [scripts that are common across wheels, like release management]
    tests/                          [tests that span multiple wheels]
    notebooks/                      [notebooks]
    src/
        wheel_name/                 [each wheel gets its own source folder]
            README.md               [wheel readme's should include dev setup instructions]
            setup.py                [setup.py defines the content and metadata of the wheel]
            azureml/                [most of our code should fall into the azureml namespace package]
                __init__.py         [__init__.py for a namespace package isn't empty, it defines the namespace package]
                subpackage_name/    [various subpackages that contain the actual code]
            other_package/          [some wheels will have code outside the azureml namespace package]
            scripts/                [scripts specific to this wheel]
            tests/                  [tests specific to this wheel]
        ...
    dataprep/
        builder/                    [gulp task generators/helpers for building Data Prep]
        Core/                       [DataPrep Engine/Lariat]
```

## Testing

python -m pytest
