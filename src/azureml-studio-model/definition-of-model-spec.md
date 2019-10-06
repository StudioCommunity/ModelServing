# Definition of model spec YAML

==========================================

Train anywhere, finetune and serve here.

## Module Definition

| Name | Type | Required | Description |
| ---- | ---- | -------- | ----------- |
| vintage | string | Yes | A specification of model type. Could be platform name (Tensorflow, etc.), library name (Pytorch, Sklearn, etc.) or format name (Onnx, etc.) |
| conda_file_path | string | Yes    | The name of module. Can not contain characters other than alphabets/digits/space. First character must be an alphabet.|
| local_dependency_path | string | No | The path contains the python packages required to load the model, will be append to sys.path when loading |
| vintage_details | [Vintage Detail](#vintage-detail) | Yes | Detail information of the model. Reference: 'vintage_details' table.|
| inputs | list<[Input](#Input)> | No | Defines input parameters of the model. Reference: 'Input Definition' table. |
| outputs | list<[Output](#Output)> | No |Defines output parameters of the model. Reference: 'Output Definition' table.|
| serving_resource_requirement | [Serving Resource Requirement](#serving-resource-requirement) | No | Resource required to serve the model, reference 'Serving Resource Requirement' table. |
| description | string | No |The detailed information that describes this module.|
| alghost_version | string | No | Version of alghost which containse model sdk |
| time_created | datetime | No | Create time of the model folder |

## Vintage Detail

Vintage Detail describes the detail information needed to load the model

### Pytorch

| Name        | Type    | Required | Description                                                  |
| ----------- | ------- | -------- | ------------------------------------------------------------ |
| model_file_path | string  | Yes      | Path of the serialized model file |
| pytorch_version | string  | Yes      | Version of pytorch |
| serialization_format | string | Yes | The format used to dump the model |
| serialization_library_version | string  | No       | The version of the serialization library. |

## Input

Input defines the input parameter of the model.

| Name        | Type    | Required | Description                                                  |
| ----------- | ------- | -------- | ------------------------------------------------------------ |
| name        | string  | Yes      | The name of this input parameter.                                      |
| type        | string  | Yes      | Defines the type of this data. Refer to [Data Types](#data-types) for details. |
| default     | dynamic | No       | The default value of this parameter. The type of this value is the same with 'type' field. This field is optional, will default to `null` if not specified. If 'type' is ndarray, the default value would be represented as equivalent nested list. |
| description | string  | No       | The detailed information that describes the input. |
| optional    | boolean | No       | Indicates where this input is optional. Default value is `false`. |

## Output

| Name        | Type   | Required | Description                                                  |
| ----------- | ------ | -------- | ------------------------------------------------------------ |
| name        | string | Yes      | The name of this Output.                                     |
| type        | string | Yes      | Defines the type of this data. Reference: 'Data Types' table. |
| description | string | No       | The detailed information that describes the Output.          |

## Data Types

Data Type is a string describes the data type of the Input/Output parameter.

| Name           | Description                                                  |
| -------------- | ------------------------------------------------------------ |
| ndarray        | Indicates that the input value is a numpy.ndarray            |
| string         | Indicates that the input value is a string.                  |
| int            | Indicates that the input value is an integer.                |
| float          | Indicates that the input value is a float.                   |
| boolean        | Indicates that the input value is a boolean value.           |

## Serving Resource Requirement

| Name      | Type                    | Required | Description                                                  |
| --------- | ----------------------- | -------- | ------------------------------------------------------------ |
| gpu_support      | boolean | No       | Set to `true` if requires GPU to run the module.             |
| cpu_core_num      | int     | No       | minimum number of cpu cores |
| memory_in_MB      | int     | No       | minimum amount of memory in MB |

## Example

model_spec.yaml:

~~~yaml
vintage: pytorch
conda_file_path: conda.yaml
local_dependency_path: local_dependency
inputs:
- name: x
  type: ndarray
  default: [10]
  description: regression feature
  optional: false
outputs:
- name: y
  type: float
  description: regression result
vintage_detail:
  model_file_path: model.pkl
  pytorch_version: 1.1.0
  torchvision_version: 0.4.0
  serialization_format: cloudpickle
  serialization_library_version: 1.1.2
serving_resource_requirement:
  gpu_support: true
  cpu_core_num: 2
  memory_in_MB: 1024
alghost_version: 0.0.83
time_created: '2019-10-01 00:00:00.000000'
~~~

folder structure:

```folder
AzureMLModel
├──model_spec.yaml
├──conda.yaml
├──model.pkl
└──local_dependency
   └──training_module
      ├──__init__.py
      ├──model.py
      └──entry.py
```
