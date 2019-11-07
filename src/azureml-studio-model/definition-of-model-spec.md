# Definition of model spec YAML

==========================================

Train anywhere, serve here.

## Model Definition

| Name | Type | Required | Description |
| ---- | ---- | -------- | ----------- |
| flavor | [Flavor](#flavor) | Yes | Description of model source information. Reference 'Flavor' table. |
| model_file | string | Yes    | The path to model(file or directory) |
| conda_file | string | Yes    | The path to conda.yaml |
| local_dependencies | list | No | The path contains the python packages required to load the model, will be append to sys.path when loading |
| inputs | list<[Input](#Input)> | No | Defines input parameters of the model. Reference: 'Input Definition' table. |
| outputs | list<[Output](#Output)> | No |Defines output parameters of the model. Reference: 'Output Definition' table.|
| serving_config | [Serving Config](#serving-config) | No | Configurations to serve the model, reference 'Serving Config' table. |
| description | string | No |The detailed information that describes this module.|
| model_spec_version | string | No | Version of model spec, current coincide with azureml-designer-model version |
| time_created | datetime | No | Create time of the model folder |

## Flavor

Describe the information so that we can load the model.

### Custom

| Name        | Type    | Required | Description                                                  |
| ----------- | ------- | -------- | ------------------------------------------------------------ |
| class | string  | Yes      | class name of the custom model class which inherits GenericModel |
| module | string  | Yes      | python module path to the module in which contains the class definition |

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

## Serving Config

| Name      | Type                    | Required | Description                                                  |
| --------- | ----------------------- | -------- | ------------------------------------------------------------ |
| gpu_support      | boolean | No       | Set to `true` if requires GPU to run the module.             |
| cpu_core_num      | float | No       | The number of cpu cores to allocate for serving the model, Can be decimal. |
| memory_in_GB      | float     | No       | The amound memory (in GB) to allocate for serving the model. Can be decimal. |

## Example

model_spec.yaml:

~~~yaml
flavor:
  module: azureml.studio.model.pytorch.cloudpickle
  class: PytorchCloudPickle
conda_file: conda.yaml
local_dependencies:
- local_dependencies/train_by_module
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
serving_config:
  gpu_support: true
  cpu_core_num: 0.1
  memory_in_GB: 0.5
model_spec_version: 0.0.83
time_created: '2019-10-01 00:00:00.000000'
~~~

folder structure:

```folder
AzureMLModel
├──model_spec.yaml
├──conda.yaml
├──model.pkl
└──local_dependencies
   └──train_by_module
      └──training_module
          ├──__init__.py
          ├──model.py
          └──entry.py
```

Custom

~~~yaml
flavor:
  module: module.path.to.cutom_model_definition
  class: CustomModelClassWhichInherentsGenericModel
model_file: model
conda_file: conda.yaml
local_dependencies:
- local_dependencies/train_by_module
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
serving_config:
  gpu_support: true
  cpu_core_num: 0.1
  memory_in_GB: 0.5
model_spec_version: 0.0.83
time_created: '2019-10-01 00:00:00.000000'
~~~

folder structure:

```folder
AzureMLModel
├──model_spec.yaml
├──conda.yaml
├──model
|  └──data.ilearner
└──local_dependencies
   └──local_module_1
      └──local_module_name
          ├──__init__.py
          ├──model.py
          └──entry.py
```

