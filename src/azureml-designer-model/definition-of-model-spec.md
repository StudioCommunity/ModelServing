# Definition of model spec YAML

==========================================

Train anywhere, serve here.

## Model Definition

| Name               | Type                              | Required | Description                                                                                               |
|--------------------|-----------------------------------|----------|-----------------------------------------------------------------------------------------------------------|
| flavor             | [Flavor](#flavor)                 | Yes      | Description of model source information. Reference 'Flavor' table.                                        |
| model_file         | string                            | Yes      | The path to model(file or directory)                                                                      |
| conda_file         | string                            | Yes      | The path to conda.yaml                                                                                    |
| local_dependencies | list                              | No       | The path contains the python packages required to load the model, will be append to sys.path when loading |
| inputs             | list<[Input](#input)>             | No       | Defines input parameters of the model. Reference: 'Input Definition' table.                               |
| outputs            | list<[Output](#output)>           | No       | Defines output parameters of the model. Reference: 'Output Definition' table.                             |
| task_type          | string                            | No       | Enumerate value of score task type, listed in [Task Types](#task-types)                                   |
| label_map_file     | string                            | No       | A csv file contains lines of "index, label" without header                                                |
| serving_config     | [Serving Config](#serving-config) | No       | Configurations to serve the model, reference 'Serving Config' table.                                      |
| description        | string                            | No       | The detailed information that describes this model.                                                       |
| time_created       | datetime                          | No       | Create time of the model folder                                                                           |

## Flavor

Describe the information so that we can load the model.

### Custom

| Name   | Type   | Required | Description                                                             |
|--------|--------|----------|-------------------------------------------------------------------------|
| name   | string | Yes      | Custom                                                                  |
| class  | string | Yes      | class name of the custom model class which inherits GenericModel        |
| module | string | Yes      | python module path to the module in which contains the class definition |

### Pytorch

| Name                 | Type   | Required | Description                                                                                                                                 |
|----------------------|--------|----------|---------------------------------------------------------------------------------------------------------------------------------------------|
| name                 | string | Yes      | A specification of model type. Could be platform name (Tensorflow, etc.), library name (Pytorch, Sklearn, etc.) or format name (Onnx, etc.) |
| serialization_method | string | Yes      | The format used to dump the model                                                                                                           |
| is_cuda              | bool   | No       | Whether or not the model need to reside on cuda                                                                                             |

#### Pytorch cloudpickle

| Name                 | Type   | Required | Description                                     |
|----------------------|--------|----------|-------------------------------------------------|
| name                 | string | Yes      | pytorch                                         |
| serialization_method | string | Yes      | cloudpickle                                     |
| is_cuda              | bool   | No       | Whether or not the model need to reside on cuda |

#### Pytorch state_dict

| Name                 | Type   | Required | Description                                                                                     |
|----------------------|--------|----------|-------------------------------------------------------------------------------------------------|
| name                 | string | Yes      | pytorch                                                                                         |
| serialization_method | string | Yes      | state_dict                                                                                      |
| is_cuda              | bool   | No       | Whether or not the model need to reside on cuda                                                 |
| class                | string | Yes      | The model class which inherits nn.Module                                                        |
| module               | string | No       | The module in which the model_class resides, if not provided, will try to load from main module |
| init_params          | dict   | No       | The keyword arguments passed to nn.Module.__init__ function                                     |

## Input

Input defines the input parameter of the model.

| Name        | Type    | Required | Description                                                                                                                                                                                                                                         |
|-------------|---------|----------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| name        | string  | Yes      | The name of this input parameter.                                                                                                                                                                                                                   |
| type        | string  | Yes      | Defines the type of this data. Refer to [Data Types](#data-types) for details.                                                                                                                                                                      |
| default     | dynamic | No       | The default value of this parameter. The type of this value is the same with 'type' field. This field is optional, will default to `null` if not specified. If 'type' is ndarray, the default value would be represented as equivalent nested list. |
| description | string  | No       | The detailed information that describes the input.                                                                                                                                                                                                  |
| optional    | boolean | No       | Indicates where this input is optional. Default value is `false`.                                                                                                                                                                                   |

## Output

| Name        | Type   | Required | Description                                                   |
|-------------|--------|----------|---------------------------------------------------------------|
| name        | string | Yes      | The name of this Output.                                      |
| type        | string | Yes      | Defines the type of this data. Reference: 'Data Types' table. |
| description | string | No       | The detailed information that describes the Output.           |

## Task Types

| Name                 | Score Result Column Name                           |
|----------------------|----------------------------------------------------|
| Regression           | Regression Assigned Labels                         |
| BinaryClassification | Binary Class Assigned Labels, Scored Probabilities |
| MultiClassification  | Scored Labels, Scored Probabilities                |
| Clustering           | Cluster Assigned Labels                            |
| ImageGeneration      | Generated Image                                    |
| TextGeneration       | Generated Text                                     |

## Data Types

Data Type is a string describes the data type of the Input/Output parameter.

| Name    | Description                                        |
|---------|----------------------------------------------------|
| ndarray | Indicates that the input value is a numpy.ndarray  |
| string  | Indicates that the input value is a string.        |
| int     | Indicates that the input value is an integer.      |
| float   | Indicates that the input value is a float.         |
| boolean | Indicates that the input value is a boolean value. |

## Serving Config

| Name         | Type    | Required | Description                                                                  |
|--------------|---------|----------|------------------------------------------------------------------------------|
| gpu_support  | boolean | No       | Set to `true` if requires GPU to run the module.                             |
| cpu_core_num | float   | No       | The number of cpu cores to allocate for serving the model, Can be decimal.   |
| memory_in_GB | float   | No       | The amound memory (in GB) to allocate for serving the model. Can be decimal. |

## Example

model_spec.yaml:

~~~yaml
flavor:
  name: pytorch
  serialization_method: cloudpickle
model_file: model.pkl
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
