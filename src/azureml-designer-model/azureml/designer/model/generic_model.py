import os
import sys

from abc import abstractmethod
import numpy as np
import pandas as pd

from . import constants
from .constants import ScoreColumnConstants
from .utils import ioutils, model_spec_utils, yamlutils
from .logger import get_logger
from .model_factory import ModelFactory
from .builtin_models.builtin_model import BuiltinModel
from .model_spec.local_dependency import LocalDependencyManager
from .model_spec.model_input import ModelInput
from .model_spec.task_type import TaskType
from .model_spec.label_map import LabelMap
from .model_spec.remote_dependency import RemoteDependencyManager
from .model_spec.serving_config import ServingConfig

logger = get_logger(__name__)


class GenericModel(object):

    core_model = None
    conda = None
    local_dependencies = None
    inputs = None
    outputs = None
    task_type = None
    label_map = None
    serving_config = None

    def __init__(self, core_model, conda=None, local_dependencies=None, inputs=None, outputs=None, task_type=None,
                 label_map=None, serving_config=None):
        self.core_model = core_model
        if not self.core_model.flavor:
            if not isinstance(core_model, BuiltinModel):
                self.core_model.flavor = {
                    "name": constants.CUSTOM_MODEL_FLAVOR_NAME,
                    "module": self.core_model.__class__.__module__,
                    "class": self.core_model.__class__.__name__
                }
            else:
                raise Exception("BuiltinModel Can't be initialized without flavor")

        self.conda = conda
        if not self.conda and isinstance(self.core_model, BuiltinModel):
            self.conda = self.core_model.default_conda
        self.local_dependencies = local_dependencies
        self.inputs = inputs
        self.outputs = outputs
        self.task_type = task_type
        self.label_map = label_map
        self.serving_config = serving_config

        if isinstance(core_model, BuiltinModel):
            # Init feature_columns
            if self.inputs:
                self._feature_columns_names = [model_input.name for model_input in self.inputs]
                logger.info(f"Loaded feature_columns_names from inputs: {self._feature_columns_names}")
            else:
                self._feature_columns_names = self.core_model.get_default_feature_columns()
                logger.info(f"Loaed feature_columns_names from default: {self._feature_columns_names}")
            if not self._feature_columns_names:
                raise Exception("Can't initialize model without feature_columns_names")

            # Init task_type
            if task_type:
                self.core_model.task_type = task_type

    def save(
        self,
        artifact_path: str = "./AzureMLModel",
        model_relative_to_artifact_path : str = "model",
        overwrite_if_exists: bool = True
    ):
        os.makedirs(artifact_path, exist_ok=overwrite_if_exists)
        model_path = os.path.join(artifact_path, model_relative_to_artifact_path)
        self.core_model.save(model_path, overwrite_if_exists=overwrite_if_exists)

        if not self.conda:
            self.conda = self.core_model.conda

        conda_file_path = None
        # TODO: Provide the option to save result of "conda env export"
        if self.conda:
            ioutils.save_conda_env(artifact_path, self.conda)
            conda_file_path = constants.CONDA_FILE_NAME
        else:
            # TODO: dump local conda env
            pass

        # In the cases where customer manually modified sys.path (e.g. sys.path.append("..")),
        # they would have to specify the code path manually.
        if not self.local_dependencies:
            self.local_dependencies = [os.path.abspath(sys.path[0])]
            logger.info(f"using sys.path[0] = {sys.path[0]} as local_dependency_path")
        local_dependency_manager = LocalDependencyManager(self.local_dependencies)
        local_dependency_manager.save(artifact_path)

        label_map_file_name = None
        if self.label_map:
            label_map_file_name = constants.LABEL_MAP_FILE_NAME
            label_map_path = os.path.join(artifact_path, label_map_file_name)
            self.label_map.save(label_map_path)

        model_spec = model_spec_utils.generate_model_spec(
            flavor=self.core_model.flavor,
            model_path=model_relative_to_artifact_path,
            conda_file_path=conda_file_path,
            local_dependencies=local_dependency_manager.copied_local_dependencies,
            inputs=self.inputs,
            outputs=self.outputs,
            task_type=self.task_type,
            label_map_path=label_map_file_name,
            serving_config=self.serving_config
        )
        model_spec_utils.save_model_spec(artifact_path, model_spec)

    @classmethod
    def load(cls, artifact_path, install_dependencies=False):
        model_spec_path = os.path.join(artifact_path, constants.MODEL_SPEC_FILE_NAME)
        logger.info(f"MODEL_FOLDER: {os.listdir(artifact_path)}")
        model_spec = yamlutils.load_yaml_file(model_spec_path)
        logger.info(f"Successfully loaded {model_spec_path}")
        
        flavor = model_spec["flavor"]
        conda = None
        inputs = None
        outputs = None
        task_type = None
        label_map = None
        serving_config = None
        
        # TODO: Use auxiliary method to handle None in loaded yaml file following Module Team
        if model_spec.get("conda_file", None):
            conda_yaml_path = os.path.join(artifact_path, model_spec["conda_file"])
            conda = yamlutils.load_yaml_file(conda_yaml_path)
            logger.info(f"Successfully loaded {conda_yaml_path}")

        local_dependencies = model_spec.get("local_dependencies", None)
        logger.info(f"local_dependencies = {local_dependencies}")
        if model_spec.get("inputs", None):
            inputs = [ModelInput.from_dict(model_input) for model_input in model_spec["inputs"]]
        if model_spec.get("outputs", None):
            outputs = [ModelInput.from_dict(model_output) for model_output in model_spec["inputs"]]
        if model_spec.get("task_type", None):
            logger.info(f"model_spec['task_type'] = {model_spec['task_type']}")
            task_type = TaskType[model_spec["task_type"]]
            logger.info(f"task_tye = {task_type}")
        if model_spec.get("label_map", None):
            load_from = os.path.join(artifact_path, model_spec["label_map"])
            label_map = LabelMap.create_from_csv(load_from)
        if model_spec.get("serving_config", None):
            serving_config = ServingConfig.from_dict(model_spec["serving_config"])

        if install_dependencies:
            logger.info("Installing dependencies")
            if conda:
                remote_dependency_manager = RemoteDependencyManager()
                remote_dependency_manager.load(conda_yaml_path)
                remote_dependency_manager.install()

            if local_dependencies:
                local_dependency_manager = LocalDependencyManager()
                local_dependency_manager.load(artifact_path, local_dependencies)
                local_dependency_manager.install()

        core_model_class = ModelFactory.get_model_class(flavor)
        core_model_path = os.path.join(artifact_path, model_spec["model_path"])
        if issubclass(core_model_class, BuiltinModel):
            core_model = core_model_class.load_with_flavor(core_model_path, model_spec.get("flavor", {}))
        else:
            core_model = core_model_class.load(core_model_path)

        return cls(
            core_model=core_model,
            conda=conda,
            local_dependencies=local_dependencies,
            inputs=inputs,
            outputs=outputs,
            task_type=task_type,
            label_map=label_map,
            serving_config=serving_config
        )
        
    @abstractmethod
    def predict(self, *args, **kwargs):
        """
        Pass args to core_model, form result Dataframe with scored label
        :param args:
        :param kwargs:
        :return:
        """
        # TODO: Some input validation and normalization here
        logger.info(f"args = {args}, kwargs = {kwargs}")
        if isinstance(self.core_model, BuiltinModel):
            # For BuiltinModel, we assume there's only one positional input parameter in args[0], which a DFD or ImageDirectory
            # The assumption is because Score Module support only one data input port
            if isinstance(args[0], pd.DataFrame):
                predict_ret_list = self.core_model.predict(args[0][self._feature_columns_names].values)
                # TODO: formulate output_df according to task_type
                output_df = pd.DataFrame(predict_ret_list)
                output_df.columns = [f"Score_{i}" for i in range(0, output_df.shape[1])]
                logger.info(f"output_df =\n{output_df}")
                return output_df
            # Else assume args[0] is a generator of ImageDirectory.iter_images()
            # Didn't use ImageDirectory object to prevent depending on azureml.designer.core
            # This logic should be refactored after we resolve cyclic dependency
            else:
                image_id_list = []
                ground_truth_label_list = []
                predict_ret_list = []
                # TODO: Implement batch inference
                for image, label, image_id in args[0]:
                    image_id_list.append(image_id)
                    ground_truth_label_list.append(label)
                    image_ndarray = np.array(image)
                    image_ndarray = np.true_divide(image_ndarray, 255.0)
                    image_ndarray = np.moveaxis(image_ndarray, -1, 0)
                    image_ndarray = np.expand_dims(image_ndarray, axis=0)
                    logger.info(f"image_ndarray.shape = {image_ndarray.shape}")
                    output = self.core_model.predict([[image_ndarray, ]])
                    predict_ret_list += output
                # Temp Workaround for Densenet Demo
                _IDENTIFIER_NAME = 'identifier'
                _LABEL_NAME = 'label'
                data_df = pd.DataFrame({_IDENTIFIER_NAME: image_id_list, _LABEL_NAME: ground_truth_label_list})
                if self.task_type == TaskType.MultiClassification:
                    logger.info(f"MultiClass Classification Task, Result Contains Scored Label and Scored Probability")
                    # From base_learner.py
                    def _gen_scored_probability_column_name(label):
                        """Generate scored probability column names with pattern "Scored Probabilities_label" """
                        return '_'.join((ScoreColumnConstants.ScoredProbabilitiesMulticlassColumnNamePattern, str(label)))

                    label_ids = [row[0] for row in predict_ret_list]
                    probs = [row[1] for row in predict_ret_list]
                    if probs:
                        class_cnt = len(probs[0])
                    else:
                        return pd.DataFrame()
                    index_to_label = self.label_map.index_to_label_dict

                    result_df = pd.DataFrame(data=probs,
                                     columns=[_gen_scored_probability_column_name(index_to_label.get(i, i)) for i in range(0, class_cnt)])
                    result_df[ScoreColumnConstants.ScoredLabelsColumnName] = [index_to_label.get(i, i) for i in label_ids]
                    result_df = pd.concat([data_df, result_df], axis=1)
                    return result_df
                else:
                    return pd.DataFrame(predict_ret_list)
        else:
            return self.core_model.predict(*args, **kwargs)

    @property
    def raw_model(self):
        if isinstance(self.core_model, BuiltinModel):
            return self.core_model.raw_model
        else:
            return None
