import os
import sys

from abc import abstractmethod
import numpy as np
import pandas as pd

from .constants import ScoreColumnConstants, ModelSpecConstants
from .utils import ioutils, model_spec_utils, yamlutils
from .logger import get_logger
from .model_factory import ModelFactory
from .builtin_models.builtin_model import BuiltinModel
from .model_spec.local_dependency import LocalDependencyManager
from .model_spec.model_input import ModelInput
from .model_spec.model_output import ModelOutput
from .model_spec.task_type import TaskType
from .model_spec.label_map import LabelMap
from .model_spec.remote_dependency import RemoteDependencyManager
from .model_spec.serving_config import ServingConfig

logger = get_logger(__name__)


class GenericModel(object):
    """
    Generic Model does the flavor-independent things, in general, save/load/predict:
    1. Save/Load model with model_spec.yaml
    2. Handle dependencies.
    3. Select Feature columns.
    4. Partition input data into mini-batches if needed.
    5. Formulate output DataFrame of which the schema can be understood by Evaluate Module
    """

    core_model = None
    conda = None
    local_dependencies = None
    inputs = None
    outputs = None
    task_type = None
    label_map = None
    serving_config = None
    # Can be set in Score Module, or just use default value
    _batch_size = 2
    # Can be set in training phase
    _label_column_name = "ground_truth_label"

    def __init__(self, core_model, conda=None, local_dependencies=None, inputs=None, outputs=None, task_type=None,
                 label_map=None, serving_config=None):
        self.core_model = core_model
        if not self.core_model.flavor:
            if not isinstance(core_model, BuiltinModel):
                self.core_model.flavor = {
                    ModelSpecConstants.FLAVOR_NAME_KEY: ModelSpecConstants.CUSTOM_MODEL_FLAVOR_NAME,
                    ModelSpecConstants.MODEL_MODULE_KEY: self.core_model.__class__.__module__,
                    ModelSpecConstants.MODEL_CLASS_KEY: self.core_model.__class__.__name__
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
                logger.info(f"Loaded feature_columns_names from default: {self._feature_columns_names}")
            if not self._feature_columns_names:
                raise Exception("Can't initialize model without feature_columns_names")

            # Init task_type
            if task_type:
                self.core_model.task_type = task_type

    def save(
        self,
        artifact_path: str = ModelSpecConstants.DEFAULT_ARTIFACT_SAVE_PATH,
        model_relative_to_artifact_path: str = ModelSpecConstants.CUSTOM_MODEL_DIRECTORY,
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
            conda_file_path = ModelSpecConstants.CONDA_FILE_NAME
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
            label_map_file_name = ModelSpecConstants.LABEL_MAP_FILE_NAME
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
        model_spec_path = os.path.join(artifact_path, ModelSpecConstants.MODEL_SPEC_FILE_NAME)
        logger.info(f"MODEL_FOLDER: {os.listdir(artifact_path)}")
        model_spec = yamlutils.load_yaml_file(model_spec_path)
        logger.info(f"Successfully loaded {model_spec_path}")
        
        flavor = model_spec[ModelSpecConstants.FLAVOR_KEY]
        conda = None
        inputs = None
        outputs = None
        task_type = None
        label_map = None
        serving_config = None
        
        # TODO: Use auxiliary method to handle None in loaded yaml file following Module Team
        if model_spec.get(ModelSpecConstants.CONDA_FILE_KEY, None):
            conda_yaml_path = os.path.join(artifact_path, model_spec[ModelSpecConstants.CONDA_FILE_KEY])
            conda = yamlutils.load_yaml_file(conda_yaml_path)
            logger.info(f"Successfully loaded {conda_yaml_path}")

        local_dependencies = model_spec.get(ModelSpecConstants.LOCAL_DEPENDENCIES_KEY, None)
        logger.info(f"local_dependencies = {local_dependencies}")
        if model_spec.get(ModelSpecConstants.INPUTS_KEY, None):
            inputs = [ModelInput.from_dict(model_input) for model_input in model_spec[ModelSpecConstants.INPUTS_KEY]]
        if model_spec.get(ModelSpecConstants.OUTPUTS_KEY, None):
            outputs = [ModelOutput.from_dict(model_output) for model_output in model_spec[ModelSpecConstants.OUTPUTS_KEY]]
        if model_spec.get(ModelSpecConstants.TASK_TYPE_KEY, None):
            task_type = TaskType[model_spec[ModelSpecConstants.TASK_TYPE_KEY]]
            logger.info(f"task_tye = {task_type}")
        if model_spec.get(ModelSpecConstants.LABEL_MAP_FILE_KEY, None):
            load_from = os.path.join(artifact_path, model_spec[ModelSpecConstants.LABEL_MAP_FILE_KEY])
            label_map = LabelMap.create_from_csv(load_from)
        if model_spec.get(ModelSpecConstants.SERVING_CONFIG_KEY, None):
            serving_config = ServingConfig.from_dict(model_spec[ModelSpecConstants.SERVING_CONFIG_KEY])

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
        core_model_path = os.path.join(artifact_path, model_spec[ModelSpecConstants.MODEL_FILE_KEY])
        if issubclass(core_model_class, BuiltinModel):
            core_model = core_model_class.load_with_flavor(core_model_path, model_spec.get(ModelSpecConstants.FLAVOR_KEY, {}))
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
        # For BuiltinModel, we assume there's only one positional input parameter in args[0],
        if isinstance(self.core_model, BuiltinModel):
            input_data = args[0]
            preprocessed_data = self._pre_process(input_data)
            if isinstance(preprocessed_data, np.ndarray):
                predict_result = self.core_model.predict(preprocessed_data)
                postprocessed_data = self._post_process(predict_result)
                return postprocessed_data
            else:
                ground_truth_label_list = []
                image_id_list = []
                predict_result = []
                batch_inputs = []
                for image, label, image_id in preprocessed_data:
                    ground_truth_label_list.append(label)
                    image_id_list.append(image_id)
                    batch_inputs.append((image,))
                    if len(batch_inputs) == self.batch_size:
                        batch_outputs = self.core_model.predict(batch_inputs)
                        predict_result += batch_outputs
                        del batch_inputs[:]
                if batch_inputs:
                    batch_outputs = self.core_model.predict(batch_inputs)
                    predict_result += batch_outputs
                logger.info(f"predict_result = {predict_result}")

                # This is a temp solution for the current implementation of ImageDirectory
                # Because ImageDirectory only provide iterator,
                # which means I can't get label and image_id in Score Module
                postprocessed_data = self._post_process(predict_result)
                origin_data_df = pd.DataFrame({"id": image_id_list, self.label_column_name: ground_truth_label_list})
                final_result = pd.concat([origin_data_df, postprocessed_data], axis=1)
                return final_result
        else:
            return self.core_model.predict(*args, **kwargs)

    def _pre_process(self, input_data):
        """
        We will deal with two types of data (DataFrame and image generator) until DataFrame++ is implemented
        For DataFrame, we select feature columns as transform to ndarray
        For image generator, just pass through, and core_model will deal with it by its own flavor-specified method
        :param input_data: DataFrame or image generator
        :return: ndarray if input_data is DataFrame, image generator otherwise
        """
        if isinstance(input_data, pd.DataFrame):
            return input_data[self._feature_columns_names].values
        else:
            return input_data

    # Form result DataFrame according to task_type
    def _post_process(self, predict_ret_list) -> pd.DataFrame:
        if self.task_type == TaskType.MultiClassification:
            logger.info(f"MultiClass Classification Task, Result Contains Scored Label and Scored Probability")

            # From base_learner.py
            def _gen_scored_probability_column_name(label):
                """Generate scored probability column names with pattern "Scored Probabilities_label" """
                return '_'.join(
                    (ScoreColumnConstants.ScoredProbabilitiesMulticlassColumnNamePattern, str(label)))

            label_ids = [row[0] for row in predict_ret_list]
            probs = [row[1] for row in predict_ret_list]
            if probs:
                class_cnt = len(probs[0])
            else:
                return pd.DataFrame()
            index_to_label = self.label_map.index_to_label_dict
            columns = [_gen_scored_probability_column_name(index_to_label.get(i, i)) for i in range(0, class_cnt)]
            result_df = pd.DataFrame(data=probs, columns=columns)
            result_df[ScoreColumnConstants.ScoredLabelsColumnName] = [index_to_label.get(i, i) for i in label_ids]
            return result_df
        else:
            return pd.DataFrame(predict_ret_list)

    @property
    def raw_model(self):
        if isinstance(self.core_model, BuiltinModel):
            return self.core_model.raw_model
        else:
            return None

    @property
    def batch_size(self):
        return self._batch_size

    @batch_size.setter
    def batch_size(self, value: int):
        self._batch_size = value

    @property
    def label_column_name(self):
        return self._label_column_name

    @label_column_name.setter
    def label_column_name(self, value: str):
        self._label_column_name = value
