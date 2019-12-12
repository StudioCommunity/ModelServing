import os
import types

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
from .model_spec.task import Task
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
    task = None
    serving_config = None
    # TODO: Decide whether or not expose this as a parameter in Score Module
    _batch_size = 2

    def __init__(self, core_model, conda=None, local_dependencies=None, inputs=None, outputs=None, task=None,
                 serving_config=None):
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
        self.task = task
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
            if self.task:
                self.core_model.task_type = self.task.task_type

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

        local_dependency_manager = LocalDependencyManager(self.local_dependencies)
        local_dependency_manager.save(artifact_path)

        task_conf = None
        if self.task:
            task_conf = self.task.save(artifact_path, overwrite_if_exists=overwrite_if_exists)

        model_spec = model_spec_utils.generate_model_spec(
            flavor=self.core_model.flavor,
            model_path=model_relative_to_artifact_path,
            conda_file_path=conda_file_path,
            local_dependencies=local_dependency_manager.copied_local_dependencies,
            inputs=self.inputs,
            outputs=self.outputs,
            task_conf=task_conf,
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
        task = None
        serving_config = None

        if ModelSpecConstants.CONDA_FILE_KEY in model_spec:
            conda_yaml_path = os.path.join(artifact_path, model_spec[ModelSpecConstants.CONDA_FILE_KEY])
            conda = yamlutils.load_yaml_file(conda_yaml_path)
            logger.info(f"Successfully loaded {conda_yaml_path}")

        local_dependencies = model_spec.get(ModelSpecConstants.LOCAL_DEPENDENCIES_KEY, None)
        logger.info(f"local_dependencies = {local_dependencies}")
        if ModelSpecConstants.INPUTS_KEY in model_spec:
            inputs = [ModelInput.from_dict(model_input) for model_input in
                      model_spec[ModelSpecConstants.INPUTS_KEY]]
        if ModelSpecConstants.OUTPUTS_KEY in model_spec:
            outputs = [ModelOutput.from_dict(model_output) for model_output in
                       model_spec[ModelSpecConstants.OUTPUTS_KEY]]
        if ModelSpecConstants.TASK_KEY in model_spec:
            task = Task.load(artifact_path, model_spec[ModelSpecConstants.TASK_KEY])
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
        logger.info(f"core_model_class = {core_model_class}")
        core_model_path = os.path.join(artifact_path, model_spec[ModelSpecConstants.MODEL_FILE_KEY])
        logger.info(f"Trying to load core_model from {core_model_path}.")
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
            task=task,
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
            if not isinstance(input_data, pd.DataFrame):
                self._feature_columns_names = ["image"]

            non_feature_df = pd.DataFrame()
            predict_result = []
            for batch_df in self._batching(input_data):
                batch_non_feature_df = batch_df.loc[:, batch_df.columns.difference(self._feature_columns_names)]
                non_feature_df = pd.concat([non_feature_df, batch_non_feature_df], ignore_index=True)
                preprocessed_data = self._pre_process(batch_df)
                predict_result += self.core_model.predict(preprocessed_data)
            postprocessed_data = self._post_process(predict_result)
            if not isinstance(input_data, pd.DataFrame):
                postprocessed_data = pd.concat([non_feature_df, postprocessed_data], axis=1)
            return postprocessed_data
        else:
            return self.core_model.predict(*args, **kwargs)

    def _batching(self, input_data) -> types.GeneratorType:
        """
        Batch input_data into DataFrames. If input_data is DataFrame, return a 1-time generator which returns it;
        If input_data is ImageDirectory's iterator, return a generator generates a DataFrame of columns:
        ["image_id", "label", "image"] with batch_size rows.
        :param input_data: DataFrame or generator which yields (image, label, image_id)
        :return: generator of DataFrame
        """
        if isinstance(input_data, pd.DataFrame):
            yield input_data
        else:
            batch_df = pd.DataFrame()
            for image, label, image_id in input_data:
                if batch_df.shape[0] == self.batch_size:
                    batch_df = pd.DataFrame()
                batch_df = batch_df.append(
                    {
                        "id": image_id,
                        self.label_column_name: label,
                        "image": image
                    }, ignore_index=True
                )
                if batch_df.shape[0] == self.batch_size:
                    yield batch_df
            yield batch_df

    def _pre_process(self, batch_df) -> np.ndarray:
        """
        Select feature columns of batch_df and transform to ndarray
        :param batch_df: DataFrame
        :return: ndarray if input_data is DataFrame, image generator otherwise
        """
        if not isinstance(batch_df, pd.DataFrame):
            raise Exception(f"_pre_process excepts DataFrame input, got {type(batch_df)}")
        return batch_df[self._feature_columns_names].values

    # Form result DataFrame according to task_type
    def _post_process(self, predict_ret_list) -> pd.DataFrame:
        if self.task_type == TaskType.MultiClassification:
            logger.info(f"MultiClass Classification Task, Result Contains Scored Label and Scored Probability")

            # From base_learner.py
            def _gen_scored_probability_column_name(label):
                """Generate scored probability column names with pattern "Scored Probabilities_label" """
                return '_'.join(
                    (ScoreColumnConstants.ScoredProbabilitiesMulticlassColumnNamePattern, str(label)))

            # MultiClassification core_model will return a list of tuples in format:
            # (label_id, [class_0_prob, class_1_prob, ...])
            label_ids = [row[0] for row in predict_ret_list]
            probs = [row[1] for row in predict_ret_list]
            if probs:
                class_cnt = len(probs[0])
            else:
                return pd.DataFrame()
            columns = [_gen_scored_probability_column_name(label) for label in 
                       self.task.label_map.inverse_transform(range(class_cnt))]
            result_df = pd.DataFrame(data=probs, columns=columns)
            result_df[ScoreColumnConstants.ScoredLabelsColumnName] = self.task.label_map.inverse_transform(label_ids)
            return result_df
        else:
            if not predict_ret_list:
                return pd.DataFrame()
            else:
                # TODO: Follow Module Team's practice to connect to Evaluate Module
                column_cnt = len(predict_ret_list[0])
                columns = [f"Score_{i}" for i in range(column_cnt)]
                return pd.DataFrame(data=predict_ret_list, columns=columns)

    @property
    def raw_model(self):
        """
        Built-in models should contain a raw_model, which is a ML framework-specified model object
        e.g. torch.nn.Module
        :return:
        """
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
        if self.task and self.task.ground_truth_column_name:
            return self.task.ground_truth_column_name
        return ModelSpecConstants.DEFAULT_GROUND_TRUTH_COLUMN_NAME

    @property
    def task_type(self):
        if self.task and self.task.task_type:
            return self.task.task_type
        return None
