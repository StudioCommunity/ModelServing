from typing import Tuple, Union
import pandas as pd
from azureml.designer.model.io import load_generic_model
from azureml.designer.model.model_spec.task_type import TaskType
from azureml.studio.core.data_frame_schema import DataFrameSchema
from azureml.studio.core.io.image_directory import ImageDirectory
from azureml.studio.core.io.data_frame_directory import DataFrameDirectory

from . import constants
from ..logger import get_logger
from ..utils import schema_utils

logger = get_logger(__name__)


class BuiltinScoreModule(object):

    def __init__(self):
        logger.info(f"Init BuiltinScoreModule")
        self.model = None
        self.append_score_column_to_output = True

    def on_init(self, trained_model: str, dataset: ImageDirectory, append_score_columns_to_output: bool) -> None:
        logger.info(f"loading generic model from {trained_model}")
        self.model = load_generic_model(trained_model, install_dependencies=True)
        logger.info(f"loaded generic model {self.model}")
        self.append_score_column_to_output = append_score_columns_to_output
        logger.info(f"self.append_score_column_to_output = {self.append_score_column_to_output}")
        logger.debug(f"dataset = {dataset}")

    def run(self,
            trained_model: str,
            dataset: Union[ImageDirectory, DataFrameDirectory],
            append_score_columns_to_output: bool) -> Tuple[DataFrameDirectory, ]:
        logger.debug(f"trained_model = {trained_model}")
        logger.debug(f"append_score_columns_to_output = {append_score_columns_to_output}")
        if not isinstance(dataset, (DataFrameDirectory, ImageDirectory)):
            raise Exception(f"Unsupported dataset type: {type(dataset)}, "
                            f"expecting DataFrameDirectory or ImageDirectory")

        if isinstance(dataset, ImageDirectory):
            result_df = self.model.predict(dataset)
        else:
            input_df = dataset.data
            output_label = self.model.predict(input_df)
            logger.debug(f"output_label = {output_label}")
            if self.append_score_column_to_output:
                if isinstance(output_label, pd.DataFrame):
                    result_df = pd.concat([input_df, output_label], axis=1)
                else:
                    result_df = input_df
                    result_df.insert(
                        loc=len(input_df.columns),
                        column=constants.SCORED_LABEL_COL_NAME,
                        value=output_label,
                        allow_duplicates=True)
            else:
                if isinstance(output_label, pd.DataFrame):
                    result_df = output_label
                else:
                    result_df = pd.DataFrame({constants.SCORED_LABEL_COL_NAME: output_label})
            logger.debug(f"result_df =\n{result_df}")
            logger.debug(f"result_df.columns = {result_df.columns}")
            if result_df.shape[0] > 0:
                for col in result_df.columns:
                    logger.debug(f"{col}: {type(result_df.loc[0][col])}")

        if self.model.task_type == TaskType.MultiClassification:
            score_columns = schema_utils.generate_score_column_meta(predict_df=result_df)
            if self.model.label_column_name in result_df.columns:
                label_column_name = self.model.label_column_name
            else:
                label_column_name = None
            meta_data = DataFrameSchema(
                column_attributes=DataFrameSchema.generate_column_attributes(df=result_df),
                score_column_names=score_columns,
                label_column_name=label_column_name
            )
            dfd = DataFrameDirectory.create(data=result_df, schema=meta_data.to_dict())
        else:
            dfd = DataFrameDirectory.create(data=result_df)
        return dfd
