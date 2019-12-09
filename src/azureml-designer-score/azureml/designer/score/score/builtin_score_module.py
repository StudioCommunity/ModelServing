import os

import pandas as pd
from azureml.designer.model.io import load_generic_model
from azureml.studio.core.io.image_directory import ImageDirectory
from azureml.studio.core.io.data_frame_directory import DataFrameDirectory

from . import constants
from ..logger import get_logger

logger = get_logger(__name__)


class BuiltinScoreModule(object):

    def __init__(self, model_path, params={}):
        logger.info(f"BuiltinScoreModule({model_path}, {params})")
        append_score_column_to_output_value_str = params.get(
            constants.APPEND_SCORE_COLUMNS_TO_OUTPUT_KEY, None
        )
        self.append_score_column_to_output = isinstance(append_score_column_to_output_value_str, str) and\
            append_score_column_to_output_value_str.lower() == "true"
        logger.info(f"self.append_score_column_to_output = {self.append_score_column_to_output}")

        self.model = load_generic_model(model_path, install_dependencies=True)
        logger.info("Generic model loaded")

    def run(self, input_directory, global_param=None):
        if not isinstance(input_directory, (DataFrameDirectory, ImageDirectory)):
            raise Exception(f"Unsupported input_directory type: {type(input_directory)}, "
                            f"expecting DataFrameDirectory or ImageDirectory")
        if isinstance(input_directory, ImageDirectory):
            result_df = self.model.predict(input_directory.iter_images())
            return result_df
        else:
            df = input_directory.data
            output_label = self.model.predict(df)
            logger.info(f"output_label = {output_label}")
            if self.append_score_column_to_output:
                if isinstance(output_label, pd.DataFrame):
                    df = pd.concat([df, output_label], axis=1)
                else:
                    df.insert(len(df.columns), constants.SCORED_LABEL_COL_NAME, output_label, True)
            else:
                if isinstance(output_label, pd.DataFrame):
                    df = output_label
                else:
                    df = pd.DataFrame({constants.SCORED_LABEL_COL_NAME: output_label})
            logger.info(f"df =\n{df}")
            logger.info(f"df.columns = {df.columns}")
            if df.shape[0] > 0:
                for col in df.columns:
                    logger.info(f"{col}: {type(df.loc[0][col])}")
            return df
