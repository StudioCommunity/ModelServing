import click
from azureml.studio.core.io.any_directory import AnyDirectory
from azureml.studio.core.io.image_directory import ImageDirectory
from azureml.studio.core.io.data_frame_directory import DataFrameDirectory
from azureml.studio.core.io.data_frame_directory import save_data_frame_to_directory
from azureml.studio.core.data_frame_schema import DataFrameSchema
from azureml.designer.model.model_spec.task_type import TaskType

from . import constants
from .builtin_score_module import BuiltinScoreModule
from ..utils import ioutils, schema_utils
from ..logger import get_logger

# python -m azureml.designer.score.module_invoker --trained-model ../dstest/model/tensorflow-minist/ --dataset ../dstest/outputs/mnist/ --scored-dataset ../dstest/outputs/mnist/ouput --append-score-columns-to-output True
# python -m azureml.designer.score.module_invoker --trained-model ../dstest/model/vgg/ --dataset ../dstest/outputs/imagenet/ --scored-dataset ../dstest/outputs/imagenet/ouput --append-score-columns-to-output True
# python -m azureml.designer.score.score.module_invoker --trained-model ./azureml/designer/score/score/tests/pytorch/InputPort1 --dataset ./azureml/designer/score/score/tests/pytorch/InputPort2 --scored-dataset ./azureml/designer/score/score/tests/pytorch/OutputPort --append-score-columns-to-output True

logger = get_logger(__name__)


@click.command()
@click.option("--trained-model", help="Path to ModelDirectory")
@click.option("--dataset", help="Path to DFD/ImageDirectory")
@click.option("--scored-dataset", help="Path to output DFD")
@click.option("--append-score-columns-to-output", default="true", help="Preserve all columns from input dataframe or not")
def entrance(trained_model: str, dataset: str, scored_dataset: str, append_score_columns_to_output: str = "true"):
    logger.info(f"append_score_columns_to_output = {append_score_columns_to_output}")
    params = {
        constants.APPEND_SCORE_COLUMNS_TO_OUTPUT_KEY: append_score_columns_to_output
    }
    score_module = BuiltinScoreModule(trained_model, params)
    any_directory = AnyDirectory.load(dataset)
    if any_directory.type == "DataFrameDirectory":
        input_dfd = DataFrameDirectory.load(dataset)
        logger.info(f"input_dfd =\n{input_dfd}")
        output_df = score_module.run(input_dfd)
    elif any_directory.type == "ImageDirectory":
        image_directory = ImageDirectory.load(dataset)
        output_df = score_module.run(image_directory)
    else:
        raise Exception(f"Unsupported directory type: {type(any_directory)}.")

    logger.info(f"output_df =\n{output_df}")
    logger.info(f"dumping to DFD {scored_dataset}")

    # TODO: Support other task types
    if score_module.model.task_type == TaskType.MultiClassification:
        predict_df = output_df
        _LABEL_NAME = 'label'
        score_columns = schema_utils.generate_score_column_meta(predict_df=predict_df)
        if score_module.model.label_column_name in predict_df.columns:
            label_column_name = score_module.model.label_column_name
        else:
            label_column_name = None
        meta_data = DataFrameSchema(
            column_attributes=DataFrameSchema.generate_column_attributes(df=predict_df),
            score_column_names=score_columns,
            label_column_name=label_column_name
        )
        save_data_frame_to_directory(scored_dataset,
                                     data=predict_df,
                                     schema=meta_data.to_dict())
    else:
        ioutils.save_dfd(output_df, scored_dataset)


if __name__ == "__main__":
    entrance()
