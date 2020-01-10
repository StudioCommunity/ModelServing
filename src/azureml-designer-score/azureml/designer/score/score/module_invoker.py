import click
from azureml.studio.core.io.any_directory import AnyDirectory
from azureml.studio.core.io.image_directory import ImageDirectory
from azureml.studio.core.io.data_frame_directory import DataFrameDirectory
from azureml.studio.core.utils.fileutils import clear_folder

from .builtin_score_module import BuiltinScoreModule
from ..logger import get_logger

logger = get_logger(__name__)


# python -m azureml.designer.score.module_invoker --trained-model ../dstest/model/tensorflow-minist/ --dataset ../dstest/outputs/mnist/ --scored-dataset ../dstest/outputs/mnist/ouput --append-score-columns-to-output True
# python -m azureml.designer.score.module_invoker --trained-model ../dstest/model/vgg/ --dataset ../dstest/outputs/imagenet/ --scored-dataset ../dstest/outputs/imagenet/ouput --append-score-columns-to-output True
# python -m azureml.designer.score.score.module_invoker --trained-model ./azureml/designer/score/score/tests/pytorch/InputPort1 --dataset ./azureml/designer/score/score/tests/pytorch/InputPort2 --scored-dataset ./azureml/designer/score/score/tests/pytorch/OutputPort --append-score-columns-to-output True
@click.command()
@click.option("--trained-model", help="Path to ModelDirectory")
@click.option("--dataset", help="Path to DFD/ImageDirectory")
@click.option("--scored-dataset", help="Path to output DFD")
@click.option("--append-score-columns-to-output", default="true",
              help="Preserve all columns from input dataframe or not")
def entrance(trained_model: str, dataset: str, append_score_columns_to_output: str, scored_dataset: str):
    logger.debug(f"append_score_columns_to_output = {append_score_columns_to_output}")
    append_score_columns_to_output_bool = isinstance(append_score_columns_to_output, str) and \
                                          append_score_columns_to_output.lower() == "true"
    logger.debug(f"append_score_columns_to_output_bool = {append_score_columns_to_output_bool}")
    score_module = BuiltinScoreModule()
    any_directory = AnyDirectory.load(dataset)
    if any_directory.type == "DataFrameDirectory":
        input_dir = DataFrameDirectory.load(dataset)
    elif any_directory.type == "ImageDirectory":
        input_dir = ImageDirectory.load(dataset)
    else:
        raise Exception(f"Unsupported directory type: {type(any_directory)}.")
    logger.info(f"loaded input_dir = {input_dir}")
    score_module.on_init(
        trained_model=trained_model,
        dataset=input_dir,
        append_score_columns_to_output=append_score_columns_to_output_bool
    )
    output_dfd = score_module.run(trained_model, input_dir, append_score_columns_to_output)
    logger.info(f"dumping to DFD {scored_dataset}")
    clear_folder(scored_dataset)
    output_dfd.dump(scored_dataset)


if __name__ == "__main__":
    entrance()
