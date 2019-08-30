import logging
import click
import pandas as pd
from torchvision import transforms as T
from ..utils import ioutils
from ..utils import datauri_utils
from ..utils import dfutils

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.DEBUG)
logging.info(f"in {__file__}")
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def tuple_parser(input_str, data_type):
    return tuple(map(data_type, input_str.strip().strip('(').strip(')').split()))

class ImagePreprocess:

    def __init__(self, params: dict = {}):
        self.image_column = str(params.get('Image Column', 'image'))
        self.target_column = str(params.get('Target Column', 'x'))
        self.target_datauri_column = str(params.get('Target DataURI Column', 'x.datauri'))

        try:
            image_size_str = str(params.get('Target Image Size', ''))
            self.target_image_size = tuple_parser(image_size_str, int)
        except:
            raise Exception('Invalid [Target Image Size] Parameter:{image_size_str}')

        self.resize_size = -1
        resize_size_str = params.get("Resize Size", "").strip()
        if resize_size_str != "":
            try:
                self.resize_size = int(params.get("Resize Size", '').strip())
            except ValueError:
                raise Exception(f"Invalid [Resize Size] Parameter: {resize_size_str}")
        
        if not self.target_column:
            self.target_column = self.image_column
        if not self.target_datauri_column:
            self.target_data_column = f"{self.target_column}_datauri"
        logger.debug(f"image_column = {self.image_column}, target_column = {self.target_column}, datauri_column = {self.target_datauri_column}")

        self.need_normaliztion = True
        try:
            normalization_mean_str = str(params.get("Normalization Mean", ""))
            if normalization_mean_str == "":
                self.need_normaliztion = False
            else:
                self.normalization_mean = tuple_parser(normalization_mean_str, float)
                normalization_std_str = str(params.get("Normalization Std", ""))
                self.normalization_std = tuple_parser(normalization_std_str, float)
        except:
            raise Exception(f"Invalid [Normalization Mean]: {normalization_mean_str} or Invalid [Normaliztion Std]: {normalization_std_str}")

        convert_to_grayscale_str = params.get("Convert to Grayscale", None)
        self.convert_to_grayscale = isinstance(convert_to_grayscale_str, str) and\
            convert_to_grayscale_str.lower() == "true"
        logger.debug(f"self.convert_to_grayscale= {self.convert_to_grayscale}")

    def run(self, input_df: pd.DataFrame, params: dict = None):
        results = []
        datauris = []
        
        for _, row in input_df.iterrows():
            img = datauri_utils.base64str_to_image(row[self.image_column])

            preprocess_operations = []
            if self.convert_to_grayscale:
                preprocess_operations.append(T.Grayscale())
            if self.resize_size != -1:
                preprocess_operations.append(T.Resize(self.resize_size))

            preprocess_operations += [
                T.CenterCrop(self.target_image_size),
                T.ToTensor()
            ]

            if self.need_normaliztion:
                normalize = T.Normalize(
                    mean=self.normalization_mean,
                    std=self.normalization_std
                )
                preprocess_operations.append(normalize)

            preprocess = T.Compose(preprocess_operations)
            image_tensor = preprocess(img).unsqueeze(0)
            results.append(image_tensor.tolist())

            if self.target_datauri_column:
                datauri = datauri_utils.tensor_to_datauri(image_tensor)
                datauris.append(datauri)

        dfutils.add_column_to_dataframe(input_df, results, self.target_column)
        if self.target_datauri_column:
            dfutils.add_column_to_dataframe(input_df, datauris, self.target_datauri_column)
        logger.info(f"input_df.columns = {input_df.columns}")
        logger.info(f"input_df = \n {input_df}")

        return input_df

@click.command()
@click.option('--input_path', default="datas/mnist")
@click.option('--output_path', default="outputs/mnist")
@click.option('--image_column', default="image")
@click.option('--target_column', default="x")
@click.option('--target_datauri_column', default="")
@click.option('--resize_size', default="")
@click.option('--target_image_size', default="")
@click.option('--normalization_mean', default="(0, 0, 0)")
@click.option('--normalization_std', default="(0.5, 0.5, 0.5)")
@click.option('--convert_to_grayscale', default="False")
def run(input_path, output_path, image_column, target_column, target_datauri_column, resize_size, target_image_size,
        normalization_mean, normalization_std, convert_to_grayscale):
    """
    This functions read base64 encoded images from df. Transform to format required by model input.
    """
    params = {
        "Image Column": image_column,
        "Target Column": target_column,
        "Target DataURI Column": target_datauri_column,
        "Resize Size": resize_size,
        "Target Image Size": target_image_size,
        "Normalization Mean": normalization_mean,
        "Normalization Std": normalization_std,
        "Convert to Grayscale": convert_to_grayscale
    }
    proccesor = ImagePreprocess(params)

    df = ioutils.read_parquet(input_path)
    result = proccesor.run(df)
    ioutils.save_parquet(result, output_path)

# mnist: python -m dstest.preprocess.preprocess  --input_path datas/mnist --output_path outputs/mnist --image_column=image --target_column=x --target_datauri_column=x.data --target_image_size=28x28
# imagenet: python -m dstest.preprocess.preprocess  --input_path datas/imagenet --output_path outputs/imagenet --image_column=image --target_column=import/images --target_datauri_column=import/images.data --target_image_size=224x224
# stargan: python -m dstest.preprocess.preprocess  --input_path inputs/stargan --output_path outputs/stargan --image_column=image --target_column=import/images --target_datauri_column=import/images.data --resize_size=300 --target_image_size=(256, 256) --normalization_mean=(0, 0, 0) --normalization_std=(0.5, 0.5, 0.5)
if __name__ == '__main__':
    run()
