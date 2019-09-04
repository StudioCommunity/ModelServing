from examples.tensorflow.gpt2.encode import BPEEncoder
from azureml.studio.score import ioutil
import os

from pip._internal import main as pipmain
pipmain(["install", "click"])
import click

import logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logging.info(f"Decode text with BPE")
logger = logging.getLogger(__name__)

DICT_PATH_KEY = "Dictionary Path"
VOCAB_PATH_KEY = "Vocabulary Path"


@click.command()
@click.option('--dict_path')
@click.option('--vocab_path')
@click.option('--encoded_token_path', default="outputs/gpt2/generated_token")
@click.option('--decoded_text_path', default="outputs/gpt2/generated_text")
def run_pipeline(dict_path, vocab_path, encoded_token_path, decoded_text_path):
    print(f'ENCODED_TOKENS_PATH: {os.listdir(encoded_token_path)}')
    df = ioutil.read_parquet(encoded_token_path)
    meta = {
        DICT_PATH_KEY: dict_path,
        VOCAB_PATH_KEY: vocab_path
    }
    encoder = BPEEncoder(meta)
    result = encoder.decode(df)
    ioutil.save_parquet(result, decoded_text_path)

# TODO(wanhan): use a better way to store dict and vocabulary
# python -m examples.tensorflow.gpt2.decode ...
if __name__ == '__main__':
    run_pipeline()
