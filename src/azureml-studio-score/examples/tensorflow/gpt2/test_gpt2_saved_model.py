import pytest

from azureml.studio.score import ioutil
from azureml.studio.score.builtin_score_module import *
from azureml.studio.score.preprocess.byte_pair_encode import BPEEncoder

"""
This test requires azureml-studio-model saved gpt-2 model under folder "model/gpt2" 
"""

ENCODED_INPUT_PATH = "outputs/gpt2/encoded_text"
MODEL_PATH = "model/gpt2"
DICT_PATH = "https://storage.googleapis.com/gpt-2/encodings/main/encoder.json"
VOCAB_PATH = "https://storage.googleapis.com/gpt-2/encodings/main/vocab.bpe"

DICT_PATH_KEY = "DictionaryPath"
VOCAB_PATH_KEY = "VocabularyPath"


def test_encode_decode():
    meta = {
        DICT_PATH_KEY: DICT_PATH,
        VOCAB_PATH_KEY: VOCAB_PATH
    }
    encoder = BPEEncoder(meta)
    raw_text = pd.DataFrame(["this is a test"])
    encoded_text = encoder.encode(raw_text)
    decoded_text = encoder.decode(encoded_text)
    assert raw_text.equals(decoded_text)


def test_builtin():
    raw_text = pd.DataFrame(["this is a test"])
    meta = {
        DICT_PATH_KEY: DICT_PATH,
        VOCAB_PATH_KEY: VOCAB_PATH
    }
    encoder = BPEEncoder(meta)
    encoded_text = encoder.encode(raw_text)
    ioutil.save_parquet(encoded_text, ENCODED_INPUT_PATH)
    df = ioutil.read_parquet(ENCODED_INPUT_PATH)
    module = BuiltinScoreModule(MODEL_PATH)
    result = module.run(df)
    print(f'Result: {result}')
    assert result is not None
    decoded_text = encoder.decode(result).values.flatten()[0]
    print(f'Decoded text lenth:{len(decoded_text)}')
    assert len(decoded_text) > 0
