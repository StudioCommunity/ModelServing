from azureml.studio.score import ioutil
from azureml.studio.score.builtin_score_module import *

ENCODED_INPUT_PATH = "outputs/gpt2/encoded_text"
MODEL_PATH = "model/gpt2"
GENERATED_TEXT_PATH = "outputs/gpt2/generated_token"


def test_builtin():
    df = ioutil.read_parquet(ENCODED_INPUT_PATH)
    module = BuiltinScoreModule(MODEL_PATH)
    result = module.run(df)
    # Result is a data frame
    ioutil.save_parquet(result, GENERATED_TEXT_PATH)

def save_raw_text_2_parquet():
    raw_test = pd.DataFrame(["this is a test"])
    ioutil.save_parquet(raw_test, ENCODED_INPUT_PATH)


# python -m examples.tensorflow.gpt2.gpt2_saved_model_test
if __name__ == '__main__':
    test_builtin()
    # save_raw_text_2_parquet()