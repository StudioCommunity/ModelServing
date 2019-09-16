# --------------------------------------------------------------------------
# TODO: Move the loading logit to azureml-studio-model and remove this file.
# --------------------------------------------------------------------------


import os
from . import constants
from azureml.visual_interface.model.tensorflow import _TFSavedModelWrapper, _TFSaverWrapper
from ..utils import ioutils


class TensorflowScoreModule(object):

    def __init__(self, model_path, config):
        tf_config = config["tensorflow"]
        
        if(tf_config.get(constants.SERIALIZATION_METHOD_KEY, "saver") == "saved_model"):
            export_dir = os.path.join(model_path, tf_config[constants.MODEL_FILE_PATH_KEY])
            tf_meta_graph_tags = tf_config["meta_graph_tags"]
            tf_signature_def_key = tf_config["signature_def_key"]
            self.wrapper = _TFSavedModelWrapper(export_dir, tf_meta_graph_tags, tf_signature_def_key)
        else:
            self.wrapper = _TFSaverWrapper(model_path, config)

    def run(self, df):
        return self.wrapper.predict(df)

    def get_schema(self):
        return self.wrapper.get_schema()


def _test_tensor(df, model_path):
    import yaml
    with open(model_path + "model_spec.yml") as fp:
        config = yaml.safe_load(fp)

    tfmodule = TensorflowScoreModule(model_path, config)
    result = tfmodule.run(df)
    print(result)

# python -m azureml.studio.score.tensorflow_score_module
if __name__ == '__main__':
    df = ioutils.read_parquet("../dstest/outputs/mnist/")
    print(df.columns)
    _test_tensor(df, "../dstest/model/tensorflow-minist/")

    # df = df.rename(columns={"x": "images"})
    # print(df.columns)
    # _test_tensor(df,"../dstest/model/tensorflow-minist-saved-model/")

    # df = ioutils.read_parquet("../dstest/outputs/mnist/")
    # df = df.rename(columns={"x": "image", "image": "image1"})
    # print(df.columns)
    # _test_tensor(df,"../dstest/model/tensorflow-mnist-cnn-estimator/")