import tensorflow as tf
import numpy as np
import os
import pandas as pd
from . import constants
from . import ioutil

def rename_col(df, col_name):
    col_pattern = col_name +"."
    df.rename(columns=lambda col : col_name if col.startswith(col_pattern) else col, inplace=True)

def get_col_schema(name, tensor):
    col = {
        "name": name,
        "dtype": tensor.dtype.name,
        "shape": tensor.shape.as_list()
    }
    return col

def load_graph(model_file_path, sess):
    with open(model_file_path, mode='rb') as f:
        fileContent = f.read()

    graph_def = tf.GraphDef()
    graph_def.ParseFromString(fileContent)
    tf.import_graph_def(graph_def)
    graph = tf.get_default_graph()
    init = tf.global_variables_initializer()
    sess.run(init)
    return graph

#df[name]
#shape = self.x_shape[name]
def array_from_df_col(col, shape):
    values = ioutil.from_df_column_to_array(col)
    if shape != None :
        target_shape = (len(values), *shape)
        # reshape if target_shape doesn't contain None
        if values.shape != target_shape and None not in target_shape:
            print(f"reshape from {values.shape} to {target_shape}.")
            values = np.array(values).reshape(target_shape)
    return values

class _TFSavedModelWrapper(object):
    """
    Wrapper class that exposes a TensorFlow model for inference via a ``predict`` function such that
    ``predict(data: pandas.DataFrame) -> pandas.DataFrame``.
    """
    def __init__(self, export_dir, tf_meta_graph_tags, tf_signature_def_key):
        tf_graph = tf.Graph()
        tf_sess = tf.Session(graph=tf_graph)

        self.tf_graph = tf_graph
        self.tf_sess = tf_sess

        with tf_graph.as_default():
          self.signature_def = self._load_tensorflow_saved_model(tf_sess, tf_meta_graph_tags, tf_signature_def_key, export_dir)
          
        # input keys in the signature definition correspond to input DataFrame column names
        self.input_tensor_mapping = {
            tensor_column_name: tf_graph.get_tensor_by_name(tensor_info.name)
            for tensor_column_name, tensor_info in self.signature_def.inputs.items()
        }
        # output keys in the signature definition correspond to output DataFrame column names
        self.output_tensors = {
            sigdef_output: tf_graph.get_tensor_by_name(tnsr_info.name)
            for sigdef_output, tnsr_info in self.signature_def.outputs.items()
        }
        print(self.input_tensor_mapping)
        print(self.output_tensors)

    def get_schema(self):
        schema = {
            "inputs": [],
            "outputs": []
            }
        for name, tensor in self.input_tensor_mapping.items():
            schema['inputs'].append(get_col_schema(name,tensor))

        for name, tensor in self.output_tensors.items():
            schema['outputs'].append(get_col_schema(name,tensor))
        
        #print(schema)
        return schema

    def predict(self, df):
      with self.tf_graph.as_default():
        feed_dict = {
            tensor: array_from_df_col(df[tensor_column_name], tensor.shape.as_list()[1:]) # TODO: check first column -1, and check if we can replace there
            for tensor_column_name, tensor in self.input_tensor_mapping.items()
        }
        raw_preds = self.tf_sess.run(self.output_tensors, feed_dict=feed_dict)
        resultdf = pd.DataFrame()
        for column_name, values in raw_preds.items():
            resultdf.insert(len(resultdf.columns), column_name, values.tolist(), True)
        return resultdf

    def _load_tensorflow_saved_model(self, sess, tf_meta_graph_tags,tf_signature_def_key, export_dir):
      meta_graph_def = tf.saved_model.loader.load(sess, tf_meta_graph_tags, export_dir)
      if tf_signature_def_key not in meta_graph_def.signature_def:
        raise Exception("Could not find signature def key %s" % tf_signature_def_key)
      return meta_graph_def.signature_def[tf_signature_def_key]

class _TFSaverWrapper(object):

    def _load_graph_from_checkpoint(self, model_path, tf_config):
        model_meta_path = os.path.join(model_path, tf_config[constants.MODEL_FILE_PATH_KEY])
        print(f"model_meta_path = {model_meta_path}, model_path = {model_path}")
        saver = tf.train.import_meta_graph(model_meta_path)
        saver.restore(self.sess,tf.train.latest_checkpoint(model_path))
        self.graph = tf.get_default_graph()
    
    def _load_graph(self, model_path, tf_config):
        model_file_path = os.path.join(model_path, tf_config[constants.MODEL_FILE_PATH_KEY])
        print(f"model_file_path = {model_file_path}, model_path = {model_path}")
        self.graph = load_graph(model_file_path, self.sess)

    def _load_inputs_outputs(self, tf_config):
        graph = self.graph
        self.x = {}
        self.x_shape = {}
        for index in range(len(tf_config["inputs"])):
            name = tf_config["inputs"][index]["name"]
            tensor = graph.get_tensor_by_name(name + ":0")
            self.x[name] = tensor
            shape = None
            if(tensor.shape.ndims == None):
                shape = tf_config["inputs"][index].get("shape", None)
            else:
                shape = tensor.shape.as_list()[1:]
            self.x_shape[name] = shape
        
        print("loaded inputs:")
        print(self.x)
        print(self.x_shape)

        self.y = []
        self.y_names = []
        for index in range(len(tf_config["outputs"])):
            name = tf_config["outputs"][index]["name"]
            # TODO: support :0 in future version. :0 means the first ouput of an op in tensorflow graph
            tensor = graph.get_tensor_by_name(name + ":0")
            self.y.append(tensor)
            self.y_names.append(name)

        print("loaded outputs:")
        print(self.y)
        print(self.y_names)

    def __init__(self, model_path, config):
        self.sess = tf.Session()
        tf_config = config["tensorflow"]
        print(tf_config)
        serialization_format = tf_config.get(constants.SERIALIZATION_METHOD_KEY, "saver")
        if(serialization_format == "saver"):
            self._load_graph_from_checkpoint(model_path, tf_config)
        else:
            self._load_graph(model_path, tf_config)
        self._load_inputs_outputs(tf_config)
        print(f"Successfully loaded model from {model_path}")

    def get_schema(self):
        schema = {
            "inputs": [],
            "outputs": []
            }
        for name, tensor in self.x.items():
            schema['inputs'].append(get_col_schema(name,tensor))

        for index in range(len(self.y_names)):
            name= self.y_names[index]
            tensor= self.y[index]
            schema['outputs'].append(get_col_schema(name,tensor))
        
        return schema

    def predict(self, df):
        predictions = self.sess.run(self.y, feed_dict= self.feed_dict(df))
        resultdf = pd.DataFrame()
        for index in range(len(self.y_names)):
            name = self.y_names[index]
            predict = predictions[index].tolist()
            resultdf.insert(len(resultdf.columns), name, predict, True)

        return resultdf

    def feed_dict(self, df):
        dict = {}
        for name, tensor in self.x.items():
            if(name not in df.columns):
                raise Exception(f"Column {name} not in input df columns: {df.columns}")
            values = array_from_df_col(df[name], self.x_shape[name])
            dict[tensor] = values
        return dict

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

# python -m builtin_score.tensorflow_score_module
if __name__ == '__main__':
    df = ioutil.read_parquet("../dstest/outputs/mnist/")
    print(df.columns)
    _test_tensor(df, "../dstest/model/tensorflow-minist/")

    # df = df.rename(columns={"x": "images"})
    # print(df.columns)
    # _test_tensor(df,"../dstest/model/tensorflow-minist-saved-model/")

    # df = ioutil.read_parquet("../dstest/outputs/mnist/")
    # df = df.rename(columns={"x": "image", "image": "image1"})
    # print(df.columns)
    # _test_tensor(df,"../dstest/model/tensorflow-mnist-cnn-estimator/")