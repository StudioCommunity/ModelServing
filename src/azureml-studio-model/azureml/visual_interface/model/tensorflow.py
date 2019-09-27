import os
import yaml
import logging
import tensorflow as tf
import pandas as pd
import numpy as np
from . import constants, utils, ioutils
from .generic import GenericModel

logger = logging.getLogger(__name__)

FLAVOR_NAME = "tensorflow"
CODE_FOLDER_NAME = "code"

# work around for cases when tensorflow couldn't get tty for warning logs
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def _get_default_conda_env():
    return utils.generate_conda_env(
        additional_pip_deps=[
            "tensorflow=={}".format(tf.__version__)
        ])


def _save_model_spec(path, model_file_path, graph_tags, signature_name, serialization_format='saved_model'):
    spec = utils.generate_default_model_spec(FLAVOR_NAME, model_file_path)
    # add meta_graph_tags
    spec[FLAVOR_NAME]['meta_graph_tags'] = graph_tags
    # add serialization_format, default serialization format is 'saved_model'
    spec[FLAVOR_NAME]['serialization_format'] = serialization_format
    # add signature_def_key
    spec[FLAVOR_NAME]['signature_def_key'] = signature_name
    with open(os.path.join(path, constants.MODEL_SPEC_FILE_NAME), 'w') as fp:
        yaml.dump(spec, fp, default_flow_style=False)


def _save_model(export_path, sess, input_tensor_list, output_tensor_list, graph_tags, signature_name):
    builder = tf.saved_model.builder.SavedModelBuilder(export_path)
    inputs_dict = dict()
    outputs_dict = dict()

    for in_tensor in input_tensor_list:
        if isinstance(in_tensor, str):
            tensor_info_input = input_tensor_list[in_tensor]
            inputs_dict[in_tensor] = tensor_info_input
        else:
            tensor_info_input = tf.saved_model.utils.build_tensor_info(in_tensor)
            inputs_dict[in_tensor.name] = tensor_info_input

    for out_tensor in output_tensor_list:
        if isinstance(out_tensor, str):
            tensor_info_output = output_tensor_list[out_tensor]
            outputs_dict[out_tensor] = tensor_info_output
        else:
            tensor_info_output = tf.saved_model.utils.build_tensor_info(out_tensor)
            outputs_dict[out_tensor.name] = tensor_info_output

    logger.info('inputs: ', inputs_dict)
    logger.info('outputs: ', outputs_dict)
    prediction_signature = (
        tf.saved_model.signature_def_utils.build_signature_def(
            inputs=inputs_dict,
            outputs=outputs_dict,
            method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))

    builder.add_meta_graph_and_variables(
        sess,
        graph_tags,
        signature_def_map={
            str(signature_name): prediction_signature
        },
        main_op=tf.tables_initializer(),
        strip_default_attrs=True)

    builder.save()


def save_model(sess, input_tensor_list, output_tensor_list, graph_tags=None, signature_name=None, conda_env=None,
               path='./AzureMLModel', code_path=None):
    """
    Save a Tensorflow model to a path on the local file system.

    :param sess: Tensorflow session.

    :param input_tensor_list: list of input tensors.
    
    :param output_tensor_list: list of output tensors.
    
    :param graph_tags: list of graph tags (optional), if not specified, default its value would be [
    tf.saved_model.tag_constants.SERVING].
    
    :param signature_name: signature name (optional), if not specified, default its value would be 'signature_name'.

    :param conda_env: Either a dictionary representation of a Conda environment or the path to a conda environment
    yaml file (optional).

    :param path: Path to a directory containing model, spec, conda yaml data (optional).

    :param code_path: Path to save users code(optional).
    """
    if not path.endswith('/'):
        path += '/'
    if not os.path.exists(path):
        os.makedirs(path)

    if not graph_tags:
        graph_tags = [tf.saved_model.tag_constants.SERVING]

    if signature_name is None or signature_name == '':
        signature_name = 'signature_name'

    model_file_path = 'model'  # sub-directory containing the tensorflow model
    _save_model(os.path.join(path, model_file_path), sess, input_tensor_list, output_tensor_list, graph_tags,
                signature_name)

    if conda_env is None:
        conda_env = _get_default_conda_env()
    utils.save_conda_env(path, conda_env)

    _save_model_spec(path, model_file_path, graph_tags, signature_name)
    utils.generate_ilearner_files(path)  # temp solution, to remove later

    if code_path is not None:
        dst_code_path = os.path.join(path, CODE_FOLDER_NAME)
        utils._copytree_include(code_path, dst_code_path, include_extensions=(".py", ))


def rename_col(df, col_name):
    col_pattern = col_name + "."
    df.rename(columns=lambda col: col_name if col.startswith(col_pattern) else col, inplace=True)


def get_col_schema(name, tensor):
    col = {
        "name": name,
        "dtype": tensor.dtype.name,
        "shape": tensor.shape.as_list()
    }
    return col


def load_graph(model_file_path, sess):
    with open(model_file_path, mode='rb') as f:
        file_content = f.read()

    graph_def = tf.GraphDef()
    graph_def.ParseFromString(file_content)
    tf.import_graph_def(graph_def)
    graph = tf.get_default_graph()
    init = tf.global_variables_initializer()
    sess.run(init)
    return graph


# df[name]
# shape = self.x_shape[name]
def array_from_df_col(col, shape):
    values = ioutils.from_df_column_to_array(col)
    if shape is not None:
        target_shape = (len(values), *shape)
        # reshape if target_shape doesn't contain None
        if values.shape != target_shape and None not in target_shape:
            logger.info(f"reshape from {values.shape} to {target_shape}.")
            values = np.array(values).reshape(target_shape)
    return values


def load_tensorflow_saved_model(sess, tf_meta_graph_tags, tf_signature_def_key, export_dir):
    meta_graph_def = tf.saved_model.loader.load(sess, tf_meta_graph_tags, export_dir)
    if tf_signature_def_key not in meta_graph_def.signature_def:
        raise Exception("Could not find signature def key %s" % tf_signature_def_key)
    return meta_graph_def.signature_def[tf_signature_def_key]


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
            self.signature_def = load_tensorflow_saved_model(tf_sess, tf_meta_graph_tags, tf_signature_def_key,
                                                             export_dir)

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
        logger.info(self.input_tensor_mapping)
        logger.info(self.output_tensors)

    def get_schema(self):
        schema = {
            "inputs": [],
            "outputs": []
        }
        for name, tensor in self.input_tensor_mapping.items():
            schema['inputs'].append(get_col_schema(name, tensor))

        for name, tensor in self.output_tensors.items():
            schema['outputs'].append(get_col_schema(name, tensor))

        return schema

    def predict(self, df):
        with self.tf_graph.as_default():
            feed_dict = {
                tensor: array_from_df_col(df[tensor_column_name], tensor.shape.as_list()[1:])
                # TODO: check first column -1, and check if we can replace there
                for tensor_column_name, tensor in self.input_tensor_mapping.items()
            }
            raw_preds = self.tf_sess.run(self.output_tensors, feed_dict=feed_dict)
            resultdf = pd.DataFrame()
            for column_name, values in raw_preds.items():
                resultdf.insert(len(resultdf.columns), column_name, values.tolist(), True)
            return resultdf


class _TFSaverWrapper(object):

    def _load_graph_from_checkpoint(self, model_path, tf_config):
        model_meta_path = os.path.join(model_path, tf_config[constants.MODEL_FILE_PATH_KEY])
        logger.info(f"model_meta_path = {model_meta_path}, model_path = {model_path}")
        saver = tf.train.import_meta_graph(model_meta_path)
        saver.restore(self.sess, tf.train.latest_checkpoint(model_path))
        self.graph = tf.get_default_graph()

    def _load_graph(self, model_path, tf_config):
        model_file_path = os.path.join(model_path, tf_config[constants.MODEL_FILE_PATH_KEY])
        logger.info(f"model_file_path = {model_file_path}, model_path = {model_path}")
        self.graph = load_graph(model_file_path, self.sess)

    def _load_inputs_outputs(self, tf_config):
        graph = self.graph
        self.x = {}
        self.x_shape = {}
        for index in range(len(tf_config["inputs"])):
            name = tf_config["inputs"][index]["name"]
            tensor = graph.get_tensor_by_name(name + ":0")
            self.x[name] = tensor
            if tensor.shape.ndims is None:
                shape = tf_config["inputs"][index].get("shape", None)
            else:
                shape = tensor.shape.as_list()[1:]
            self.x_shape[name] = shape

        logger.info("loaded inputs:")
        logger.info(self.x)
        logger.info(self.x_shape)

        self.y = []
        self.y_names = []
        for index in range(len(tf_config["outputs"])):
            name = tf_config["outputs"][index]["name"]
            # TODO: support :0 in future version. :0 means the first ouput of an op in tensorflow graph
            tensor = graph.get_tensor_by_name(name + ":0")
            self.y.append(tensor)
            self.y_names.append(name)

        logger.info("loaded outputs:")
        logger.info(self.y)
        logger.info(self.y_names)

    def __init__(self, model_path, config):
        self.sess = tf.Session()
        tf_config = config["tensorflow"]
        logger.info(tf_config)
        serialization_format = tf_config.get(constants.SERIALIZATION_METHOD_KEY, "saver")
        if serialization_format == "saver":
            self._load_graph_from_checkpoint(model_path, tf_config)
        else:
            self._load_graph(model_path, tf_config)
        self._load_inputs_outputs(tf_config)
        logger.info(f"Successfully loaded model from {model_path}")

    def get_schema(self):
        schema = {
            "inputs": [],
            "outputs": []
        }
        for name, tensor in self.x.items():
            schema['inputs'].append(get_col_schema(name, tensor))

        for index in range(len(self.y_names)):
            name = self.y_names[index]
            tensor = self.y[index]
            schema['outputs'].append(get_col_schema(name, tensor))

        return schema

    def predict(self, df):
        predictions = self.sess.run(self.y, feed_dict=self.feed_dict(df))
        resultdf = pd.DataFrame()
        for index in range(len(self.y_names)):
            name = self.y_names[index]
            predict = predictions[index].tolist()
            resultdf.insert(len(resultdf.columns), name, predict, True)

        return resultdf

    def feed_dict(self, df):
        dictionary = {}
        for name, tensor in self.x.items():
            if name not in df.columns:
                raise Exception(f"Column {name} not in input df columns: {df.columns}")
            values = array_from_df_col(df[name], self.x_shape[name])
            dictionary[tensor] = values
        return dictionary


class _TensorflowWrapper(GenericModel):
    def __init__(self, model_path, config):
        super().__init__()
        tf_config = config["tensorflow"]

        if tf_config.get(constants.SERIALIZATION_METHOD_KEY, "saver") == "saved_model":
            export_dir = os.path.join(model_path, tf_config[constants.MODEL_FILE_PATH_KEY])
            tf_meta_graph_tags = tf_config["meta_graph_tags"]
            tf_signature_def_key = tf_config["signature_def_key"]
            self.wrapper = _TFSavedModelWrapper(export_dir, tf_meta_graph_tags, tf_signature_def_key)
        else:
            self.wrapper = _TFSaverWrapper(model_path, config)

    def predict(self, df):
        return self.wrapper.predict(df)

    def get_schema(self):
        return self.wrapper.get_schema()


def load_model(tf_sess, artifact_path="./AzureMLModel"):
    model_conf = utils._get_configuration(artifact_path)
    tf_config = model_conf['tensorflow']
    tf_meta_graph_tags = tf_config["meta_graph_tags"]
    tf_signature_def_key = tf_config["signature_def_key"]
    export_dir = os.path.join(artifact_path, tf_config[constants.MODEL_FILE_PATH_KEY])
    return load_tensorflow_saved_model(tf_sess, tf_meta_graph_tags, tf_signature_def_key, export_dir)


def _load_generic_model(artifact_path) -> _TensorflowWrapper:
    model_conf = utils._get_configuration(artifact_path)
    return _TensorflowWrapper(artifact_path, model_conf)
