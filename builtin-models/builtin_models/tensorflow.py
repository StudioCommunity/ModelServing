import os
import yaml
import json
import tensorflow as tf
import builtin_models.constants as constants
import builtin_models.utils as utils

FLAVOR_NAME = "tensorflow"


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

    print('inputs: ', inputs_dict)
    print('outputs: ', outputs_dict)
    prediction_signature = (
        tf.saved_model.signature_def_utils.build_signature_def(
            inputs=inputs_dict,
            outputs=outputs_dict,
            method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))

    builder.add_meta_graph_and_variables(
        sess, 
        graph_tags,
        signature_def_map= {
            str(signature_name): prediction_signature
        },
        main_op=tf.tables_initializer(),
        strip_default_attrs=True)

    builder.save()


def save_model(sess, input_tensor_list, output_tensor_list, graph_tags=None, signature_name=None, conda_env=None, path='./model/'):
    """
    Save a Tensorflow model to a path on the local file system.

    :param sess: Tensorflow session.

    :param input_tensor_list: list of input tensors.
    
    :param output_tensor_list: list of output tensors.
    
    :param graph_tags: list of graph tags (optional), if not specified, default its value would be [tf.saved_model.tag_constants.SERVING].
    
    :param signature_name: signature name (optional), if not specified, default its value would be 'signature_name'.

    :param conda_env: Either a dictionary representation of a Conda environment or the path to a conda environment yaml file (optional). 

    :param path: Path to a directory containing model, spec, conda yaml data (optional).
    """
    if(not path.endswith('/')):
        path += '/'
    if not os.path.exists(path):
        os.makedirs(path)

    if graph_tags == None or len(graph_tags) == 0:
        graph_tags = [tf.saved_model.tag_constants.SERVING]

    if signature_name is None or signature_name == '':
        signature_name = 'signature_name'

    model_file_path = 'model' # sub-directory containing the tensorflow model
    _save_model(os.path.join(path, model_file_path), sess, input_tensor_list, output_tensor_list, graph_tags, signature_name)

    if conda_env is None:
        conda_env = _get_default_conda_env()
    utils.save_conda_env(path, conda_env)

    _save_model_spec(path, model_file_path, graph_tags, signature_name)
    utils.generate_ilearner_files(path) # temp solution, to remove later

