import unittest

import azureml.studio.model.tensorflow
import azureml.studio.model.generic

import shutil

import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# The mnist model
x = tf.placeholder(tf.float32, [None,784], name="x")
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

y = tf.nn.softmax(tf.matmul(x,W) + b, name="y")
y_ = tf.placeholder(tf.float32, [None, 10])
y_label = tf.argmax(y, 1, name= "y_label")

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
tf.summary.scalar('cross_entropy', cross_entropy)

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

merged = tf.summary.merge_all()

for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    summary, _ = sess.run([merged, train_step], feed_dict={x: batch_xs, y_: batch_ys})

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict = {x: mnist.test.images, y_:mnist.test.labels}))


def prepare_input():
    batch_xs, batch_ys = mnist.train.next_batch(2)
    print(batch_xs)
    input = pd.DataFrame()
    input["x:0"] = [batch_xs[0]]
    return input


class TestTensorflowModel(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        # remove if exists
        shutil.rmtree('./AzureMLModel', ignore_errors=True)
        shutil.rmtree('MNIST_data', ignore_errors=True)

    @classmethod
    def tearDownClass(self) -> None:
        # remove if exists
        shutil.rmtree('./AzureMLModel', ignore_errors=True)
        shutil.rmtree('MNIST_data', ignore_errors=True)

    # Currently use hard code to control test run order
    # TODO(wanhan): find a better way
    def test_1_save_model(self):
        azureml.studio.model.tensorflow.save_model(sess, [x], [y])

    def test_2_load_model(self):
        # load model(with tensorflow API)
        tf_graph = tf.Graph()
        tf_sess = tf.Session(graph=tf_graph)
        signature_def = azureml.studio.model.tensorflow.load_model(tf_sess)
        # input keys in the signature definition correspond to input DataFrame column names
        input_tensor_mapping = {
            tensor_column_name: tf_graph.get_tensor_by_name(tensor_info.name)
            for tensor_column_name, tensor_info in signature_def.inputs.items()
        }
        # output keys in the signature definition correspond to output DataFrame column names
        output_tensors = {
            sigdef_output: tf_graph.get_tensor_by_name(tnsr_info.name)
            for sigdef_output, tnsr_info in signature_def.outputs.items()
        }
        print(input_tensor_mapping)
        print(output_tensors)

    def test_3_predict(self):
        # load model(with generic API)
        loaded_generic_model = azureml.studio.model.generic.load()
        print(dir(loaded_generic_model))
        df = prepare_input()
        pred = loaded_generic_model.predict(df)
        print(np.argmax(pred['y:0'][0]))


if __name__ == '__main__':
    unittest.main()
