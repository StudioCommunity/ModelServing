# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

#! /usr/bin/env python
r"""Train and export a simple Softmax Regression TensorFlow model.

The model is from the TensorFlow "MNIST For ML Beginner" tutorial. This program
simply follows all its training instructions, and uses TensorFlow SavedModel to
export the trained model with proper signatures that can be loaded by standard
tensorflow_model_server.

Usage: mnist_saved_model.py [--training_iteration=x] [--model_version=y] \
    export_dir
"""

from __future__ import print_function

import os
import sys
import json

# This is a placeholder for a Google-internal import.
import tensorflow as tf
import yaml

from .mnist_input_data import read_data_sets
from builtin_models.tensorflow import save_model

tf.app.flags.DEFINE_integer('training_iteration', 1000,
                            'number of training iterations.')
tf.app.flags.DEFINE_string('model_version', 'mnist', 'version number of the model.')
tf.app.flags.DEFINE_string('work_dir', '/tmp', 'Working directory.')
tf.app.flags.DEFINE_string('export_dir', '', 'export_dir.')
FLAGS = tf.app.flags.FLAGS


def main(_):
  if len(sys.argv) < 2:
    print('Usage: mnist_saved_model.py [--training_iteration=x] '
          '[--model_version=y] [--export_dir=/path/to/dir/]')
    sys.exit(-1)
  if FLAGS.training_iteration <= 0:
    print('Please specify a positive value for training iteration.')
    sys.exit(-1)
  if FLAGS.model_version == '':
    print('Please specify a positive value for version number.')
    sys.exit(-1)
  if FLAGS.export_dir == '':
    print('Please specify a value for export_dir.')
    sys.exit(-1)

  # Train model
  print('Training model...')
  mnist = read_data_sets(FLAGS.work_dir, one_hot=True)
  sess = tf.InteractiveSession()
  serialized_tf_example = tf.placeholder(tf.string, name='tf_example')
  feature_configs = {'x': tf.FixedLenFeature(shape=[784], dtype=tf.float32),}
  tf_example = tf.parse_example(serialized_tf_example, feature_configs)
  x = tf.identity(tf_example['x'], name='x')  # use tf.identity() to assign name
  y_ = tf.placeholder('float', shape=[None, 10])
  w = tf.Variable(tf.zeros([784, 10]))
  b = tf.Variable(tf.zeros([10]))
  sess.run(tf.global_variables_initializer())
  y = tf.nn.softmax(tf.matmul(x, w) + b, name='y')
  cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
  train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
  values, indices = tf.nn.top_k(y, 10)
  table = tf.contrib.lookup.index_to_string_table_from_tensor(
      tf.constant([str(i) for i in range(10)]))
  prediction_classes = table.lookup(tf.to_int64(indices))
  for _ in range(FLAGS.training_iteration):
    batch = mnist.train.next_batch(50)
    train_step.run(feed_dict={x: batch[0], y_: batch[1]})
  correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
  print('training accuracy %g' % sess.run(
      accuracy, feed_dict={
          x: mnist.test.images,
          y_: mnist.test.labels
      }))
  print('Done training!')

  # using builtin_model to save model
  save_model(sess, [x], [y], path = FLAGS.export_dir)
  print('Done exporting!')


# python -m dstest.tensorflow.saved_model.mnist_saved_model --export_dir=model/tensorflow-minist-saved-model/
if __name__ == '__main__':
  tf.app.run()
