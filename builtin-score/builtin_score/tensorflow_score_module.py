import tensorflow as tf
import numpy as np
import os

class TensorflowScoreModule(object):

    def __init__(self, model_meta_path):
        model_path = os.path.dirname(model_meta_path)
        self.class_names = ["class:{}".format(str(i)) for i in range(10)]
        self.sess = tf.Session()
        saver = tf.train.import_meta_graph(model_meta_path)
        saver.restore(self.sess,tf.train.latest_checkpoint(model_path))

        graph = tf.get_default_graph()
        # TODO: fix the hard-code x,y here
        self.x = graph.get_tensor_by_name("x:0")
        self.y = graph.get_tensor_by_name("y:0")

        print(f"Successfully loaded model from {model_path}")

    def run(self, df):
        return self.predict(df.values, None)

    def predict(self,X,feature_names):
        predictions = self.sess.run(self.y,feed_dict={self.x:X})
        return predictions.astype(np.float64)

