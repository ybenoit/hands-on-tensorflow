"""Tests the graph freezing tool."""
#from __future__ import absolute_import
#from __future__ import division
#from __future__ import print_function

import tensorflow as tf
import numpy as np

from tutorials.solutions.mnist_2_deep_neural_net import graph


class GraphTest(tf.test.TestCase):
    """
    Test the graph part of the graph
    """

    def test_add_inference_step(self):
        with tf.Graph().as_default():
            x = tf.placeholder(tf.float32, [2, 2], name='x_input')

            self.assertIsNotNone(graph.create_inference_step(x, num_pixels=2, num_dense1=4, num_dense2=3, num_dense3=2,
                                                             num_classes=2))

    def test_run_inference(self):
        with tf.Graph().as_default():
            with tf.Session() as sess:
                # Input variables
                x = tf.Variable([[1.0, 1.0], [1.0, 1.0]], dtype=tf.float32)

                # Accuracy
                softmax = graph.create_inference_step(x, num_pixels=2, num_dense1=4, num_dense2=3, num_dense3=2,
                                                      num_classes=2)

                # Evaluate results
                tf.initialize_all_variables().run()
                self.assertAllClose(sess.run(softmax), [[0.5, 0.5], [0.5, 0.5]], rtol=1e-2)

    def test_add_loss_step(self):
        with tf.Graph().as_default():
            logits = tf.placeholder(tf.float32, [2, 2], name='logits')
            labels = tf.placeholder(tf.float32, [2, 2], name='labels')

            self.assertIsNotNone(graph.add_loss_step(logits, labels))

    def test_run_loss(self):
        with tf.Graph().as_default():
            with tf.Session() as sess:
                # Input variables
                logits = tf.Variable([[np.exp(0.1), np.exp(0.9)], [np.exp(0.9), np.exp(0.1)]], dtype=tf.float32)
                labels = tf.Variable([[1.0, 0.0], [1.0, 0.0]], dtype=tf.float32)

                # Cross Entropy
                cross_entropy = graph.add_loss_step(logits, labels)

                # Evaluate results
                tf.initialize_all_variables().run()
                self.assertEqual(sess.run(cross_entropy), -0.25)

    def test_add_train_step(self):
        with tf.Graph().as_default():
            loss = tf.Variable([1.0], dtype=tf.float32)

            self.assertIsNotNone(graph.add_train_step(loss, 0.1))

    # TODO: test_run_train_step

if __name__ == '__main__':
    tf.test.main()
