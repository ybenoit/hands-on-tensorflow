"""Loss Testing"""

import tensorflow as tf
import numpy as np

from tutorials.exercices.mnist_2_deep_neural_net import graph


class GraphLossTest(tf.test.TestCase):
    """
    Test the addition of the loss step in the graph
    """

    def test_add_loss_step(self):
        with tf.Graph().as_default():
            # Given
            logits = tf.placeholder(tf.float32, [2, 2], name='logits')
            labels = tf.placeholder(tf.float32, [2, 2], name='labels')

            # When
            cross_entropy = graph.add_loss_step(logits, labels)

            # Then
            self.assertIsNotNone(cross_entropy)
            self.assertEqual(type(cross_entropy).__name__, "Tensor")

    def test_run_loss(self):
        with tf.Graph().as_default():
            with tf.Session() as sess:
                # Given
                logits = tf.Variable([[np.exp(0.1), np.exp(0.9)], [np.exp(0.9), np.exp(0.1)]], dtype=tf.float32)
                labels = tf.Variable([[1.0, 0.0], [1.0, 0.0]], dtype=tf.float32)

                # When
                cross_entropy = graph.add_loss_step(logits, labels)

                # Then
                tf.initialize_all_variables().run()
                self.assertEqual(sess.run(cross_entropy), -0.25)


if __name__ == '__main__':
    tf.test.main()
