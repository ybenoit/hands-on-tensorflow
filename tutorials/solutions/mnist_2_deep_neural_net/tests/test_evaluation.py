"""Tests the graph freezing tool."""
#from __future__ import absolute_import
#from __future__ import division
#from __future__ import print_function

import tensorflow as tf

from tutorials.solutions.mnist_2_deep_neural_net import evaluation


class EvaluationTest(tf.test.TestCase):
    """
    Test the evaluation part of the graph
    """

    def test_add_evaluation_step(self):
        with tf.Graph().as_default():
            final = tf.placeholder(tf.float32, [1], name='final')
            gt = tf.placeholder(tf.float32, [1], name='gt')

            self.assertIsNotNone(evaluation.evaluate(final, gt))

    def test_run_evaluation(self):
        with tf.Graph().as_default():
            with tf.Session() as sess:
                # Input variables
                final = tf.Variable([[0.1, 0.9], [0.1, 0.9]], dtype=tf.float32)
                gt = tf.Variable([[0.6, 0.4], [0.2, 0.8]], dtype=tf.float32)

                # Accuracy
                accuracy = evaluation.evaluate(final, gt)

                # Evaluate results
                tf.initialize_all_variables().run()
                self.assertAllClose(sess.run(accuracy), 0.5)


if __name__ == '__main__':
    tf.test.main()
