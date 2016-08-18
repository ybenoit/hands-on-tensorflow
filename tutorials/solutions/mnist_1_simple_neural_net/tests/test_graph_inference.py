"""Inference Testing"""

import tensorflow as tf

from tutorials.solutions.mnist_1_simple_neural_net import graph


class GraphInferenceTest(tf.test.TestCase):
    """
    Test the creation of the inference step in the graph
    """

    def test_add_inference_step(self):
        with tf.Graph().as_default():
            # Given
            x = tf.placeholder(tf.float32, [2, 2], name='x_input')

            # When
            softmax = graph.create_inference_step(x, num_pixels=2, num_classes=2)

            # Then
            self.assertIsNotNone(softmax)
            self.assertEqual(type(softmax).__name__, "Tensor")

    def test_run_inference(self):
        with tf.Graph().as_default():
            with tf.Session() as sess:
                # Given
                x = tf.Variable([[1.0, 1.0], [1.0, 1.0]], dtype=tf.float32)

                # When
                softmax = graph.create_inference_step(x, num_pixels=2, num_classes=2)
                tf.initialize_all_variables().run()

                # Then
                self.assertAllClose(sess.run(softmax), [[0.5, 0.5], [0.5, 0.5]])


if __name__ == '__main__':
    tf.test.main()
