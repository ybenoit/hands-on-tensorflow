"""Inference Testing"""

import tensorflow as tf

from hands_on_tensorflow.exercices.mnist_2_deep_neural_net import graph


class GraphInferenceTest(tf.test.TestCase):
    """
    Test the creation of the inference step in the graph
    """

    def test_add_inference_step(self):
        with tf.Graph().as_default():
            # Given
            x = tf.placeholder(tf.float32, [2, 2], name='x_input')

            # When
            softmax = graph.create_inference_step(x, num_pixels=2, num_dense1=4, num_dense2=3, num_dense3=2,
                                                  num_classes=2)

            # Then
            self.assertIsNotNone(softmax)
            self.assertEqual(type(softmax).__name__, "Tensor")

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
                self.assertAllClose(sess.run(softmax), [[0.5, 0.5], [0.5, 0.5]], rtol=1e-2, atol=1e-2)

if __name__ == '__main__':
    tf.test.main()
