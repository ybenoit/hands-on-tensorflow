"""Train Testing"""

import tensorflow as tf

from hands_on_tensorflow.solutions.mnist_2_simple_neural_net import graph


class GraphTrainTest(tf.test.TestCase):
    """
    Test the addition of the training step in the graph
    """

    def test_add_train_step(self):
        with tf.Graph().as_default():
            # Given
            loss = tf.Variable([1.0], dtype=tf.float32)

            # When
            train_op = graph.add_train_step(loss, 0.1)

            # Then
            self.assertIsNotNone(train_op)
            self.assertEqual(type(train_op).__name__, "Operation")

    def test_run_train(self):
        with tf.Graph().as_default():
            with self.test_session():
                # Given
                var0 = tf.Variable([1.0, 2.0], dtype=tf.float32)
                var1 = tf.Variable([3.0, 4.0], dtype=tf.float32)
                cost = 5 * var0 + 3 * var1

                # When
                train_op = graph.add_train_step(cost, 3.0)
                tf.initialize_all_variables().run()

                # Then
                # Fetch params to validate initial values
                self.assertAllClose([1.0, 2.0], var0.eval())
                self.assertAllClose([3.0, 4.0], var1.eval())

                # Run one training step
                train_op.run()

                # Validate updated params
                self.assertAllClose([-14., -13.], var0.eval())
                self.assertAllClose([-6., -5.], var1.eval())


if __name__ == '__main__':
    tf.test.main()
