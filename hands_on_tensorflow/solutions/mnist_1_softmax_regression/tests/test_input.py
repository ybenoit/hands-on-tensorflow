"""Input Testing"""

import tensorflow as tf

from hands_on_tensorflow.solutions.mnist_1_softmax_regression import input


class InputTest(tf.test.TestCase):
    """
    Test the evaluation part of the graph
    """

    def test_input_placeholders(self):
        with tf.Graph().as_default():
            # Given
            image_pixels = 100
            num_classes = 10

            # When
            x, y_ = input.input_placeholders(image_pixels, num_classes)

            # Then
            self.assertIsNotNone(x)
            self.assertEqual(type(x).__name__, "Tensor")
            self.assertEqual(x.__dict__["_shape"].as_list(), [None, 100])
            self.assertIsNotNone(y_)
            self.assertEqual(type(y_).__name__, "Tensor")
            self.assertEqual(y_.__dict__["_shape"].as_list(), [None, 10])

if __name__ == '__main__':
    tf.test.main()
