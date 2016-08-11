"""
Input functions for MNIST data.

1. input_reader() - Loads the MNIST data.
2. input_placeholders() - Creates input placeholders

This file is not meant to be run.
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


def input_reader(data_dir, one_hot=True):
    """
    Read the mnist dataset (downloads it if not present in the provided directory)

    :param data_dir: Directory in which the data is stored
    :param one_hot: Boolean specifying if the labels should be one-hot encoded or non
    :return: The mnist dataset
    """
    return input_data.read_data_sets(data_dir, one_hot=one_hot)


def input_placeholders(image_pixels, num_classes):
    """
    Initialises all input placeholders needed in the graph

    :param image_pixels: Total number of pixels in the image
    :param num_classes: Number of classes to predict
    :return: Input and labels placeholders
    """

    with tf.name_scope('input'):

        # Images input
        x = tf.placeholder(tf.float32, [None, image_pixels], name="x_input")

        # Labels input
        y_ = tf.placeholder(tf.float32, [None, num_classes])

    return x, y_
