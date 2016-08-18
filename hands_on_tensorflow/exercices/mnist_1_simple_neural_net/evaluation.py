"""
Evaluates a MNIST Convolutional Neural Network.

Implements the evaluate pattern for model evaluation.

1. evaluate() - Adds the evaluation Ops to evaluate the model accuracy.

This file is not meant to be run.
"""

import tensorflow as tf


def evaluate(logits, labels):
    """
    * Sets up the evaluation Ops.
    * Creates a summarizer to track the accuracy over time in TensorBoard.
    * The Op returned by this function is what must be passed to the `sess.run()` call to cause the model to be evaluated.

    :param logits: Logits tensor, from inference().
    :param labels: True labels.
    :return: accuracy: The Op for evaluating accuracy.
    """

    with tf.name_scope('accuracy'):
        with tf.name_scope('correct_prediction'):
            correct_prediction = "YOUR CODE HERE"

        with tf.name_scope('accuracy'):
            accuracy = "YOUR CODE HERE"

        tf.scalar_summary('accuracy', accuracy)

    return accuracy
