"""
Builds a MNIST Neural Network.

Implements the inference/loss/training pattern for model building.

1. inference() - Builds the model as far as is required for running the network forward to make predictions.
2. loss() - Adds to the inference model the layers required to generate loss.
3. training() - Adds to the loss model the Ops required to generate and apply gradients.

This file is not meant to be run.
"""

import re
import tensorflow as tf

# If a model is trained with multiple GPUs, prefix all Op names with tower_name
# to differentiate the operations. Note that this prefix is removed from the
# names of the summaries when visualizing a model.
TOWER_NAME = 'tower'


def create_inference_step(x, num_pixels, num_classes):
    """
    Build the graph as far as is required for running the network forward to make predictions.

    :param x: Images placeholder, from inputs().
    :param num_pixels: Number of pixels in the original image
    :param num_classes: Number of classes to predict
    :return: softmax: Output tensor with the computed logits.
    """

    with tf.name_scope("softmax"):

        # Model parameters
        weights = "YOUR CODE HERE"
        biases = "YOUR CODE HERE"

        softmax = "YOUR CODE HERE"
        _activation_summary(softmax)

    return softmax


def add_loss_step(logits, labels):
    """
    Adds to the inference graph the ops required to generate loss (cross-entropy).

    :param logits: Logits tensor, float - [batch_size, NUM_CLASSES].
    :param labels: Labels tensor, int32 - [batch_size].
    :return: loss: Loss tensor of type float.
    """

    with tf.name_scope('cross_entropy'):
        diff = "YOUR CODE HERE"

        with tf.name_scope('total'):
            cross_entropy = "YOUR CODE HERE"

        tf.scalar_summary('cross entropy', cross_entropy)

    return cross_entropy


def add_train_step(loss, learning_rate):
    """
    * Sets up the training Ops.
    * Creates a summarizer to track the loss over time in TensorBoard.
    * Creates an optimizer and applies the gradients to all trainable variables.
    * The Op returned by this function is what must be passed to the `sess.run()` call to cause the model to train.

    :param loss: Loss tensor, from loss().
    :param learning_rate: Learning Rate.
    :return: train_op: The Op for training.
    """

    with tf.name_scope('train'):

        # Optimizer
        optimizer = "YOUR CODE HERE"

        # Create a variable to track the global step.
        global_step = tf.Variable(0, name='global_step', trainable=False)

        # Use the optimizer to apply the gradients that minimize the loss (and also increment the global step counter)
        # as a single training step.
        train_op = "YOUR CODE HERE"

    return train_op


def _activation_summary(x):
    """
    Helper to create summaries for activations.
    * Creates a summary that provides a histogram of activations.
    * Creates a summary that measures the sparsity of activations.

    :param x: Tensor
    """

    # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training session. This helps the clarity of
    # presentation on tensorboard.
    tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
    tf.histogram_summary(tensor_name + '/activations', x)
    tf.scalar_summary(tensor_name + '/sparsity', tf.nn.zero_fraction(x))

