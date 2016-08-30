"""
Trains a MNIST Neural Network.

Reports summaries in TensorBoard
"""

import os
import shutil

import tensorflow as tf

from hands_on_tensorflow.solutions.mnist_1_softmax_regression import evaluation, input, graph

"""
Global variables
"""
# The MNIST images are always 28x28 pixels.
IMAGE_SIZE = 28
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE

# The MNIST dataset has 10 classes, representing the digits 0 through 9.
NUM_CLASSES = 10

"""
TensorFlow flags
"""
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('max_steps', 2000, 'Number of steps to run trainer.')
flags.DEFINE_integer('batch_size', 100, 'Train batch size.')
flags.DEFINE_float('learning_rate', 0.1, 'Initial learning rate.')
flags.DEFINE_string('data_dir',  os.path.join(os.path.dirname(__file__), "../../data/MNIST_data/"), 'Data Directory')
flags.DEFINE_string('summaries_dir', '/tmp/mnist_dense_logs', 'Summaries directory')


def main():

    # Create an InteractiveSession
    sess = tf.InteractiveSession()

    # Remove tensorboard previous directory
    if os.path.exists(FLAGS.summaries_dir):
        shutil.rmtree(FLAGS.summaries_dir)

    """
    Step 1 - Input data management
    """

    # MNIST data
    mnist = input.input_reader(FLAGS.data_dir)

    # Input placeholders
    x, y_ = input.input_placeholders(IMAGE_PIXELS, NUM_CLASSES)

    # Reshape images for visualization
    x_reshaped = tf.reshape(x, [-1, IMAGE_SIZE, IMAGE_SIZE, 1])
    tf.image_summary('input', x_reshaped, NUM_CLASSES, name="y_input")

    """
    Step 2 - Building the graph
    """

    # Inference
    softmax = graph.create_inference_step(x=x, num_pixels=IMAGE_PIXELS, num_classes=NUM_CLASSES)

    # Loss
    cross_entropy = graph.add_loss_step(softmax, y_)

    # Train step
    train_step = graph.add_train_step(cross_entropy, FLAGS.learning_rate)

    """
    Step 3 - Build the evaluation step
    """

    # Model Evaluation
    accuracy = evaluation.evaluate(softmax, y_)

    """
    Step 4 - Merge all summaries for TensorBoard generation
    """

    # Merge all the summaries and write them out to /tmp/mnist_dense_logs (by default)
    merged = tf.merge_all_summaries()
    train_writer = tf.train.SummaryWriter(FLAGS.summaries_dir + '/train', sess.graph)
    validation_writer = tf.train.SummaryWriter(FLAGS.summaries_dir + '/validation')

    """
    Step 5 - Train the model, and write summaries
    """

    # Initialize all variables
    tf.initialize_all_variables().run()

    # All other steps, run train_step on training data, & add training summaries
    for i in range(FLAGS.max_steps):

        # Load next batch of data
        x_batch, y_batch = mnist.train.next_batch(FLAGS.batch_size)

        # Run summaries and train_step
        summary, _ = sess.run([merged, train_step], feed_dict={x: x_batch, y_: y_batch})

        # Add summaries to train writer
        train_writer.add_summary(summary, i)

        # Every 10th step, measure validation-set accuracy, and write validation summaries
        if i % 10 == 0:
            # Run summaries and mesure accuracy on validation set
            summary, acc_valid = sess.run([merged, accuracy],
                                          feed_dict={x: mnist.validation.images, y_: mnist.validation.labels})

            # Add summaries to validation writer
            validation_writer.add_summary(summary, i)

            print('Validation Accuracy at step %s: %s' % (i, acc_valid))

    # Measure accuracy on test set
    acc_test = sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})
    print('Accuracy on test set: %s' % acc_test)


if __name__ == '__main__':
    main()
