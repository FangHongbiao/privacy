# Copyright 2018, The TensorFlow Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Training a CNN on MNIST with differentially private SGD optimizer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags

from distutils.version import LooseVersion

import numpy as np
import tensorflow as tf

from privacy.analysis import privacy_ledger
from privacy.analysis.rdp_accountant import compute_rdp_from_ledger
from privacy.analysis.rdp_accountant import get_privacy_spent
from privacy.optimizers import dp_optimizer

if LooseVersion(tf.__version__) < LooseVersion('2.0.0'):
    GradientDescentOptimizer = tf.train.GradientDescentOptimizer
else:
    GradientDescentOptimizer = tf.optimizers.SGD  # pylint: disable=invalid-name

FLAGS = flags.FLAGS

flags.DEFINE_boolean(
    'dpsgd', True, 'If True, train with DP-SGD. If False, '
                   'train with vanilla SGD.')
flags.DEFINE_float('learning_rate', .15, 'Learning rate for training')
flags.DEFINE_float('noise_multiplier', 1.1,
                   'Ratio of the standard deviation to the clipping norm')
flags.DEFINE_float('l2_norm_clip', 1.0, 'Clipping norm')
flags.DEFINE_integer('batch_size', 256, 'Batch size')
flags.DEFINE_integer('epochs', 60, 'Number of epochs')
flags.DEFINE_integer(
    'microbatches', 256, 'Number of microbatches '
                         '(must evenly divide batch_size)')
flags.DEFINE_string('model_dir', None, 'Model directory')

num_classes = 10
X = tf.placeholder(tf.float32, [FLAGS.batch_size, 227, 227, 3])
Y = tf.placeholder(tf.float32, [FLAGS.batch_size, num_classes])


def cnn_model_fn(features, labels, mode):
    """Model function for a CNN."""

    # Define CNN architecture using tf.keras.layers.
    input_layer = tf.reshape(features['x'], [-1, 28, 28, 1])
    y = tf.keras.layers.Conv2D(16, 8,
                               strides=2,
                               padding='same',
                               activation='relu').apply(input_layer)
    y = tf.keras.layers.MaxPool2D(2, 1).apply(y)
    y = tf.keras.layers.Conv2D(32, 4,
                               strides=2,
                               padding='valid',
                               activation='relu').apply(y)
    y = tf.keras.layers.MaxPool2D(2, 1).apply(y)
    y = tf.keras.layers.Flatten().apply(y)
    y = tf.keras.layers.Dense(32, activation='relu').apply(y)
    logits = tf.keras.layers.Dense(10).apply(y)

    # Calculate accuracy.
    correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # Calculate loss as a vector (to support microbatches in DP-SGD).
    vector_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels, logits=logits)
    # Define mean of loss across minibatch (for reporting through tf.Estimator).
    scalar_loss = tf.reduce_mean(vector_loss)

    if FLAGS.dpsgd:
        ledger = privacy_ledger.PrivacyLedger(
            population_size=60000,
            selection_probability=(FLAGS.batch_size / 60000))

        # Use DP version of GradientDescentOptimizer. Other optimizers are
        # available in dp_optimizer. Most optimizers inheriting from
        # tf.train.Optimizer should be wrappable in differentially private
        # counterparts by calling dp_optimizer.optimizer_from_args().
        optimizer = dp_optimizer.DPGradientDescentGaussianOptimizer(
            l2_norm_clip=FLAGS.l2_norm_clip,
            noise_multiplier=FLAGS.noise_multiplier,
            num_microbatches=FLAGS.microbatches,
            ledger=ledger,
            learning_rate=FLAGS.learning_rate)
        opt_loss = vector_loss
    else:
        optimizer = GradientDescentOptimizer(learning_rate=FLAGS.learning_rate)

        opt_loss = scalar_loss
    global_step = tf.train.get_global_step()
    train_op = optimizer.minimize(loss=opt_loss, global_step=global_step)
    # In the following, we pass the mean of the loss (scalar_loss) rather than
    # the vector_loss because tf.estimator requires a scalar loss. This is only
    # used for evaluation and debugging by tf.estimator. The actual loss being
    # minimized is opt_loss defined above and passed to optimizer.minimize().
    return train_op, opt_loss, accuracy


def load_mnist():
    """Loads MNIST and preprocesses to combine training and validation data."""
    train, test = tf.keras.datasets.mnist.load_data()
    train_data, train_labels = train
    test_data, test_labels = test

    train_data = np.array(train_data, dtype=np.float32) / 255
    test_data = np.array(test_data, dtype=np.float32) / 255

    train_labels = np.array(train_labels, dtype=np.int32)
    test_labels = np.array(test_labels, dtype=np.int32)

    assert train_data.min() == 0.
    assert train_data.max() == 1.
    assert test_data.min() == 0.
    assert test_data.max() == 1.
    assert train_labels.ndim == 1
    assert test_labels.ndim == 1

    return train_data, train_labels, test_data, test_labels


def generate_next_batch(data, label, batch_size, shffule=False):
    if shffule:
        permutation = np.random.permutation(batch_size)
        data = data[permutation]
        label = label[permutation]
    size = len(data) // batch_size
    for i in range(size):
        yield (data[i * batch_size: (i + 1) * batch_size], label[i * batch_size: (i + 1) * batch_size])


train_op, opt_loss, opt_accuracy = cnn_model_fn(X, Y, None)


def main(unused_argv):
    gpu_options = tf.GPUOptions(allow_growth=True)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        sess.run(tf.global_variables_initializer())
        tf.logging.set_verbosity(tf.logging.INFO)
        if FLAGS.dpsgd and FLAGS.batch_size % FLAGS.microbatches != 0:
            raise ValueError('Number of microbatches should divide evenly batch_size')

        # Load training and test data.
        train_data, train_labels, test_data, test_labels = load_mnist()

        # Training loop.
        steps_per_epoch = 60000 // FLAGS.batch_size
        for epoch in range(1, FLAGS.epochs + 1):

            print("At epcho %d" % epoch)

            # Train the model for one epoch.
            gen = generate_next_batch(data=train_data, label=train_labels, shffule=True)
            for img_batch, label_batch in gen:
                sess.run(train_op, feed_dict={X: img_batch,
                                              Y: label_batch})

            gen = generate_next_batch(data=test_data, label=test_labels, shffule=False)
            test_acc = 0.
            test_loss = 0.
            test_count = 0
            for img_batch, label_batch in gen:
                loss, acc = sess.run([opt_loss, opt_accuracy], feed_dict={X: img_batch,
                                                                          Y: label_batch})
                test_acc += acc
                test_loss += loss
                test_count += 1
            test_acc /= test_count
            test_loss /= test_count
            print('Test accuracy: %.3f, Test loss : %.3f,  after %d epochs' % (test_acc, test_loss, epoch))


if __name__ == '__main__':
    app.run(main)
