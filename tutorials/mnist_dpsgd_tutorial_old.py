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


# nohup python -u mnist_dpsgd_tutorial_old.py --gpu 0 --dpsgd Flase > nohup_mnist_dpsgd_tutorial_old_None_0614 2>&1  &
# nohup python -u mnist_dpsgd_tutorial_old.py --gpu 0 --dpsgd True --method sgd --batch_size 256 > nohup_mnist_dpsgd_tutorial_old_sgd_0614 2>&1  &
# nohup python -u mnist_dpsgd_tutorial_old.py --gpu 1 --dpsgd True --method adam --learning_rate 0.001 --noise_multiplier 0.1 --l2_norm_clip 1.0 --batch_size 256 > nohup_mnist_dpsgd_tutorial_old_adam_0.1_0614 2>&1  &
# nohup python -u mnist_dpsgd_tutorial_old.py --gpu 1 --dpsgd True --method adagrad --batch_size 256 > nohup_mnist_dpsgd_tutorial_old_adagrad_0614 2>&1  &

"""Training a CNN on MNIST with differentially private SGD optimizer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from time import strftime, localtime

import os
from absl import app
from tensorflow.python.platform import flags

from distutils.version import LooseVersion

import numpy as np
import tensorflow as tf

from privacy.analysis import privacy_ledger
from privacy.analysis.rdp_accountant import compute_rdp_from_ledger
from privacy.analysis.rdp_accountant import get_privacy_spent
from privacy.optimizers import dp_optimizer

from scipy.io import loadmat as loadmat
from six.moves import urllib
import tarfile
import gzip
import sys
import subprocess


if LooseVersion(tf.__version__) < LooseVersion('2.0.0'):
    GradientDescentOptimizer = tf.train.GradientDescentOptimizer
    AdamOptimizer = tf.train.AdamOptimizer
    AdagradOptimizer = tf.train.AdagradOptimizer
    MomentumOptimizer = tf.train.MomentumOptimizer
else:
    GradientDescentOptimizer = tf.optimizers.SGD  # pylint: disable=invalid-name

FLAGS = flags.FLAGS

flags.DEFINE_string(
    'dataset', "mnist", 'choose: mnist or cifar10 or svhn')

flags.DEFINE_string(
    'model', "deep", 'choose: trival or deep or lenet')

flags.DEFINE_boolean(
    'dpsgd', True, 'If True, train with DP-SGD. If False, '
                   'train with vanilla SGD.')
flags.DEFINE_boolean('use_nesterov', True, 'If True, train with use_nesterov.')
flags.DEFINE_string(
    'method', "sgd", 'opt method '
                     'train with dp SGD.')
flags.DEFINE_float(
    'momentum', 0.9, 'momentum'
                     'train with momentum.')

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
flags.DEFINE_string('gpu', '1', 'set gpu')

os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu

model_base_path = '/home/fanghb/dptestlog/'

params = {
    'dpsgd': FLAGS.dpsgd,
    'method': FLAGS.method,
    'learning_rate': FLAGS.learning_rate,
    'noise_multiplier': FLAGS.noise_multiplier,
    'l2_norm_clip': FLAGS.l2_norm_clip,
    'batch_size': FLAGS.batch_size,
    'epochs': FLAGS.epochs,
    'microbatches': FLAGS.microbatches,
    'model_dir': FLAGS.model_dir,
    'momentum': FLAGS.momentum,
    'gpu': FLAGS.gpu,
    'use_nesterov': FLAGS.use_nesterov

}

print(strftime("%Y-%m-%d %H:%M:%S", localtime()))
print(params)

# TODO 模型和tensorboard存放的位置
filewriter_path = model_base_path + str(FLAGS.dpsgd) + "_" + FLAGS.method + "/tensorboard"
checkpoint_path = model_base_path + str(FLAGS.dpsgd) + "_" + FLAGS.method + "/checkpoints"

# Create parent path if it doesn't exist
if not os.path.isdir(checkpoint_path):
    os.makedirs(checkpoint_path)
if not os.path.isdir(filewriter_path):
    os.makedirs(filewriter_path)

if FLAGS.dataset == "mnist":
    X = tf.placeholder(tf.float32, [FLAGS.batch_size, 28, 28])
elif FLAGS.dataset == "cifar10":
    X = tf.placeholder(tf.float32, [FLAGS.batch_size, 32, 32, 3])
elif FLAGS.dataset == "svhn":
    X = tf.placeholder(tf.float32, [FLAGS.batch_size, 32, 32, 3])

Y = tf.placeholder(tf.int64, [FLAGS.batch_size])


def trival(input_layer):
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
    return logits


def letnet(input_layer):
    pass


def alexnet(input_layer):
    pass


def deep(input_layer):
    conv1_1 = tf.layers.conv2d(input_layer,
                               32,  # 输出的通道数(也就是卷积核的数量)
                               (3, 3),  # 卷积核大小
                               padding='same',
                               activation=tf.nn.relu,
                               name='conv1_1'
                               )
    conv1_2 = tf.layers.conv2d(conv1_1,
                               32,  # 输出的通道数(也就是卷积核的数量)
                               (3, 3),  # 卷积核大小
                               padding='same',
                               activation=tf.nn.relu,
                               name='conv1_2'
                               )
    # 池化层 图像输出为: 16 * 16
    pooling1 = tf.layers.max_pooling2d(conv1_2,
                                       (2, 2),  # 核大小
                                       (2, 2),  # 步长
                                       name='pool1'
                                       )
    conv2_1 = tf.layers.conv2d(pooling1,
                               32,  # 输出的通道数
                               (3, 3),  # 卷积核大小
                               padding='same',
                               activation=tf.nn.relu,
                               name='conv2_1'
                               )
    conv2_2 = tf.layers.conv2d(conv2_1,
                               32,  # 输出的通道数
                               (3, 3),  # 卷积核大小
                               padding='same',
                               activation=tf.nn.relu,
                               name='conv2_2'
                               )
    # 池化层 图像输出为 8 * 8
    pooling2 = tf.layers.max_pooling2d(conv2_2,
                                       (2, 2),  # 核大小
                                       (2, 2),  # 步长
                                       name='pool2'
                                       )
    conv3_1 = tf.layers.conv2d(pooling2,
                               32,  # 输出的通道数
                               (3, 3),  # 卷积核大小
                               padding='same',
                               activation=tf.nn.relu,
                               name='conv3_1'
                               )
    conv3_2 = tf.layers.conv2d(conv3_1,
                               32,  # 输出的通道数
                               (3, 3),  # 卷积核大小
                               padding='same',
                               activation=tf.nn.relu,
                               name='conv3_2'
                               )
    # 池化层 输出为 4 * 4 * 32
    pooling3 = tf.layers.max_pooling2d(conv3_2,
                                       (2, 2),  # 核大小
                                       (2, 2),  # 步长
                                       name='pool3'
                                       )
    # 展平
    flatten = tf.contrib.layers.flatten(pooling3)
    logits = tf.layers.dense(flatten, 10)
    return logits


def cnn_model_fn(features, labels):
    """Model function for a CNN."""

    # Define CNN architecture using tf.keras.layers.
    if FLAGS.dataset == "mnist":
        input_layer = tf.reshape(features, [-1, 28, 28, 1])
    elif FLAGS.dataset == "cifar10":
        input_layer = features
        # input_layer = tf.reshape(features, [-1, 32, 32, 3])
    elif FLAGS.dataset == "svhn":
        input_layer = tf.reshape(features, [-1, 32, 32, 3])

    # y = tf.keras.layers.Conv2D(16, 8,
    #                            strides=2,
    #                            padding='same',
    #                            activation='relu').apply(input_layer)
    # y = tf.keras.layers.MaxPool2D(2, 1).apply(y)
    # y = tf.keras.layers.Conv2D(32, 4,
    #                            strides=2,
    #                            padding='valid',
    #                            activation='relu').apply(y)
    # y = tf.keras.layers.MaxPool2D(2, 1).apply(y)
    # y = tf.keras.layers.Flatten().apply(y)
    # y = tf.keras.layers.Dense(32, activation='relu').apply(y)

    if FLAGS.model == "trival":
        logits = trival(input_layer=input_layer)
    elif FLAGS.model == "deep":
        logits = deep(input_layer=input_layer)
        # input_layer = tf.reshape(features, [-1, 32, 32, 3])
    elif FLAGS.model == "letnet":
        logits = trival(input_layer=input_layer)

    # Calculate accuracy.
    correct_pred = tf.equal(tf.argmax(logits, 1), labels)
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
        if FLAGS.method == 'sgd':
            optimizer = dp_optimizer.DPGradientDescentGaussianOptimizer(
                l2_norm_clip=FLAGS.l2_norm_clip,
                noise_multiplier=FLAGS.noise_multiplier,
                num_microbatches=FLAGS.microbatches,
                ledger=ledger,
                learning_rate=FLAGS.learning_rate)
        elif FLAGS.method == 'adam':
            optimizer = dp_optimizer.DPAdamGaussianOptimizer(
                l2_norm_clip=FLAGS.l2_norm_clip,
                noise_multiplier=FLAGS.noise_multiplier,
                num_microbatches=FLAGS.microbatches,
                ledger=ledger,
                learning_rate=FLAGS.learning_rate,
                unroll_microbatches=True)
        elif FLAGS.method == 'adagrad':
            optimizer = dp_optimizer.DPAdagradGaussianOptimizer(
                l2_norm_clip=FLAGS.l2_norm_clip,
                noise_multiplier=FLAGS.noise_multiplier,
                num_microbatches=FLAGS.microbatches,
                ledger=ledger,
                learning_rate=FLAGS.learning_rate)
        elif FLAGS.method == 'momentum':
            optimizer = dp_optimizer.DPMomentumGaussianOptimizer(
                l2_norm_clip=FLAGS.l2_norm_clip,
                noise_multiplier=FLAGS.noise_multiplier,
                num_microbatches=FLAGS.microbatches,
                ledger=ledger,
                learning_rate=FLAGS.learning_rate,
                momentum=FLAGS.momentum,
                use_nesterov=FLAGS.use_nesterov
            )

        else:
            raise ValueError('method must be sgd or adam or adagrad or momentum')
        opt_loss = vector_loss
    else:
        if FLAGS.method == 'sgd':
            optimizer = GradientDescentOptimizer(learning_rate=FLAGS.learning_rate)
        elif FLAGS.method == 'adam':
            optimizer = AdamOptimizer(learning_rate=FLAGS.learning_rate)
        elif FLAGS.method == 'adagrad':
            optimizer = AdagradOptimizer(learning_rate=FLAGS.learning_rate)
        elif FLAGS.method == 'momentum':
            optimizer = MomentumOptimizer(learning_rate=FLAGS.learning_rate, momentum=FLAGS.momentum,
                                          use_nesterov=FLAGS.use_nesterov)
        else:
            raise ValueError('method must be sgd or adam or adagrad or momentum')
        opt_loss = scalar_loss
    global_step = tf.train.get_global_step()
    train_op = optimizer.minimize(loss=opt_loss, global_step=global_step)
    # In the following, we pass the mean of the loss (scalar_loss) rather than
    # the vector_loss because tf.estimator requires a scalar loss. This is only
    # used for evaluation and debugging by tf.estimator. The actual loss being
    # minimized is opt_loss defined above and passed to optimizer.minimize().
    return train_op, scalar_loss, accuracy


def load_mnist():
    """Loads MNIST and preprocesses to combine training and validation data."""
    train, test = tf.keras.datasets.mnist.load_data()
    train_data, train_labels = train
    test_data, test_labels = test

    train_data = np.array(train_data, dtype=np.float32) / 255
    test_data = np.array(test_data, dtype=np.float32) / 255

    train_labels = np.array(train_labels, dtype=np.int32)
    test_labels = np.array(test_labels, dtype=np.int32)

    print(train_labels.shape, test_data.shape)

    assert train_data.min() == 0.
    assert train_data.max() == 1.
    assert test_data.min() == 0.
    assert test_data.max() == 1.
    assert train_labels.ndim == 1
    assert test_labels.ndim == 1

    return train_data, train_labels, test_data, test_labels


def load_cifar10():
    """Loads MNIST and preprocesses to combine training and validation data."""
    train, test = tf.keras.datasets.cifar10.load_data()
    train_data, train_labels = train
    test_data, test_labels = test

    train_data = np.array(train_data, dtype=np.float32) / 255
    test_data = np.array(test_data, dtype=np.float32) / 255

    train_labels = np.array(train_labels, dtype=np.int32)
    test_labels = np.array(test_labels, dtype=np.int32)

    train_labels = np.reshape(train_labels, (-1,))
    test_labels = np.reshape(test_labels, (-1,))

    print(train_labels.shape, test_data.shape)

    assert train_data.min() == 0.
    assert train_data.max() == 1.
    assert test_data.min() == 0.
    assert test_data.max() == 1.
    assert train_labels.ndim == 1
    assert test_labels.ndim == 1

    return train_data, train_labels, test_data, test_labels








# def load_svhn(one_hot = False):
#     image_size = 32
#     num_labels = 10
#     import scipy.io as sio
#     train = sio.loadmat('train_32x32.mat')
#     test = sio.loadmat('test_32x32.mat')
#
#     train_data = train['X']
#     train_label = train['y']
#     test_data = test['X']
#     test_label = test['y']
#
#     train_data = np.swapaxes(train_data, 0, 3)
#     train_data = np.swapaxes(train_data, 2, 3)
#     train_data = np.swapaxes(train_data, 1, 2)
#     test_data = np.swapaxes(test_data, 0, 3)
#     test_data = np.swapaxes(test_data, 2, 3)
#     test_data = np.swapaxes(test_data, 1, 2)
#
#     test_data = test_data / 255.
#     train_data = train_data / 255.
#
#     for i in range(train_label.shape[0]):
#         if train_label[i][0] == 10:
#             train_label[i][0] = 0
#
#     for i in range(test_label.shape[0]):
#         if test_label[i][0] == 10:
#             test_label[i][0] = 0
#
#     if one_hot:
#         train_labels = (np.arange(num_labels) == train_label[:, ]).astype(np.float32)
#         test_labels = (np.arange(num_labels) == test_label[:, ]).astype(np.float32)
#
#     return train_data, train_labels, test_data, test_labels


def load_svhn():
    from scipy.io import loadmat as load
    train = load('train_32x32.mat')
    test = load('test_32x32.mat')
    train_data = train['X']
    train_labels = train['y']
    test_data = test['X']
    test_labels = test['y']

    train_data = np.array(train_data, dtype=np.float32) / 255
    test_data = np.array(test_data, dtype=np.float32) / 255

    train_labels = np.array(train_labels, dtype=np.int32)
    test_labels = np.array(test_labels, dtype=np.int32)

    train_labels = np.reshape(train_labels, (-1,))
    test_labels = np.reshape(test_labels, (-1,))

    print(train_labels.shape, test_data.shape)

    # assert train_data.min() == 0.
    # assert train_data.max() == 1.
    # assert test_data.min() == 0.
    # assert test_data.max() == 1.
    # assert train_labels.ndim == 1
    # assert test_labels.ndim == 1

    return train_data, train_labels, test_data, test_labels




def generate_next_batch(data, label, batch_size, shffule=False):
    print(type(data), data.shape, label.shape, batch_size, "---", data.shape[0], data.shape[1], data.shape[2])
    if shffule:
        permutation = np.random.permutation(data.shape[0])
        data = data[permutation]
        label = label[permutation]
    size = data.shape[0] // batch_size
    print(batch_size, "()()", size, "()", data.shape[0])
    for i in range(size):
        yield (data[i * batch_size: (i + 1) * batch_size], label[i * batch_size: (i + 1) * batch_size])


train_op, opt_loss, opt_accuracy = cnn_model_fn (X, Y)

tf.summary.scalar('loss', opt_loss)
tf.summary.scalar('accuracy', opt_accuracy)

if FLAGS.dataset == "mnist":
    train_data, train_labels, test_data, test_labels = load_mnist()
elif FLAGS.dataset == "cifar10":
    train_data, train_labels, test_data, test_labels = load_cifar10()
elif FLAGS.dataset == "svhn":
    train_data, train_labels, test_data, test_labels =load_svhn()


def main(unused_argv):
    merged = tf.summary.merge_all()
    # Initialize the FileWriter
    writer = tf.summary.FileWriter(filewriter_path)

    # Initialize an saver for store model checkpoints
    saver = tf.train.Saver(var_list=tf.trainable_variables())

    gpu_options = tf.GPUOptions(allow_growth=True)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        sess.run(tf.global_variables_initializer())
        tf.logging.set_verbosity(tf.logging.INFO)
        if FLAGS.dpsgd and FLAGS.batch_size % FLAGS.microbatches != 0:
            raise ValueError('Number of microbatches should divide evenly batch_size')

        # Load training and test data.
        # print(train_data.shape, train_labels.shape)
        # Training loop.
        for epoch in range(1, FLAGS.epochs + 1):

            print("At epcho %d" % epoch)

            # Train the model for one epoch.
            gen = generate_next_batch(data=train_data, label=train_labels, batch_size=FLAGS.batch_size, shffule=True)
            train_acc = 0.
            train_loss = 0.
            train_count = 0
            for img_batch, label_batch in gen:
                _, loss, acc = sess.run([train_op, opt_loss, opt_accuracy], feed_dict={X: img_batch,
                                                                                       Y: label_batch})
                train_acc += acc
                train_loss += loss
                train_count += 1
            train_acc /= train_count
            train_loss /= train_count
            print('Train accuracy: {}, Train loss : {},  after {} epochs'.format(train_acc, train_loss, epoch))

            gen = generate_next_batch(data=test_data, label=test_labels, batch_size=FLAGS.batch_size, shffule=False)
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
            print('Test accuracy: {}, Test loss : {},  after {} epochs'.format(test_acc, test_loss, epoch))


if __name__ == '__main__':
    app.run(main)
