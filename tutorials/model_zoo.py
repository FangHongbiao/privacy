import tensorflow as tf


def trival(input_layer, y):
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
