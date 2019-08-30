from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from transformer.model.transformer import LayerNormalization


class CNNNetwork(tf.layers.Layer):
    """ CNN network """

    def __init__(self, hidden_size, progression, kernel_size, dropout, train):
        super(CNNNetwork, self).__init__()
        self.filter_double_conv_layer = [
            DoubleConv2DLayer(int((hidden_size) * (progression ** n)), # Number of filter for 1st Conv2D
                              int((hidden_size) * (progression ** n)), # Number of filter for 2nd Conv2D
                              kernel_size, # Kernel Size # Hidden size for apply LayerNormalization to entry
                              dropout, #Dropout
                              train) for n in range(4)
        ]

        self.filter_max_pool_layer = [
            tf.layers.MaxPooling2D(
                2, 2, name="max_layer") for n in range(3)
        ]

        self.filter_max_pool_layer.append(tf.layers.MaxPooling2D(
            [3, 1], [2, 1], name="max_last_layer"))

    def call(self, x):
        y = x

        for i in range(4):
            y = self.filter_double_conv_layer[i](y)
            y = self.filter_max_pool_layer[i](y)
        output = tf.unstack(y,axis=1)
        return output


class DoubleConv2DLayer(tf.layers.Layer):

    def __init__(self, size_1, size_2, kernel_size, dropout, train):
        # Create 3 Conv2D, 2 + 1 for Residual Connection
        super(DoubleConv2DLayer, self).__init__()
        self.layer_1 = tf.layers.Conv2D(
            size_1, kernel_size, padding='same',
            activation=tf.nn.relu)

        self.layer_2 = tf.layers.Conv2D(
            size_2, kernel_size, padding='same',
            activation=tf.nn.relu)

        self.residual = tf.layers.Conv2D(
            size_2, 1, padding='same',
            activation=None)

        self.dropout = dropout
        self.train = train

        # Create 2 normalization layer
        self.layer_norm = LayerNormalization(size_1)
        self.layer_norm_2 = LayerNormalization(size_2)

    def call(self, x):
        #Apply Layer norm + Conv2D
        y1 = self.layer_1(x)
        y1 = self.layer_norm(y1)

        y2 = self.layer_2(y1)
        y3 = self.residual(x)

        # Postprocessing: apply dropout and residual connection
        if self.train:
            y2 = tf.nn.dropout(y2, 1 - self.dropout)

        y = y2 + y3

        return self.layer_norm_2(y)

"""
inputs = tf.ones([1, 80, 300, 1])

encoder = CNNNetwork(16, 2, [3, 3], 0.2, True)

test = encoder(inputs)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    print(sess.run(test))
    print(test[0].shape)
    print(len(test))"""