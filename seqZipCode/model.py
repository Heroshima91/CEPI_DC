# Some code was borrowed from https://github.com/petewarden/tensorflow_makefile/blob/master/tensorflow/models/image/mnist/convolutional.py
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim

BATCH_SIZE = 100
SEQ_LENGTH = 11
tf.logging.set_verbosity(tf.logging.INFO)

# Create model of CNN with slim api
def inference(inputs, params, is_training=True):
    batch_norm_params = {'is_training': is_training, 'decay': 0.9, 'updates_collections': None}
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        normalizer_fn=slim.batch_norm,
                        normalizer_params=batch_norm_params,
                        ):
        x = tf.reshape(inputs, [-1, 28, 100, 1])
        net = slim.conv2d(x, params['output_size'][0], params['kernel_size'], scope='conv1')
        net = slim.max_pool2d(net, [2, 2], scope='pool1')
        net = slim.conv2d(net, params['output_size'][1], params['kernel_size'], scope='conv2')
        net = slim.max_pool2d(net, [2, 2], scope='pool2')
        net = slim.conv2d(net, params['output_size'][2], params['kernel_size'], scope='conv3')
        net = slim.max_pool2d(net, [7, 2], scope='pool3')
        net = tf.squeeze(net, axis=1)
        net = slim.fully_connected(net, params['output_size'][3], scope='fc1')
        net = slim.dropout(net, params['drop_out'])
        net = slim.fully_connected(net, params['output_size'][4], scope='fc2')
        net = slim.dropout(net, params['drop_out'])
        outputs = slim.fully_connected(net, 11, activation_fn=None, normalizer_fn=None, scope='fco')
        return outputs

"""
    get_loss computes the CTC loss for the model
"""
def get_loss(logits, sequence, sequence_length):
    losses = tf.nn.ctc_loss(labels=sequence,inputs=logits,sequence_length=sequence_length, time_major=True)
    losses = tf.reduce_mean(losses)
    tf.summary.scalar('ctc_loss', losses)
    return losses

"""
    get_train_op computes the AdamOptimizer with the learning rate and minimize the loss function
"""
def get_train_op(loss,learning_rate):
    global_step = tf.train.get_or_create_global_step()
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    return optimizer.minimize(loss, global_step=global_step)


def model_fn(features, labels, params, mode):
    logits = inference(features, params, is_training=(mode == tf.estimator.ModeKeys.TRAIN))
    logits = tf.transpose(logits, [1, 0, 2])
    sequence_length = tf.ones([BATCH_SIZE], dtype=tf.dtypes.int32) * SEQ_LENGTH

    if mode == tf.estimator.ModeKeys.PREDICT:
        decoded, log_probabilities = tf.nn.ctc_beam_search_decoder(
            logits,
            sequence_length,
            top_paths=1,
            beam_width=1,
            merge_repeated=False
        )
        prediction = tf.sparse_tensor_to_dense(decoded[0], default_value=-1)
        print(prediction)
        return tf.estimator.EstimatorSpec(mode=mode, predictions=prediction)

    else:
        labels = tf.string_split(labels, delimiter="").values
        labels = tf.string_to_number(labels, tf.int32)
        labels = tf.reshape(labels, [BATCH_SIZE, 5])
        zero = tf.constant(-1, dtype=tf.int32)
        idx = tf.where(tf.not_equal(labels, zero))
        sequence = tf.SparseTensor(idx, tf.gather_nd(labels, idx), labels.get_shape())

        loss = get_loss(logits=logits, sequence=sequence, sequence_length=sequence_length)

        if mode == tf.estimator.ModeKeys.TRAIN:
            # get loss and training op
            train_op = get_train_op(loss,params['learning_rate'])
            return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

        if mode == tf.estimator.ModeKeys.EVAL:
            decoded, log_probabilities = tf.nn.ctc_beam_search_decoder(
                logits,
                sequence_length,
                top_paths=1,
                beam_width=1,
                merge_repeated=False
            )
            cast = tf.cast(decoded[0], tf.int32)
            prediction = tf.sparse_tensor_to_dense(cast, default_value=-1)
            prediction = tf.concat(prediction,axis=1)
            paddings = tf.constant([[0, 0, ], [0, 5]])
            prediction = tf.pad(prediction, paddings, "CONSTANT",constant_values=-1)
            prediction = prediction[:, :5]
            eval_metrics = {
                'accuracy': tf.metrics.accuracy(
                    tf.sparse_tensor_to_dense(sequence),
                    prediction,
                    name='accuracy'
                )
            }
            return tf.estimator.EstimatorSpec(mode, predictions=prediction, loss=loss, eval_metric_ops=eval_metrics)