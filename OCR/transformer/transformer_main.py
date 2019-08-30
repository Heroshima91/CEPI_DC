# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Train and evaluate the Transformer model.

See README for description of setting the training schedule and evaluating the
BLEU score.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tempfile

# pylint: disable=g-bad-import-order
from six.moves import xrange  # pylint: disable=redefined-builtin
from absl import app as absl_app
from absl import flags
import tensorflow as tf
# pylint: enable=g-bad-import-order

from transformer.model import model_params
from transformer.model import transformer
from transformer.utils import metrics
from transformer.model import cnn_layer
from transformer.utils import tokenizer

PARAMS_MAP = {
    "tiny": model_params.TINY_PARAMS,
    "base": model_params.BASE_PARAMS,
    "big": model_params.BIG_PARAMS,
}

DEFAULT_TRAIN_EPOCHS = 10
INF = int(1e9)
BLEU_DIR = "bleu"

# Dictionary containing tensors that are logged by the logging hooks. Each item
# maps a string to the tensor name.
TENSORS_TO_LOG = {
    "learning_rate": "model/get_train_op/learning_rate/learning_rate",
    "cross_entropy_loss": "model/cross_entropy"}


def model_fn(features, labels, mode, params):
  """Defines how to train, evaluate and predict from the transformer model.
    labels: liste de 4 tensors shape [batch_size, seq_len, 1], un tenseur pour chaque ligne
  """
  with tf.variable_scope("model"):
    inputs, targets = features, labels



    # Create model and get output logits.
    conv = cnn_layer.CNNNetwork(4, 2, [3,3], 0.5, mode == tf.estimator.ModeKeys.TRAIN)
    model = transformer.Transformer(params, mode == tf.estimator.ModeKeys.TRAIN)

    conv_output = conv(inputs)

    logits = model(conv_output[0],targets)

    # When in prediction mode, the labels/targets is None. The model output
    # is the prediction
    if mode == tf.estimator.ModeKeys.PREDICT:
      if params["use_tpu"]:
        raise NotImplementedError("Prediction is not yet supported on TPUs.")
      return tf.estimator.EstimatorSpec(
          tf.estimator.ModeKeys.PREDICT,
          predictions=logits,
          #export_outputs={
              #"translate": tf.estimator.export.PredictOutput(logits)
          #})
      )

    # Explicitly set the shape of the logits for XLA (TPU). This is needed
    # because the logits are passed back to the host VM CPU for metric
    # evaluation, and the shape of [?, ?, vocab_size] is too vague. However
    # it is known from Transformer that the first two dimensions of logits
    # are the dimensions of targets. Note that the ambiguous shape of logits is
    # not a problem when computing xentropy, because padded_cross_entropy_loss
    # resolves the shape on the TPU.
    logits.set_shape(targets.shape.as_list() + logits.shape.as_list()[2:])

    # Calculate model loss.
    # xentropy contains the cross entropy loss of every nonpadding token in the
    # targets.

    xentropy, weights = metrics.padded_cross_entropy_loss(
        logits, targets, params["label_smoothing"], params["vocab_size"])
    loss = tf.reduce_sum(xentropy) / tf.reduce_sum(weights)

    if mode == tf.estimator.ModeKeys.EVAL:
      if params["use_tpu"]:
        # host call functions should only have tensors as arguments.
        # This lambda pre-populates params so that metric_fn is
        # TPUEstimator compliant.
        raise NotImplementedError("Prediction is not yet supported on TPUs.")
      return tf.estimator.EstimatorSpec(
          mode=mode, loss=loss, predictions={"predictions": logits},
          eval_metric_ops=metrics.get_eval_metrics(logits, labels, params))
    else:
      train_op, metric_dict = get_train_op_and_metrics(loss, params)

      record_scalars(metric_dict)
      return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)


def record_scalars(metric_dict):
  for key, value in metric_dict.items():
    tf.contrib.summary.scalar(name=key, tensor=value)


def get_learning_rate(learning_rate, hidden_size, learning_rate_warmup_steps):
  """Calculate learning rate with linear warmup and rsqrt decay."""
  with tf.name_scope("learning_rate"):
    warmup_steps = tf.to_float(learning_rate_warmup_steps)
    step = tf.to_float(tf.train.get_or_create_global_step())

    learning_rate *= (hidden_size ** -0.5)
    # Apply linear warmup
    learning_rate *= tf.minimum(1.0, step / warmup_steps)
    # Apply rsqrt decay
    learning_rate *= tf.rsqrt(tf.maximum(step, warmup_steps))

    # Create a named tensor that will be logged using the logging hook.
    # The full name includes variable and names scope. In this case, the name
    # is model/get_train_op/learning_rate/learning_rate
    tf.identity(learning_rate, "learning_rate")

    return learning_rate


def get_train_op_and_metrics(loss, params):
  """Generate training op and metrics to save in TensorBoard."""
  with tf.variable_scope("get_train_op"):
    learning_rate = get_learning_rate(
        learning_rate=params["learning_rate"],
        hidden_size=params["hidden_size"],
        learning_rate_warmup_steps=params["learning_rate_warmup_steps"])

    # Create optimizer. Use LazyAdamOptimizer from TF contrib, which is faster
    # than the TF core Adam optimizer.
    optimizer = tf.contrib.estimator.TowerOptimizer(tf.contrib.opt.LazyAdamOptimizer(
        learning_rate,
        beta1=params["optimizer_adam_beta1"],
        beta2=params["optimizer_adam_beta2"],
        epsilon=params["optimizer_adam_epsilon"]))

    # Calculate and apply gradients using LazyAdamOptimizer.
    global_step = tf.train.get_global_step()
    tvars = tf.trainable_variables()
    gradients = optimizer.compute_gradients(
        loss, tvars, colocate_gradients_with_ops=True)
    minimize_op = optimizer.apply_gradients(
        gradients, global_step=global_step, name="train")
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    train_op = tf.group(minimize_op, update_ops)

    train_metrics = {"learning_rate": learning_rate}

    if not params["use_tpu"]:
      # gradient norm is not included as a summary when running on TPU, as
      # it can cause instability between the TPU and the host controller.
      gradient_norm = tf.global_norm(list(zip(*gradients))[0])
      train_metrics["global_norm/gradient_norm"] = gradient_norm

    return train_op, train_metrics
