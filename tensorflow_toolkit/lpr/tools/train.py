#!/usr/bin/env python3
#
# Copyright (C) 2019 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.

import random
import os
import argparse
import numpy as np
import tensorflow.compat.v1 as tf
from lpr.trainer import CTCUtils, inference, InputData
from tfutils.helpers import load_module

OPTIMIZER_CLS_NAMES = {
    "Adagrad": tf.train.AdagradOptimizer,
    "Adam": tf.train.AdamOptimizer,
    "Ftrl": tf.train.FtrlOptimizer,
    "Momentum": lambda learning_rate: tf.train.MomentumOptimizer(learning_rate, momentum=0.9),  # pylint: disable=line-too-long
    "RMSProp": tf.train.RMSPropOptimizer,
    "SGD": tf.train.GradientDescentOptimizer,
}

OPTIMIZER_SUMMARIES = [
    "learning_rate",
    "loss",
    "gradients",
    "gradient_norm",
    "global_gradient_norm",
]

def _clip_gradients_by_norm(grads_and_vars, clip_gradients):
  """Clips gradients by global norm."""
  gradients, variables = zip(*grads_and_vars)
  clipped_gradients, _ = tf.clip_by_global_norm(gradients, clip_gradients)
  return list(zip(clipped_gradients, variables))

def _multiply_gradients(grads_and_vars, gradient_multipliers):
  """Multiply specified gradients."""
  multiplied_grads_and_vars = []
  for grad, var in grads_and_vars:
    if (grad is not None and
        (var in gradient_multipliers or var.name in gradient_multipliers)):
      key = var if var in gradient_multipliers else var.name
      multiplier = gradient_multipliers[key]
      if isinstance(grad, tf.IndexedSlices):
        grad_values = grad.values * multiplier
        grad = tf.IndexedSlices(grad_values, grad.indices, grad.dense_shape)
      else:
        grad *= tf.cast(multiplier, grad.dtype)
    multiplied_grads_and_vars.append((grad, var))
  return multiplied_grads_and_vars

def _add_scaled_noise_to_gradients(grads_and_vars, gradient_noise_scale):
  """Adds scaled noise from a 0-mean normal distribution to gradients."""
  gradients, variables = zip(*grads_and_vars)
  noisy_gradients = []
  for gradient in gradients:
    if gradient is None:
      noisy_gradients.append(None)
      continue
    if isinstance(gradient, tf.IndexedSlices):
      gradient_shape = gradient.dense_shape
    else:
      gradient_shape = gradient.get_shape()
    noise = tf.truncated_normal(gradient_shape) * gradient_noise_scale
    noisy_gradients.append(gradient + noise)
  return list(zip(noisy_gradients, variables))

def optimize_loss(loss,
                  global_step,
                  learning_rate,
                  optimizer,
                  gradient_noise_scale=None,
                  gradient_multipliers=None,
                  clip_gradients=None,
                  learning_rate_decay_fn=None,
                  update_ops=None,
                  variables=None,
                  name=None,
                  summaries=None,
                  colocate_gradients_with_ops=False,
                  increment_global_step=True):
  """Given loss and parameters for optimizer, returns a training op.
  Various ways of passing optimizers include:
  - by string specifying the name of the optimizer. See OPTIMIZER_CLS_NAMES
      for full list. E.g. `optimize_loss(..., optimizer='Adam')`.
  - by function taking learning rate `Tensor` as argument and returning an
      `Optimizer` instance. E.g. `optimize_loss(...,
      optimizer=lambda lr: tf.compat.v1.train.MomentumOptimizer(lr,
      momentum=0.5))`.
    Alternatively, if `learning_rate` is `None`, the function takes no
    arguments. E.g. `optimize_loss(..., learning_rate=None,
      optimizer=lambda: tf.compat.v1.train.MomentumOptimizer(0.5,
      momentum=0.5))`.
  - by a subclass of `Optimizer` having a single-argument constructor
      (the argument is the learning rate), such as AdamOptimizer or
      AdagradOptimizer. E.g. `optimize_loss(...,
      optimizer=tf.compat.v1.train.AdagradOptimizer)`.
  - by an instance of a subclass of `Optimizer`.
      E.g., `optimize_loss(...,
      optimizer=tf.compat.v1.train.AdagradOptimizer(0.5))`.
  Args:
    loss: Scalar `Tensor`.
    global_step: Scalar int `Tensor`, step counter to update on each step unless
      `increment_global_step` is `False`. If not supplied, it will be fetched
      from the default graph (see `tf.compat.v1.train.get_global_step` for
      details). If it has not been created, no step will be incremented with
      each weight update. `learning_rate_decay_fn` requires `global_step`.
    learning_rate: float or `Tensor`, magnitude of update per each training
      step. Can be `None`.
    optimizer: string, class or optimizer instance, used as trainer. string
      should be name of optimizer, like 'SGD', 'Adam', 'Adagrad'. Full list in
      OPTIMIZER_CLS_NAMES constant. class should be sub-class of `tf.Optimizer`
      that implements `compute_gradients` and `apply_gradients` functions.
      optimizer instance should be instantiation of `tf.Optimizer` sub-class and
      have `compute_gradients` and `apply_gradients` functions.
    gradient_noise_scale: float or None, adds 0-mean normal noise scaled by this
      value.
    gradient_multipliers: dict of variables or variable names to floats. If
      present, gradients for specified variables will be multiplied by given
      constant.
    clip_gradients: float, callable or `None`. If a float is provided, a global
      clipping is applied to prevent the norm of the gradient from exceeding
      this value. Alternatively, a callable can be provided, e.g.,
      `adaptive_clipping_fn()`.  This callable takes a list of `(gradients,
      variables)` tuples and returns the same thing with the gradients modified.
    learning_rate_decay_fn: function, takes `learning_rate` and `global_step`
      `Tensor`s, returns `Tensor`. Can be used to implement any learning rate
      decay functions.
                            For example: `tf.compat.v1.train.exponential_decay`.
                              Ignored if `learning_rate` is not supplied.
    update_ops: list of update `Operation`s to execute at each step. If `None`,
      uses elements of UPDATE_OPS collection. The order of execution between
      `update_ops` and `loss` is non-deterministic.
    variables: list of variables to optimize or `None` to use all trainable
      variables.
    name: The name for this operation is used to scope operations and summaries.
    summaries: List of internal quantities to visualize on tensorboard. If not
      set, the loss, the learning rate, and the global norm of the gradients
      will be reported. The complete list of possible values is in
      OPTIMIZER_SUMMARIES.
    colocate_gradients_with_ops: If True, try colocating gradients with the
      corresponding op.
    increment_global_step: Whether to increment `global_step`. If your model
      calls `optimize_loss` multiple times per training step (e.g. to optimize
      different parts of the model), use this arg to avoid incrementing
      `global_step` more times than necessary.
  Returns:
    Training op.
  Raises:
    ValueError: if:
        * `loss` is an invalid type or shape.
        * `global_step` is an invalid type or shape.
        * `learning_rate` is an invalid type or value.
        * `optimizer` has the wrong type.
        * `clip_gradients` is neither float nor callable.
        * `learning_rate` and `learning_rate_decay_fn` are supplied, but no
          `global_step` is available.
        * `gradients` is empty.
  """
  loss = tf.convert_to_tensor(loss)
  tf.debugging.assert_scalar(loss)
  if global_step is None:
    global_step = tf.train.get_global_step()
  else:
    tf.train.assert_global_step(global_step)
  with tf.variable_scope(name, "OptimizeLoss", [loss, global_step]):
    # Update ops take UPDATE_OPS collection if not provided.
    if update_ops is None:
      update_ops = set(tf.get_collection(tf.GraphKeys.UPDATE_OPS))
    # Make sure update ops are ran before computing loss.
    if update_ops:
      loss = tf.with_dependencies(list(update_ops), loss)

    # Learning rate variable, with possible decay.
    lr = None
    if learning_rate is not None:
      if (isinstance(learning_rate, tf.Tensor) and
          learning_rate.get_shape().ndims == 0):
        lr = learning_rate
      elif isinstance(learning_rate, float):
        if learning_rate < 0.0:
          raise ValueError("Invalid learning_rate %s.", learning_rate)
        lr = tf.get_variable(
            "learning_rate", [],
            trainable=False,
            initializer=tf.constant_initializer(learning_rate))
      else:
        raise ValueError("Learning rate should be 0d Tensor or float. "
                         "Got %s of type %s" %
                         (str(learning_rate), str(type(learning_rate))))
    if summaries is None:
      summaries = ["loss", "learning_rate", "global_gradient_norm"]
    else:
      for summ in summaries:
        if summ not in OPTIMIZER_SUMMARIES:
          raise ValueError("Summaries should be one of [%s], you provided %s." %
                           (", ".join(OPTIMIZER_SUMMARIES), summ))
    if learning_rate is not None and learning_rate_decay_fn is not None:
      if global_step is None:
        raise ValueError("global_step is required for learning_rate_decay_fn.")
      lr = learning_rate_decay_fn(lr, global_step)
      if "learning_rate" in summaries:
        tf.summary.scalar("learning_rate", lr)

    # Create optimizer, given specified parameters.
    if isinstance(optimizer, str):
      if lr is None:
        raise ValueError("Learning rate is None, but should be specified if "
                         "optimizer is string (%s)." % optimizer)
      if optimizer not in OPTIMIZER_CLS_NAMES:
        raise ValueError(
            "Optimizer name should be one of [%s], you provided %s." %
            (", ".join(OPTIMIZER_CLS_NAMES), optimizer))
      opt = OPTIMIZER_CLS_NAMES[optimizer](learning_rate=lr)
    elif (isinstance(optimizer, type) and
          issubclass(optimizer, tf.train.Optimizer)):
      if lr is None:
        raise ValueError("Learning rate is None, but should be specified if "
                         "optimizer is class (%s)." % optimizer)
      opt = optimizer(learning_rate=lr)
    elif isinstance(optimizer, tf.train.Optimizer):
      opt = optimizer
    elif callable(optimizer):
      if learning_rate is not None:
        opt = optimizer(lr)
      else:
        opt = optimizer()
      if not isinstance(opt, tf.train.Optimizer):
        raise ValueError("Unrecognized optimizer: function should return "
                         "subclass of Optimizer. Got %s." % str(opt))
    else:
      raise ValueError("Unrecognized optimizer: should be string, "
                       "subclass of Optimizer, instance of "
                       "subclass of Optimizer or function with one argument. "
                       "Got %s." % str(optimizer))

    # All trainable variables, if specific variables are not specified.
    if variables is None:
      variables = tf.trainable_variables()

    # Compute gradients.
    gradients = opt.compute_gradients(
        loss,
        variables,
        colocate_gradients_with_ops=colocate_gradients_with_ops)

    # Optionally add gradient noise.
    if gradient_noise_scale is not None:
      gradients = _add_scaled_noise_to_gradients(gradients,
                                                 gradient_noise_scale)

    # Multiply some gradients.
    if gradient_multipliers is not None:
      gradients = _multiply_gradients(gradients, gradient_multipliers)
      if not gradients:
        raise ValueError(
            "Empty list of (gradient, var) pairs encountered. This is most "
            "likely to be caused by an improper value of gradient_multipliers.")

    if "global_gradient_norm" in summaries or "gradient_norm" in summaries:
      tf.summary.scalar("global_norm/gradient_norm",
                     tf.global_norm(list(zip(*gradients))[0]))

    # Optionally clip gradients by global norm.
    if isinstance(clip_gradients, float):
      gradients = _clip_gradients_by_norm(gradients, clip_gradients)
    elif callable(clip_gradients):
      gradients = clip_gradients(gradients)
    elif clip_gradients is not None:
      raise ValueError("Unknown type %s for clip_gradients" %
                       type(clip_gradients))

    # Add scalar summary for loss.
    if "loss" in summaries:
      tf.summary.scalar("loss", loss)

    # Add histograms for variables, gradients and gradient norms.
    for gradient, variable in gradients:
      if isinstance(gradient, tf.IndexedSlices):
        grad_values = gradient.values
      else:
        grad_values = gradient

      if grad_values is not None:
        var_name = variable.name.replace(":", "_")
        if "gradients" in summaries:
          tf.summary.histogram("gradients/%s" % var_name, grad_values)
        if "gradient_norm" in summaries:
          tf.summary.scalar("gradient_norm/%s" % var_name,
                         tf.global_norm([grad_values]))

    if clip_gradients is not None and ("global_gradient_norm" in summaries or
                                       "gradient_norm" in summaries):
      tf.summary.scalar("global_norm/clipped_gradient_norm",
                     tf.global_norm(list(zip(*gradients))[0]))

    # Create gradient updates.
    grad_updates = opt.apply_gradients(
        gradients,
        global_step=global_step if increment_global_step else None,
        name="train")

    # Ensure the train_tensor computes grad_updates.
    train_tensor = tf.with_dependencies([grad_updates], loss)

    return train_tensor


def parse_args():
  parser = argparse.ArgumentParser(description='Perform training of a model')
  parser.add_argument('path_to_config', help='Path to a config.py')
  parser.add_argument('--init_checkpoint', default=None, help='Path to checkpoint')
  return parser.parse_args()

# pylint: disable=too-many-locals, too-many-statements
def train(config, init_checkpoint):
  if hasattr(config.train, 'random_seed'):
    np.random.seed(config.train.random_seed)
    tf.set_random_seed(config.train.random_seed)
    random.seed(config.train.random_seed)

  if hasattr(config.train.execution, 'CUDA_VISIBLE_DEVICES'):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = config.train.execution.CUDA_VISIBLE_DEVICES

  CTCUtils.vocab = config.vocab
  CTCUtils.r_vocab = config.r_vocab

  input_train_data = InputData(batch_size=config.train.batch_size,
                               input_shape=config.input_shape,
                               file_list_path=config.train.file_list_path,
                               apply_basic_aug=config.train.apply_basic_aug,
                               apply_stn_aug=config.train.apply_stn_aug,
                               apply_blur_aug=config.train.apply_blur_aug)


  graph = tf.Graph()
  with graph.as_default():
    global_step = tf.Variable(0, name='global_step', trainable=False)
    input_data, input_labels = input_train_data.input_fn()

    prob = inference(config.rnn_cells_num, input_data, config.num_classes)
    prob = tf.transpose(prob, (1, 0, 2))  # prepare for CTC

    data_length = tf.fill([tf.shape(prob)[1]], tf.shape(prob)[0])  # input seq length, batch size
    ctc = tf.py_func(CTCUtils.compute_ctc_from_labels, [input_labels], [tf.int64, tf.int64, tf.int64])
    ctc_labels = tf.to_int32(tf.SparseTensor(ctc[0], ctc[1], ctc[2]))

    predictions = tf.to_int32(
      tf.nn.ctc_beam_search_decoder(prob, data_length, merge_repeated=False, beam_width=10)[0][0])
    tf.sparse_tensor_to_dense(predictions, default_value=-1, name='d_predictions')
    tf.reduce_mean(tf.edit_distance(predictions, ctc_labels, normalize=False), name='error_rate')

    loss = tf.reduce_mean(
      tf.nn.ctc_loss(inputs=prob, labels=ctc_labels, sequence_length=data_length, ctc_merge_repeated=True), name='loss')

    learning_rate = tf.train.piecewise_constant(global_step, [150000, 200000],
                                                [config.train.learning_rate, 0.1 * config.train.learning_rate,
                                                 0.01 * config.train.learning_rate])
    opt_loss = tf.contrib.layers.optimize_loss(loss, global_step, learning_rate, config.train.opt_type,
                                               config.train.grad_noise_scale, name='train_step')

    tf.global_variables_initializer()
    saver = tf.train.Saver(max_to_keep=1000, write_version=tf.train.SaverDef.V2, save_relative_paths=True)

  conf = tf.ConfigProto()
  if hasattr(config.train.execution, 'per_process_gpu_memory_fraction'):
    conf.gpu_options.per_process_gpu_memory_fraction = config.train.execution.per_process_gpu_memory_fraction
  if hasattr(config.train.execution, 'allow_growth'):
    conf.gpu_options.allow_growth = config.train.execution.allow_growth

  session = tf.Session(graph=graph, config=conf)
  coordinator = tf.train.Coordinator()
  threads = tf.train.start_queue_runners(sess=session, coord=coordinator)

  session.run('init')

  if init_checkpoint:
    tf.logging.info('Initialize from: ' + init_checkpoint)
    saver.restore(session, init_checkpoint)
  else:
    lastest_checkpoint = tf.train.latest_checkpoint(config.model_dir)
    if lastest_checkpoint:
      tf.logging.info('Restore from: ' + lastest_checkpoint)
      saver.restore(session, lastest_checkpoint)

  writer = None
  if config.train.need_to_save_log:
    writer = tf.summary.FileWriter(config.model_dir, session.graph)

  graph.finalize()


  for i in range(config.train.steps):
    curr_step, curr_learning_rate, curr_loss, curr_opt_loss = session.run([global_step, learning_rate, loss, opt_loss])

    if i % config.train.display_iter == 0:
      if config.train.need_to_save_log:

        writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag='train/loss',
                                                              simple_value=float(curr_loss)),
                                             tf.Summary.Value(tag='train/learning_rate',
                                                              simple_value=float(curr_learning_rate)),
                                             tf.Summary.Value(tag='train/optimization_loss',
                                                              simple_value=float(curr_opt_loss))
                                             ]),
                           curr_step)
        writer.flush()

      tf.logging.info('Iteration: ' + str(curr_step) + ', Train loss: ' + str(curr_loss))

    if ((curr_step % config.train.save_checkpoints_steps == 0 or curr_step == config.train.steps)
        and config.train.need_to_save_weights):
      saver.save(session, config.model_dir + '/model.ckpt-{:d}.ckpt'.format(curr_step))

  coordinator.request_stop()
  coordinator.join(threads)
  session.close()


def main(_):
  args = parse_args()
  cfg = load_module(args.path_to_config)
  train(cfg, args.init_checkpoint)


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run(main)
