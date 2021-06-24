# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
"""Keras convolution layers and image transformation layers.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.eager import context
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras import activations
from tensorflow.python.keras import backend
from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.engine.input_spec import InputSpec
# imports for backwards namespace compatibility
# pylint: disable=unused-import
from tensorflow.python.keras.layers.pooling import AveragePooling1D
from tensorflow.python.keras.layers.pooling import AveragePooling2D
from tensorflow.python.keras.layers.pooling import AveragePooling3D
from tensorflow.python.keras.layers.pooling import MaxPooling1D
from tensorflow.python.keras.layers.pooling import MaxPooling2D
from tensorflow.python.keras.layers.pooling import MaxPooling3D
# pylint: enable=unused-import
from tensorflow.python.keras.utils import conv_utils
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import nn_ops
from tensorflow.python.util.tf_export import keras_export
import tensorflow as tf

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.utils import generic_utils
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import standard_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.util import nest
from tensorflow.python.util import tf_inspect

class DecomposedDense(Layer):
  """Just your regular densely-connected NN layer.
  `Dense` implements the operation:
  `output = activation(dot(input, kernel) + bias)`
  where `activation` is the element-wise activation function
  passed as the `activation` argument, `kernel` is a weights matrix
  created by the layer, and `bias` is a bias vector created by the layer
  (only applicable if `use_bias` is `True`).
  Note: If the input to the layer has a rank greater than 2, then
  it is flattened prior to the initial dot product with `kernel`.
  Example:
  ```python
  # as first layer in a sequential model:
  model = Sequential()
  model.add(Dense(32, input_shape=(16,)))
  # now the model will take as input arrays of shape (*, 16)
  # and output arrays of shape (*, 32)
  # after the first layer, you don't need to specify
  # the size of the input anymore:
  model.add(Dense(32))
  ```
  Arguments:
    units: Positive integer, dimensionality of the output space.
    activation: Activation function to use.
      If you don't specify anything, no activation is applied
      (ie. "linear" activation: `a(x) = x`).
    use_bias: Boolean, whether the layer uses a bias vector.
    kernel_initializer: Initializer for the `kernel` weights matrix.
    bias_initializer: Initializer for the bias vector.
    kernel_regularizer: Regularizer function applied to
      the `kernel` weights matrix.
    bias_regularizer: Regularizer function applied to the bias vector.
    activity_regularizer: Regularizer function applied to
      the output of the layer (its "activation")..
    kernel_constraint: Constraint function applied to
      the `kernel` weights matrix.
    bias_constraint: Constraint function applied to the bias vector.
  Input shape:
    N-D tensor with shape: `(batch_size, ..., input_dim)`.
    The most common situation would be
    a 2D input with shape `(batch_size, input_dim)`.
  Output shape:
    N-D tensor with shape: `(batch_size, ..., units)`.
    For instance, for a 2D input with shape `(batch_size, input_dim)`,
    the output would have shape `(batch_size, units)`.
  """

  def __init__(self,
               units,
               activation=None,
               use_bias=True,
               kernel_initializer='glorot_uniform',
               bias_initializer='zeros',
               kernel_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               bias_constraint=None,
               theta_sup=None,
               theta_unsup=None,
               bias=None,
               l1_thres=None,
               **kwargs):
    if 'input_shape' not in kwargs and 'input_dim' in kwargs:
      kwargs['input_shape'] = (kwargs.pop('input_dim'),)

    super(DecomposedDense, self).__init__(
        activity_regularizer=regularizers.get(activity_regularizer), **kwargs)
    self.units = int(units)
    self.activation = activations.get(activation)
    self.use_bias = use_bias
    self.kernel_initializer = initializers.get(kernel_initializer)
    self.bias_initializer = initializers.get(bias_initializer)
    self.kernel_regularizer = regularizers.get(kernel_regularizer)
    self.bias_regularizer = regularizers.get(bias_regularizer)
    self.kernel_constraint = constraints.get(kernel_constraint)
    self.bias_constraint = constraints.get(bias_constraint)

    self.supports_masking = True
    self.input_spec = InputSpec(min_ndim=2)

    self.theta_sup = theta_sup
    self.theta_unsup = theta_unsup
    self.bias = bias
    self.l1_thres = l1_thres

  def build(self, input_shape):
    dtype = dtypes.as_dtype(self.dtype or K.floatx())
    if not (dtype.is_floating or dtype.is_complex):
      raise TypeError('Unable to build `Dense` layer with non-floating point '
                      'dtype %s' % (dtype,))
    input_shape = tensor_shape.TensorShape(input_shape)
    if tensor_shape.dimension_value(input_shape[-1]) is None:
      raise ValueError('The last dimension of the inputs to `Dense` '
                       'should be defined. Found `None`.')
    last_dim = tensor_shape.dimension_value(input_shape[-1])
    self.input_spec = InputSpec(min_ndim=2,
                                axes={-1: last_dim})
    if self.use_bias:
      # self.bias = self.add_weight(
      #     'bias',
      #     shape=[self.units,],
      #     initializer=self.bias_initializer,
      #     regularizer=self.bias_regularizer,
      #     constraint=self.bias_constraint,
      #     dtype=self.dtype,
      #     trainable=True)
      pass
    else:
      self.bias = None
    self.built = True

  # def l1_pruning(self, weights, hyp):
  #   hard_threshold = tf.cast(tf.greater(tf.abs(weights), hyp), tf.float32)
  #   return tf.multiply(weights, hard_threshold)
  
  def call(self, inputs):
    # theta_unsup = self.theta_unsup #if tf.keras.backend.learning_phase() else self.l1_pruning(self.theta_unsup, self.l1_thres)
    if tf.keras.backend.learning_phase() is not None:
      theta_sup = self.theta_sup
      theta_unsup = self.theta_unsup 
    else: 
      #############
      # Normal
      #############
      theta_sup = self.theta_sup
      hard_threshold = tf.cast(tf.greater(tf.abs(self.theta_unsup), self.l1_thres), tf.float32)
      theta_unsup = tf.multiply(self.theta_unsup, hard_threshold)
      #############
      # No Sigma
      #############
      # theta_sup = self.theta_sup*0
      # hard_threshold = tf.cast(tf.greater(tf.abs(self.theta_unsup), self.l1_thres), tf.float32)
      # theta_unsup = tf.multiply(self.theta_unsup, hard_threshold)
      #############
      # No Psi
      #############
      # theta_sup = self.theta_sup
      # theta_unsup = 0
    ######################### Decomposed Kernel #########################
    self.my_theta = 0.7*theta_sup + 0.3*theta_unsup
    # if tf.keras.backend.learning_phase()==0:
    #   # self.my_theta = self.theta_sup
    #   self.my_theta = theta_unsup
    #####################################################################
    rank = len(inputs.shape)
    if rank > 2:
      # Broadcasting is required for the inputs.
      outputs = standard_ops.tensordot(inputs, self.my_theta, [[rank - 1], [0]])
      # Reshape the output back to the original ndim of the input.
      if not context.executing_eagerly():
        shape = inputs.shape.as_list()
        output_shape = shape[:-1] + [self.units]
        outputs.set_shape(output_shape)
    else:
      inputs = math_ops.cast(inputs, self._compute_dtype)
      if K.is_sparse(inputs):
        outputs = sparse_ops.sparse_tensor_dense_matmul(inputs, self.my_theta)
      else:
        outputs = gen_math_ops.mat_mul(inputs, self.my_theta)
    if self.use_bias:
      outputs = nn.bias_add(outputs, self.bias)
    if self.activation is not None:
      return self.activation(outputs)  # pylint: disable=not-callable
    return outputs

  def compute_output_shape(self, input_shape):
    input_shape = tensor_shape.TensorShape(input_shape)
    input_shape = input_shape.with_rank_at_least(2)
    if tensor_shape.dimension_value(input_shape[-1]) is None:
      raise ValueError(
          'The innermost dimension of input_shape must be defined, but saw: %s'
          % input_shape)
    return input_shape[:-1].concatenate(self.units)

  def get_config(self):
    config = {
        'units': self.units,
        'activation': activations.serialize(self.activation),
        'use_bias': self.use_bias,
        'kernel_initializer': initializers.serialize(self.kernel_initializer),
        'bias_initializer': initializers.serialize(self.bias_initializer),
        'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
        'bias_regularizer': regularizers.serialize(self.bias_regularizer),
        'activity_regularizer':
            regularizers.serialize(self.activity_regularizer),
        'kernel_constraint': constraints.serialize(self.kernel_constraint),
        'bias_constraint': constraints.serialize(self.bias_constraint)
    }
    # base_config = super(Dense, self).get_config()


    base_config = super(Dense, self).get_config()


    return dict(list(base_config.items()) + list(config.items()))


class DecomposedConv(Layer):
  """Abstract N-D convolution layer (private, used as implementation base).
  This layer creates a convolution kernel that is convolved
  (actually cross-correlated) with the layer input to produce a tensor of
  outputs. If `use_bias` is True (and a `bias_initializer` is provided),
  a bias vector is created and added to the outputs. Finally, if
  `activation` is not `None`, it is applied to the outputs as well.
  Note: layer attributes cannot be modified after the layer has been called
  once (except the `trainable` attribute).
  Arguments:
    rank: An integer, the rank of the convolution, e.g. "2" for 2D convolution.
    filters: Integer, the dimensionality of the output space (i.e. the number
      of filters in the convolution).
    kernel_size: An integer or tuple/list of n integers, specifying the
      length of the convolution window.
    strides: An integer or tuple/list of n integers,
      specifying the stride length of the convolution.
      Specifying any stride value != 1 is incompatible with specifying
      any `dilation_rate` value != 1.
    padding: One of `"valid"`,  `"same"`, or `"causal"` (case-insensitive).
    data_format: A string, one of `channels_last` (default) or `channels_first`.
      The ordering of the dimensions in the inputs.
      `channels_last` corresponds to inputs with shape
      `(batch_size, ..., channels)` while `channels_first` corresponds to
      inputs with shape `(batch_size, channels, ...)`.
    dilation_rate: An integer or tuple/list of n integers, specifying
      the dilation rate to use for dilated convolution.
      Currently, specifying any `dilation_rate` value != 1 is
      incompatible with specifying any `strides` value != 1.
    activation: Activation function to use.
      If you don't specify anything, no activation is applied.
    use_bias: Boolean, whether the layer uses a bias.
    kernel_initializer: An initializer for the convolution kernel.
    bias_initializer: An initializer for the bias vector. If None, the default
      initializer will be used.
    kernel_regularizer: Optional regularizer for the convolution kernel.
    bias_regularizer: Optional regularizer for the bias vector.
    activity_regularizer: Optional regularizer function for the output.
    kernel_constraint: Optional projection function to be applied to the
        kernel after being updated by an `Optimizer` (e.g. used to implement
        norm constraints or value constraints for layer weights). The function
        must take as input the unprojected variable and must return the
        projected variable (which must have the same shape). Constraints are
        not safe to use when doing asynchronous distributed training.
    bias_constraint: Optional projection function to be applied to the
        bias after being updated by an `Optimizer`.
    trainable: Boolean, if `True` the weights of this layer will be marked as
      trainable (and listed in `layer.trainable_weights`).
    name: A string, the name of the layer.
  """

# def __init__(self,
#                filters,
#                kernel_size,
#                strides=(1, 1),
#                padding='valid',
#                data_format=None,
#                dilation_rate=(1, 1),
#                activation=None,
#                use_bias=True,
#                kernel_initializer='glorot_uniform',
#                bias_initializer='zeros',
#                kernel_regularizer=None,
#                bias_regularizer=None,
#                activity_regularizer=None,
#                kernel_constraint=None,
#                bias_constraint=None,
#                **kwargs):
#     super(Conv2D, self).__init__(
#         rank=2,
#         filters=filters,
#         kernel_size=kernel_size,
#         strides=strides,
#         padding=padding,
#         data_format=data_format,
#         dilation_rate=dilation_rate,
#         activation=activations.get(activation),
#         use_bias=use_bias,
#         kernel_initializer=initializers.get(kernel_initializer),
#         bias_initializer=initializers.get(bias_initializer),
#         kernel_regularizer=regularizers.get(kernel_regularizer),
#         bias_regularizer=regularizers.get(bias_regularizer),
#         activity_regularizer=regularizers.get(activity_regularizer),
#         kernel_constraint=constraints.get(kernel_constraint),
#         bias_constraint=constraints.get(bias_constraint),
#         **kwargs)
  def __init__(self, 
               filters,
               kernel_size,
               rank=2,
               strides=(1, 1),
               padding='valid',
               data_format=None,
               dilation_rate=(1, 1),
               activation=None,
               use_bias=True,
               kernel_initializer='glorot_uniform',
               bias_initializer='zeros',
               kernel_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               bias_constraint=None,
               trainable=True,
               name=None,
               theta_sup=None,
               theta_unsup=None,
               bias=None,
               l1_thres=None,
               **kwargs):
    super(DecomposedConv, self).__init__(
        trainable=trainable,
        name=name,
        activity_regularizer=regularizers.get(activity_regularizer),
        **kwargs)
    self.rank = rank
    self.filters = filters
    self.kernel_size = conv_utils.normalize_tuple(
        kernel_size, rank, 'kernel_size')
    self.strides = conv_utils.normalize_tuple(strides, rank, 'strides')
    self.padding = conv_utils.normalize_padding(padding)
    if (self.padding == 'causal' and not isinstance(self,
                                                    (Conv1D, SeparableConv1D))):
      raise ValueError('Causal padding is only supported for `Conv1D`'
                       'and ``SeparableConv1D`.')
    self.data_format = conv_utils.normalize_data_format(data_format)
    self.dilation_rate = conv_utils.normalize_tuple(
        dilation_rate, rank, 'dilation_rate')
    self.activation = activations.get(activation)
    self.use_bias = use_bias
    self.kernel_initializer = initializers.get(kernel_initializer)
    self.bias_initializer = initializers.get(bias_initializer)
    self.kernel_regularizer = regularizers.get(kernel_regularizer)
    self.bias_regularizer = regularizers.get(bias_regularizer)
    self.kernel_constraint = constraints.get(kernel_constraint)
    self.bias_constraint = constraints.get(bias_constraint)
    self.input_spec = InputSpec(ndim=self.rank + 2)

    self.theta_sup = theta_sup
    self.theta_unsup = theta_unsup
    self.bias = bias
    self.l1_thres = l1_thres

  def build(self, input_shape):
    input_shape = tensor_shape.TensorShape(input_shape)
    input_channel = self._get_input_channel(input_shape)
    kernel_shape = self.kernel_size + (input_channel, self.filters)

    if self.use_bias:
      # self.bias = self.add_weight(
      #     name='bias',
      #     shape=(self.filters,),
      #     initializer=self.bias_initializer,
      #     regularizer=self.bias_regularizer,
      #     constraint=self.bias_constraint,
      #     trainable=True,
      #     dtype=self.dtype)
      pass
    else:
      self.bias = None
    
    channel_axis = self._get_channel_axis()
    self.input_spec = InputSpec(ndim=self.rank + 2,
                                axes={channel_axis: input_channel})

    self._build_conv_op_input_shape = input_shape
    self._build_input_channel = input_channel
    self._padding_op = self._get_padding_op()
    self._conv_op_data_format = conv_utils.convert_data_format(
        self.data_format, self.rank + 2)
   
    self.built = True

  # def l1_pruning(self, weights, hyp):
  #   hard_threshold = tf.cast(tf.greater(tf.abs(weights), hyp), tf.float32)
  #   return tf.multiply(weights, hard_threshold)
  
  def call(self, inputs):
    if tf.keras.backend.learning_phase() is not None:
      theta_sup = self.theta_sup
      theta_unsup = self.theta_unsup 
    else: 
      #############
      # Normal
      #############
      theta_sup = self.theta_sup
      hard_threshold = tf.cast(tf.greater(tf.abs(self.theta_unsup), self.l1_thres), tf.float32)
      theta_unsup = tf.multiply(self.theta_unsup, hard_threshold)
      #############
      # No Sigma
      #############
      # theta_sup = self.theta_sup*0
      # hard_threshold = tf.cast(tf.greater(tf.abs(self.theta_unsup), self.l1_thres), tf.float32)
      # theta_unsup = tf.multiply(self.theta_unsup, hard_threshold)
      #############
      # No Psi
      #############
      # theta_sup = self.theta_sup
      # theta_unsup = self.theta_sup*0
    ######################### Decomposed Kernel #########################
    self.my_theta = 0.7*theta_sup + 0.3*theta_unsup
    # if tf.keras.backend.learning_phase()==0:
    #   # self.my_theta = self.theta_sup
    #   self.my_theta = theta_unsup
    #####################################################################

    # if self._recreate_conv_op(inputs):
    self._convolution_op = nn_ops.Convolution(
        inputs.get_shape(),
        filter_shape=self.my_theta.shape,
        dilation_rate=self.dilation_rate,
        strides=self.strides,
        padding=self._padding_op,
        data_format=self._conv_op_data_format)

    # Apply causal padding to inputs for Conv1D.
    if self.padding == 'causal' and self.__class__.__name__ == 'Conv1D':
      inputs = array_ops.pad(inputs, self._compute_causal_padding())
   
    outputs = self._convolution_op(inputs, self.my_theta)

    if self.use_bias:
      if self.data_format == 'channels_first':
        if self.rank == 1:
          # nn.bias_add does not accept a 1D input tensor.
          bias = array_ops.reshape(self.bias, (1, self.filters, 1))
          outputs += bias
        else:
          outputs = nn.bias_add(outputs, self.bias, data_format='NCHW')
      else:
        outputs = nn.bias_add(outputs, self.bias, data_format='NHWC')

    if self.activation is not None:
      return self.activation(outputs)
    return outputs

  def compute_output_shape(self, input_shape):
    input_shape = tensor_shape.TensorShape(input_shape).as_list()
    if self.data_format == 'channels_last':
      space = input_shape[1:-1]
      new_space = []
      for i in range(len(space)):
        new_dim = conv_utils.conv_output_length(
            space[i],
            self.kernel_size[i],
            padding=self.padding,
            stride=self.strides[i],
            dilation=self.dilation_rate[i])
        new_space.append(new_dim)
      return tensor_shape.TensorShape([input_shape[0]] + new_space +
                                      [self.filters])
    else:
      space = input_shape[2:]
      new_space = []
      for i in range(len(space)):
        new_dim = conv_utils.conv_output_length(
            space[i],
            self.kernel_size[i],
            padding=self.padding,
            stride=self.strides[i],
            dilation=self.dilation_rate[i])
        new_space.append(new_dim)
      return tensor_shape.TensorShape([input_shape[0], self.filters] +
                                      new_space)

  def get_config(self):
    config = {
        'filters': self.filters,
        'kernel_size': self.kernel_size,
        'strides': self.strides,
        'padding': self.padding,
        'data_format': self.data_format,
        'dilation_rate': self.dilation_rate,
        'activation': activations.serialize(self.activation),
        'use_bias': self.use_bias,
        'kernel_initializer': initializers.serialize(self.kernel_initializer),
        'bias_initializer': initializers.serialize(self.bias_initializer),
        'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
        'bias_regularizer': regularizers.serialize(self.bias_regularizer),
        'activity_regularizer':
            regularizers.serialize(self.activity_regularizer),
        'kernel_constraint': constraints.serialize(self.kernel_constraint),
        'bias_constraint': constraints.serialize(self.bias_constraint)
    }
    base_config = super(Conv, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  def _compute_causal_padding(self):
    """Calculates padding for 'causal' option for 1-d conv layers."""
    left_pad = self.dilation_rate[0] * (self.kernel_size[0] - 1)
    if self.data_format == 'channels_last':
      causal_padding = [[0, 0], [left_pad, 0], [0, 0]]
    else:
      causal_padding = [[0, 0], [0, 0], [left_pad, 0]]
    return causal_padding

  def _get_channel_axis(self):
    if self.data_format == 'channels_first':
      return 1
    else:
      return -1

  def _get_input_channel(self, input_shape):
    channel_axis = self._get_channel_axis()
    if input_shape.dims[channel_axis].value is None:
      raise ValueError('The channel dimension of the inputs '
                       'should be defined. Found `None`.')
    return int(input_shape[channel_axis])

  def _get_padding_op(self):
    if self.padding == 'causal':
      op_padding = 'valid'
    else:
      op_padding = self.padding
    if not isinstance(op_padding, (list, tuple)):
      op_padding = op_padding.upper()
    return op_padding

  def _recreate_conv_op(self, inputs):
    """Recreate conv_op if necessary.
    Check if the input_shape in call() is different from that in build().
    For the values that are not None, if they are different, recreate
    the _convolution_op to avoid the stateful behavior.
    Args:
      inputs: The input data to call() method.
    Returns:
      `True` or `False` to indicate whether to recreate the conv_op.
    """
    call_input_shape = inputs.get_shape()
    for axis in range(1, len(call_input_shape)):
      if (call_input_shape[axis] is not None
          and self._build_conv_op_input_shape[axis] is not None
          and call_input_shape[axis] != self._build_conv_op_input_shape[axis]):
        return True
    return False



