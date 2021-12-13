import numpy as np
import pandas as pd
import tensorflow as tf
from base import Base

class FM(Base):
  """Factorization Machine

  Args:
    input_config(dict): This arg is to input the dataset configuration and for the construction of the input.
    latent_k(int): This arg is to declare the embedding dimension of sparse features.
    random_state(int): This arg is to initialize the random seed. Default 42.
    data_type(object): This arg is to declare the weighting data type in tensorflow. Default tf.float32.

  Attributes:
    w0(tensor): This attribute is for the initialization of bias weight.
    w(tensor): This attribute is for the initialization of linear weight.
    v(tensor): This attribute is for the initialization of embedding weight.
  """

  def __init__(self, input_config, latent_k, random_state=42, data_type=tf.float32):
    super().__init__(input_config=input_config, latent_k=latent_k, random_state=random_state, data_type=data_type)
    self.w0 = tf.Variable(tf.zeros([1], name='w0', dtype=data_type))
    self.w = tf.Variable(tf.zeros([self.feat_size], name='w', dtype=data_type))
    self.v = tf.Variable(tf.random.normal([self.latent_k, self.feat_size], mean=0.0, stddev=0.01, name='v', dtype=data_type, seed=random_state))

  def call(self, inputs):
    """Calculation of factorization machine of each calling in each training step.

    Args:
      inputs(dict): Dictionary of tensors of dataset for calculation.

    Returns:
      output: Predicted value of each observations.
    """
    input_layers = [self.input_layers[input_layer](ts) for input_layer, ts in inputs.items()]
    inputs = tf.keras.layers.Concatenate(name='inputs', dtype=tf.float32)(input_layers)
    axis = 1 if len(inputs.shape) > 1 else None

    linear = tf.reduce_sum(self.cal_linear(inputs, axis), axis=axis, keepdims=False)
    fm = tf.reduce_sum(self.cal_fm(inputs, axis), axis=axis, keepdims=False)

    output = self.w0 + linear + 0.5 * fm

    return output

  def cal_linear(self, inputs):
    """Calculation of linear part of factorization machine

    Args:
      inputs(dict): Dictionary of tensors of dataset for calculation.
    
    Returns:
      linear: Predicted value of linear part of each observations.
    """
    linear = inputs * self.w
    
    return linear

  def cal_fm(self, inputs):
    """Calculation of fm part of factorization machine

    Args:
      inputs(dict): Dictionary of tensors of dataset for calculation.
    
    Returns:
      linear: Predicted value of fm part of each observations.
    """
    fm = tf.subtract(
        tf.pow(tf.tensordot(inputs, tf.transpose(self.v), axes=1), 2),
        tf.tensordot(tf.pow(inputs, 2), tf.transpose(tf.pow(self.v, 2)), axes=1)
    )
    
    return fm

  def get_bias(self):
    """Return of bias weight

    Returns:
      w0: Bias weight.
    """
    return self.w0

  def get_linear_weight(self):
    """Return of linear weight

    Returns:
      w: Linear weight.
    """
    return self.w

  def get_embedding_weight(self):
    """Return of embedding weight

    Returns:
      v: Embedding weight.
    """
    return self.v