import numpy as np
import pandas as pd
import tensorflow as tf
from collections import OrderedDict
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
        self.v = self._init_v(input_config, random_state, data_type)

    def _init_v(self, config, random_state, data_type):
        """Initialization of features latent vector

        Args:
            config(dict): Dataset configuration
            random_state(int): Random see
            data_type(object): Weighting data type

        Returns:
            v_dict: Dictionary of features latent vector
        """
        v_dict = OrderedDict()

        for col in config:
            dims = len(config[col]['unique_val']) + 1 if 'unique_val' in config[col] else 1
            v_dict[col] = tf.Variable(tf.random.normal([dims, self.latent_k], mean=0.0, stddev=0.01, name='v_{}'.format(col), dtype=data_type, seed=random_state))

        return v_dict

    def call(self, inputs):
        """Calculation of factorization machine of each calling in each training step.

        Args:
            inputs(dict): Dictionary of tensors of dataset for calculation.

        Returns:
            output: Predicted value of each observations.
        """
        linear = self.cal_linear(inputs)
        fm = self.cal_fm(inputs)
        output = self.w0 + linear + 0.5 * fm

        return output

    def cal_linear(self, inputs):
        """Calculation of linear part of factorization machine

        Args:
            inputs(dict): Dictionary of tensors of dataset for calculation.
        
        Returns:
            linear: Predicted value of linear part of each observations.
        """
        inputs = [input_layer(inputs[layer_name]) for layer_name, input_layer in self.input_layers.items()]
        linear_inputs = tf.keras.layers.Concatenate(name='linear_inputs', dtype=tf.float32)(inputs)
        linear_axis = 1 if len(linear_inputs.shape) > 1 else None
        linear = linear_inputs * self.w
        linear = tf.reduce_sum(linear, axis=linear_axis, keepdims=True)

        return linear

    def cal_fm(self, inputs):
        """Calculation of fm part of factorization machine

        Args:
            inputs(dict): Dictionary of tensors of dataset for calculation.
        
        Returns:
            fm: Predicted value of fm part of each observations.
        """
        a = [tf.pow(tf.tensordot(input_layer(inputs[layer_name]), self.v[layer_name], axes=1), 2) for layer_name, input_layer in self.input_layers.items()]
        a = tf.stack(a, axis=1)
        b = [tf.tensordot(tf.pow(input_layer(inputs[layer_name]), 2), tf.pow(self.v[layer_name], 2), axes=1) for layer_name, input_layer in self.input_layers.items()]
        b = tf.stack(b, axis=1)

        fm = tf.subtract(a, b)
        fm = tf.reduce_sum(tf.reduce_sum(fm, axis=1), axis=1, keepdims=True)

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