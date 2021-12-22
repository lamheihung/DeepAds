import numpy as np
import pandas as pd
import tensorflow as tf
from collections import OrderedDict
from base import Base

class DeepCrossing(Base):
    """DeepCrossing Recommender

    Args:
        residual_level(int): This arg is to input the number of level in mutiple residual unit layer.
        input_config(dict): This arg is to input the dataset configuration and for the construction of the input.
        latent_k(int): This arg is to declare the embedding dimension of sparse features.
        random_state(int): This arg is to initialize the random seed. Default 42.
        data_type(object): This arg is to declare the weighting data type in tensorflow. Default tf.float32.

    Attributes:
        v(dict(tensor)): This attribute is to store the dictionary of features latent vector.
        r(list(tensor)): This attribute is to store the list of residual unit layer.
        output_layer(layer): This attribute is to store the output layer of the model.
    """
    def __init__(self, residual_level, input_config, latent_k, random_state=42, data_type=tf.float32):
        super().__init__(input_config=input_config, latent_k=latent_k, random_state=random_state, data_type=data_type)
        self.v, self.r = self._init_model_structure(residual_level, input_config, random_state, data_type)
        self.output_layer = tf.keras.layers.Dense(units=1, activation='sigmoid')

    def _init_model_structure(self, residual_level, config, random_state, data_type):
        """Initialization of model structure

        Args:
            residual_level(int): Number of level in mutiple residual unit layer.
            config(dict): Dataset configuration.
            random_state(int): Random seed.
            data_type(object): Weighting data type.

        Returns:
            v_dict: Dictionary of features latent vector.
            mutiple_residual_unit: List of layer in mutiple residual units layer.
        """
        v_dict = OrderedDict()
        size = 0

        for col in config:
            dims = len(config[col]['unique_val']) + 1 if 'unique_val' in config[col] else 1
            size += self.latent_k if 'unique_val' in config[col] else 1
            v_dict[col] = tf.Variable(tf.random.normal([dims, self.latent_k], mean=0.0, stddev=0.01, name='v_{}'.format(col), dtype=data_type, seed=random_state))

        mutiple_residual_unit = [tf.keras.layers.Dense(size, activation='relu') for _ in range(residual_level)]

        return v_dict, mutiple_residual_unit

    def call(self, inputs):
        """Calculation of DeepCrossing of each calling in each training step.

        Args:
            inputs(dict): Dictionary of tensors of dataset for calculation.

        Returns:
            output: Predicted value of each observations:
        """
        inputs = [tf.tensordot(input_layer(inputs[layer_name]), self.v[layer_name], axes=1) for layer_name, input_layer in self.input_layers.items()]
        inputs = tf.keras.layers.Concatenate(name='inputs', dtype=tf.float32)(inputs)
        residual_layer = self.cal_residual(inputs)
        output = self.output_layer(residual_layer)

        return output

    def cal_residual(self, inputs):
        """Calculation of multiple residual units part of DeepCrossing Recommender

        Args:
            inputs(vector): Concatenated input vector.
        
        Returns:
            inputs: Predicted value of multiple residual units part of each observations.
        """
        for layer in self.r:
            inputs = layer(inputs) + inputs

        return inputs