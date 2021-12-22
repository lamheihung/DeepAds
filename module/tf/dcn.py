import numpy as np
import pandas as pd
import tensorflow as tf
from collections import OrderedDict
from base import Base

class DCN(Base):
    """Deep & Cross Recommender

    Args:
        cross_level(int): This arg is to input the number of level in the CROSS part.
        dnn_unit(list(int)): This arg is to input the number of unit of each layer in DNN part.
        activation(str): This arg is to input the activation of each layer in DNN part.
        is_bn(bool): This arg is to input the usage of BatchNormalization.
        is_dropout(float): This arg is to input the dropout weight.
        input_config(dict): This arg is to input the dataset configuration and for the construction of the input.
        latent_k(int): This arg is to declare the embedding dimension of sparse features.
        random_state(int): This arg is to initialize the random seed. Default 42.
        data_type(object): This arg is to declare the weighting data type in tensorflow. Default tf.float32.

    Attributes:
        v(dict(tensor)): This attribute is to store the dictionary of features latent vector.
        cross_w(list(tensor)): This attribute is to store ...
        cross_w0(list(tensor)): This attribute is to store ...
        dnn_layers(list(layer)): This attribute is to store the model structure of DNN part.
        output_layer(layer): This attribute is to store the output layer of the model.
    """
    def __init__(self, cross_level, dnn_unit, activation, is_bn, is_dropout, input_config, latent_k, random_state=42, data_type=tf.float32):
        super().__init__(input_config=input_config, latent_k=latent_k, random_state=random_state, data_type=data_type)
        self.v, self.cross_w, self.cross_w0 = self._init_cross_struct(cross_level, input_config, random_state, data_type)
        self.dnn_layers = self._init_dnn_struct(dnn_unit, activation, is_bn, is_dropout)
        self.output_layer = tf.keras.layers.Dense(units=1, activation='sigmoid')

    def _init_cross_struct(self, cross_level, config, random_state, data_type):
        """Initalization of model of CROSS part

        Args:
            cross_level(int): Number of level in CROSS part.
            config(dict): Dataset configuration.
            random_state(int): Random seed.
            data_type(object): Weighting data type.

        Returns:
            v_dict: Dictionary of features latent vector
            cross_w: List of weight of cross layer
            cross_w0: List of bias of cross layer
        """
        v_dict = OrderedDict()
        size = 0

        for col in config:
            dims = len(config[col]['unique_val']) + 1 if 'unique_val' in config[col] else 1
            size += self.latent_k
            v_dict[col] = tf.Variable(tf.random.normal([dims, self.latent_k], mean=0.0, stddev=0.01, name='v_{}'.format(col), dtype=data_type, seed=random_state))

        cross_w = [tf.Variable(tf.zeros([size], dtype=data_type, name='w_{}'.format(i))) for i in range(cross_level)]
        cross_w0 = [tf.Variable(tf.zeros([size], dtype=data_type, name='w0_{}'.format(i))) for i in range(cross_level)]

        return v_dict, cross_w, cross_w0

    def _init_dnn_struct(self, dnn_unit, activation, is_bn, is_dropout):
        """Initalization of model of DNN part

        Args:
            dnn_unit(list(int)): This arg is to input the number of unit of each layer in DNN part.
            activation(str): This arg is to input the activation of each layer in DNN part.
            is_bn(bool): This arg is to input the usage of BatchNormalization.
            is_dropout(float): This arg is to input the dropout weight.

        Returns:
            dnn_list: List of layer for the DNN part.
        """
        dnn_list = list()

        for unit in dnn_unit:
            dnn_list += [tf.keras.layers.Dense(unit, activation=activation)]
            if is_bn:
                dnn_list += [tf.keras.layers.BatchNormalization()]
            if is_dropout:
                dnn_list += [tf.keras.layers.Dropout(is_dropout)]
        
        return dnn_list

    def call(self, inputs):
        """Calculation of Deep & Cross of each calling in each training step.

        Args:
            inputs(dict): Dictionary of tensors of dataset for calculation.

        Returns:
            output: Predicted value of each observations:
        """
        inputs = [tf.tensordot(input_layer(inputs[layer_name]), self.v[layer_name], axes=1) for layer_name, input_layer in self.input_layers.items()]
        inputs = tf.keras.layers.Concatenate(name='dnn_inputs', dtype=tf.float32)(inputs)
        cross = self.cal_cross(inputs)
        dnn = self.cal_dnn(inputs)
        output = tf.keras.layers.Concatenate(name='output')([cross, dnn])
        output = self.output_layer(output)
    
        return output

    def cal_cross(self, inputs):
        """Calculation of cross part of Deep & Cross Recommender

        Args:
            inputs(vector): Concatenated input vector.
        
        Returns:
            vector: Predicted value of cross part of each observations.
        """
        for _idx, (w, w0) in enumerate(zip(self.cross_w, self.cross_w0)):
            if _idx == 0:
                vector = tf.tensordot(inputs, tf.transpose(inputs), axes=1) * w + w0 + inputs
            else:
                vector = tf.tensordot(inputs, tf.transpose(vector), axes=1) * w + w0 + vector

        return vector

    def cal_dnn(self, inputs):
        """Calculation of dnn part of Deep & Cross Recommender

        Args:
            inputs(vector): Concatenated input vector.
        
        Returns:
            dnn: Predicted value of dnn part of each observations.
        """
        for _idx, layer in enumerate(self.dnn_layers):
            if _idx == 0:
                dnn = layer(inputs)
            else:
                dnn = layer(dnn)
        
        return dnn