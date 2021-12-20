import numpy as np
import pandas as pd
import tensorflow as tf

from fm import FM

class DeepFM(FM):
    """Deep Factorization Machine

    Args:
        dnn_unit(list(int)): This arg is to input the number of unit of each layer in DNN part.
        activation(str): This arg is to input the activation of each layer in DNN part.
        is_bn(bool): This arg is to input the usage of BatchNormalization.
        is_dropout(float): This arg is to input the dropout weight.
        input_config(dict): This arg is to input the dataset configuration and for the construction of the input.
        latent_k(int): This arg is to declare the embedding dimension of sparse features.
        random_state(int): This arg is to initialize the random seed. Default 42.
        data_type(object): This arg is to declare the weighting data type in tensorflow. Default tf.float32.

    Attributes:
        dnn_layers(list(layer)): This attribute is to store the model structure of DNN part.
        output_layer(layer): This attribute is to store the output layer of the model.
    """
    def __init__(self, dnn_unit, activation, is_bn, is_dropout, input_config, latent_k, random_state=42, data_type=tf.float32):
        super().__init__(input_config=input_config, latent_k=latent_k, random_state=random_state, data_type=data_type)
        self.dnn_layers = self._init_dnn_struct(dnn_unit, activation, is_bn, is_dropout)
        self.output_layer = tf.keras.layers.Dense(units=1, activation='sigmoid')

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
        """Calculation of deep factorization machine of each calling in each training step.

        Args:
            inputs(dict): Dictionary of tensors of dataset for calculation.

        Returns:
            output: Predicted value of each observations.
        """
        linear = self.cal_linear(inputs)
        fm = self.cal_fm(inputs)
        dnn = self.cal_dnn(inputs)
        output = tf.keras.layers.Concatenate(name='output')([linear, fm, dnn])
        output = self.output_layer(output)

        return output

    def cal_dnn(self, inputs):
        """Calculation of dnn part of deep factorization machine

        Args:
            inputs(dict): Dictionary of tensors of dataset for calculation.
        
        Returns:
            linear: Predicted value of linear part of each observations.
        """
        inputs = [tf.tensordot(input_layer(inputs[layer_name]), self.v[layer_name], axes=1) for layer_name, input_layer in self.input_layers.items()]
        dnn = tf.keras.layers.Concatenate(name='dnn_inputs', dtype=tf.float32)(inputs)

        for layer in self.dnn_layers:
            dnn = layer(dnn)
        
        return dnn