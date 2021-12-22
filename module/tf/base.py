import numpy as np
import pandas as pd
from collections import OrderedDict
import tensorflow as tf

class Base(tf.keras.Model):
    """Base model for deepads tf module

    Args:
        input_config(dict): This arg is to input the dataset configuration and for the construction of the input.
        latent_k(int): This arg is to declare the embedding dimension of sparse features.
        random_state(int): This arg is to initialize the random seed. Default 42.
        data_type(object): This arg is to declare the weighting data type in tensorflow. Default tf.float32.

    Attributes:
        input_config(dict): This attribute is to storte the input_config argument.
        feat_size(int): This attribute is to store the length of feature size after encoding.
        latent_k(int): This attribute is to store the latent_k argument.
        input_layers(dict): This attribute is to store the model input layer structure.
    """
    def __init__(self, input_config, latent_k, random_state=42, data_type=tf.float32):
        super().__init__()
        self.input_config = input_config
        self.feat_size = sum([len(input_config[col]['unique_val']) + 1 if 'unique_val' in input_config[col] else 1 for col in input_config])
        self.latent_k = latent_k
        self.input_layers = self._init_input_layers(input_config) 

    def _init_input_layers(self, config):
        """Construction of model input layer

        Args:
            config(dict): Dictionary of dataset configuration including data type and unique value

        Returns:
            input_dict: the model input layer structure.
        """
        input_dict = OrderedDict()

        for col in config:
            if config[col]['data_type'] == tf.string:
                input_dict[col] = tf.keras.layers.StringLookup(name='{}_lookup'.format(col), vocabulary=config[col]['unique_val'], mask_token=None, output_mode='one_hot')
            else:
                input_dict[col] = tf.keras.layers.InputLayer(input_shape=(1,), name=col, dtype=config[col]['data_type'])

        return input_dict

    def train_step(self, data):
        """
        """
        x, y = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        self.compiled_metrics.update_state(y, y_pred)

        return {m.name: m.result() for m in self.metrics}