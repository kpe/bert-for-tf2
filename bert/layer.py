# coding=utf-8
#
# created by kpe on 28.Mar.2019 at 12:46
#

from __future__ import absolute_import, division, print_function

import tensorflow as tf

import params_flow as pf
from params_flow.activations import gelu


class Layer(pf.Layer):
    """ Common abstract base layer for all BERT layers. """
    class Params(pf.Layer.Params):
        initializer_range = 0.02

    def create_initializer(self):
        return tf.keras.initializers.TruncatedNormal(stddev=self.params.initializer_range)

    @staticmethod
    def get_activation(activation_string):
        if not isinstance(activation_string, str):
            return activation_string
        if not activation_string:
            return None

        act = activation_string.lower()
        if act == "linear":
            return None
        elif act == "relu":
            return tf.nn.relu
        elif act == "gelu":
            return gelu
        elif act == "tanh":
            return tf.tanh
        else:
            raise ValueError("Unsupported activation: %s" % act)
