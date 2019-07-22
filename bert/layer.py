# coding=utf-8
#
# created by kpe on 28.Mar.2019 at 12:46
#

from __future__ import absolute_import, division, print_function

import tensorflow as tf
from params_flow import Layer
from params_flow.activations import gelu


class Layer(Layer):
    """ Common abstract base layer for all BERT layers. """
    class Params(Layer.Params):
        initializer_range = 0.02

    def create_initializer(self):
        return tf.compat.v1.initializers.truncated_normal(stddev=self.params.initializer_range)
        # return tf.compat.v2.initializers.TruncatedNormal(stddev=self.params.initializer_range)
        # TODO: TF < v2.0
        # return tf.truncated_normal_initializer(stddev=self.params.initializer_range)

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
