# coding=utf-8
#
# created by mrinaald on 17.Oct.2020 at 12:33
#

from __future__ import absolute_import, division, print_function

import tensorflow as tf
import params_flow as pf

from bert.layer import Layer


class BertPoolerLayer(Layer):
    class Params(Layer.Params):
        hidden_size = 768

    def _construct(self, **kwargs):
        super()._construct(**kwargs)
        self.pooler_layer = None

    def build(self, input_shape):
        # Input Shape: (BATCH_SIZE, SEQUENCE_LENGTH, HIDDEN_SIZE)
        assert len(input_shape) == 3

        self.input_spec = tf.keras.layers.InputSpec(shape=input_shape)
        self.pooler_layer = tf.keras.layers.Dense(units=self.params.hidden_size,
                                                  activation='tanh',
                                                  kernel_initializer=self.create_initializer(),
                                                  name="dense")

        super(BertPoolerLayer, self).build(input_shape)

    def call(self, inputs, mask=None, training=None):
        first_token_tensor = inputs[:, 0, :]

        pooled_output = self.pooler_layer(first_token_tensor)
        return pooled_output
