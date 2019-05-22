# coding=utf-8
#
# created by kpe on 28.Mar.2019 at 12:33
#

from __future__ import absolute_import, division, print_function

from tensorflow.python import keras

import bert
from bert.embeddings import BertEmbeddingsLayer
from bert.transformer import TransformerEncoderLayer


class BertModelLayer(bert.Layer):
    """
    BERT Model (arXiv:1810.04805).

    See: https://arxiv.org/pdf/1810.04805.pdf

    """
    class Params(BertEmbeddingsLayer.Params,
                 TransformerEncoderLayer.Params):
        pass

    # noinspection PyUnusedLocal
    def _construct(self, params: Params):
        self.embeddings_layer           = None
        self.encoders_layer             = None

        self.support_masking = True

    def build(self, input_shape):
        if isinstance(input_shape, list):
            assert len(input_shape) == 2
            input_ids_shape, token_type_ids_shape = input_shape
            self.input_spec = [keras.layers.InputSpec(shape=input_ids_shape),
                               keras.layers.InputSpec(shape=token_type_ids_shape)]
        else:
            input_ids_shape = input_shape
            self.input_spec = keras.layers.InputSpec(shape=input_ids_shape)

        self.embeddings_layer = BertEmbeddingsLayer.from_params(
            self.params,
            name="embeddings"
        )

        # create all transformer encoder sub-layers
        self.encoders_layer = TransformerEncoderLayer.from_params(
            self.params,
            name="encoder"
        )

        super(BertModelLayer, self).build(input_shape)

    def call(self, inputs, mask=None, training=None):
        if mask is None:
            mask = self.embeddings_layer.compute_mask(inputs)

        embedding_output = self.embeddings_layer(inputs, mask=mask, training=training)
        output           = self.encoders_layer(embedding_output, mask=mask, training=training)
        return output   # [B, seq_len, hidden_size]

