# coding=utf-8
#
# created by kpe on 28.Mar.2019 at 12:33
#

from __future__ import absolute_import, division, print_function

import tensorflow as tf
import params_flow as pf

from tensorflow import keras
from tensorflow.keras import backend as K

import bert


class PositionEmbeddingLayer(bert.Layer):
    class Params(bert.Layer.Params):
        max_position_embeddings  = 512
        hidden_size              = 128

    # noinspection PyUnusedLocal
    def _construct(self, params: Params):
        self.embedding_table = None

    # noinspection PyAttributeOutsideInit
    def build(self, input_shape):
        # input_shape: () of seq_len
        if input_shape is not None:
            assert input_shape.ndims == 0
            self.input_spec = keras.layers.InputSpec(shape=input_shape, dtype='int32')
        else:
            self.input_spec = keras.layers.InputSpec(shape=(), dtype='int32')

        self.embedding_table = self.add_weight(name="embeddings",
                                               dtype=K.floatx(),
                                               shape=[self.params.max_position_embeddings, self.params.hidden_size],
                                               initializer=self.create_initializer())
        super(PositionEmbeddingLayer, self).build(input_shape)

    # noinspection PyUnusedLocal
    def call(self, inputs, **kwargs):
        # just return the embedding after verifying
        # that seq_len is less than max_position_embeddings
        seq_len = inputs

        assert_op = tf.compat.v1.assert_less_equal(seq_len, self.params.max_position_embeddings)
        # TODO: TF < v2.0
        # assert_op = tf.assert_less_equal(seq_len, self.params.max_position_embeddings)

        with tf.control_dependencies([assert_op]):
            # slice to seq_len
            full_position_embeddings = tf.slice(self.embedding_table,
                                                [0, 0],
                                                [seq_len, -1])
        output = full_position_embeddings
        return output


class BertEmbeddingsLayer(bert.Layer):
    class Params(PositionEmbeddingLayer.Params):
        vocab_size               = None
        use_token_type           = True
        use_position_embeddings  = True
        token_type_vocab_size    = 2
        hidden_size              = 768
        hidden_dropout           = 0.1

        #
        # ALBERT support - set embedding_size (or None for BERT)
        #
        embedding_size               = None   # None for BERT, not None for ALBERT
        project_embeddings_with_bias = True   # in ALBERT - True for Google, False for brightmart/albert_zh
        project_position_embeddings  = True   # in ALEBRT - True for Google, False for brightmart/albert_zh

        extra_tokens_vocab_size  = None  # size of the extra (task specific) token vocabulary (using negative token ids)

    # noinspection PyUnusedLocal
    def _construct(self, params: Params):
        self.word_embeddings_layer        = None
        self.word_embeddings_2_layer      = None   # for ALBERT
        self.word_embeddings_2_layer_bias = None   # for ALBERT
        self.extra_word_embeddings_layer  = None   # for task specific tokens (negative token ids)
        self.token_type_embeddings_layer = None
        self.position_embeddings_layer   = None
        self.layer_norm_layer = None
        self.dropout_layer    = None

        self.support_masking = True

    # noinspection PyAttributeOutsideInit
    def build(self, input_shape):
        if isinstance(input_shape, list):
            assert len(input_shape) == 2
            input_ids_shape, token_type_ids_shape = input_shape
            self.input_spec = [keras.layers.InputSpec(shape=input_ids_shape),
                               keras.layers.InputSpec(shape=token_type_ids_shape)]
        else:
            input_ids_shape = input_shape
            self.input_spec = keras.layers.InputSpec(shape=input_ids_shape)

        # use either hidden_size for BERT or embedding_size for ALBERT
        embedding_size = self.params.hidden_size if self.params.embedding_size is None else self.params.embedding_size

        self.word_embeddings_layer = keras.layers.Embedding(
            input_dim=self.params.vocab_size,
            output_dim=embedding_size,
            mask_zero=True,
            name="word_embeddings"
        )
        if self.params.extra_tokens_vocab_size is not None:
            self.extra_word_embeddings_layer = keras.layers.Embedding(
                input_dim=self.params.extra_tokens_vocab_size + 1,  # +1 is for a <pad>/0 vector
                output_dim=embedding_size,
                mask_zero=True,
                name="extra_word_embeddings"
            )

        if self.params.embedding_size is not None:
            # ALBERT word embeddings projection
            self.word_embeddings_2_layer = self.add_weight(name="word_embeddings_2/embeddings",
                                                           shape=[self.params.embedding_size,
                                                                  self.params.hidden_size],
                                                           dtype=K.floatx())
            if self.params.project_embeddings_with_bias:
                self.word_embeddings_2_layer_bias = self.add_weight(
                    name="word_embeddings_2/bias",
                    shape=[self.params.hidden_size],
                    dtype=K.floatx())

        position_embedding_size = embedding_size if self.params.project_position_embeddings else self.params.hidden_size

        if self.params.use_token_type:
            self.token_type_embeddings_layer = keras.layers.Embedding(
                input_dim=self.params.token_type_vocab_size,
                output_dim=position_embedding_size,
                mask_zero=False,
                name="token_type_embeddings"
            )
        if self.params.use_position_embeddings:
            self.position_embeddings_layer = PositionEmbeddingLayer.from_params(
                self.params,
                name="position_embeddings",
                hidden_size=position_embedding_size
            )

        self.layer_norm_layer = pf.LayerNormalization(name="LayerNorm")
        self.dropout_layer    = keras.layers.Dropout(rate=self.params.hidden_dropout)

        super(BertEmbeddingsLayer, self).build(input_shape)

    def call(self, inputs, mask=None, training=None):
        if isinstance(inputs, list):
            assert 2 == len(inputs), "Expecting inputs to be a [input_ids, token_type_ids] list"
            input_ids, token_type_ids = inputs
        else:
            input_ids      = inputs
            token_type_ids = None

        input_ids = tf.cast(input_ids, dtype=tf.int32)

        if self.extra_word_embeddings_layer is not None:
            extra_tokens = tf.where(input_ids < 0, -input_ids, tf.zeros_like(input_ids))
            token_ids    = tf.where(input_ids >= 0, input_ids, tf.zeros_like(input_ids))
            token_output = self.word_embeddings_layer(token_ids)
            extra_output = self.extra_word_embeddings_layer(extra_tokens)
            embedding_output = tf.where(tf.expand_dims(input_ids >= 0, -1), token_output, extra_output)
        else:
            embedding_output = self.word_embeddings_layer(input_ids)

        def project_embedding(emb):
            if self.word_embeddings_2_layer is not None:  # ALBERT: project embedding to hidden_size
                emb = tf.matmul(emb, self.word_embeddings_2_layer)
                if self.word_embeddings_2_layer_bias is not None:
                    emb += self.word_embeddings_2_layer_bias
            return emb

        if not self.params.project_position_embeddings:   # ALBERT: for brightmart/albert_zh weights
            embedding_output = project_embedding(embedding_output)

        if token_type_ids is not None:
            token_type_ids    = tf.cast(token_type_ids, dtype=tf.int32)
            embedding_output += self.token_type_embeddings_layer(token_type_ids)

        if self.position_embeddings_layer is not None:
            seq_len  = input_ids.shape.as_list()[1]
            emb_size = embedding_output.shape[-1]

            pos_embeddings = self.position_embeddings_layer(seq_len)
            # broadcast over all dimension except the last two [..., seq_len, width]
            broadcast_shape = [1] * (embedding_output.shape.ndims - 2) + [seq_len, emb_size]
            embedding_output += tf.reshape(pos_embeddings, broadcast_shape)

        embedding_output = self.layer_norm_layer(embedding_output)
        embedding_output = self.dropout_layer(embedding_output, training=training)

        if self.params.project_position_embeddings:         # ALBERT: for google weights
            embedding_output = project_embedding(embedding_output)

        return embedding_output   # [B, seq_len, hidden_size]

    def compute_mask(self, inputs, mask=None):
        if isinstance(inputs, list):
            assert 2 == len(inputs), "Expecting inputs to be a [input_ids, token_type_ids] list"
            input_ids, token_type_ids = inputs
        else:
            input_ids      = inputs
            token_type_ids = None

        # if not self.mask_zero:
        #   return None

        return tf.not_equal(input_ids, 0)
