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
    def _construct(self, **kwargs):
        super()._construct(**kwargs)
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

        assert_op = tf.compat.v2.debugging.assert_less_equal(seq_len, self.params.max_position_embeddings)

        with tf.control_dependencies([assert_op]):
            # slice to seq_len
            full_position_embeddings = tf.slice(self.embedding_table,
                                                [0, 0],
                                                [seq_len, -1])
        output = full_position_embeddings
        return output


class EmbeddingsProjector(bert.Layer):
    class Params(bert.Layer.Params):
        hidden_size                  = 768
        embedding_size               = None   # None for BERT, not None for ALBERT
        project_embeddings_with_bias = True   # in ALBERT - True for Google, False for brightmart/albert_zh

    # noinspection PyUnusedLocal
    def _construct(self, **kwargs):
        super()._construct(**kwargs)
        self.projector_layer      = None   # for ALBERT
        self.projector_bias_layer = None   # for ALBERT

    def build(self, input_shape):
        emb_shape = input_shape
        self.input_spec = keras.layers.InputSpec(shape=emb_shape)
        assert emb_shape[-1] == self.params.embedding_size

        # ALBERT word embeddings projection
        self.projector_layer = self.add_weight(name="projector",
                                               shape=[self.params.embedding_size,
                                                      self.params.hidden_size],
                                               dtype=K.floatx())
        if self.params.project_embeddings_with_bias:
            self.projector_bias_layer = self.add_weight(name="bias",
                                                        shape=[self.params.hidden_size],
                                                        dtype=K.floatx())
        super(EmbeddingsProjector, self).build(input_shape)

    def call(self, inputs, **kwargs):
        input_embedding = inputs
        assert input_embedding.shape[-1] == self.params.embedding_size

        # ALBERT: project embedding to hidden_size
        output = tf.matmul(input_embedding, self.projector_layer)
        if self.projector_bias_layer is not None:
            output = tf.add(output, self.projector_bias_layer)

        return output


class BertEmbeddingsLayer(bert.Layer):
    class Params(PositionEmbeddingLayer.Params,
                 EmbeddingsProjector.Params):
        vocab_size               = None
        use_token_type           = True
        use_position_embeddings  = True
        token_type_vocab_size    = 2
        hidden_size              = 768
        hidden_dropout           = 0.1

        extra_tokens_vocab_size  = None  # size of the extra (task specific) token vocabulary (using negative token ids)

        #
        # ALBERT support - set embedding_size (or None for BERT)
        #
        embedding_size               = None   # None for BERT, not None for ALBERT
        project_embeddings_with_bias = True   # in ALBERT - True for Google, False for brightmart/albert_zh
        project_position_embeddings  = True   # in ALEBRT - True for Google, False for brightmart/albert_zh

        mask_zero                    = False

    # noinspection PyUnusedLocal
    def _construct(self, **kwargs):
        super()._construct(**kwargs)
        self.word_embeddings_layer       = None
        self.extra_word_embeddings_layer = None   # for task specific tokens (negative token ids)
        self.token_type_embeddings_layer = None
        self.position_embeddings_layer   = None
        self.word_embeddings_projector_layer = None   # for ALBERT
        self.layer_norm_layer = None
        self.dropout_layer    = None

        self.support_masking = self.params.mask_zero

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
            mask_zero=self.params.mask_zero,
            name="word_embeddings"
        )
        if self.params.extra_tokens_vocab_size is not None:
            self.extra_word_embeddings_layer = keras.layers.Embedding(
                input_dim=self.params.extra_tokens_vocab_size + 1,  # +1 is for a <pad>/0 vector
                output_dim=embedding_size,
                mask_zero=self.params.mask_zero,
                embeddings_initializer=self.create_initializer(),
                name="extra_word_embeddings"
            )

        # ALBERT word embeddings projection
        if self.params.embedding_size is not None:
            self.word_embeddings_projector_layer = EmbeddingsProjector.from_params(
                self.params, name="word_embeddings_projector")

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
            token_mask   = tf.cast(tf.greater_equal(input_ids, 0), tf.int32)
            extra_mask   = tf.cast(tf.less(input_ids, 0), tf.int32)
            token_ids    = token_mask * input_ids
            extra_tokens = extra_mask * (-input_ids)
            token_output = self.word_embeddings_layer(token_ids)
            extra_output = self.extra_word_embeddings_layer(extra_tokens)
            embedding_output = tf.add(token_output,
                                      extra_output * tf.expand_dims(tf.cast(extra_mask, K.floatx()), axis=-1))
        else:
            embedding_output = self.word_embeddings_layer(input_ids)

        # ALBERT: for brightmart/albert_zh weights - project only token embeddings
        if not self.params.project_position_embeddings:
            if self.word_embeddings_projector_layer:
                embedding_output = self.word_embeddings_projector_layer(embedding_output)

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

        # ALBERT: for google-research/albert weights - project all embeddings
        if self.params.project_position_embeddings:
            if self.word_embeddings_projector_layer:
                embedding_output = self.word_embeddings_projector_layer(embedding_output)

        return embedding_output   # [B, seq_len, hidden_size]

    def compute_mask(self, inputs, mask=None):
        if isinstance(inputs, list):
            assert 2 == len(inputs), "Expecting inputs to be a [input_ids, token_type_ids] list"
            input_ids, token_type_ids = inputs
        else:
            input_ids      = inputs
            token_type_ids = None

        if not self.support_masking:
            return None

        return tf.not_equal(input_ids, 0)
