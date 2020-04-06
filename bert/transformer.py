# coding=utf-8
#
# created by kpe on 20.Mar.2019 at 16:30
#

from __future__ import absolute_import, division, print_function

import tensorflow as tf
from tensorflow.python import keras

from params_flow import LayerNormalization

from bert.attention import AttentionLayer
from bert.layer import Layer


class ProjectionLayer(Layer):
    class Params(Layer.Params):
        hidden_size        = None
        hidden_dropout     = 0.1
        initializer_range  = 0.02
        adapter_size       = None       # bottleneck size of the adapter - arXiv:1902.00751
        adapter_activation = "gelu"
        adapter_init_scale = 1e-3

    def _construct(self, **kwargs):
        super()._construct(**kwargs)
        self.dense      = None
        self.dropout    = None
        self.layer_norm = None

        self.adapter_down = None
        self.adapter_up   = None

        self.supports_masking = True

    # noinspection PyAttributeOutsideInit
    def build(self, input_shape):
        assert isinstance(input_shape, list) and 2 == len(input_shape)
        out_shape, residual_shape = input_shape
        self.input_spec = [keras.layers.InputSpec(shape=out_shape),
                           keras.layers.InputSpec(shape=residual_shape)]

        self.dense = keras.layers.Dense(units=self.params.hidden_size,
                                        kernel_initializer=self.create_initializer(),
                                        name="dense")
        self.dropout    = keras.layers.Dropout(rate=self.params.hidden_dropout)
        self.layer_norm = LayerNormalization(name="LayerNorm")

        if self.params.adapter_size is not None:
            self.adapter_down = keras.layers.Dense(units=self.params.adapter_size,
                                                   kernel_initializer=tf.keras.initializers.TruncatedNormal(
                                                       stddev=self.params.adapter_init_scale),
                                                   activation=self.get_activation(self.params.adapter_activation),
                                                   name="adapter-down")
            self.adapter_up   = keras.layers.Dense(units=self.params.hidden_size,
                                                   kernel_initializer=tf.keras.initializers.TruncatedNormal(
                                                       stddev=self.params.adapter_init_scale),
                                                   name="adapter-up")

        super(ProjectionLayer, self).build(input_shape)

    def call(self, inputs, mask=None, training=None, **kwargs):
        output, residual = inputs
        output = self.dense(output)
        output = self.dropout(output, training=training)

        if self.adapter_down is not None:
            adapted = self.adapter_down(output)
            adapted = self.adapter_up(adapted)
            output = tf.add(output, adapted)

        output = self.layer_norm(tf.add(output, residual))
        return output


class TransformerSelfAttentionLayer(Layer):
    class Params(ProjectionLayer.Params,
                 AttentionLayer.Params):
        hidden_size         = None
        num_heads           = None
        hidden_dropout      = None
        attention_dropout   = 0.1
        initializer_range   = 0.02

    def _construct(self, **kwargs):
        super()._construct(**kwargs)
        params = self.params
        if params.hidden_size % params.num_heads != 0:
            raise ValueError("The hidden_size:[{}] is not a multiple of num_heads:[{}]".format(params.hidden_size,
                                                                                               params.num_heads))
        self.size_per_head = params.hidden_size // params.num_heads
        assert params.size_per_head is None or self.size_per_head == params.size_per_head

        self.attention_layer     = None
        self.attention_projector = None

        self.supports_masking = True

    def build(self, input_shape):
        self.input_spec = keras.layers.InputSpec(shape=input_shape)

        self.attention_layer = AttentionLayer.from_params(
            self.params,
            size_per_head=self.size_per_head,
            name="self",
        )
        self.attention_projector = ProjectionLayer.from_params(
            self.params,
            name="output",
        )

        super(TransformerSelfAttentionLayer, self).build(input_shape)

    def call(self, inputs, mask=None, training=None):
        layer_input = inputs

        #
        # TODO: is it OK to recompute the 3D attention mask in each attention layer
        #
        attention_head   = self.attention_layer(layer_input, mask=mask, training=training)
        attention_output = self.attention_projector([attention_head, layer_input], mask=mask, training=training)

        return attention_output


class SingleTransformerEncoderLayer(Layer):
    """
    Multi-headed, single layer for the Transformer from 'Attention is All You Need' (arXiv: 1706.03762).

    See also: https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/models/transformer.py
    """

    class Params(TransformerSelfAttentionLayer.Params,
                 ProjectionLayer.Params):
        intermediate_size       = None
        intermediate_activation = "gelu"

    def _construct(self, **kwargs):
        super()._construct(**kwargs)
        params = self.params
        if params.hidden_size % params.num_heads != 0:
            raise ValueError("The hidden_size:[{}] is not a multiple of num_heads:[{}]".format(params.hidden_size,
                                                                                               params.num_heads))
        self.size_per_head = params.hidden_size // params.num_heads

        self.self_attention_layer = None
        self.intermediate_layer   = None
        self.output_projector     = None

        self.supports_masking = True

    def build(self, input_shape):
        self.input_spec = keras.layers.InputSpec(shape=input_shape)  # [B, seq_len, hidden_size]

        self.self_attention_layer = TransformerSelfAttentionLayer.from_params(
            self.params,
            name="attention"
        )
        self.intermediate_layer = keras.layers.Dense(
            name="intermediate",
            units=self.params.intermediate_size,
            activation=self.get_activation(self.params.intermediate_activation),
            kernel_initializer=self.create_initializer()
        )
        self.output_projector = ProjectionLayer.from_params(
            self.params,
            name="output",
        )

        super(SingleTransformerEncoderLayer, self).build(input_shape)

    def call(self, inputs, mask=None, training=None):
        layer_input = inputs

        attention_output    = self.self_attention_layer(layer_input, mask=mask, training=training)

        # intermediate
        intermediate_output = self.intermediate_layer(attention_output)

        # output
        layer_output = self.output_projector([intermediate_output, attention_output], mask=mask)

        return layer_output


class TransformerEncoderLayer(Layer):
    """
    Multi-headed, multi-layer Transformer from 'Attention is All You Need' (arXiv: 1706.03762).

    Implemented for BERT, with support for ALBERT (sharing encoder layer params).

    See also: https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/models/transformer.py
    """

    class Params(SingleTransformerEncoderLayer.Params):
        num_layers     = None
        out_layer_ndxs = None   # [-1]

        shared_layer   = False  # False for BERT, True for ALBERT

    def _construct(self, **kwargs):
        super()._construct(**kwargs)
        self.encoder_layers   = []
        self.shared_layer     = None  # for ALBERT
        self.supports_masking = True

    def build(self, input_shape):
        self.input_spec = keras.layers.InputSpec(shape=input_shape)

        # create all transformer encoder sub-layers
        if self.params.shared_layer:
            # ALBERT: share params
            self.shared_layer = SingleTransformerEncoderLayer.from_params(self.params, name="layer_shared")
        else:
            # BERT
            for layer_ndx in range(self.params.num_layers):
                encoder_layer = SingleTransformerEncoderLayer.from_params(
                    self.params,
                    name="layer_{}".format(layer_ndx),
                )
                self.encoder_layers.append(encoder_layer)

        super(TransformerEncoderLayer, self).build(input_shape)

    def call(self, inputs, mask=None, training=None):
        layer_output = inputs

        layer_outputs = []
        for layer_ndx in range(self.params.num_layers):
            encoder_layer = self.encoder_layers[layer_ndx] if self.encoder_layers else self.shared_layer
            layer_input = layer_output

            layer_output = encoder_layer(layer_input, mask=mask, training=training)
            layer_outputs.append(layer_output)

        if self.params.out_layer_ndxs is None:
            # return the final layer only
            final_output = layer_output
        else:
            final_output = []
            for ndx in self.params.out_layer_ndxs:
                final_output.append(layer_outputs[ndx])
            final_output = tuple(final_output)

        return final_output


