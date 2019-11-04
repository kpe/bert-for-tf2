# coding=utf-8
#
# created by kpe on 10.Oct.2019 at 15:41
#

from __future__ import absolute_import, division, print_function
import unittest

import os
import tempfile

import numpy as np
import tensorflow as tf
from tensorflow import keras

import bert

from .test_common import AbstractBertTest, MiniBertFactory


class AlbertTest(AbstractBertTest):

    def setUp(self) -> None:
        tf.compat.v1.reset_default_graph()
        tf.compat.v1.enable_eager_execution()
        print("Eager Execution:", tf.executing_eagerly())

    def test_albert(self):
        bert_params = bert.BertModelLayer.Params(hidden_size=32,
                                                 vocab_size=67,
                                                 max_position_embeddings=64,
                                                 num_layers=1,
                                                 num_heads=1,
                                                 intermediate_size=4,
                                                 use_token_type=False,

                                                 embedding_size=16,  # using ALBERT instead of BERT
                                                 project_embeddings_with_bias=False,
                                                 shared_layer=True,
                                                 )

        def to_model(bert_params):
            l_bert = bert.BertModelLayer.from_params(bert_params)

            token_ids = keras.layers.Input(shape=(21,))
            seq_out = l_bert(token_ids)
            model = keras.Model(inputs=[token_ids], outputs=seq_out)

            model.build(input_shape=(None, 21))
            l_bert.apply_adapter_freeze()

            return model

        model = to_model(bert_params)
        model.summary()

        print(len(model.trainable_weights))
        self.assertEqual(21, len(model.trainable_weights))
        for weight in model.trainable_weights:
            print(weight.name, weight.shape)

        # adapter-ALBERT  :-)

        bert_params.adapter_size = 16

        model = to_model(bert_params)
        model.summary()
        print(len(model.trainable_weights))
        self.assertEqual(15, len(model.trainable_weights))
        for weight in model.trainable_weights:
            print(weight.name, weight.shape)

    def test_albert_load_base_google_weights(self):  # for coverage mainly
        albert_model_name = "albert_base"
        albert_dir = bert.fetch_tfhub_albert_model(albert_model_name, ".models")
        model_params = bert.albert_params(albert_model_name)

        l_bert = bert.BertModelLayer.from_params(model_params, name="albert")

        model = keras.models.Sequential([
            keras.layers.InputLayer(input_shape=(8,), dtype=tf.int32, name="input_ids"),
            l_bert,
            keras.layers.Lambda(lambda x: x[:, 0, :]),
            keras.layers.Dense(2),
        ])
        model.build(input_shape=(None, 8))
        model.compile(optimizer=keras.optimizers.Adam(),
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[keras.metrics.SparseCategoricalAccuracy(name="acc")])

        bert.load_albert_weights(l_bert, albert_dir)

        model.summary()
