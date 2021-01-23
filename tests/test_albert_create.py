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
                                                 project_embeddings_with_bias=True,
                                                 shared_layer=True,
                                                 extra_tokens_vocab_size=3,
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

        print("trainable_weights:", len(model.trainable_weights))
        for weight in model.trainable_weights:
            print(weight.name, weight.shape)
        self.assertEqual(23, len(model.trainable_weights))

        # adapter-ALBERT  :-)

        bert_params.adapter_size = 16

        model = to_model(bert_params)
        model.summary()

        print("trainable_weights:", len(model.trainable_weights))
        for weight in model.trainable_weights:
            print(weight.name, weight.shape)
        self.assertEqual(15, len(model.trainable_weights))

        print("non_trainable_weights:", len(model.non_trainable_weights))
        for weight in model.non_trainable_weights:
            print(weight.name, weight.shape)
        self.assertEqual(16, len(model.non_trainable_weights))

    def test_albert_with_pooler(self):
        bert_params = bert.BertModelLayer.Params(hidden_size=32,
                                                 vocab_size=67,
                                                 max_position_embeddings=64,
                                                 num_layers=1,
                                                 num_heads=1,
                                                 intermediate_size=4,
                                                 use_token_type=False,

                                                 embedding_size=16,  # using ALBERT instead of BERT
                                                 project_embeddings_with_bias=True,
                                                 shared_layer=True,
                                                 extra_tokens_vocab_size=3,

                                                 return_pooler_output=True
                                                 )


        def to_model(bert_params):
            l_bert = bert.BertModelLayer.from_params(bert_params)

            token_ids = keras.layers.Input(shape=(21,))
            _, pooler_out = l_bert(token_ids)
            model = keras.Model(inputs=[token_ids], outputs=pooler_out)

            model.build(input_shape=(None, 21))
            l_bert.apply_adapter_freeze()

            return model

        model = to_model(bert_params)
        model.summary()

        print("trainable_weights:", len(model.trainable_weights))
        print(bert_params.adapter_size)
        for weight in model.trainable_weights:
            print(weight.name, weight.shape)
        self.assertEqual(25, len(model.trainable_weights))

        # adapter-ALBERT  :-)

        bert_params.adapter_size = 16

        model = to_model(bert_params)
        model.summary()

        print("trainable_weights:", len(model.trainable_weights))
        for weight in model.trainable_weights:
            print(weight.name, weight.shape)
        self.assertEqual(15, len(model.trainable_weights))

        print("non_trainable_weights:", len(model.non_trainable_weights))
        for weight in model.non_trainable_weights:
            print(weight.name, weight.shape)
        self.assertEqual(18, len(model.non_trainable_weights))

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

        # return_pooler_output
        model_params.return_pooler_output = True
        l_bert = bert.BertModelLayer.from_params(model_params, name="albert_with_pooler")

        inp = keras.layers.Input(shape=(8,), dtype=tf.int32, name="input_ids")
        _, pooler_out = l_bert(inp)
        out = keras.layers.Dense(2)(pooler_out)
        model = keras.Model(inputs=[inp], outputs=out)
        model.compile(optimizer=keras.optimizers.Adam(),
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[keras.metrics.SparseCategoricalAccuracy(name="acc")])

        bert.load_albert_weights(l_bert, albert_dir)

        model.summary()

    def test_albert_params(self):
        albert_model_name = "albert_base"
        albert_dir = bert.fetch_tfhub_albert_model(albert_model_name, ".models")
        dir_params = bert.albert_params(albert_dir)
        dir_params.attention_dropout = 0.1  # diff between README and assets/albert_config.json
        dir_params.hidden_dropout = 0.1
        name_params = bert.albert_params(albert_model_name)
        self.assertEqual(name_params, dir_params)

        # coverage
        model_params = dir_params
        model_params.vocab_size = model_params.vocab_size
        model_params.adapter_size = 1
        l_bert = bert.BertModelLayer.from_params(model_params, name="albert")
        l_bert(tf.zeros((1, 128)))
        bert.load_albert_weights(l_bert, albert_dir)

        # coverage: return_pooler_output
        model_params = dir_params
        model_params.vocab_size = model_params.vocab_size
        model_params.return_pooler_output = True
        l_bert = bert.BertModelLayer.from_params(model_params, name="albert_with_pooler")
        l_bert(tf.zeros((1, 128)))
        bert.load_albert_weights(l_bert, albert_dir)

    def test_albert_zh_fetch_and_load(self):
        albert_model_name = "albert_tiny"
        albert_dir = bert.fetch_brightmart_albert_model(albert_model_name, ".models")

        model_params = bert.params_from_pretrained_ckpt(albert_dir)
        model_params.vocab_size = model_params.vocab_size + 2
        model_params.adapter_size = 1
        l_bert = bert.BertModelLayer.from_params(model_params, name="albert")
        l_bert(tf.zeros((1, 128)))
        res = bert.load_albert_weights(l_bert, albert_dir)
        self.assertTrue(len(res) > 0)

    def test_coverage(self):
        try:
            bert.fetch_google_bert_model("not-existent_bert_model", ".models")
        except:
            pass

        albert_model_name = "albert_tiny"
        albert_dir = bert.fetch_brightmart_albert_model(albert_model_name, ".models")

        model_params = bert.params_from_pretrained_ckpt(albert_dir)
        model_params.vocab_size = model_params.vocab_size + 2
        l_bert = bert.BertModelLayer.from_params(model_params, name="albert")

        seq_len = 128
        input_ids_shape = (1, seq_len)
        token_type_ids_shape = (1, seq_len)

        output_shape = l_bert.compute_output_shape(input_ids_shape)
        self.assertTrue(len(output_shape) == 3)
        self.assertTrue(output_shape[0] == 1)
        self.assertTrue(output_shape[1] == seq_len)
        self.assertTrue(output_shape[2] == model_params.hidden_size)

        output_shape = l_bert.compute_output_shape([input_ids_shape, token_type_ids_shape])
        self.assertTrue(len(output_shape) == 3)
        self.assertTrue(output_shape[0] == 1)
        self.assertTrue(output_shape[1] == seq_len)
        self.assertTrue(output_shape[2] == model_params.hidden_size)

        # return_pooler_output
        model_params.return_pooler_output = True
        l_bert = bert.BertModelLayer.from_params(model_params, name="albert")

        output_shape = l_bert.compute_output_shape(input_ids_shape)
        self.assertTrue(isinstance(output_shape, tuple))
        self.assertTrue(len(output_shape) == 2)
        self.assertTrue(output_shape[0][0] == 1)
        self.assertTrue(output_shape[0][1] == seq_len)
        self.assertTrue(output_shape[0][2] == model_params.hidden_size)
        self.assertTrue(output_shape[1][0] == 1)
        self.assertTrue(output_shape[1][1] == model_params.hidden_size)

        output_shape = l_bert.compute_output_shape([input_ids_shape, token_type_ids_shape])
        self.assertTrue(isinstance(output_shape, tuple))
        self.assertTrue(len(output_shape) == 2)
        self.assertTrue(output_shape[0][0] == 1)
        self.assertTrue(output_shape[0][1] == seq_len)
        self.assertTrue(output_shape[0][2] == model_params.hidden_size)
        self.assertTrue(output_shape[1][0] == 1)
        self.assertTrue(output_shape[1][1] == model_params.hidden_size)
