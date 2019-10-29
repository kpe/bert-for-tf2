# coding=utf-8
#
# created by kpe on 10.Oct.2019 at 16:26
#

from __future__ import absolute_import, division, print_function

import unittest

import os

import tensorflow as tf
from tensorflow import keras

import bert


class TestAlbertLoadWeights(unittest.TestCase):

    def test_chinese_weights(self):
        #bert_ckpt_dir = ".models/albert_base_zh/"
        #bert_ckpt_file = bert_ckpt_dir + "albert_model.ckpt"
        #bert_config_file = bert_ckpt_dir + "albert_config_base.json"

        print("Eager Execution:", tf.executing_eagerly())

        albert_model_name = "albert_base"
        albert_dir = bert.fetch_brightmart_albert_model(albert_model_name, ".models")
        albert_ckpt = os.path.join(albert_dir, "albert_model.ckpt")

        bert_params = bert.params_from_pretrained_ckpt(albert_dir)
        l_bert = bert.BertModelLayer.from_params(bert_params, name="bert")

        l_input_ids      = keras.layers.Input(shape=(128,), dtype='int32', name="input_ids")
        l_token_type_ids = keras.layers.Input(shape=(128,), dtype='int32', name="token_type_ids")
        output = l_bert([l_input_ids, l_token_type_ids])
        output = keras.layers.Lambda(lambda x: x[:, 0, :])(output)
        output = keras.layers.Dense(2)(output)
        model  = keras.Model(inputs=[l_input_ids, l_token_type_ids], outputs=output)

        model.build(input_shape=(None, 128))
        model.compile(optimizer=keras.optimizers.Adam(),
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[keras.metrics.SparseCategoricalAccuracy(name="acc")])

        for weight in l_bert.weights:
            print(weight.name)

        bert.load_albert_weights(l_bert, albert_ckpt)

        model.summary()

    def test_google_weights(self):
        albert_model_name = "albert_base"
        albert_dir = bert.fetch_tfhub_albert_model(albert_model_name, ".models")

        albert_params = bert.albert_params(albert_model_name)
        l_bert = bert.BertModelLayer.from_params(albert_params, name="albert")

        l_input_ids      = keras.layers.Input(shape=(128,), dtype='int32', name="input_ids")
        l_token_type_ids = keras.layers.Input(shape=(128,), dtype='int32', name="token_type_ids")
        output = l_bert([l_input_ids, l_token_type_ids])
        output = keras.layers.Lambda(lambda x: x[:, 0, :])(output)
        output = keras.layers.Dense(2)(output)
        model  = keras.Model(inputs=[l_input_ids, l_token_type_ids], outputs=output)

        model.build(input_shape=(None, 128))
        model.compile(optimizer=keras.optimizers.Adam(),
                      loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=[keras.metrics.SparseCategoricalAccuracy(name="acc")])

        for weight in l_bert.weights:
            print(weight.name)

        bert.load_albert_weights(l_bert, albert_dir)

        model.summary()
