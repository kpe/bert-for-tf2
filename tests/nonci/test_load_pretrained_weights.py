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


class TestLoadPreTrainedWeights(unittest.TestCase):

    def build_model(self, bert_params):
        l_bert = bert.BertModelLayer.from_params(bert_params, name="bert")

        l_input_ids = keras.layers.Input(shape=(128,), dtype='int32', name="input_ids")
        l_token_type_ids = keras.layers.Input(shape=(128,), dtype='int32', name="token_type_ids")
        output = l_bert([l_input_ids, l_token_type_ids])
        output = keras.layers.Lambda(lambda x: x[:, 0, :])(output)
        output = keras.layers.Dense(2)(output)
        model = keras.Model(inputs=[l_input_ids, l_token_type_ids], outputs=output)

        model.build(input_shape=(None, 128))
        model.compile(optimizer=keras.optimizers.Adam(),
                      loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=[keras.metrics.SparseCategoricalAccuracy(name="acc")])

        for weight in l_bert.weights:
            print(weight.name)

        return model, l_bert

    def test_bert_google_weights(self):
        bert_model_name = "uncased_L-12_H-768_A-12"
        bert_dir = bert.fetch_google_bert_model(bert_model_name, ".models")
        bert_ckpt = os.path.join(bert_dir, "bert_model.ckpt")

        bert_params = bert.params_from_pretrained_ckpt(bert_dir)
        model, l_bert = self.build_model(bert_params)

        skipped_weight_value_tuples = bert.load_bert_weights(l_bert, bert_ckpt)
        self.assertEqual(0, len(skipped_weight_value_tuples))
        model.summary()

    def test_albert_chinese_weights(self):
        albert_model_name = "albert_base"
        albert_dir = bert.fetch_brightmart_albert_model(albert_model_name, ".models")
        albert_ckpt = os.path.join(albert_dir, "albert_model.ckpt")

        albert_params = bert.params_from_pretrained_ckpt(albert_dir)
        model, l_bert = self.build_model(albert_params)

        skipped_weight_value_tuples = bert.load_albert_weights(l_bert, albert_ckpt)
        self.assertEqual(0, len(skipped_weight_value_tuples))
        model.summary()

    def test_albert_google_weights(self):
        albert_model_name = "albert_base"
        albert_dir = bert.fetch_tfhub_albert_model(albert_model_name, ".models")

        albert_params = bert.albert_params(albert_model_name)
        model, l_bert = self.build_model(albert_params)

        skipped_weight_value_tuples = bert.load_albert_weights(l_bert, albert_dir)
        self.assertEqual(0, len(skipped_weight_value_tuples))
        model.summary()

    def test_albert_google_weights_non_tfhub(self):
        albert_model_name = "albert_base_v2"
        albert_dir = bert.fetch_google_albert_model(albert_model_name, ".models")
        model_ckpt = os.path.join(albert_dir, "model.ckpt-best")

        albert_params = bert.albert_params(albert_dir)
        model, l_bert = self.build_model(albert_params)

        skipped_weight_value_tuples = bert.load_albert_weights(l_bert, model_ckpt)
        self.assertEqual(0, len(skipped_weight_value_tuples))
        model.summary()
