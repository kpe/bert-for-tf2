# coding=utf-8
#
# created by kpe on 05.06.2019 at 9:01 PM
#

from __future__ import division, absolute_import, print_function

import os

import unittest

import tensorflow as tf
import bert


class TestMultiLang(unittest.TestCase):
    def setUp(self) -> None:
        self.bert_name = "multilingual_L-12_H-768_A-12"
        self.bert_ckpt_dir = bert.fetch_google_bert_model(self.bert_name, fetch_dir=".models")
        self.bert_ckpt_file = os.path.join(self.bert_ckpt_dir, "bert_model.ckpt")
        self.bert_config_file = os.path.join(self.bert_ckpt_dir, "bert_config.json")

    def test_multi(self):
        print(self.bert_ckpt_dir)
        bert_params = bert.loader.params_from_pretrained_ckpt(self.bert_ckpt_dir)
        bert_params.adapter_size = 32
        l_bert = bert.BertModelLayer.from_params(bert_params, name="bert")

        max_seq_len=128
        l_input_ids      = tf.keras.layers.Input(shape=(max_seq_len,), dtype='int32', name="input_ids")
        l_token_type_ids = tf.keras.layers.Input(shape=(max_seq_len,), dtype='int32', name="token_type_ids")
        output = l_bert([l_input_ids, l_token_type_ids])

        model = tf.keras.Model(inputs=[l_input_ids, l_token_type_ids], outputs=output)
        model.build(input_shape=[(None, max_seq_len),
                                 (None, max_seq_len)])

        bert.load_stock_weights(l_bert, self.bert_ckpt_file)

        model.summary()

