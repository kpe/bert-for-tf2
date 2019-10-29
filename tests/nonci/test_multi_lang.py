# coding=utf-8
#
# created by kpe on 05.06.2019 at 9:01 PM
#

from __future__ import division, absolute_import, print_function

import os

import unittest

import tensorflow as tf
from tensorflow.python import keras
from bert import BertModelLayer
from bert.loader import StockBertConfig, load_stock_weights

class TestMultiLang(unittest.TestCase):
    model_dir = os.path.join(os.path.dirname(__file__),'../../', '.models/multilingual_L-12_H-768_A-12/')

    def test_multi(self):
        model_dir = self.model_dir
        print(model_dir)

        bert_config_file = os.path.join(model_dir, "bert_config.json")
        bert_ckpt_file   = os.path.join(model_dir, "bert_model.ckpt")

        with tf.io.gfile.GFile(bert_config_file, "r") as reader:
            stock_params = StockBertConfig.from_json_string(reader.read())
            bert_params  = stock_params.to_bert_model_layer_params()

        l_bert = BertModelLayer.from_params(bert_params, name="bert")

        max_seq_len=128
        l_input_ids      = keras.layers.Input(shape=(max_seq_len,), dtype='int32', name="input_ids")
        l_token_type_ids = keras.layers.Input(shape=(max_seq_len,), dtype='int32', name="token_type_ids")
        output = l_bert([l_input_ids, l_token_type_ids])

        model = keras.Model(inputs=[l_input_ids, l_token_type_ids], outputs=output)
        model.build(input_shape=[(None, max_seq_len),
                                 (None, max_seq_len)])

        load_stock_weights(l_bert, bert_ckpt_file)

