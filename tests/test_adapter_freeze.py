# coding=utf-8
#
# created by kpe on 09.08.2019 at 10:38 PM
#

from __future__ import division, absolute_import, print_function


import unittest

import os
import tempfile

import numpy as np
import tensorflow as tf
from tensorflow import keras

import bert

from .test_common import AbstractBertTest, MiniBertFactory

tf.enable_eager_execution()

class AdapterFreezeTest(AbstractBertTest):

    def test_adapter_freezing(self):
        bert_params = bert.BertModelLayer.Params(hidden_size=32,
                                                 vocab_size=67,
                                                 max_position_embeddings=64,
                                                 num_layers=1,
                                                 num_heads=1,
                                                 intermediate_size=4,
                                                 use_token_type=False)

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
        self.assertEqual(20, len(model.trainable_weights))
        for weight in model.trainable_weights:
            print(weight.name, weight.shape)

        bert_params.adapter_size = 16

        model = to_model(bert_params)
        model.summary()
        print(len(model.trainable_weights))
        self.assertEqual(14, len(model.trainable_weights))
        for weight in model.trainable_weights:
            print(weight.name, weight.shape)


    def test_freeze(self):
        model_dir = tempfile.TemporaryDirectory().name
        os.makedirs(model_dir)
        save_path = MiniBertFactory.create_mini_bert_weights(model_dir)
        tokenizer = bert.FullTokenizer(vocab_file=os.path.join(model_dir, "vocab.txt"), do_lower_case=True)

        # prepare input
        max_seq_len  = 24
        input_str_batch    = ["hello, bert!", "how are you doing!"]

        input_ids, token_type_ids = self.prepare_input_batch(input_str_batch, tokenizer, max_seq_len)

        bert_ckpt_file   = os.path.join(model_dir, "bert_model.ckpt")

        bert_params = bert.params_from_pretrained_ckpt(model_dir)
        bert_params.adapter_size = 4
        l_bert = bert.BertModelLayer.from_params(bert_params)

        model = keras.models.Sequential([
            l_bert,
        ])


        model.build(input_shape=(None, max_seq_len))

        model.summary()
        l_bert.apply_adapter_freeze()
        model.summary()

        bert.load_stock_weights(l_bert, bert_ckpt_file)
        #l_bert.embeddings_layer.trainable = False

        model.summary()

        orig_weight_values = []
        for weight in l_bert.weights:
            orig_weight_values.append(weight.numpy())

        model.compile(optimizer=keras.optimizers.Adam(),
                      loss=keras.losses.mean_squared_error)

        orig_pred = model.predict(input_ids)
        model.fit(x=input_ids, y=np.zeros_like(orig_pred),
          batch_size=2,
          epochs=4)

        for ndx, weight in enumerate(l_bert.weights):
            print("{}: {}".format(
                    np.array_equal(weight.numpy(), orig_weight_values[ndx]),
                    weight.name))

        model.summary()






