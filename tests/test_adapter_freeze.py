# coding=utf-8
#
# created by kpe on 09.08.2019 at 10:38 PM
#

from __future__ import division, absolute_import, print_function


import unittest

from tensorflow import keras

import bert


class AdapterFreezeTest(unittest.TestCase):

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




