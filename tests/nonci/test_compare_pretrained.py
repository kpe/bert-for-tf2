# coding=utf-8
#
# created by kpe on 27.Mar.2019 at 15:37
#

from __future__ import absolute_import, division, print_function

import unittest
import os

import numpy as np

import tensorflow as tf
from tensorflow.python import keras


import bert
from bert.tokenization.bert_tokenization import FullTokenizer

tf.compat.v1.disable_eager_execution()


class TestCompareBertsOnPretrainedWeight(unittest.TestCase):
    def setUp(self) -> None:
        self.bert_name = "uncased_L-12_H-768_A-12"
        self.bert_ckpt_dir = bert.fetch_google_bert_model(self.bert_name, fetch_dir=".models")
        self.bert_ckpt_file = os.path.join(self.bert_ckpt_dir, "bert_model.ckpt")
        self.bert_config_file = os.path.join(self.bert_ckpt_dir, "bert_config.json")

    def test_bert_original_weights(self):
        print("bert checkpoint: ", self.bert_ckpt_file)
        bert_vars = tf.train.list_variables(self.bert_ckpt_file)
        for ndx, var in enumerate(bert_vars):
            print("{:3d}".format(ndx), var)

    def create_bert_model(self, max_seq_len=18):
        bert_params = bert.loader.params_from_pretrained_ckpt(self.bert_ckpt_dir)
        l_bert = bert.BertModelLayer.from_params(bert_params, name="bert")

        input_ids      = keras.layers.Input(shape=(max_seq_len,), dtype='int32', name="input_ids")
        token_type_ids = keras.layers.Input(shape=(max_seq_len,), dtype='int32', name="token_type_ids")
        output = l_bert([input_ids, token_type_ids])

        model = keras.Model(inputs=[input_ids, token_type_ids], outputs=output)

        return model, l_bert, (input_ids, token_type_ids)

    def test_keras_weights(self):
        max_seq_len = 18
        model, l_bert, inputs = self.create_bert_model(18)

        model.build(input_shape=[(None, max_seq_len),
                                 (None, max_seq_len)])

        model.summary()

        for ndx, var in enumerate(l_bert.trainable_variables):
            print("{:3d}".format(ndx), var.name, var.shape)

        #for ndx, var in enumerate(model.trainable_variables):
        #    print("{:3d}".format(ndx), var.name, var.shape)

    def test___compare_weights(self):

        tf.compat.v1.reset_default_graph()

        max_seq_len = 18
        model, l_bert, inputs = self.create_bert_model(18)
        model.build(input_shape=[(None, max_seq_len),
                                 (None, max_seq_len)])

        stock_vars = tf.train.list_variables(self.bert_ckpt_file)
        stock_vars = {name: list(shape) for name, shape in stock_vars}

        keras_vars = model.trainable_variables
        keras_vars = {var.name.split(":")[0]: var.shape.as_list() for var in keras_vars}

        matched_vars   = set()
        unmatched_vars = set()
        shape_errors   = set()

        for name in stock_vars:
            bert_name  = name
            keras_name = bert.loader.map_from_stock_variale_name(bert_name)
            if keras_name in keras_vars:
                if keras_vars[keras_name] == stock_vars[bert_name]:
                    matched_vars.add(bert_name)
                else:
                    shape_errors.add(bert_name)
            else:
                unmatched_vars.add(bert_name)

        print("bert -> keras:")
        print("     matched count:", len(matched_vars))
        print("   unmatched count:", len(unmatched_vars))
        print(" shape error count:", len(shape_errors))

        print("unmatched:\n", "\n ".join(unmatched_vars))

        self.assertEqual(197, len(matched_vars))
        self.assertEqual(9, len(unmatched_vars))
        self.assertEqual(0, len(shape_errors))

        matched_vars   = set()
        unmatched_vars = set()
        shape_errors   = set()

        for name in keras_vars:
            keras_name = name
            bert_name  = bert.loader.map_to_stock_variable_name(keras_name)
            if bert_name in stock_vars:
                if stock_vars[bert_name] == keras_vars[keras_name]:
                    matched_vars.add(keras_name)
                else:
                    shape_errors.add(keras_name)
            else:
                unmatched_vars.add(keras_name)

        print("keras -> bert:")
        print("     matched count:", len(matched_vars))
        print("   unmatched count:", len(unmatched_vars))
        print(" shape error count:", len(shape_errors))

        print("unmatched:\n", "\n ".join(unmatched_vars))
        self.assertEqual(197, len(matched_vars))
        self.assertEqual(0, len(unmatched_vars))
        self.assertEqual(0, len(shape_errors))



    def predict_on_keras_model(self, input_ids, input_mask, token_type_ids):
        max_seq_len = input_ids.shape[-1]
        model, l_bert, k_inputs = self.create_bert_model(max_seq_len)
        model.build(input_shape=[(None, max_seq_len),
                                 (None, max_seq_len)])
        bert.load_stock_weights(l_bert, self.bert_ckpt_file)
        k_res = model.predict([input_ids, token_type_ids])
        return k_res

    def predict_on_stock_model(self, input_ids, input_mask, token_type_ids):
        from tests.ext.modeling import BertModel, BertConfig, get_assignment_map_from_checkpoint

        tf.compat.v1.reset_default_graph()

        tf_placeholder = tf.compat.v1.placeholder

        max_seq_len       = input_ids.shape[-1]
        pl_input_ids      = tf.compat.v1.placeholder(tf.int32, shape=(1, max_seq_len))
        pl_mask           = tf.compat.v1.placeholder(tf.int32, shape=(1, max_seq_len))
        pl_token_type_ids = tf.compat.v1.placeholder(tf.int32, shape=(1, max_seq_len))

        bert_config = BertConfig.from_json_file(self.bert_config_file)
        tokenizer = FullTokenizer(vocab_file=os.path.join(self.bert_ckpt_dir, "vocab.txt"))

        s_model = BertModel(config=bert_config,
                               is_training=False,
                               input_ids=pl_input_ids,
                               input_mask=pl_mask,
                               token_type_ids=pl_token_type_ids,
                               use_one_hot_embeddings=False)

        tvars = tf.compat.v1.trainable_variables()
        (assignment_map, initialized_var_names) = get_assignment_map_from_checkpoint(tvars, self.bert_ckpt_file)
        tf.compat.v1.train.init_from_checkpoint(self.bert_ckpt_file, assignment_map)

        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())

            s_res = sess.run(
                s_model.get_sequence_output(),
                feed_dict={pl_input_ids:      input_ids,
                           pl_token_type_ids: token_type_ids,
                           pl_mask:           input_mask,
                           })
        return s_res

    def test_direct_keras_to_stock_compare(self):
        from tests.ext.modeling import BertModel, BertConfig, get_assignment_map_from_checkpoint

        bert_config = BertConfig.from_json_file(self.bert_config_file)
        tokenizer = FullTokenizer(vocab_file=os.path.join(self.bert_ckpt_dir, "vocab.txt"))

        # prepare input
        max_seq_len  = 6
        input_str    = "Hello, Bert!"
        input_tokens = tokenizer.tokenize(input_str)
        input_tokens = ["[CLS]"] + input_tokens + ["[SEP]"]
        input_ids    = tokenizer.convert_tokens_to_ids(input_tokens)
        input_ids      = input_ids             + [0]*(max_seq_len - len(input_tokens))
        input_mask     = [1]*len(input_tokens) + [0]*(max_seq_len - len(input_tokens))
        token_type_ids = [0]*len(input_tokens) + [0]*(max_seq_len - len(input_tokens))

        input_ids      = np.array([input_ids], dtype=np.int32)
        input_mask     = np.array([input_mask], dtype=np.int32)
        token_type_ids = np.array([token_type_ids], dtype=np.int32)

        print("   tokens:", input_tokens)
        print("input_ids:{}/{}:{}".format(len(input_tokens), max_seq_len, input_ids), input_ids.shape, token_type_ids)

        s_res = self.predict_on_stock_model(input_ids, input_mask, token_type_ids)
        k_res = self.predict_on_keras_model(input_ids, input_mask, token_type_ids)

        np.set_printoptions(precision=9, threshold=20, linewidth=200, sign="+", floatmode="fixed")
        print("s_res", s_res.shape)
        print("k_res", k_res.shape)

        print("s_res:\n {}".format(s_res[0, :2, :10]), s_res.dtype)
        print("k_res:\n {}".format(k_res[0, :2, :10]), k_res.dtype)

        adiff = np.abs(s_res-k_res).flatten()
        print("diff:", np.max(adiff), np.argmax(adiff))
        self.assertTrue(np.allclose(s_res, k_res, atol=1e-6))


