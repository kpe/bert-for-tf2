# coding=utf-8
#
# created by kpe on 25.Jul.2019 at 13:30
#

from __future__ import absolute_import, division, print_function


import os
import string
import unittest
import tempfile

import tensorflow as tf
import numpy as np

from tensorflow.python import keras

from bert.tokenization import FullTokenizer, validate_case_matches_checkpoint


class MiniBertFactory:

    @staticmethod
    def create_mini_bert_weights(model_dir):
        from bert.loader import StockBertConfig

        bert_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
        bert_config = StockBertConfig(
            attention_probs_dropout_prob = 0.1,
            hidden_act                   = "gelu",
            hidden_dropout_prob          = 0.1,
            hidden_size                  = 8,
            initializer_range            = 0.02,
            intermediate_size            = 32,
            max_position_embeddings      = 32,
            num_attention_heads          = 2,
            num_hidden_layers            = 2,
            type_vocab_size              = 2,
            vocab_size                   = len(string.ascii_lowercase)*2 + len(bert_tokens)
        )

        print("creating mini BERT at:", model_dir)

        bert_config_file = os.path.join(model_dir, "bert_config.json")
        bert_vocab_file  = os.path.join(model_dir, "vocab.txt")

        with open(bert_config_file, "w") as f:
            f.write(bert_config.to_json_string())
        with open(bert_vocab_file, "w") as f:
            f.write("\n".join(list(string.ascii_lowercase) + bert_tokens))
            f.write("\n".join(["##"+tok for tok in list(string.ascii_lowercase)]))

        with tf.Graph().as_default():
            _ = MiniBertFactory.create_stock_bert_graph(bert_config_file, 16)
            saver = tf.compat.v1.train.Saver(max_to_keep=1, save_relative_paths=True)

            with tf.compat.v1.Session() as sess:
                sess.run(tf.compat.v1.global_variables_initializer())
                ckpt_path = os.path.join(model_dir, "bert_model.ckpt")
                save_path = saver.save(sess, ckpt_path, write_meta_graph=True)
                print("saving to:", save_path)

        validate_case_matches_checkpoint(True, save_path)

        return save_path

    @staticmethod
    def create_stock_bert_graph(bert_config_file, max_seq_len):
        from tests.ext.modeling import BertModel, BertConfig

        tf_placeholder = tf.compat.v1.placeholder

        pl_input_ids      = tf_placeholder(tf.int32, shape=(1, max_seq_len))
        pl_mask           = tf_placeholder(tf.int32, shape=(1, max_seq_len))
        pl_token_type_ids = tf_placeholder(tf.int32, shape=(1, max_seq_len))

        bert_config = BertConfig.from_json_file(bert_config_file)
        s_model = BertModel(config=bert_config,
                            is_training=False,
                            input_ids=pl_input_ids,
                            input_mask=pl_mask,
                            token_type_ids=pl_token_type_ids,
                            use_one_hot_embeddings=False)

        return s_model, pl_input_ids, pl_mask, pl_token_type_ids


class AbstractBertTest(unittest.TestCase):

    @staticmethod
    def create_mini_bert_weights():
        model_dir = tempfile.TemporaryDirectory().name
        # model_dir = "/tmp/mini_bert/";
        os.makedirs(model_dir, exist_ok=True)
        save_path = MiniBertFactory.create_mini_bert_weights(model_dir)
        print("mini_bert save_path", save_path)
        print("\n\t".join([""] + os.listdir(model_dir)))
        return model_dir
