# coding=utf-8
#
# created by kpe on 04.11.2019 at 2:07 PM
#

from __future__ import division, absolute_import, print_function

import unittest

import os
import tempfile

import numpy as np
import tensorflow as tf

import bert

from .test_common import AbstractBertTest, MiniBertFactory


class TestExtendSegmentVocab(AbstractBertTest):

    def setUp(self) -> None:
        tf.compat.v1.reset_default_graph()
        tf.compat.v1.enable_eager_execution()
        print("Eager Execution:", tf.executing_eagerly())

    def test_extend_pretrained_tokens(self):
        model_dir = tempfile.TemporaryDirectory().name
        os.makedirs(model_dir)
        save_path = MiniBertFactory.create_mini_bert_weights(model_dir)
        tokenizer = bert.FullTokenizer(vocab_file=os.path.join(model_dir, "vocab.txt"), do_lower_case=True)

        ckpt_dir = os.path.dirname(save_path)
        bert_params = bert.params_from_pretrained_ckpt(ckpt_dir)

        self.assertEqual(bert_params.token_type_vocab_size, 2)
        bert_params.extra_tokens_vocab_size = 3

        l_bert = bert.BertModelLayer.from_params(bert_params)
        # we dummy call the layer once in order to instantiate the weights
        l_bert([np.array([[1, 1, 0]]), np.array([[1, 0, 0]])], mask=[[True, True, False]])

        mismatched = bert.load_stock_weights(l_bert, save_path)
        self.assertEqual(0, len(mismatched), "token_type embeddings should have mismatched shape")

        l_bert([np.array([[1, -3, 0]]), np.array([[1, 0, 0]])], mask=[[True, True, False]])

