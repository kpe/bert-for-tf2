# coding=utf-8
#
# created by kpe on 30.Jul.2019 at 16:41
#

from __future__ import absolute_import, division, print_function


import unittest

import bert

import tensorflow as tf

tf.enable_eager_execution()


class TestAttention(unittest.TestCase):

    def test_attention(self):
        am = bert.AttentionLayer.create_attention_mask(from_shape=[2, 3, 5],   # B,S,..
                                                       input_mask=[[2], [1]]   # B,seq_len
                                                       )
        print(am)  # [batch_size, from_seq_len, seq_len]

