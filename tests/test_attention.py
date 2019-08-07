# coding=utf-8
#
# created by kpe on 30.Jul.2019 at 16:41
#

from __future__ import absolute_import, division, print_function


import unittest

import random

import bert

import numpy as np
import tensorflow as tf
from tensorflow import keras


tf.enable_eager_execution()


class TestAttention(unittest.TestCase):

    def test_attention(self):
        am = bert.AttentionLayer.create_attention_mask(from_shape=[2, 3, 5],   # B,S,..
                                                       input_mask=[[2], [1]]   # B,seq_len
                                                       )
        print(am)  # [batch_size, from_seq_len, seq_len]

    def test_compute_shape(self):
        l_att = bert.AttentionLayer(num_heads=2, size_per_head=2)
        l_att.compute_output_shape(input_shape=(16, 8, 2))