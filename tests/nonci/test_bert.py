# coding=utf-8
#
# created by kpe on 26.Mar.2019 at 14:11
#

from __future__ import absolute_import, division, print_function

import random

import unittest

import numpy as np
import tensorflow as tf

from tensorflow import keras

from bert import BertModelLayer


tf.compat.v1.enable_eager_execution()


class MaskFlatten(keras.layers.Flatten):

    def __init__(self, **kwargs):
        self.supports_masking = True
        super(MaskFlatten, self).__init__(**kwargs)

    def compute_mask(self, _, mask=None):
        return mask


def parity_ds_generator(batch_size=32, max_len=10, max_int=4, modulus=2):
    """
    Generates a parity calculation dataset (seq -> sum(seq) mod 2),
    where seq is a sequence of length less than max_len
    of integers in [1..max_int).
    """
    while True:
        data = np.zeros((batch_size, max_len))
        tag = np.zeros(batch_size, dtype='int32')
        for i in range(batch_size):
            datum_len = random.randint(1, max_len - 1)
            total = 0
            for j in range(datum_len):
                data[i, j] = random.randint(1, max_int)
                total += data[i, j]
            tag[i] = total % modulus
        yield data, tag                  # ([batch_size, max_len], [max_len])


class RawBertTest(unittest.TestCase):

    def test_simple(self):
        max_seq_len = 10
        bert = BertModelLayer(
            vocab_size=5,
            max_position_embeddings=10,
            hidden_size=15,
            num_layers=2,
            num_heads=5,
            intermediate_size=4,
            use_token_type=False
        )
        model = keras.Sequential([
            bert,
            keras.layers.Lambda(lambda x: x[:, -0, ...]),        # [B, 2]
            keras.layers.Dense(units=2, activation="softmax"),   # [B, 10, 2]
        ])

        model.build(input_shape=(None, max_seq_len))

        model.compile(optimizer=keras.optimizers.Adam(lr=0.002),
                      loss=keras.losses.sparse_categorical_crossentropy,
                      metrics=[keras.metrics.sparse_categorical_accuracy]
                      )

        model.summary(line_length=120)

        for ndx, var in enumerate(model.trainable_variables):
            print("{:5d}".format(ndx), var.name, var.shape, var.dtype)

        model.fit_generator(generator=parity_ds_generator(64, max_seq_len),
                            steps_per_epoch=100,
                            epochs=10,
                            validation_data=parity_ds_generator(32, max_seq_len),  # TODO: can't change max_seq_len (but transformer alone can)
                            validation_steps=10,
                            callbacks=[
                                keras.callbacks.EarlyStopping(monitor='val_sparse_categorical_accuracy', patience=5),
                            ],
                            )



