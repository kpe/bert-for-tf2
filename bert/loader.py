# coding=utf-8
#
# created by kpe on 28.Mar.2019 at 14:01
#

from __future__ import absolute_import, division, print_function

import os

import tensorflow as tf
from tensorflow import keras

import params

from bert.model import BertModelLayer


def map_from_stock_variale_name(name, prefix="bert"):
    name = name.split(":")[0]
    ns   = name.split("/")
    pns  = prefix.split("/")

    if ns[:len(pns)] != pns:
        return None

    name = "/".join(["bert"] + ns[len(pns):])
    ns = name.split("/")

    if ns[0] != "bert":
        return None
    if ns[1] not in ["encoder", "embeddings"]:
        return None
    if ns[1] == "embeddings":
        if ns[2] == "LayerNorm":
            return name
        else:
            return name + "/embeddings"
    if ns[1] == "encoder":
        if ns[3] == "intermediate":
            return "/".join(ns[:4] + ns[5:])
        else:
            return name
    return None


def map_to_stock_variable_name(name, prefix="bert"):
    name = name.split(":")[0]
    ns   = name.split("/")
    pns  = prefix.split("/")

    if ns[:len(pns)] != pns:
        return None

    name = "/".join(["bert"] + ns[len(pns):])
    ns   = name.split("/")

    if ns[1] not in ["encoder", "embeddings"]:
        return None
    if ns[1] == "embeddings":
        if ns[2] == "LayerNorm":
            return name
        else:
            return "/".join(ns[:-1])
    if ns[1] == "encoder":
        if ns[3] == "intermediate":
            return "/".join(ns[:4] + ["dense"] + ns[4:])
        else:
            return name
    return None


class StockBertConfig(params.Params):
    attention_probs_dropout_prob = None,  # 0.1
    hidden_act                   = None,  # "gelu"
    hidden_dropout_prob          = None,  # 0.1,
    hidden_size                  = None,  # 768,
    initializer_range            = None,  # 0.02,
    intermediate_size            = None,  # 3072,
    max_position_embeddings      = None,  # 512,
    num_attention_heads          = None,  # 12,
    num_hidden_layers            = None,  # 12,
    type_vocab_size              = None,  # 2,
    vocab_size                   = None,  # 30522

    def to_bert_model_layer_params(self):
        return map_stock_config_to_params(self)


def map_stock_config_to_params(bc):
    """
    Converts the original BERT config dictionary
    to a `BertModelLayer.Params` instance.
    :return: a `BertModelLayer.Params` instance.
    """
    bert_params = BertModelLayer.Params(
        num_layers=bc.num_hidden_layers,
        num_heads=bc.num_attention_heads,
        hidden_size=bc.hidden_size,
        hidden_dropout=bc.hidden_dropout_prob,
        attention_dropout=bc.attention_probs_dropout_prob,

        intermediate_size=bc.intermediate_size,
        intermediate_activation=bc.hidden_act,

        vocab_size=bc.vocab_size,
        use_token_type=True,
        use_position_embeddings=True,
        token_type_vocab_size=bc.type_vocab_size,
        max_position_embeddings=bc.max_position_embeddings,
    )
    return bert_params


def params_from_pretrained_ckpt(bert_ckpt_dir):
    bert_config_file = os.path.join(bert_ckpt_dir, "bert_config.json")

    with tf.io.gfile.GFile(bert_config_file, "r") as reader:
        bc = StockBertConfig.from_json_string(reader.read())
        bert_params = map_stock_config_to_params(bc)

    return bert_params


def load_stock_weights(bert: BertModelLayer, ckpt_file):
    assert isinstance(bert, BertModelLayer), "Expecting a BertModelLayer instance as first argument"
    assert tf.compat.v1.train.checkpoint_exists(ckpt_file), "Checkpoint does not exist: {}".format(ckpt_file)
    ckpt_reader = tf.train.load_checkpoint(ckpt_file)

    bert_prefix = bert.weights[0].name.split("/")[0]

    skip_count = 0
    weight_value_tuples = []

    bert_params = bert.weights
    param_values = keras.backend.batch_get_value(bert.weights)
    for ndx, (param_value, param) in enumerate(zip(param_values, bert_params)):
        stock_name = map_to_stock_variable_name(param.name, bert_prefix)

        if ckpt_reader.has_tensor(stock_name):
            ckpt_value = ckpt_reader.get_tensor(stock_name)

            if param_value.shape != ckpt_value.shape:
                raise ValueError("Layer weight shape:[{}] not compatible "
                                 "with checkpoint:[{}] shape:{}".format(param.shape, stock_name, ckpt_value.shape))

            weight_value_tuples.append((param, ckpt_value))
        else:
            print("loader: No value for:[{}], i.e.:[{}] in:[{}]".format(param.name, stock_name, ckpt_file))
            skip_count += 1
    keras.backend.batch_set_value(weight_value_tuples)

    print("Done loading {} BERT weights from: {} into {} (prefix:{}). "
          "Count of weights not found in the checkpoint was: {}".format(
        len(weight_value_tuples),
        ckpt_file,
        bert, bert_prefix, skip_count))
