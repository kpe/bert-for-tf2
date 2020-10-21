# coding=utf-8
#
# created by kpe on 28.Mar.2019 at 14:01
#

from __future__ import absolute_import, division, print_function

import os
import re

import tensorflow as tf
from tensorflow import keras

import params_flow as pf
import params

from bert.model import BertModelLayer

_verbose = os.environ.get('VERBOSE', 1)  # verbose print per default
trace = print if int(_verbose) else lambda *a, **k: None

bert_models_google = {
    "uncased_L-12_H-768_A-12":      "https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip",
    "uncased_L-24_H-1024_A-16":     "https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-24_H-1024_A-16.zip",
    "cased_L-12_H-768_A-12":        "https://storage.googleapis.com/bert_models/2018_10_18/cased_L-12_H-768_A-12.zip",
    "cased_L-24_H-1024_A-16":       "https://storage.googleapis.com/bert_models/2018_10_18/cased_L-24_H-1024_A-16.zip",
    "multi_cased_L-12_H-768_A-12":  "https://storage.googleapis.com/bert_models/2018_11_23/multi_cased_L-12_H-768_A-12.zip",
    "multilingual_L-12_H-768_A-12": "https://storage.googleapis.com/bert_models/2018_11_03/multilingual_L-12_H-768_A-12.zip",
    "chinese_L-12_H-768_A-12":  "https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip",
    "wwm_uncased_L-24_H-1024_A-16": "https://storage.googleapis.com/bert_models/2019_05_30/wwm_uncased_L-24_H-1024_A-16.zip",
    "wwm_cased_L-24_H-1024_A-16":   "https://storage.googleapis.com/bert_models/2019_05_30/wwm_cased_L-24_H-1024_A-16.zip",
}


def fetch_google_bert_model(model_name: str, fetch_dir: str):
    if model_name not in bert_models_google:
        raise ValueError("BERT model with name:[{}] not found, try one of:{}".format(
            model_name, bert_models_google))
    else:
        fetch_url = bert_models_google[model_name]

    fetched_file = pf.utils.fetch_url(fetch_url, fetch_dir=fetch_dir)
    fetched_dir = pf.utils.unpack_archive(fetched_file)
    fetched_dir = os.path.join(fetched_dir, model_name)
    return fetched_dir


def map_from_stock_variale_name(name, prefix="bert"):
    name = name.split(":")[0]
    ns   = name.split("/")
    pns  = prefix.split("/")

    # assert ns[0] == "bert"

    name = "/".join(pns + ns[1:])
    ns = name.split("/")

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
        elif ns[2] == "word_embeddings_projector":
            ns[2] = "word_embeddings_2"
            if ns[3] == "projector":
                ns[3] = "embeddings"
                return "/".join(ns[:-1])
            return "/".join(ns)
        else:
            return "/".join(ns[:-1])
    if ns[1] == "encoder":
        if ns[3] == "intermediate":
            return "/".join(ns[:4] + ["dense"] + ns[4:])
        else:
            return name
    return None


class StockBertConfig(params.Params):
    attention_probs_dropout_prob = None  # 0.1
    hidden_act                   = None  # "gelu"
    hidden_dropout_prob          = None  # 0.1,
    hidden_size                  = None  # 768,
    initializer_range            = None  # 0.02,
    intermediate_size            = None  # 3072,
    max_position_embeddings      = None  # 512,
    num_attention_heads          = None  # 12,
    num_hidden_layers            = None  # 12,
    type_vocab_size              = None  # 2,
    vocab_size                   = None  # 30522

    # ALBERT params
    # directionality             = None  # "bidi"
    # pooler_fc_size             = None  # 768,
    # pooler_num_attention_heads = None  # 12,
    # pooler_num_fc_layers       = None  # 3,
    # pooler_size_per_head       = None  # 128,
    # pooler_type                = None  # "first_token_transform",
    ln_type                      = None  # "postln"   # used for detecting brightmarts weights
    embedding_size               = None  # 128

    def to_bert_model_layer_params(self):
        return map_stock_config_to_params(self)


def map_stock_config_to_params(bc):
    """
    Converts the original BERT or ALBERT config dictionary
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

        embedding_size=bc.embedding_size,
        shared_layer=bc.embedding_size is not None,
    )
    return bert_params


def params_from_pretrained_ckpt(bert_ckpt_dir):
    json_config_files = tf.io.gfile.glob(os.path.join(bert_ckpt_dir, "*_config*.json"))
    if len(json_config_files) != 1:
        raise ValueError("Can't glob for BERT config json at: {}/*_config*.json".format(bert_ckpt_dir))

    config_file_name = os.path.basename(json_config_files[0])
    bert_config_file = os.path.join(bert_ckpt_dir, config_file_name)

    with tf.io.gfile.GFile(bert_config_file, "r") as reader:
        bc = StockBertConfig.from_json_string(reader.read())
        bert_params = map_stock_config_to_params(bc)
        is_brightmart_weights = bc["ln_type"] is not None
        bert_params.project_position_embeddings  = not is_brightmart_weights  # ALBERT: False for brightmart/weights
        bert_params.project_embeddings_with_bias = not is_brightmart_weights  # ALBERT: False for brightmart/weights

    return bert_params


def _checkpoint_exists(ckpt_path):
    cktp_files = tf.io.gfile.glob(ckpt_path + "*")
    return len(cktp_files) > 0


def bert_prefix(bert: BertModelLayer):
    re_bert = re.compile(r'(.*)/(embeddings|encoder)/(.+):0')
    match = re_bert.match(bert.weights[0].name)
    assert match, "Unexpected bert layer: {} weight:{}".format(bert, bert.weights[0].name)
    prefix = match.group(1)
    return prefix


def load_stock_weights(bert: BertModelLayer, ckpt_path, map_to_stock_fn=map_to_stock_variable_name):
    """
    Use this method to load the weights from a pre-trained BERT checkpoint into a bert layer.

    :param bert: a BertModelLayer instance within a built keras model.
    :param ckpt_path: checkpoint path, i.e. `uncased_L-12_H-768_A-12/bert_model.ckpt` or `albert_base_zh/albert_model.ckpt`
    :return: list of weights with mismatched shapes. This can be used to extend
    the segment/token_type embeddings.
    """
    assert isinstance(bert, BertModelLayer), "Expecting a BertModelLayer instance as first argument"
    assert _checkpoint_exists(ckpt_path), "Checkpoint does not exist: {}".format(ckpt_path)
    assert len(bert.weights) > 0, "BertModelLayer weights have not been instantiated yet. " \
                                  "Please add the layer in a Keras model and call model.build() first!"

    ckpt_reader = tf.train.load_checkpoint(ckpt_path)

    stock_weights = set(ckpt_reader.get_variable_to_dtype_map().keys())

    prefix = bert_prefix(bert)

    loaded_weights = set()
    skip_count = 0
    weight_value_tuples = []
    skipped_weight_value_tuples = []

    bert_params = bert.weights
    param_values = keras.backend.batch_get_value(bert.weights)
    for ndx, (param_value, param) in enumerate(zip(param_values, bert_params)):
        stock_name = map_to_stock_fn(param.name, prefix)

        if ckpt_reader.has_tensor(stock_name):
            ckpt_value = ckpt_reader.get_tensor(stock_name)

            if param_value.shape != ckpt_value.shape:
                trace("loader: Skipping weight:[{}] as the weight shape:[{}] is not compatible "
                      "with the checkpoint:[{}] shape:{}".format(param.name, param.shape,
                                                                 stock_name, ckpt_value.shape))
                skipped_weight_value_tuples.append((param, ckpt_value))
                continue

            weight_value_tuples.append((param, ckpt_value))
            loaded_weights.add(stock_name)
        else:
            trace("loader: No value for:[{}], i.e.:[{}] in:[{}]".format(param.name, stock_name, ckpt_path))
            skip_count += 1
    keras.backend.batch_set_value(weight_value_tuples)

    trace("Done loading {} BERT weights from: {} into {} (prefix:{}). "
          "Count of weights not found in the checkpoint was: [{}]. "
          "Count of weights with mismatched shape: [{}]".format(
              len(weight_value_tuples), ckpt_path, bert, prefix, skip_count, len(skipped_weight_value_tuples)))

    trace("Unused weights from checkpoint:",
          "\n\t" + "\n\t".join(sorted(stock_weights.difference(loaded_weights))))

    return skipped_weight_value_tuples  # (bert_weight, value_from_ckpt)
