# coding=utf-8
#
# created by kpe on 28.10.2019 at 2:02 PM
#

from __future__ import division, absolute_import, print_function

import os
import re
import urllib
import params_flow as pf

import tensorflow as tf
from tensorflow import keras

from bert import BertModelLayer, loader

albert_models_tfhub = {
    "albert_base":    "https://tfhub.dev/google/albert_base/1?tf-hub-format=compressed",
    "albert_large":   "https://tfhub.dev/google/albert_large/1?tf-hub-format=compressed",
    "albert_xlarge":  "https://tfhub.dev/google/albert_xlarge/1?tf-hub-format=compressed",
    "albert_xxlarge": "https://tfhub.dev/google/albert_xxlarge/1?tf-hub-format=compressed",
}

albert_models_brightmart = {
    "albert_tiny":        "https://storage.googleapis.com/albert_zh/albert_tiny.zip",
    "albert_tiny_489k":   "https://storage.googleapis.com/albert_zh/albert_tiny_489k.zip",
    "albert_base":        "https://storage.googleapis.com/albert_zh/albert_base_zh.zip",
    "albert_base_36k":    "https://storage.googleapis.com/albert_zh/albert_base_zh_additional_36k_steps.zip",
    "albert_large":       "https://storage.googleapis.com/albert_zh/albert_large_zh.zip",
    "albert_xlarge":      "https://storage.googleapis.com/albert_zh/albert_xlarge_zh_177k.zip",
    "albert_xlarge_183k": "https://storage.googleapis.com/albert_zh/albert_xlarge_zh_183k.zip",
}

config_albert_base = {
    "attention_probs_dropout_prob": 0.1,
    "hidden_act": "gelu",
    "hidden_dropout_prob": 0.1,
    "embedding_size": 128,
    "hidden_size": 768,
    "initializer_range": 0.02,
    "intermediate_size": 3072,
    "max_position_embeddings": 512,
    "num_attention_heads": 12,
    "num_hidden_layers": 12,
    "num_hidden_groups": 1,
    "net_structure_type": 0,
    "gap_size": 0,
    "num_memory_blocks": 0,
    "inner_group_num": 1,
    "down_scale_factor": 1,
    "type_vocab_size": 2,
    "vocab_size": 30000
}

config_albert_large = {
    "attention_probs_dropout_prob": 0.1,
    "hidden_act": "gelu",
    "hidden_dropout_prob": 0.1,
    "embedding_size": 128,
    "hidden_size": 1024,
    "initializer_range": 0.02,
    "intermediate_size": 4096,
    "max_position_embeddings": 512,
    "num_attention_heads": 16,
    "num_hidden_layers": 24,
    "num_hidden_groups": 1,
    "net_structure_type": 0,
    "gap_size": 0,
    "num_memory_blocks": 0,
    "inner_group_num": 1,
    "down_scale_factor": 1,
    "type_vocab_size": 2,
    "vocab_size": 30000
}
config_albert_xlarge = {
    "attention_probs_dropout_prob": 0.1,
    "hidden_act": "gelu",
    "hidden_dropout_prob": 0.1,
    "embedding_size": 128,
    "hidden_size": 2048,
    "initializer_range": 0.02,
    "intermediate_size": 8192,
    "max_position_embeddings": 512,
    "num_attention_heads": 16,
    "num_hidden_layers": 24,
    "num_hidden_groups": 1,
    "net_structure_type": 0,
    "gap_size": 0,
    "num_memory_blocks": 0,
    "inner_group_num": 1,
    "down_scale_factor": 1,
    "type_vocab_size": 2,
    "vocab_size": 30000
}

config_albert_xxlarge = {
    "attention_probs_dropout_prob": 0,
    "hidden_act": "gelu",
    "hidden_dropout_prob": 0,
    "embedding_size": 128,
    "hidden_size": 4096,
    "initializer_range": 0.02,
    "intermediate_size": 16384,
    "max_position_embeddings": 512,
    "num_attention_heads": 64,
    "num_hidden_layers": 12,
    "num_hidden_groups": 1,
    "net_structure_type": 0,
    "layers_to_keep": [],
    "gap_size": 0,
    "num_memory_blocks": 0,
    "inner_group_num": 1,
    "down_scale_factor": 1,
    "type_vocab_size": 2,
    "vocab_size": 30000
}

albert_models_config = {
    "albert_base":    config_albert_base,
    "albert_large":   config_albert_large,
    "albert_xlarge":  config_albert_xlarge,
    "albert_xxlarge": config_albert_xxlarge,
}


def albert_params(albert_model: str):
    albert_config = albert_models_config[albert_model]
    stock_config = loader.StockBertConfig(**albert_config)
    params = loader.map_stock_config_to_params(stock_config)
    return params


def fetch_brightmart_albert_model(model_name: str, fetch_dir: str):
    if model_name not in albert_models_brightmart:
        raise ValueError("ALBERT model with name:[{}] not found at brightmart/albert_zh, try one of:{}".format(
            model_name, albert_models_brightmart))
    else:
        fetch_url = albert_models_brightmart[model_name]

    fetched_file = pf.utils.fetch_url(fetch_url, fetch_dir=fetch_dir)
    fetched_dir = pf.utils.unpack_archive(fetched_file)
    return fetched_dir


def fetch_tfhub_albert_model(albert_model: str, fetch_dir: str):
    """
    Fetches a pre-trained ALBERT model from TFHub.
    :param albert_model: TFHub model URL or a model name like albert_base, albert_large, etc.
    :param fetch_dir:
    :return:
    """
    if albert_model.startswith("http"):
        fetch_url = albert_model
    elif albert_model not in albert_models_tfhub:
        raise ValueError("ALBERT model with name:[{}] not found in tfhub/google, try one of:{}".format(
            albert_model, albert_models_tfhub))
    else:
        fetch_url = albert_models_tfhub[albert_model]

    name, version = urllib.parse.urlparse(fetch_url).path.split("/")[-2:]
    local_file_name = "{}.tar.gz".format(name)

    print("Fetching ALBERT model: {} version: {}".format(name, version))

    fetched_file = pf.utils.fetch_url(fetch_url, fetch_dir=fetch_dir, local_file_name=local_file_name)
    fetched_dir = pf.utils.unpack_archive(fetched_file)

    return fetched_dir


def map_to_tfhub_albert_variable_name(name, prefix="bert"):

    name = re.compile("encoder/layer_shared/intermediate/(?=kernel|bias)").sub(
        "encoder/transformer/group_0/inner_group_0/ffn_1/intermediate/dense/", name)
    name = re.compile("encoder/layer_shared/output/dense/(?=kernel|bias)").sub(
        "encoder/transformer/group_0/inner_group_0/ffn_1/intermediate/output/dense/", name)

    name = name.replace("encoder/layer_shared/output/dense",               "encoder/transformer/group_0/inner_group_0/ffn_1/intermediate/output/dense")
    name = name.replace("encoder/layer_shared/attention/output/LayerNorm", "encoder/transformer/group_0/inner_group_0/LayerNorm")
    name = name.replace("encoder/layer_shared/output/LayerNorm", "encoder/transformer/group_0/inner_group_0/LayerNorm_1")
    name = name.replace("encoder/layer_shared/attention",        "encoder/transformer/group_0/inner_group_0/attention_1")

    name = name.replace("embeddings/word_embeddings_projector/projector",
                        "encoder/embedding_hidden_mapping_in/kernel")
    name = name.replace("embeddings/word_embeddings_projector/bias",
                        "encoder/embedding_hidden_mapping_in/bias")

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


def _is_tfhub_model(tfhub_model_path):
    try:
        assets_files     = tf.io.gfile.glob(os.path.join(tfhub_model_path, "assets/*"))
        variables_files  = tf.io.gfile.glob(os.path.join(tfhub_model_path, "variables/variables.*"))
        pb_files = tf.io.gfile.glob(os.path.join(tfhub_model_path, "*.pb"))
    except tf.errors.NotFoundError:
        assets_files, variables_files, pb_files = [], [], []

    return len(pb_files) >= 2 and len(assets_files) >= 1 and len(variables_files) >= 2


def load_albert_weights(bert: BertModelLayer, tfhub_model_path, tags=[]):
    """
    Use this method to load the weights from a pre-trained BERT checkpoint into a bert layer.

    :param bert: a BertModelLayer instance within a built keras model.
    :param ckpt_path: checkpoint path, i.e. `uncased_L-12_H-768_A-12/bert_model.ckpt` or `albert_base_zh/albert_model.ckpt`
    :return: list of weights with mismatched shapes. This can be used to extend
    the segment/token_type embeddings.
    """
    if not _is_tfhub_model(tfhub_model_path):
        print("Loading brightmart/albert_zh weights...")
        return loader.load_stock_weights(bert, tfhub_model_path)

    assert isinstance(bert, BertModelLayer), "Expecting a BertModelLayer instance as first argument"
    prefix = loader.bert_prefix(bert)

    with tf.Graph().as_default():
        sm = tf.compat.v2.saved_model.load(tfhub_model_path, tags=tags)
        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            stock_values = {v.name.split(":")[0]: v.read_value() for v in sm.variables}
            stock_values = sess.run(stock_values)

    # print("\n".join([str((n, v.shape)) for n,v in stock_values.items()]))

    loaded_weights = set()
    skip_count = 0
    weight_value_tuples = []
    skipped_weight_value_tuples = []

    bert_params = bert.weights
    param_values = keras.backend.batch_get_value(bert.weights)
    for ndx, (param_value, param) in enumerate(zip(param_values, bert_params)):
        stock_name = map_to_tfhub_albert_variable_name(param.name, prefix)

        if stock_name in stock_values:
            ckpt_value = stock_values[stock_name]

            if param_value.shape != ckpt_value.shape:
                print("loader: Skipping weight:[{}] as the weight shape:[{}] is not compatible "
                      "with the checkpoint:[{}] shape:{}".format(param.name, param.shape,
                                                                 stock_name, ckpt_value.shape))
                skipped_weight_value_tuples.append((param, ckpt_value))
                continue

            weight_value_tuples.append((param, ckpt_value))
            loaded_weights.add(stock_name)
        else:
            print("loader: No value for:[{}], i.e.:[{}] in:[{}]".format(param.name, stock_name, tfhub_model_path))
            skip_count += 1
    keras.backend.batch_set_value(weight_value_tuples)

    print("Done loading {} BERT weights from: {} into {} (prefix:{}). "
          "Count of weights not found in the checkpoint was: [{}]. "
          "Count of weights with mismatched shape: [{}]".format(
              len(weight_value_tuples), tfhub_model_path, bert, prefix, skip_count, len(skipped_weight_value_tuples)))
    print("Unused weights from saved model:",
          "\n\t" + "\n\t".join(sorted(set(stock_values.keys()).difference(loaded_weights))))

    return skipped_weight_value_tuples  # (bert_weight, value_from_ckpt)

