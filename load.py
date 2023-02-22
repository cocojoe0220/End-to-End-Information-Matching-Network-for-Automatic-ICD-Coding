# -*- coding: utf-8 -*-
"""

@author: zgh

"""

from classifier_multi_label_seq2seq_attention.hyperparameters import Hyperparamters as hp


def label2onehot(string):
    string = '|' if string == '' else string
    string_list = list(str(string).split('/')) + ['E']
    return [int(hp.dict_label2id.get(l)) for l in string_list]


def normalization_label(label_ids):
    max_length = max([len(l) for l in label_ids])
    return [l + [0] * (max_length - len(l)) if len(l) < max_length else l for i, l in enumerate(label_ids)]
