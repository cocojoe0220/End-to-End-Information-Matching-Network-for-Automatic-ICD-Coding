# -*- coding: utf-8 -*-
"""
Created on Thu May 30 17:12:37 2019

@author: cm
"""

import os

# os.environ["CUDA_VISIBLE_DEVICES"] = '-1'
pwd = os.path.dirname(os.path.abspath(__file__))
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import tensorflow as tf
from classifier_multi_label_seq2seq_attention.networks import NetworkAlbertSeq2Seq
from classifier_multi_label_seq2seq_attention.hyperparameters import Hyperparamters as hp
from classifier_multi_label_seq2seq_attention.classifier_utils import get_feature_test


class ModelAlbertSeq2seq(object, ):
    """
    Load Network Albert Seq2seq model
    """

    def __init__(self):
        self.albert, self.sess = self.load_model()

    @staticmethod
    def load_model():
        with tf.Graph().as_default():
            sess = tf.Session()
            with sess.as_default():
                albert = NetworkAlbertSeq2Seq(is_training=False)
                saver = tf.train.Saver()
                sess.run(tf.global_variables_initializer())
                checkpoint_dir = os.path.abspath(os.path.join(pwd, hp.file_load_model))
                ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
                saver.restore(sess, checkpoint_dir)
        return albert, sess


MODEL = ModelAlbertSeq2seq()
print('Load model finished!')


def get_label(sentence):
    """
    Prediction of the sentence's sentiment.
    """

    feature = get_feature_test(sentence)
    fd = {MODEL.albert.input_ids: [feature[0]],
          MODEL.albert.input_masks: [feature[1]],
          MODEL.albert.segment_ids: [feature[2]],
          }
    output = MODEL.sess.run(MODEL.albert.predictions, feed_dict=fd)
    return sorted([hp.dict_id2label[i] for i in output[0][:, 0] if i != 1])


def get_labels(sentences):
    """
    Prediction of some sentences's sentiment.
    """
    features = [get_feature_test(str(sentence)) for sentence in sentences]
    fd = {MODEL.albert.input_ids: [feature[0] for feature in features],
          MODEL.albert.input_masks: [feature[1] for feature in features],
          MODEL.albert.segment_ids: [feature[2] for feature in features]}
    outputs = MODEL.sess.run(MODEL.albert.predictions, feed_dict=fd)
    return [sorted([hp.dict_id2label[i] for i in output[:, 0] if i != 1]) for output in outputs]


if __name__ == '__main__':
    #f = open('data_preprocessing/test.csv', 'r', encoding='utf-8')
    f = open('zfb.csv', 'r', encoding='utf-8')
    lines = f.readlines()
    data = []
    labels_set = []
    for line in lines:
        line = line.replace('\n', '').split(',')
        data.append(line[0])
        labels = []
        for label in line[1].split('/'):
            labels.append(label)
        labels_set.append(labels)

    accuracy_set = []
    error_rate_set = []
    for index in range(1, len(data)):
        sentence = data[index]
        predict = get_label(sentence)
        labels = set(labels_set[index])
        print(sentence)
        print('原标签：')
        print(labels)
        print('模型推断标签：')
        print(set(predict))
        print('-' * 20)
        diff_between_predict_labels = set(predict).difference(labels)
        diff_between_labels_predict = labels.difference(set(predict))
        accuracy = (len(labels) - len(diff_between_labels_predict)) / len(labels)
        error_rate = len(diff_between_predict_labels) / len(predict)
        accuracy_set.append(accuracy)
        error_rate_set.append(error_rate)

    sum = 0
    for accuracy in accuracy_set:
        sum = sum + accuracy
    print('avg accuracy : ' + str(sum / len(accuracy_set)))

    sum = 0
    for error_rate in error_rate_set:
        sum = sum + error_rate
    print('avg error_rate : ' + str(sum / len(error_rate_set)))