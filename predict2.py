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
    # Test
    '''
    sentences = ['全麻成功后，仰卧位，腰背部垫枕，消毒术区，铺盖无菌单，取右上腹肋下斜切口，入腹。无腹腔游离液体，探查盆腔、腹膜后、肠系膜、大网膜、前腹壁无肿大淋巴结和质硬肿物，肝脏结节样硬化改变，表面未触及明显占位，胆囊常大，肝十二指肠韧带无肿大淋巴结。离断肝圆韧带，断端结扎，离断肝镰状韧带至肝上下腔静脉前方，用术中超声检查明确肝S8段膈面靠近第二肝门处肝被膜下一个实体性肿物，直径约3厘米。依次游离右侧肝结肠、肝肾、三角和冠状韧带等，至右肝完全游离，处于术者左手掌控中，右肝后垫纱布，牵拉肝圆韧带断端，再次用术中超声检查全肝，未见其他占位病变，明确病灶部位，靠近肝右静脉主干，病灶周边脉管分支予以标记，考虑合并结节样肝硬化，遂决定行局部根治性切除。距离病灶边缘约2厘米标记约定切除线。肝十二指肠韧带环绕导尿管，预备阻断，肝针缝合牵拉肝组织，用双极电凝边切边凝，按预定切除线离断肝组织，遇有S8段供应肿瘤侧的脉管分支予以结扎后切断。逐渐分离至肝右静脉主干，肿瘤位于肝右静脉靠S8段一侧，将供应肿瘤的静脉分支血管，予以结扎后切断。沿着肝右静脉主干管壁细致分离，将肝肿物连同周围部分正常肝组织，完整切除。移出腹腔术区。检查肝创面，可见少量肝静脉侧壁渗血和周围肝组织渗血，静脉壁用无损伤线予以缝合止血，周围肝组织电凝止血，至肝创面无活动性出血和胆汁渗漏。右肝膈下靠近肝创面处和右肝下文氏孔附近各留置引流管一枚，分别引出体外并固定，查点纱布和器械无误，可吸收线缝合腹壁切口各层，关闭腹腔，术毕。手术顺利，麻醉满意，术中出血约20毫升，输注血浆400毫升，无输血反应，尿量300毫升。切除之标本剖开见肝肿物，黄白质中，边界清楚，无子灶，经家属过目送术后病理检查。未阻断入肝血流。术后病人安返病房。']
    for sentence in sentences:
        print(get_label(sentence))
    '''
    f = open('test.txt', 'r', encoding='utf8')
    lines = f.readlines()
    for line in lines:
        print(get_label(line))
    '''
    sentence='患者取俯卧位，后枕部肿物约0.6cm×0.5cm大小，沿肿物边缘画标记线，范围约0.8cm×0.7cm大小，2%碘伏常规消毒，铺无菌手术巾，2%利多卡因局部注射麻醉，麻醉满意后，沿标记线切除肿物，钝性剥离肿物，深达筋膜层，双极电凝止血。用4-0慕丝线逐层缝合，纱布覆盖、包扎。患者取仰卧位，右额部肿物约0.8cm×0.7cm大小，沿肿物边缘画标记线，范围约1.0cm×0.9cm大小，2%碘伏常规消毒，铺无菌手术巾，2%利多卡因局部注射麻醉，麻醉满意后，沿标记线切除肿物，钝性剥离肿物，深达筋膜层，双极电凝止血。用4-0慕丝线逐层缝合，纱布覆盖、包扎。患者取仰卧位，右眉部肿物约0.5cm×0.4cm大小，沿肿物边缘画标记线，范围约0.7cm×0.6cm大小，2%碘伏常规消毒，铺无菌手术巾，2%利多卡因局部注射麻醉，麻醉满意后，沿标记线切除肿物，钝性剥离肿物，深达筋膜层，双极电凝止血。用4-0慕丝线逐层缝合，纱布覆盖、包扎。患者取仰卧位，右下颌肿物约0.6cm×0.4cm大小，沿肿物边缘画标记线，范围约0.8cm×0.6cm大小，2%碘伏常规消毒，铺无菌手术巾，2%利多卡因局部注射麻醉，麻醉满意后，沿标记线切除肿物，钝性剥离肿物，深达筋膜层，双极电凝止血。用4-0慕丝线逐层缝合，纱布覆盖、包扎。患者取仰卧位，前胸部肿物约0.5cm×0.5cm大小，沿肿物边缘画标记线，范围约0.7cm×0.6cm大小，2%碘伏常规消毒，铺无菌手术巾，2%利多卡因局部注射麻醉，麻醉满意后，沿标记线切除肿物，钝性剥离肿物，深达筋膜层，双极电凝止血。用4-0慕丝线逐层缝合，纱布覆盖、包扎。患者取仰卧位，右腰部肿物约0.8cm×0.6cm大小，沿肿物边缘画标记线，范围约1.0cm×0.8cm大小，2%碘伏常规消毒，铺无菌手术巾，2%利多卡因局部注射麻醉，麻醉满意后，沿标记线切除肿物，钝性剥离肿物，深达筋膜层，双极电凝止血。用4-0慕丝线逐层缝合，纱布覆盖、包扎。患者取仰卧位，左上臂肿物约0.9cm×0.7cm大小，沿肿物边缘画标记线，范围约1.1cm×0.9cm大小，2%碘伏常规消毒，铺无菌手术巾，2%利多卡因局部注射麻醉，麻醉满意后，沿标记线切除肿物，钝性剥离肿物，深达筋膜层，双极电凝止血。用4-0慕丝线逐层缝合，纱布覆盖、包扎。确定包扎稳固且无出血后，送患者安返病房。术中出血量1ml输血量0ml输液量0ml切除组织标本经福尔马林浸泡后送病理检查术中麻醉满意，手术顺利，清点敷料和器械无误，术后病人清醒，安返病房。手术医师：李冰手签记录医师：王良民/孙靖手签'
    print(get_label(sentence))
    '''