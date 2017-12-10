import os, datetime
import numpy as np
import tensorflow as tf
import CNNModels
import CNNModels2

class vgg_seg1:
    def __init__(self, x, y, seg_labels, obj_class, lam, keep_dropout, train_phase):
        self.logits_class, self.logits_seg = CNNModels2.VGG_Seg1(x, keep_dropout, train_phase, num_classes=100, batch_norm=True, seg=True, num_seg_classes=176, random_init_seg_score_fr=True, debug=True)
        self.loss_seg = loss_seg(seg_labels, self.logits_seg)
        self.loss_class = loss_class(y, self.logits_class)
        self.loss = self.loss_class + lam * self.loss_seg

def loss_class(label_class, logits_class):
    return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label_class, logits=logits_class))

def loss_seg(label_seg, logits_seg):
    label_seg = tf.nn.softmax(label_seg)
    return tf.nn.softmax_cross_entropy_with_logits(labels=label_seg, logits=logits_seg)

