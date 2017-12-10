import os, datetime
import numpy as np
import tensorflow as tf
import CNNModels
import CNNModels2

class vgg_seg1:
    def __init__(self, x, y, seg_labels, obj_class, lam, keep_dropout, train_phase):
        self.logits_class, logits_seg = CNNModels2.VGG_Seg1(x, keep_dropout, train_phase, num_classes=100, num_seg_classes=176, random_init_seg_score_fr=True, debug=True)
        self.loss_class =loss_class(seg_labels, self.logits_class)

def loss_class(y,logits):
    return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits))
