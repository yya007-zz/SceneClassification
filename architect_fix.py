import os, datetime
import numpy as np
import tensorflow as tf
import CNNModels_fix

class vgg_seg1:
    def __init__(self, x, y, seg_labels, keep_dropout, train_phase):
        self.prob_seg_class, self.prob_pure_class, self.prob_class, self.logits_seg = CNNModels_fix.VGG_Seg1(x, keep_dropout, train_phase, debug=True)
        self.loss_seg = loss_seg_norm(seg_labels, self.logits_seg)
        self.loss_class = loss_class(y, self.prob_class)
        self.loss_seg_class = loss_class(y, self.prob_seg_class)
        self.loss_pure_class = loss_class(y, self.prob_pure_class)

class vgg_seg1_mask:
    def __init__(self, x, y, seg_labels, keep_dropout, train_phase):
        self.prob_seg_class, self.prob_pure_class, self.prob_class, self.logits_seg = CNNModels_fix.VGG_Seg1_Mask(x, keep_dropout, train_phase, debug=True)
        self.loss_seg = loss_seg_norm(seg_labels, self.logits_seg)
        self.loss_class = loss_class(y, self.prob_class)
        self.loss_seg_class = loss_class(y, self.prob_seg_class)
        self.loss_pure_class = loss_class(y, self.prob_pure_class)

def loss_seg_norm(y, logits):
    newy = y/tf.tile(tf.expand_dims(tf.reduce_sum(y,axis=-1), axis=-1), (1,1,1,176))
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=newy, logits=logits))

def loss_class(y, prob):
    newy = tf.one_hot(y, 100)
    return -tf.reduce_mean(tf.reduce_sum(newy * tf.log(prob + 1e-8), axis=-1))

#def loss_class(y, prob):
#    return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=prob))

