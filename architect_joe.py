import os, datetime
import numpy as np
import tensorflow as tf
import CNNModels_joe

class vgg_seg1:
    def __init__(self, x, seg_labels, keep_dropout, train_phase):
        self.logits_seg = CNNModels_joe.VGG_Seg1(x, keep_dropout, train_phase, num_seg_classes=176, batch_norm=True, seg=True, random_init_seg_score_fr=False, debug=True)
        #self.loss = loss_seg(seg_labels, self.logits_seg)
        self.loss = loss_seg_tanh(seg_labels, self.logits_seg)

def loss_seg_tanh(y, logits):
    newl= tf.nn.tanh(logits)
    return tf.reduce_mean(tf.reduce_sum(tf.abs(y-newl),-1))

def loss_seg(label_seg, logits_seg):
    newy= tf.nn.softmax(label_seg)
    newl= tf.nn.softmax(logits_seg)
    return tf.reduce_mean(tf.reduce_sum(tf.abs(newy-newl),-1))
    #label_seg = tf.nn.softmax(label_seg)
    #return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=label_seg, logits=logits_seg))

