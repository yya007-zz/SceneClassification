import os, datetime
import numpy as np
import tensorflow as tf
import CNNModels_joe
#from architect import *

class vgg_seg1:
    def __init__(self, x, seg_labels, keep_dropout, train_phase):
        self.logits_seg = CNNModels_joe.VGG_Seg1(x, keep_dropout, train_phase, num_seg_classes=176, batch_norm=True, seg=True, random_init_seg_score_fr=False, debug=True)
        #self.loss = loss_seg(seg_labels, self.logits_seg)
        self.loss = loss_seg_norm(seg_labels, self.logits_seg)

def loss_seg_norm(y, logits):
    newy = y/tf.tile(tf.expand_dims(tf.reduce_sum(y,axis=-1), axis=-1), (1,1,1,176))
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=newy, logits=logits))
