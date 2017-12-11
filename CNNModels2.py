import os
import logging
from math import ceil
import sys

import numpy as np
import tensorflow as tf 

from layers import *

def VGG_Seg1(x, keep_dropout, train_phase, num_classes, num_seg_classes, batch_norm=True, seg=False, random_init_seg_score_fr=False,
          debug=False):
    with tf.name_scope('Processing'):
        if debug:
            x = tf.Print(x, [tf.shape(x)],
                           message='Shape of input image: ',
                           summarize=4, first_n=1)

    conv1_1 = conv_layer(x, train_phase, "conv1_1",batch_norm)
    conv1_2 = conv_layer(conv1_1, train_phase, "conv1_2",batch_norm)
    pool1 = max_pool(conv1_2, 'pool1', debug)

    conv2_1 = conv_layer(pool1, train_phase, "conv2_1",batch_norm)
    conv2_2 = conv_layer(conv2_1, train_phase, "conv2_2",batch_norm)
    pool2 = max_pool(conv2_2, 'pool2', debug)

    conv3_1 = conv_layer(pool2, train_phase, "conv3_1",batch_norm)
    conv3_2 = conv_layer(conv3_1, train_phase, "conv3_2",batch_norm)
    conv3_3 = conv_layer(conv3_2, train_phase, "conv3_3",batch_norm)
    pool3 = max_pool(conv3_3, 'pool3', debug)

    conv4_1 = conv_layer(pool3, train_phase, "conv4_1",batch_norm)
    conv4_2 = conv_layer(conv4_1, train_phase, "conv4_2",batch_norm)
    conv4_3 = conv_layer(conv4_2, train_phase, "conv4_3",batch_norm)
    pool4 = max_pool(conv4_3, 'pool4', debug)

    conv5_1 = conv_layer(pool4, train_phase, "conv5_1",batch_norm)
    conv5_2 = conv_layer(conv5_1, train_phase, "conv5_2",batch_norm)
    conv5_3 = conv_layer(conv5_2, train_phase, "conv5_3",batch_norm)
    pool5 = max_pool(conv5_3, 'pool5', debug)


    # classification final 3 layers
    fc6_class = fc_layer(pool5, "fc6_class", "fc6", use="vgg")
    if batch_norm:
        fc6_class = batch_norm_layer(fc6_class, train_phase, 'bn6')
    fc6_class = tf.cond(train_phase, lambda: tf.nn.dropout(fc6_class, keep_dropout), lambda:fc6_class)

    fc7_class = fc_layer(fc6_class, "fc7_class", "fc7", use="vgg")
    if batch_norm:
        fc7_class = batch_norm_layer(fc7_class, train_phase, 'bn7')
    fc7_class = tf.cond(train_phase, lambda: tf.nn.dropout(fc7_class, keep_dropout), lambda:fc7_class)

    logits_class = fc_layer(fc7_class, "score_fr_class", "score_fr", num_classes=num_classes, relu=False, use="vgg")

    # segmentation final 3 layers
    fc6_seg = fc_layer(pool5, "fc6_seg", "fc6", use="seg")
    fc6_seg = tf.cond(train_phase, lambda: tf.nn.dropout(fc6_seg, keep_dropout), lambda:fc6_seg)


    fc7_seg = fc_layer(fc6_seg, "fc7_seg", "fc7", use="seg")
    fc7_seg = tf.cond(train_phase, lambda: tf.nn.dropout(fc7_seg, keep_dropout), lambda:fc7_seg)


    if random_init_seg_score_fr:
        logits_seg = score_layer(fc7_seg, "score_fr_seg",
                                          num_seg_classes)
    else:
        logits_seg = fc_layer(fc7_seg, "score_fr_seg", "score_fr",
                                       num_classes=num_seg_classes,
                                       relu=False)

    return logits_class, logits_seg
