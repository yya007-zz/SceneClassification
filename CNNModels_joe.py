import os
import logging
from math import ceil
import sys

import numpy as np
import tensorflow as tf 

from layers import *


def VGG_Seg1(x, keep_dropout, train_phase, num_classes = 100, batch_norm=True, num_classes_seg=176, debug=False):
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

    # pure classification part
    fc6 = fc_layer(pool5, "fc6", "fc6", use="vgg")
    if batch_norm:
        fc6 = batch_norm_layer(fc6, train_phase, 'bn6')
    fc6 = tf.cond(train_phase,lambda: tf.nn.dropout(fc6, keep_dropout),lambda: fc6)
   
    fc7 = fc_layer(fc6, "fc7", "fc7", use="vgg")
    if batch_norm:
        fc7 = batch_norm_layer(fc7, train_phase, 'bn7')
    fc7 = tf.cond(train_phase,lambda: tf.nn.dropout(fc7, keep_dropout),lambda: fc7)

    logits_pure_class = fc_layer(fc7, "score_pure_class", "score_fr", num_classes=num_classes,relu=False,use="vgg")
    prob_pure_class = tf.nn.softmax(logits_pure_class)

    # segmentation part of network
    fc6_seg = fc_layer(pool5, "fc6_seg", "fc6", use="seg")
    fc6_seg = tf.cond(train_phase, lambda: tf.nn.dropout(fc6_seg, keep_dropout), lambda:fc6_seg)


    fc7_seg = fc_layer(fc6_seg, "fc7_seg", "fc7", use="seg")
    fc7_seg = tf.cond(train_phase, lambda: tf.nn.dropout(fc7_seg, keep_dropout), lambda:fc7_seg)


    logits_seg = score_layer(fc7_seg, "score_fr_seg",
                                      num_classes_seg)

    #forking part from seg to classification
    logits_seg_class = rand_init_fc_layer(fc7_seg, "score_seg_class", 100)
    print "seg class output: ", logits_seg_class.get_shape().as_list()
    prob_seg_class = tf.nn.softmax(logits_seg_class)
    prob_class = tf.add(prob_pure_class, prob_seg_class) / 2.

    return prob_class, logits_seg
    #return prob_pure_class, logits_seg
    #return logits_pure_class, logits_seg

def VGG_Seg1_Mask(x, keep_dropout, train_phase, num_classes = 100, batch_norm=True, num_classes_seg=176, debug=False):
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

    # pure classification part
    fc6 = fc_layer(pool5, "fc6", "fc6", use="vgg")
    if batch_norm:
        fc6 = batch_norm_layer(fc6, train_phase, 'bn6')
    fc6 = tf.cond(train_phase,lambda: tf.nn.dropout(fc6, keep_dropout),lambda: fc6)
   
    fc7 = fc_layer(fc6, "fc7", "fc7", use="vgg")
    if batch_norm:
        fc7 = batch_norm_layer(fc7, train_phase, 'bn7')
    fc7 = tf.cond(train_phase,lambda: tf.nn.dropout(fc7, keep_dropout),lambda: fc7)

    logits_pure_class = fc_layer(fc7, "score_pure_class", "score_fr", num_classes=num_classes,relu=False,use="vgg")
    prob_pure_class = tf.nn.softmax(logits_pure_class)

    # segmentation part of network
    fc6_seg = fc_layer(pool5, "fc6_seg", "fc6", use="seg")
    fc6_seg = tf.cond(train_phase, lambda: tf.nn.dropout(fc6_seg, keep_dropout), lambda:fc6_seg)


    fc7_seg = fc_layer(fc6_seg, "fc7_seg", "fc7", use="seg")
    fc7_seg = tf.cond(train_phase, lambda: tf.nn.dropout(fc7_seg, keep_dropout), lambda:fc7_seg)


    logits_seg = score_layer(fc7_seg, "score_fr_seg",
                                      num_classes_seg)

    #forking part from seg to classification
    logits_seg_class = rand_init_fc_layer(fc7_seg, "score_seg_class", 100)
    print "seg class output: ", logits_seg_class.get_shape().as_list()
    prob_seg_class = tf.nn.softmax(logits_seg_class)

    #introduce mask
    seg_dist = dist('./data/new_train.txt')
    shifted_dist = (seg_dist - min(seg_dist)) / (max(seg_dist) - min(seg_dist)) * 2. - 2.
    mask_init = tf.constant_initializer(value=shifted_dist, dtype=tf.float32)

    weight_mask = tf.get_variable(name="weight_mask_var", initializer=mask_init, shape=[100])
    weight_mask = tf.minimum(tf.maximum(tf.sigmoid(weight_mask), 0.01), 0.99, name="weight_mask")
    prob_class = prob_pure_class * (1. - weight_mask) + prob_seg_class * weight_mask

    return prob_class, logits_seg

def VGG_Seg1_Fix_Mask(x, keep_dropout, train_phase, num_classes = 100, batch_norm=True, num_classes_seg=176, debug=False):
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

    # pure classification part
    fc6 = fc_layer(pool5, "fc6", "fc6", use="vgg")
    if batch_norm:
        fc6 = batch_norm_layer(fc6, train_phase, 'bn6')
    fc6 = tf.cond(train_phase,lambda: tf.nn.dropout(fc6, keep_dropout),lambda: fc6)
   
    fc7 = fc_layer(fc6, "fc7", "fc7", use="vgg")
    if batch_norm:
        fc7 = batch_norm_layer(fc7, train_phase, 'bn7')
    fc7 = tf.cond(train_phase,lambda: tf.nn.dropout(fc7, keep_dropout),lambda: fc7)

    logits_pure_class = fc_layer(fc7, "score_pure_class", "score_fr", num_classes=num_classes,relu=False,use="vgg")
    prob_pure_class = tf.nn.softmax(logits_pure_class)

    # segmentation part of network
    fc6_seg = fc_layer(pool5, "fc6_seg", "fc6", use="seg")
    fc6_seg = tf.cond(train_phase, lambda: tf.nn.dropout(fc6_seg, keep_dropout), lambda:fc6_seg)


    fc7_seg = fc_layer(fc6_seg, "fc7_seg", "fc7", use="seg")
    fc7_seg = tf.cond(train_phase, lambda: tf.nn.dropout(fc7_seg, keep_dropout), lambda:fc7_seg)


    logits_seg = score_layer(fc7_seg, "score_fr_seg",
                                      num_classes_seg)

    #forking part from seg to classification
    logits_seg_class = rand_init_fc_layer(fc7_seg, "score_seg_class", 100)
    print "seg class output: ", logits_seg_class.get_shape().as_list()
    prob_seg_class = tf.nn.softmax(logits_seg_class)

    #introduce mask
    seg_dist = dist('./data/new_train.txt')
    shifted_dist = (seg_dist - min(seg_dist)) / (max(seg_dist) - min(seg_dist)) * 0.5
    weight_mask = tf.constant(shift_dist, name="weight_mask")
    prob_class = prob_pure_class * (1. - weight_mask) + prob_seg_class * weight_mask

    return prob_class, logits_seg

def dist(filename):
    dist = np.zeros(100)
    with open(filename, 'r') as f:
        for line in f:
            dist[int(line.rsplit()[2])] += 1.
    dist = dist / sum(dist)
    return dist
