import os
import logging
from math import ceil
import sys

import numpy as np
import tensorflow as tf 

from layers import *

def VGG(x, keep_dropout, train_phase, num_classes, batch_norm=True, seg=False, seg_mode=0, num_classes_seg=176, debug=False):
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

    if not seg:
        fc6 = fc_layer(pool5, "fc6", "fc6", use="vgg")
        if batch_norm:
            fc6 = batch_norm_layer(fc6, train_phase, 'bn6')
        fc6 = tf.cond(train_phase,lambda: tf.nn.dropout(fc6, keep_dropout),lambda: fc6)
       
        fc7 = fc_layer(fc6, "fc7", "fc7", use="vgg")
        if batch_norm:
            fc7 = batch_norm_layer(fc7, train_phase, 'bn7')
        fc7 = tf.cond(train_phase,lambda: tf.nn.dropout(fc7, keep_dropout),lambda: fc7)

        logits_class = fc_layer(fc7, "score_fr", "score_fr", num_classes=num_classes,relu=False,use="vgg")
    
        return logits_class
    else:
        if seg_mode==1:
            fc6 = fc_layer(pool5, "fc6", "fc6", use="vgg")
            if batch_norm:
                fc6 = batch_norm_layer(fc6, train_phase, 'bn6')
            fc6 = tf.cond(train_phase,lambda: tf.nn.dropout(fc6, keep_dropout),lambda: fc6)
           
            fc7 = fc_layer(fc6, "fc7", "fc7", use="vgg")
            if batch_norm:
                fc7 = batch_norm_layer(fc7, train_phase, 'bn7')
            fc7 = tf.cond(train_phase,lambda: tf.nn.dropout(fc7, keep_dropout),lambda: fc7)

            logits_class = fc_layer(fc7, "score_fr", "score_fr", num_classes=num_classes,relu=False,use="vgg")
            logits_seg = fc_layer(fc7, "score_fr_2", "score_fr", num_classes=num_classes_seg,relu=False,use="vgg")
        
        elif seg_mode==2:
            fc8 = fc_layer(pool5, "fc8", "fc6", use="vgg")
            if batch_norm:
                fc8 = batch_norm_layer(fc8, train_phase, 'bn8')
            fc8 = tf.cond(train_phase,lambda: tf.nn.dropout(fc8, keep_dropout),lambda: fc8)
            
            fc9 = fc_layer(fc8, "fc9", "fc7", use="vgg")
            if batch_norm:
                fc9 = batch_norm_layer(fc9, train_phase, 'bn9')
            fc9 = tf.cond(train_phase,lambda: tf.nn.dropout(fc9, keep_dropout),lambda: fc9)

            logits_seg = fc_layer(fc9, "score_fr_2", "score_fr", num_classes=num_classes_seg,relu=False,use="vgg")
            
            fc6 = fc_layer(pool5, "fc6", "fc6", use="vgg")
            if batch_norm:
                fc6 = batch_norm_layer(fc6, train_phase, 'bn6')
            fc6 = tf.cond(train_phase,lambda: tf.nn.dropout(fc6, keep_dropout),lambda: fc6)
            
            fc6 = tf.add(fc6,fc8)
            
            fc7 = fc_layer(fc6, "fc7", "fc7", use="vgg")
            if batch_norm:
                fc7 = batch_norm_layer(fc7, train_phase, 'bn7')
            fc7 = tf.cond(train_phase,lambda: tf.nn.dropout(fc7, keep_dropout),lambda: fc7)

            logits_class = fc_layer(fc7, "score_fr", "score_fr", num_classes=num_classes,relu=False,use="vgg")

        else:
            fc8 = fc_layer(pool5, "fc8", "fc6", use="vgg")
            if batch_norm:
                fc8 = batch_norm_layer(fc8, train_phase, 'bn8')
            fc8 = tf.cond(train_phase,lambda: tf.nn.dropout(fc8, keep_dropout),lambda: fc8)
           
            fc9 = fc_layer(fc8, "fc9", "fc7", use="vgg")
            if batch_norm:
                fc9 = batch_norm_layer(fc9, train_phase, 'bn9')
            fc9 = tf.cond(train_phase,lambda: tf.nn.dropout(fc9, keep_dropout),lambda: fc9)

            logits_seg = fc_layer(fc9, "score_fr_2", "score_fr", num_classes=num_classes_seg,relu=False,use="vgg")


            fc6 = fc_layer(pool5, "fc6", "fc6", use="vgg")
            if batch_norm:
                fc6 = batch_norm_layer(fc6, train_phase, 'bn6')
            fc6 = tf.cond(train_phase,lambda: tf.nn.dropout(fc6, keep_dropout),lambda: fc6)
            
            fc7 = fc_layer(fc6, "fc7", "fc7", use="vgg")
            if batch_norm:
                fc7 = batch_norm_layer(fc7, train_phase, 'bn7')
            fc7 = tf.cond(train_phase,lambda: tf.nn.dropout(fc7, keep_dropout),lambda: fc7)

            logits_class = fc_layer(fc7, "score_fr", "score_fr", num_classes=num_classes,relu=False,use="vgg")

      
                   
        return logits_class,logits_seg

def alexnet(x, keep_dropout, train_phase):
    weights = {
        'wc1': tf.Variable(tf.random_normal([11, 11, 3, 96], stddev=np.sqrt(2./(11*11*3)))),
        'wc2': tf.Variable(tf.random_normal([5, 5, 96, 256], stddev=np.sqrt(2./(5*5*96)))),
        'wc3': tf.Variable(tf.random_normal([3, 3, 256, 384], stddev=np.sqrt(2./(3*3*256)))),
        'wc4': tf.Variable(tf.random_normal([3, 3, 384, 256], stddev=np.sqrt(2./(3*3*384)))),
        'wc5': tf.Variable(tf.random_normal([3, 3, 256, 256], stddev=np.sqrt(2./(3*3*256)))),

        'wf6': tf.Variable(tf.random_normal([7*7*256, 4096], stddev=np.sqrt(2./(7*7*256)))),
        'wf7': tf.Variable(tf.random_normal([4096, 4096], stddev=np.sqrt(2./4096))),
        'wo': tf.Variable(tf.random_normal([4096, 100], stddev=np.sqrt(2./4096)))
    }

    biases = {
        'bo': tf.Variable(tf.ones(100))
    }

    # Conv + ReLU + Pool, 224->55->27
    conv1 = tf.nn.conv2d(x, weights['wc1'], strides=[1, 4, 4, 1], padding='SAME')
    conv1 = batch_norm_layer(conv1, train_phase, 'bn1')
    conv1 = tf.nn.relu(conv1)
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

    # Conv + ReLU  + Pool, 27-> 13
    conv2 = tf.nn.conv2d(pool1, weights['wc2'], strides=[1, 1, 1, 1], padding='SAME')
    conv2 = batch_norm_layer(conv2, train_phase, 'bn2')
    conv2 = tf.nn.relu(conv2)
    pool2 = tf.nn.max_pool(conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

    # Conv + ReLU, 13-> 13
    conv3 = tf.nn.conv2d(pool2, weights['wc3'], strides=[1, 1, 1, 1], padding='SAME')
    conv3 = batch_norm_layer(conv3, train_phase, 'bn3')
    conv3 = tf.nn.relu(conv3)

    # Conv + ReLU, 13-> 13
    conv4 = tf.nn.conv2d(conv3, weights['wc4'], strides=[1, 1, 1, 1], padding='SAME')
    conv4 = batch_norm_layer(conv4, train_phase, 'bn4')
    conv4 = tf.nn.relu(conv4)

    # Conv + ReLU + Pool, 13->6
    conv5 = tf.nn.conv2d(conv4, weights['wc5'], strides=[1, 1, 1, 1], padding='SAME')
    conv5 = batch_norm_layer(conv5, train_phase, 'bn5')
    conv5 = tf.nn.relu(conv5)
    pool5 = tf.nn.max_pool(conv5, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

    # FC + ReLU + Dropout
    fc6 = tf.reshape(pool5, [-1, weights['wf6'].get_shape().as_list()[0]])
    fc6 = tf.matmul(fc6, weights['wf6'])
    fc6 = batch_norm_layer(fc6, train_phase, 'bn6')
    fc6 = tf.nn.relu(fc6)
    fc6 = tf.nn.dropout(fc6, keep_dropout)
    
    # FC + ReLU + Dropout
    fc7 = tf.matmul(fc6, weights['wf7'])
    fc7 = batch_norm_layer(fc7, train_phase, 'bn7')
    fc7 = tf.nn.relu(fc7)
    fc7 = tf.nn.dropout(fc7, keep_dropout)

    # Output FC
    out = tf.add(tf.matmul(fc7, weights['wo']), biases['bo'])
    
    return out

def VGG_SegNet(x, keep_dropout, train_phase, num_classes = 100, batch_norm=True, num_classes_seg=176, debug=False):
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
    prob_seg_class = tf.nn.softmax(logits_seg_class)

    # introduce mask
    # initialize mask
    dist = np.zeros(100)
    with open('./data/new_train.txt', 'r') as f:
        for line in f:
            dist[int(line.rsplit()[2])] += 1.
    seg_dist = dist / sum(dist)
    shifted_dist = (seg_dist - min(seg_dist)) / (max(seg_dist) - min(seg_dist)) * 3.5 - 3.5
    mask_init = tf.constant_initializer(value=shifted_dist, dtype=tf.float32)

    weight_mask = tf.get_variable(name="weight_mask_var", initializer=mask_init, shape=[100])
    weight_mask = tf.minimum(tf.maximum(tf.sigmoid(weight_mask), 0.01), 0.99, name="weight_mask")
    prob_class = prob_pure_class * (1. - weight_mask) + prob_seg_class * weight_mask

    return prob_class, logits_seg
