import os
import logging
from math import ceil
import sys

import numpy as np
import tensorflow as tf 

import layers

def FCN(x, keep_prob, train_phase, num_classes, random_init_fc8=False,
          debug=False):
    with tf.name_scope('Processing'):
        if debug:
            x = tf.Print(x, [tf.shape(x)],
                           message='Shape of input image: ',
                           summarize=4, first_n=1)

    conv1_1 = _conv_layer(x, "conv1_1")
    conv1_2 = _conv_layer(conv1_1, "conv1_2")
    pool1 = _max_pool(conv1_2, 'pool1', debug)

    conv2_1 = _conv_layer(pool1, "conv2_1")
    conv2_2 = _conv_layer(conv2_1, "conv2_2")
    pool2 = _max_pool(conv2_2, 'pool2', debug)

    conv3_1 = _conv_layer(pool2, "conv3_1")
    conv3_2 = _conv_layer(conv3_1, "conv3_2")
    conv3_3 = _conv_layer(conv3_2, "conv3_3")
    pool3 = _max_pool(conv3_3, 'pool3', debug)

    conv4_1 = _conv_layer(pool3, "conv4_1")
    conv4_2 = _conv_layer(conv4_1, "conv4_2")
    conv4_3 = _conv_layer(conv4_2, "conv4_3")
    pool4 = _max_pool(conv4_3, 'pool4', debug)

    conv5_1 = _conv_layer(pool4, "conv5_1")
    conv5_2 = _conv_layer(conv5_1, "conv5_2")
    conv5_3 = _conv_layer(conv5_2, "conv5_3")
    pool5 = _max_pool(conv5_3, 'pool5', debug)

    fc6 = _fc_layer(pool5, "fc6")

    if train_phase:
        fc6 = tf.nn.dropout(fc6, keep_prob)

    fc7 = _fc_layer(fc6, "fc7")
    if train_phase:
        fc7 = tf.nn.dropout(fc7, keep_prob)

    if random_init_fc8:
        score_fr = _score_layer(fc7, "score_fr",
                                          num_classes)
    else:
        score_fr = _fc_layer(fc7, "score_fr",
                                       num_classes=num_classes,
                                       relu=False)

    # pred = tf.argmax(score_fr, dimension=3)

    print score_fr.get_shape().aslist()
    upscore5 = _upscore_layer(score_fr,
                                        shape=tf.shape(pool4),
                                        num_classes=num_classes,
                                        debug=debug, name='upscore5',
                                        ksize=4, stride=2)
    print upscore5.get_shape().aslist()
                                        
    score_pool4 = _score_layer(pool4, "score_pool4",
                                         num_classes=num_classes)
    fuse_pool4 = tf.add(upscore5, score_pool4)

    upscore4 = _upscore_layer(fuse_pool4,
                                        shape=tf.shape(pool3),
                                        num_classes=num_classes,
                                        debug=debug, name='upscore4',
                                        ksize=4, stride=2)
    
    score_pool3 = _score_layer(pool3, "score_pool3",
                                         num_classes=num_classes)
    fuse_pool3 = tf.add(upscore4, score_pool3)

    upscore3 = _upscore_layer(fuse_pool3,
                                         shape=tf.shape(pool2),
                                         num_classes=num_classes,
                                         debug=debug, name='upscore3',
                                         ksize=4, stride=2)
    score_pool2 = _score_layer(pool2, "score_pool2",
                                         num_classes=num_classes)
    fuse_pool2 = tf.add(upscore3, score_pool2)

    upscore2 = _upscore_layer(fuse_pool2,
                                         shape=tf.shape(pool1),
                                         num_classes=num_classes,
                                         debug=debug, name='upscore2',
                                         ksize=4, stride=2)
    score_pool1 = _score_layer(pool1, "score_pool1",num_classes=num_classes)
    
    fuse_pool1 = tf.add(upscore2, score_pool1)
    

    upscore1 = _upscore_layer(fuse_pool1,
                                         shape=tf.shape(bgr),
                                         num_classes=num_classes,
                                         debug=debug, name='upscore1',
                                         ksize=4, stride=2)
                                         
    # pred_up = tf.argmax(upscore1, dimension=3)
    return upscore1

