import os
import logging
from math import ceil
import sys

import numpy as np
import tensorflow as tf 
from tensorflow.contrib.layers.python.layers import batch_norm

wd = 5e-4
data_dict = np.load("./data/pretrained/vgg16.npy", encoding='latin1').item()

def batch_norm_layer(x, train_phase, scope_bn):
    return batch_norm(x, decay=0.9, center=True, scale=True,
    updates_collections=None,
    is_training=train_phase,
    reuse=None,
    trainable=True,
    scope=scope_bn)

def VGG16_Simple(x,keep_dropout,train_phase,num_classes):
    # conv1_1
    with tf.name_scope('conv1_1') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 3, 64], dtype=tf.float32,
                                                 stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(x, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                             trainable=True, name='biases')
        out = tf.nn.bias_add(conv, biases)
        conv1_1 = tf.nn.relu(out, name=scope)

    # conv1_2
    with tf.name_scope('conv1_2') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 64, 64], dtype=tf.float32,
                                                 stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(conv1_1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                             trainable=True, name='biases')
        out = tf.nn.bias_add(conv, biases)
        conv1_2 = tf.nn.relu(out, name=scope)

    # pool1
    pool1 = tf.nn.max_pool(conv1_2,
                           ksize=[1, 2, 2, 1],
                           strides=[1, 2, 2, 1],
                           padding='SAME',
                           name='pool1')

    # conv2_1
    with tf.name_scope('conv2_1') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 64, 128], dtype=tf.float32,
                                                 stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32),
                             trainable=True, name='biases')
        out = tf.nn.bias_add(conv, biases)
        conv2_1 = tf.nn.relu(out, name=scope)

    # conv2_2
    with tf.name_scope('conv2_2') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 128, 128], dtype=tf.float32,
                                                 stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(conv2_1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32),
                             trainable=True, name='biases')
        out = tf.nn.bias_add(conv, biases)
        conv2_2 = tf.nn.relu(out, name=scope)

    # pool2
    pool2 = tf.nn.max_pool(conv2_2,
                           ksize=[1, 2, 2, 1],
                           strides=[1, 2, 2, 1],
                           padding='SAME',
                           name='pool2')

    # conv3_1
    with tf.name_scope('conv3_1') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 128, 256], dtype=tf.float32,
                                                 stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                             trainable=True, name='biases')
        out = tf.nn.bias_add(conv, biases)
        conv3_1 = tf.nn.relu(out, name=scope)

    # conv3_2
    with tf.name_scope('conv3_2') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32,
                                                 stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(conv3_1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                             trainable=True, name='biases')
        out = tf.nn.bias_add(conv, biases)
        conv3_2 = tf.nn.relu(out, name=scope)

    # conv3_3
    with tf.name_scope('conv3_3') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32,
                                                 stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(conv3_2, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                             trainable=True, name='biases')
        out = tf.nn.bias_add(conv, biases)
        conv3_3 = tf.nn.relu(out, name=scope)

    # pool3
    pool3 = tf.nn.max_pool(conv3_3,
                           ksize=[1, 2, 2, 1],
                           strides=[1, 2, 2, 1],
                           padding='SAME',
                           name='pool3')

    # conv4_1
    with tf.name_scope('conv4_1') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 512], dtype=tf.float32,
                                                 stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(pool3, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                             trainable=True, name='biases')
        out = tf.nn.bias_add(conv, biases)
        conv4_1 = tf.nn.relu(out, name=scope)

    # conv4_2
    with tf.name_scope('conv4_2') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                 stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(conv4_1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                             trainable=True, name='biases')
        out = tf.nn.bias_add(conv, biases)
        conv4_2 = tf.nn.relu(out, name=scope)

    # conv4_3
    with tf.name_scope('conv4_3') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                 stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(conv4_2, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                             trainable=True, name='biases')
        out = tf.nn.bias_add(conv, biases)
        conv4_3 = tf.nn.relu(out, name=scope)

    # pool4
    pool4 = tf.nn.max_pool(conv4_3,
                           ksize=[1, 2, 2, 1],
                           strides=[1, 2, 2, 1],
                           padding='SAME',
                           name='pool4')

    # conv5_1
    with tf.name_scope('conv5_1') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                 stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(pool4, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                             trainable=True, name='biases')
        out = tf.nn.bias_add(conv, biases)
        conv5_1 = tf.nn.relu(out, name=scope)

    # conv5_2
    with tf.name_scope('conv5_2') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                 stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(conv5_1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                             trainable=True, name='biases')
        out = tf.nn.bias_add(conv, biases)
        conv5_2 = tf.nn.relu(out, name=scope)

    # conv5_3
    with tf.name_scope('conv5_3') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                 stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(conv5_2, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                             trainable=True, name='biases')
        out = tf.nn.bias_add(conv, biases)
        conv5_3 = tf.nn.relu(out, name=scope)

    # pool5
    pool5 = tf.nn.max_pool(conv5_3,
                           ksize=[1, 2, 2, 1],
                           strides=[1, 2, 2, 1],
                           padding='SAME',
                           name='pool4')
    # fc1
    with tf.name_scope('fc1') as scope:
        shape = int(np.prod(pool5.get_shape()[1:]))
        fc1w = tf.Variable(tf.truncated_normal([shape, 4096],
                                                     dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
        fc1b = tf.Variable(tf.constant(1.0, shape=[4096], dtype=tf.float32),
                             trainable=True, name='biases')
        pool5_flat = tf.reshape(pool5, [-1, shape])
        fc1l = tf.nn.bias_add(tf.matmul(pool5_flat, fc1w), fc1b)
        fc1 = tf.nn.relu(fc1l)
        fc1 = tf.cond(train_phase,lambda: tf.nn.dropout(fc1, keep_dropout),lambda: fc1)

    # fc2
    with tf.name_scope('fc2') as scope:
        fc2w = tf.Variable(tf.truncated_normal([4096, 4096],
                                                     dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
        fc2b = tf.Variable(tf.constant(1.0, shape=[4096], dtype=tf.float32),
                             trainable=True, name='biases')
        fc2l = tf.nn.bias_add(tf.matmul(fc1, fc2w), fc2b)
        fc2 = tf.nn.relu(fc2l)
        fc2 = tf.cond(train_phase,lambda: tf.nn.dropout(fc2, keep_dropout),lambda: fc2)


    # fc3
    with tf.name_scope('fc3') as scope:
        fc3w = tf.Variable(tf.truncated_normal([4096, num_classes],
                                                     dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
        fc3b = tf.Variable(tf.constant(1.0, shape=[num_classes], dtype=tf.float32),
                             trainable=True, name='biases')
        fc3l = tf.nn.bias_add(tf.matmul(fc2, fc3w), fc3b)

    return fc3l

def VGG(x, keep_dropout, train_phase, num_classes, batch_norm=True, seg=False, num_classes_seg=0, debug=False):
    conv1_1 = _conv_layer(x, train_phase, "conv1_1",batch_norm)
    conv1_2 = _conv_layer(conv1_1, train_phase, "conv1_2",batch_norm)
    pool1 = _max_pool(conv1_2, 'pool1', debug)

    conv2_1 = _conv_layer(pool1, train_phase, "conv2_1",batch_norm)
    conv2_2 = _conv_layer(conv2_1, train_phase, "conv2_2",batch_norm)
    pool2 = _max_pool(conv2_2, 'pool2', debug)

    conv3_1 = _conv_layer(pool2, train_phase, "conv3_1",batch_norm)
    conv3_2 = _conv_layer(conv3_1, train_phase, "conv3_2",batch_norm)
    conv3_3 = _conv_layer(conv3_2, train_phase, "conv3_3",batch_norm)
    pool3 = _max_pool(conv3_3, 'pool3', debug)

    conv4_1 = _conv_layer(pool3, train_phase, "conv4_1",batch_norm)
    conv4_2 = _conv_layer(conv4_1, train_phase, "conv4_2",batch_norm)
    conv4_3 = _conv_layer(conv4_2, train_phase, "conv4_3",batch_norm)
    pool4 = _max_pool(conv4_3, 'pool4', debug)

    conv5_1 = _conv_layer(pool4, train_phase, "conv5_1",batch_norm)
    conv5_2 = _conv_layer(conv5_1, train_phase, "conv5_2",batch_norm)
    conv5_3 = _conv_layer(conv5_2, train_phase, "conv5_3",batch_norm)
    pool5 = _max_pool(conv5_3, 'pool5', debug)

    fc6 = _fc_layer(pool5, "fc6", use="vgg")
    if batch_norm:
        fc6 = batch_norm_layer(fc6, train_phase, 'bn6')
    fc6 = tf.cond(train_phase,lambda: tf.nn.dropout(fc6, keep_dropout),lambda: fc6)
   
    fc7 = _fc_layer(fc6, "fc7", use="vgg")
    if batch_norm:
        fc7 = batch_norm_layer(fc7, train_phase, 'bn7')
    fc7 = tf.cond(train_phase,lambda: tf.nn.dropout(fc7, keep_dropout),lambda: fc7)

    class_logits = _fc_layer(fc7, "score_fr",num_classes=num_classes,relu=False,use="vgg")

    logits_seg = None
    if seg:
        fc8 = _fc_layer(pool5, "fc8", use="vgg")
        fc8 = batch_norm_layer(fc8, train_phase, 'bn8')
        fc8 = tf.cond(train_phase,lambda: tf.nn.dropout(fc8, keep_dropout),lambda: fc8)
       
        fc9 = _fc_layer(fc9, "fc9", use="vgg")
        fc9 = batch_norm_layer(fc9, train_phase, 'bn9')
        fc9 = tf.cond(train_phase,lambda: tf.nn.dropout(fc9, keep_dropout),lambda: fc9)

        logits_seg = _fc_layer(fc9, "score_fr",num_classes=num_classes_seg,relu=False,use="vgg")

    return class_logits，logits_seg

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

    upscore5 = _upscore_layer(score_fr,
                                        shape=tf.shape(pool4),
                                        num_classes=num_classes,
                                        debug=debug, name='upscore5',
                                        ksize=4, stride=2)
                                        
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

def _max_pool( bottom, name, debug):
    pool = tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                          padding='SAME', name=name)

    if debug:
        pool = tf.Print(pool, [tf.shape(pool)],
                        message='Shape of %s' % name,
                        summarize=4, first_n=1)
    return pool

def _conv_layer( bottom, train_phase, name, batch_norm=False):
    with tf.variable_scope(name) as scope:
        filt = get_conv_filter(name)
        conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')

        if batch_norm:
            conv = batch_norm_layer(conv, train_phase, 'bn-'+name)
        conv_biases = get_bias(name)
        bias = tf.nn.bias_add(conv, conv_biases)

        relu = tf.nn.relu(bias)
        # Add summary to Tensorboard
        _activation_summary(relu)
        return relu

def _fc_layer( bottom, name, num_classes=None,
              relu=True, debug=False,use=""):
    with tf.variable_scope(name) as scope:
        
        if use=="vgg":
            shape = bottom.get_shape().as_list()
            dim = 1
            for d in shape[1:]:
                 dim *= d
            bottom = tf.reshape(bottom, [-1, dim])
            if name == 'fc6':
                filt = get_fc_weight(name)
            elif name == 'score_fr':
                name = 'fc8'  # Name of score_fr layer in VGG Model
                filt = get_fc_weight_reshape(name, [4096, 1000],num_classes=num_classes)
            else:
                filt = get_fc_weight(name)
            conv = tf.matmul(bottom, filt)
        else:
            shape = bottom.get_shape().as_list()

            if name == 'fc6':
                filt = get_fc_weight_reshape(name, [7, 7, 512, 4096])
            elif name == 'score_fr':
                name = 'fc8'  # Name of score_fr layer in VGG Model
                filt = get_fc_weight_reshape(name, [1, 1, 4096, 1000],
                                                  num_classes=num_classes)
            else:
                filt = get_fc_weight_reshape(name, [1, 1, 4096, 4096])

            _add_wd_and_summary(filt, wd, "fc_wlosses")
            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')

        conv_biases = get_bias(name, num_classes=num_classes)
        bias = tf.nn.bias_add(conv, conv_biases)

        if relu:
            bias = tf.nn.relu(bias)
        _activation_summary(bias)

        if debug:
            bias = tf.Print(bias, [tf.shape(bias)],
                            message='Shape of %s' % name,
                            summarize=4, first_n=1)
        return bias

def _score_layer( bottom, name, num_classes):
    with tf.variable_scope(name) as scope:
        # get number of input channels
        in_features = bottom.get_shape()[3].value
        #print name,bottom.get_shape().as_list()
        shape = [1, 1, in_features, num_classes]
        # He initialization Sheme
        if name == "score_fr":
            num_input = in_features
            stddev = (2 / num_input)**0.5
        #elif name == "score_pool4":
        #    stddev = 0.001
        #elif name == "score_pool3":
        #    stddev = 0.0001
        else:
            stddev = 0.001
        # Apply convolution
        w_decay = wd

        weights = _variable_with_weight_decay(shape, stddev, w_decay,
                                                   decoder=True)
        conv = tf.nn.conv2d(bottom, weights, [1, 1, 1, 1], padding='SAME')
        # Apply bias
        conv_biases = _bias_variable([num_classes], constant=0.0)
        bias = tf.nn.bias_add(conv, conv_biases)

        _activation_summary(bias)

        return bias

def _upscore_layer( bottom, shape,
                   num_classes, name, debug,
                   ksize=4, stride=2):
    strides = [1, stride, stride, 1]
    with tf.variable_scope(name):
        in_features = bottom.get_shape()[3].value

        if shape is None:
            # Compute shape out of Bottom
            in_shape = tf.shape(bottom)

            h = ((in_shape[1] - 1) * stride) + 1
            w = ((in_shape[2] - 1) * stride) + 1
            new_shape = [in_shape[0], h, w, num_classes]
        else:
            new_shape = [shape[0], shape[1], shape[2], num_classes]
        output_shape = tf.stack(new_shape)

        logging.debug("Layer: %s, Fan-in: %d" % (name, in_features))
        f_shape = [ksize, ksize, num_classes, in_features]

        # create
        num_input = ksize * ksize * in_features / stride
        stddev = (2 / num_input)**0.5

        weights = get_deconv_filter(f_shape)
        _add_wd_and_summary(weights, wd, "fc_wlosses")
        deconv = tf.nn.conv2d_transpose(bottom, weights, output_shape,
                                        strides=strides, padding='SAME')

        if debug:
            deconv = tf.Print(deconv, [tf.shape(deconv)],
                              message='Shape of %s' % name,
                              summarize=4, first_n=1)

    _activation_summary(deconv)
    return deconv

def get_deconv_filter( f_shape):
    width = f_shape[0]
    heigh = f_shape[0]
    f = ceil(width/2.0)
    c = (2 * f - 1 - f % 2) / (2.0 * f)
    bilinear = np.zeros([f_shape[0], f_shape[1]])
    for x in range(width):
        for y in range(heigh):
            value = (1 - abs(x / f - c)) * (1 - abs(y / f - c))
            bilinear[x, y] = value
    weights = np.zeros(f_shape)
    for i in range(f_shape[2]):
        weights[:, :, i, i] = bilinear

    init = tf.constant_initializer(value=weights,
                                   dtype=tf.float32)
    var = tf.get_variable(name="up_filter", initializer=init,
                          shape=weights.shape)
    return var

def get_conv_filter( name):
    init = tf.constant_initializer(value=data_dict[name][0],
                                   dtype=tf.float32)
    shape = data_dict[name][0].shape
    #print('Layer name: %s' % name)
    #print('Layer shape: %s' % str(shape))
    var = tf.get_variable(name="filter", initializer=init, shape=shape)
    if not tf.get_variable_scope().reuse:

        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd,name='weight_loss')
        #weight_decay = tf.mul(tf.nn.l2_loss(var), wd,name='weight_loss')
        tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES,
                             weight_decay)
    _variable_summaries(var)
    return var

def get_bias( name, num_classes=None):
    bias_wights = data_dict[name][1]
    shape = data_dict[name][1].shape
    if name == 'fc8':
        bias_wights = _bias_reshape(bias_wights, shape[0],
                                         num_classes)
        shape = [num_classes]
    init = tf.constant_initializer(value=bias_wights,
                                   dtype=tf.float32)
    var = tf.get_variable(name="biases", initializer=init, shape=shape)
    _variable_summaries(var)
    return var

def get_fc_weight( name):
    init = tf.constant_initializer(value=data_dict[name][0],
                                   dtype=tf.float32)
    shape = data_dict[name][0].shape
    var = tf.get_variable(name="weights", initializer=init, shape=shape)
    if not tf.get_variable_scope().reuse:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd,
                                   name='weight_loss')
        tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES,
                             weight_decay)
    _variable_summaries(var)
    return var

def _bias_reshape( bweight, num_orig, num_new):
    """ Build bias weights for filter produces with `_summary_reshape`
    """
    n_averaged_elements = num_orig//num_new
    avg_bweight = np.zeros(num_new)
    for i in range(0, num_orig, n_averaged_elements):
        start_idx = i
        end_idx = start_idx + n_averaged_elements
        avg_idx = start_idx//n_averaged_elements
        if avg_idx == num_new:
            break
        avg_bweight[avg_idx] = np.mean(bweight[start_idx:end_idx])
    return avg_bweight

def _summary_reshape( fweight, shape, num_new):
    """ Produce weights for a reduced fully-connected layer.
    FC8 of VGG produces 1000 classes. Most semantic segmentation
    task require much less classes. This reshapes the original weights
    to be used in a fully-convolutional layer which produces num_new
    classes. To archive this the average (mean) of n adjanced classes is
    taken.
    Consider reordering fweight, to perserve semantic meaning of the
    weights.
    Args:
      fweight: original weights
      shape: shape of the desired fully-convolutional layer
      num_new: number of new classes
    Returns:
      Filter weights for `num_new` classes.
    """
    lastdim=len(shape)-1
    num_orig = shape[lastdim]
    shape[lastdim] = num_new
    assert(num_new < num_orig)
    n_averaged_elements = num_orig//num_new
    avg_fweight = np.zeros(shape)
    for i in range(0, num_orig, n_averaged_elements):
        start_idx = i
        end_idx = start_idx + n_averaged_elements
        avg_idx = start_idx//n_averaged_elements
        if avg_idx == num_new:
            break
        if lastdim==3:
            avg_fweight[:, :, :, avg_idx] = np.mean(fweight[:, :, :, start_idx:end_idx], axis=3)
        if lastdim==1:
            avg_fweight[:, avg_idx] = np.mean(fweight[:, start_idx:end_idx], axis=1)
    return avg_fweight

def _variable_with_weight_decay( shape, stddev, wd, decoder=False):
    """Helper to create an initialized Variable with weight decay.
    Note that the Variable is initialized with a truncated normal
    distribution.
    A weight decay is added only if one is specified.
    Args:
      name: name of the variable
      shape: list of ints
      stddev: standard deviation of a truncated Gaussian
      wd: add L2Loss weight decay multiplied by this float. If None, weight
          decay is not added for this Variable.
    Returns:
      Variable Tensor
    """

    initializer = tf.truncated_normal_initializer(stddev=stddev)
    var = tf.get_variable('weights', shape=shape,
                          initializer=initializer)

    collection_name = tf.GraphKeys.REGULARIZATION_LOSSES
    if wd and (not tf.get_variable_scope().reuse):
        weight_decay = tf.multiply(
            tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection(collection_name, weight_decay)
    _variable_summaries(var)
    return var

def _add_wd_and_summary( var, wd, collection_name=None):
    if collection_name is None:
        collection_name = tf.GraphKeys.REGULARIZATION_LOSSES
    if wd and (not tf.get_variable_scope().reuse):
        weight_decay = tf.multiply(
            tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection(collection_name, weight_decay)
    _variable_summaries(var)
    return var

def _bias_variable( shape, constant=0.0):
    initializer = tf.constant_initializer(constant)
    var = tf.get_variable(name='biases', shape=shape,
                          initializer=initializer)
    _variable_summaries(var)
    return var

def get_fc_weight_reshape( name, shape, num_classes=None):
    #print('Layer name: %s' % name)
    #print('Layer shape: %s' % shape)
    weights = data_dict[name][0]
    weights = weights.reshape(shape)
    if num_classes is not None:
        weights = _summary_reshape(weights, shape,
                                        num_new=num_classes)
    init = tf.constant_initializer(value=weights,
                                   dtype=tf.float32)
    var = tf.get_variable(name="weights", initializer=init, shape=shape)
    return var


def _activation_summary(x):
    """Helper to create summaries for activations.
    Creates a summary that provides a histogram of activations.
    Creates a summary that measure the sparsity of activations.
    Args:
      x: Tensor
    Returns:
      nothing
    """
    # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
    # session. This helps the clarity of presentation on tensorboard.
    tensor_name = x.op.name
    # tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
    tf.summary.histogram(tensor_name + '/activations', x)
    tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))


def _variable_summaries(var):
    """Attach a lot of summaries to a Tensor."""
    if not tf.get_variable_scope().reuse:
        name = var.op.name
        logging.info("Creating Summary for: %s" % name)
        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(var)
            tf.summary.scalar(name + '/mean', mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_sum(tf.square(var - mean)))
            tf.summary.scalar(name + '/sttdev', stddev)
            tf.summary.scalar(name + '/max', tf.reduce_max(var))
            tf.summary.scalar(name + '/min', tf.reduce_min(var))
            tf.summary.histogram(name, var)
