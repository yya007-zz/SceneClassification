import os, datetime
import numpy as np
import tensorflow as tf
import CNNModels
import CNNModels2

class alexnet_model:
    def __init__(self, x, y, seg_labels, obj_class, keep_dropout, train_phase):
        self.logits_class=CNNModels.alexnet(x, keep_dropout, train_phase)
        self.loss_class =loss_class(y,self.logits_class)

class vgg_simple_model:
    def __init__(self, x, y, seg_labels, obj_class, keep_dropout, train_phase):
        self.logits_class=CNNModels.VGG16_Simple(x, keep_dropout, train_phase)
        self.loss_class =loss_class(y,logits_class)
        self.loss_seg = 0

class vgg_model:
    def __init__(self, x, y, seg_labels, obj_class, keep_dropout, train_phase):
        self.logits_class = CNNModels.VGG(x, keep_dropout, train_phase, num_classes=100)
        self.loss_class =loss_class(y,self.logits_class)
        self.loss_seg = 0

class vgg_bn_model:
    def __init__(self, x, y, seg_labels, obj_class, keep_dropout, train_phase):
        self.logits_class=CNNModels.VGG(x, keep_dropout, train_phase, num_classes=100, batch_norm=True)
        self.loss_class =loss_class(y,self.logits_class)
        self.loss_seg = 0

class vgg_seg2_model:
    def __init__(self, x, y, seg_labels, obj_class, keep_dropout, train_phase):
        self.logits_class,logits_seg=CNNModels.VGG(x, keep_dropout, train_phase, num_classes=100, seg=True, num_classes_seg=176)
        self.loss_class =loss_class(y,self.logits_class)
        self.loss_seg = loss_seg(obj_class,logits_seg)

class vgg_bn_seg2_model:
    def __init__(self, x, y, seg_labels, obj_class, keep_dropout, train_phase):
        self.logits_class,logits_seg=CNNModels.VGG(x, keep_dropout, train_phase,num_classes=100, batch_norm=True, seg=True, num_classes_seg=176)
        self.loss_class =loss_class(y,self.logits_class)
        self.loss_seg = loss_seg(obj_class,logits_seg)

        # self.obj_class = tf.nn.softmax(obj_class)
        # self.obj_class_ = tf.nn.softmax(logits_seg)

class vgg_bn_seg2_1_model:
    def __init__(self, x, y, seg_labels, obj_class, keep_dropout, train_phase):
        self.logits_class,logits_seg=CNNModels.VGG(x, keep_dropout, train_phase,num_classes=100, batch_norm=True, seg=True, seg_mode=1, num_classes_seg=176)
        self.loss_class =loss_class(y,self.logits_class)
        self.loss_seg = loss_seg(obj_class,logits_seg)
        
class vgg_bn_seg2_2_model:
    def __init__(self, x, y, seg_labels, obj_class, keep_dropout, train_phase):
        self.logits_class,logits_seg=CNNModels.VGG(x, keep_dropout, train_phase,num_classes=100, batch_norm=True, seg=True, seg_mode=2, num_classes_seg=176)
        self.loss_class =loss_class(y,self.logits_class)
        self.loss_seg = loss_seg(obj_class,logits_seg)

def loss_class(y,logits):
    return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits))

def loss_seg(y,logits):
    return loss_seg_l1(y,logits)
    # newy= y/ tf.reduce_sum(logits, -1)
    # newl= logits/ tf.reduce_sum(logits, -1)
    # newl= tf.nn.tanh(logits)
    # return tf.reduce_mean(tf.reduce_sum(tf.abs(newy-newl),-1))
    # return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=newy, logits=logits))

def loss_seg_en(y, logits):
    newy= tf.nn.softmax(label_seg)
    newl= tf.nn.softmax(logits_seg)
    return tf.reduce_mean(tf.reduce_sum(tf.abs(newy-newl),-1))

def loss_seg_l1(y, logits):
    newy= tf.nn.softmax(y)
    newl= tf.nn.softmax(logits_seg)
    return tf.reduce_mean(tf.reduce_sum(tf.abs(newy-newl),-1))