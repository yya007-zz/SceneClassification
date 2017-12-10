import os, datetime
import numpy as np
import tensorflow as tf
import CNNModels
import CNNModels2

class alexnet_model:
    def __init__(self, x, y, seg_labels, obj_class, lam, keep_dropout, train_phase):
        self.logits_class=CNNModels.alexnet(x, keep_dropout, train_phase)
        self.loss_class =loss_class(y,self.logits_class)
        self.loss=self.loss_class

class vgg_simple_model:
    def __init__(self, x, y, seg_labels, obj_class, lam, keep_dropout, train_phase):
        self.logits_class=CNNModels.VGG16_Simple(x, keep_dropout, train_phase)
        self.loss_class =loss_class(y,logits_class)
        self.loss_seg = 0
        self.loss=self.loss_class+lam*self.loss_seg

class vgg_model:
    def __init__(self, x, y, seg_labels, obj_class, lam, keep_dropout, train_phase):
        self.logits_class = CNNModels.VGG(x, keep_dropout, train_phase, num_classes=100)
        self.loss_class =loss_class(y,self.logits_class)
        self.loss_seg = 0
        self.loss=self.loss_class+lam*self.loss_seg

class vgg_bn_model:
    def __init__(self, x, y, seg_labels, obj_class, lam, keep_dropout, train_phase):
        self.logits_class=CNNModels.VGG(x, keep_dropout, train_phase, num_classes=100, batch_norm=True)
        self.loss_class =loss_class(y,self.logits_class)
        self.loss_seg = 0
        self.loss=self.loss_class+lam*self.loss_seg

class vgg_seg2_model:
    def __init__(self, x, y, seg_labels, obj_class, lam, keep_dropout, train_phase):
        self.logits_class,logits_seg=CNNModels.VGG(x, keep_dropout, train_phase, num_classes=100, seg=True, num_classes_seg=176)
        self.loss_class =loss_class(y,self.logits_class)
        self.loss_seg = loss_seg(obj_class,logits_seg)
        print("wwwwwww",self.loss_seg.get_shape().as_list()) 
        self.loss=self.loss_class+lam*self.loss_seg

class vgg_bn_seg2_model:
    def __init__(self, x, y, seg_labels, obj_class, lam, keep_dropout, train_phase):
        self.logits_class,logits_seg=CNNModels.VGG(x, keep_dropout, train_phase,num_classes=100, batch_norm=True, seg=True, num_classes_seg=176)
        self.loss_class =loss_class(y,self.logits_class)
        self.loss_seg = loss_seg(obj_class,logits_seg)

        self.loss=self.loss_class+lam*self.loss_seg

def loss_class(y,logits):
    return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits))

def loss_seg(y,logits):
    y=tf.nn.softmax(y)
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits))
